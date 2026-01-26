"""
Diagnose why banditGPT-Hybrid has non-monotonic behavior.
We'll run multiple trials for λ=1.0 and λ=2.0 to see variance.
"""
import sys
from pathlib import Path
import json
import gzip
import joblib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import CorrallingRouter
from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    DEFAULT_MODEL_REGISTRY_PATH,
)

def load_model_costs():
    """Load model costs from models.json."""
    with open(DEFAULT_MODEL_REGISTRY_PATH, 'r') as f:
        data = json.load(f)
    
    costs = {}
    for model_data in data.get("models", []):
        input_cost = model_data.get("price_1m_input", 0) / 1_000_000 * 100
        output_cost = model_data.get("price_1m_output", 0) / 1_000_000 * 200
        total_cost = input_cost + output_cost
        costs[model_data["openrouter_id"]] = {
            "name": model_data["name"],
            "cost": total_cost
        }
    return costs

def load_holdout_data():
    """Load holdout data."""
    prompt_rewards = defaultdict(lambda: {})
    with gzip.open(CANONICAL_HOLDOUT_DATA_PATH, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt = entry["prompt"]
                model_id = entry["model_id"]
                score = entry.get("raw_score", 0.0)
                prompt_rewards[prompt][model_id] = score
    
    filtered_data = []
    for prompt, rewards in prompt_rewards.items():
        if len(rewards) == 2:
            filtered_data.append({"prompt": prompt, "rewards": rewards})
    return filtered_data

class CostAwareLinUCBRouter(SimpleLinUCBRouter):
    def __init__(self, models, warmup_priors, model_costs, alpha=1.0, cost_penalty=0.0):
        super().__init__(models, warmup_priors, alpha)
        self.model_costs = model_costs
        self.cost_penalty = cost_penalty
        
        all_costs = [mc["cost"] for mc in model_costs.values()]
        min_cost = min(all_costs)
        max_cost = max(all_costs)
        self.cost_range = max_cost - min_cost
        
        self.normalized_costs = {}
        for model_id, mc in model_costs.items():
            if self.cost_range > 0:
                self.normalized_costs[model_id] = (mc["cost"] - min_cost) / self.cost_range
            else:
                self.normalized_costs[model_id] = 0.0

    def select_model(self, context):
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            normalized_cost = self.normalized_costs.get(model, 0.0)
            ucb_scores[model] = expected_reward - self.cost_penalty * normalized_cost + self.alpha * uncertainty
        return max(ucb_scores, key=ucb_scores.get)

class CostAwareTabulaRasaRouter:
    def __init__(self, models, context_dim, model_costs, alpha=1.0, cost_penalty=0.0):
        self.models = models
        self.alpha = alpha
        self.A = {m: np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
        self.model_costs = model_costs
        self.cost_penalty = cost_penalty

        all_costs = [mc["cost"] for mc in model_costs.values()]
        min_cost = min(all_costs)
        max_cost = max(all_costs)
        self.cost_range = max_cost - min_cost
        
        self.normalized_costs = {}
        for model_id, mc in model_costs.items():
            if self.cost_range > 0:
                self.normalized_costs[model_id] = (mc["cost"] - min_cost) / self.cost_range
            else:
                self.normalized_costs[model_id] = 0.0

    def select_model(self, context):
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            normalized_cost = self.normalized_costs.get(model, 0.0)
            ucb_scores[model] = expected_reward - self.cost_penalty * normalized_cost + self.alpha * uncertainty
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, context, model, reward):
        context = context.reshape(-1, 1)
        self.A[model] += context @ context.T
        self.b[model] += reward * context.flatten()

def run_bandit_trial(eval_data, encoder, pca, warmup_priors, model_costs, 
                     learning_rate, cost_penalty, gamma, seed):
    """Run a single bandit trial with given parameters."""
    np.random.seed(seed)
    
    models = list(eval_data[0]["rewards"].keys())
    scaled_priors = apply_gamma_scaling(warmup_priors, gamma)
    context_dim = scaled_priors["A"][list(scaled_priors["A"].keys())[0]].shape[0]
    
    warmup_expert = CostAwareLinUCBRouter(
        models=models,
        warmup_priors=scaled_priors,
        model_costs=model_costs,
        alpha=1.0,
        cost_penalty=cost_penalty
    )
    
    tabula_rasa_expert = CostAwareTabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        model_costs=model_costs,
        alpha=1.0,
        cost_penalty=cost_penalty
    )
    
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=learning_rate
    )
    
    total_reward = 0.0
    total_cost = 0.0
    model_selections = {m: 0 for m in models}
    expert_weights_history = []
    
    for prompt_data in eval_data:
        prompt = prompt_data["prompt"]
        rewards = prompt_data["rewards"]
        
        context = embed_prompt(prompt, encoder, pca)
        selected_model = router.select_model(context)
        reward = rewards[selected_model]
        
        router.update(context, selected_model, reward)
        
        total_reward += reward
        total_cost += model_costs[selected_model]["cost"]
        model_selections[selected_model] += 1
        expert_weights_history.append(router.weights.copy())
    
    n = len(eval_data)
    return {
        "reward": total_reward / n,
        "cost": total_cost / n,
        "model_selections": model_selections,
        "expert_weights_final": router.weights,
        "expert_weights_history": expert_weights_history
    }

def main():
    print("="*70)
    print("DIAGNOSING BANDITGPT INSTABILITY")
    print("="*70)
    
    # Load resources
    print("\n📦 Loading resources...")
    model_costs = load_model_costs()
    eval_data = load_holdout_data()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    print(f"  ✓ Loaded {len(eval_data)} prompts")
    
    # Test parameters
    gamma = 0.01
    learning_rate = 0.1
    num_trials = 20  # More trials to see variance
    
    # Test λ=1.0 and λ=2.0 (the problematic points)
    lambda_values = [1.0, 2.0]
    
    print("\n" + "="*70)
    print("RUNNING MULTIPLE TRIALS FOR λ=1.0 AND λ=2.0")
    print("="*70)
    
    for lambda_val in lambda_values:
        print(f"\n--- λ={lambda_val} (gamma={gamma}, lr={learning_rate}) ---")
        
        results = []
        for trial in tqdm(range(num_trials), desc=f"  λ={lambda_val}"):
            result = run_bandit_trial(
                eval_data, encoder, pca, warmup_priors, model_costs,
                learning_rate, lambda_val, gamma, seed=42 + trial
            )
            results.append(result)
        
        # Analyze results
        rewards = [r["reward"] for r in results]
        costs = [r["cost"] for r in results]
        
        print(f"\n  Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f} (range: {min(rewards):.4f} - {max(rewards):.4f})")
        print(f"  Cost:   ${np.mean(costs):.6f} ± ${np.std(costs):.6f} (range: ${min(costs):.6f} - ${max(costs):.6f})")
        
        # Check model selection distribution
        all_models = list(results[0]["model_selections"].keys())
        for model in all_models:
            selections = [r["model_selections"][model] for r in results]
            avg_pct = np.mean(selections) / len(eval_data) * 100
            print(f"  {model}: {avg_pct:.1f}% ± {np.std(selections)/len(eval_data)*100:.1f}%")
        
        # Check expert weights
        final_weights = np.array([r["expert_weights_final"] for r in results])
        print(f"  Expert 0 (Warmup): {np.mean(final_weights[:, 0]):.3f} ± {np.std(final_weights[:, 0]):.3f}")
        print(f"  Expert 1 (Tabula Rasa): {np.mean(final_weights[:, 1]):.3f} ± {np.std(final_weights[:, 1]):.3f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nIf λ=2.0 consistently outperforms λ=1.0, the issue is:")
    print("  1. High variance in bandit learning (even with 5 trials)")
    print("  2. Non-smooth relationship between λ and performance")
    print("  3. Possible interaction between λ and expert weights")

if __name__ == "__main__":
    main()

