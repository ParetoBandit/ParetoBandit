"""
Test the two-phase burn-in protocol to ensure it works as expected.
"""
import sys
from pathlib import Path
import json
import gzip
import joblib
import numpy as np
from collections import defaultdict
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
    CANONICAL_DEV_DATA_PATH,
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
    
    # Normalize costs
    all_costs = [c["cost"] for c in costs.values()]
    min_cost, max_cost = min(all_costs), max(all_costs)
    cost_range = max_cost - min_cost
    
    for model_id in costs:
        costs[model_id]["normalized_cost"] = (costs[model_id]["cost"] - min_cost) / cost_range if cost_range > 0 else 0.0
    
    return costs

def load_data(filepath):
    """Load and filter data."""
    prompt_rewards = defaultdict(lambda: {})
    with gzip.open(filepath, 'rt') as f:
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

    def select_model(self, context):
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            normalized_cost = self.model_costs[model]["normalized_cost"]
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

    def select_model(self, context):
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            normalized_cost = self.model_costs[model]["normalized_cost"]
            ucb_scores[model] = expected_reward - self.cost_penalty * normalized_cost + self.alpha * uncertainty
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, context, model, reward):
        context = context.reshape(-1, 1)
        self.A[model] += context @ context.T
        self.b[model] += reward * context.flatten()

def test_burnin(train_data, eval_data, encoder, pca, warmup_priors, model_costs, cost_penalty):
    """Run the two-phase burn-in protocol."""
    gamma = 0.01
    learning_rate = 0.1
    
    models = list(train_data[0]["rewards"].keys())
    scaled_priors = apply_gamma_scaling(warmup_priors, gamma)
    context_dim = scaled_priors["A"][list(scaled_priors["A"].keys())[0]].shape[0]
    
    print(f"\n=== Testing λ={cost_penalty} ===")
    
    # PHASE 1: BURN-IN with λ=0
    print(f"Phase 1: Burn-in on dev set (N={len(train_data)}, λ=0)")
    warmup_expert = CostAwareLinUCBRouter(
        models=models, warmup_priors=scaled_priors,
        model_costs=model_costs, alpha=1.0, cost_penalty=0.0
    )
    tabula_rasa_expert = CostAwareTabulaRasaRouter(
        models=models, context_dim=context_dim,
        model_costs=model_costs, alpha=1.0, cost_penalty=0.0
    )
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models, learning_rate=learning_rate
    )
    
    # Track model selections during burn-in
    burnin_selections = {m: 0 for m in models}
    for prompt_data in train_data[:100]:  # Use subset for speed
        context = embed_prompt(prompt_data["prompt"], encoder, pca)
        selected_model = router.select_model(context)
        reward = prompt_data["rewards"][selected_model]
        router.update(context, selected_model, reward)
        burnin_selections[selected_model] += 1
    
    print(f"  Burn-in model usage: {burnin_selections}")
    print(f"  Expert weights after burn-in: Warmup={router.weights[0]:.3f}, TR={router.weights[1]:.3f}")
    
    # PHASE 2: EVALUATION with λ
    print(f"Phase 2: Evaluation on holdout (N={len(eval_data)}, λ={cost_penalty})")
    warmup_expert.cost_penalty = cost_penalty
    tabula_rasa_expert.cost_penalty = cost_penalty
    
    # Evaluate WITHOUT updates
    total_reward = 0.0
    total_cost = 0.0
    eval_selections = {m: 0 for m in models}
    
    for prompt_data in eval_data[:100]:  # Use subset for speed
        context = embed_prompt(prompt_data["prompt"], encoder, pca)
        selected_model = router.select_model(context)
        reward = prompt_data["rewards"][selected_model]
        
        # NO UPDATE
        total_reward += reward
        total_cost += model_costs[selected_model]["cost"]
        eval_selections[selected_model] += 1
    
    print(f"  Eval model usage: {eval_selections}")
    print(f"  Avg reward: {total_reward/100:.4f}, Avg cost: ${total_cost/100:.6f}")
    
    return total_reward / 100, total_cost / 100

def main():
    print("="*70)
    print("TESTING TWO-PHASE BURN-IN PROTOCOL")
    print("="*70)
    
    # Load resources
    print("\n📦 Loading resources...")
    model_costs = load_model_costs()
    train_data = load_data(CANONICAL_DEV_DATA_PATH)
    eval_data = load_data(CANONICAL_HOLDOUT_DATA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    print(f"  ✓ Train: {len(train_data)} prompts")
    print(f"  ✓ Eval: {len(eval_data)} prompts")
    print(f"  ✓ Models: {list(model_costs.keys())}")
    
    # Test with different cost penalties
    for cost_penalty in [0.0, 1.0, 5.0]:
        test_burnin(train_data, eval_data, encoder, pca, warmup_priors, model_costs, cost_penalty)
    
    print("\n" + "="*70)
    print("EXPECTED BEHAVIOR:")
    print("  - λ=0: Should select GPT-4 frequently (quality focus)")
    print("  - λ=1: Should balance GPT-4 and Mixtral")
    print("  - λ=5: Should prefer Mixtral (cost focus)")
    print("="*70)

if __name__ == "__main__":
    main()

