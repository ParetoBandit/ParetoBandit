"""
Diagnose Corralling Expert Weights During Sensitivity Experiment
================================================================
Check if the Tabula Rasa expert is dominating, which would explain
why n_eff changes don't matter.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit_gpt.router import BanditRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER, 
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEV_DATA_PATH_ALL_MODELS
)
from utils.aligned_evaluator import AlignedEvaluator

WARMUP_MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"
NEIGHBOR_MODEL = "openai/gpt-4-turbo"
TOTAL_STEPS = 1000
RELEASE_STEP = 300

def create_model_registry(models):
    all_models = {
        "mistralai/mixtral-8x7b-instruct": {
            "input_cost_per_m": 0.5, "output_cost_per_m": 1.5,
            "description": "Efficient sparse mixture-of-experts model."
        },
        "openai/gpt-4-turbo": {
            "input_cost_per_m": 10.0, "output_cost_per_m": 30.0,
            "description": "High-intelligence flagship model."
        },
        "openai/gpt-5.1": {
            "input_cost_per_m": 15.0, "output_cost_per_m": 45.0,
            "description": "Next-generation flagship model."
        }
    }
    return {k: v for k, v in all_models.items() if k in models}

def load_real_data():
    required_models = WARMUP_MODELS + [NEW_MODEL]
    evaluator = AlignedEvaluator.from_jsonl_gz(
        DEV_DATA_PATH_ALL_MODELS,
        required_models=required_models
    )
    filtered_data = [item for item in evaluator if all(m in item.rewards for m in required_models)]
    return AlignedEvaluator(filtered_data)

def run_with_weight_tracking(n_effective: float, seed: int = 42):
    """Run experiment and track Corralling expert weights over time."""
    np.random.seed(seed)
    
    evaluator = load_real_data()
    rng = np.random.RandomState(seed)
    indices = np.arange(len(evaluator.data))
    rng.shuffle(indices)
    shuffled_data = [evaluator.data[i] for i in indices]
    
    router = BanditRouter.create(
        model_registry=create_model_registry(WARMUP_MODELS),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        alpha=2.0,
        pca_path=DEFAULT_PCA_PATH
    )
    
    # Track weights
    weight_history = []
    reward_history = []
    
    for t, item in enumerate(shuffled_data):
        if t >= TOTAL_STEPS: break
        
        # Release new model
        if t == RELEASE_STEP:
            A_neighbor = router.bandit.A[NEIGHBOR_MODEL].copy()
            b_neighbor = router.bandit.b[NEIGHBOR_MODEL].copy()
            theta_neighbor = router.bandit.A_inv[NEIGHBOR_MODEL] @ b_neighbor
            
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = n_effective * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = n_effective * theta_neighbor
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.bandit.last_update[NEW_MODEL] = router.bandit.t
            router.registry[NEW_MODEL] = create_model_registry([NEW_MODEL])[NEW_MODEL]
            
            if router.corralling_router:
                router.corralling_router.add_model(NEW_MODEL)
                for expert in router.corralling_router.experts:
                    if hasattr(expert, 'add_model'):
                        expert_type = type(expert).__name__
                        if 'TabulaRasa' in expert_type:
                            expert.add_model(NEW_MODEL, 0.5)
                        else:
                            transfer_A = n_effective * np.eye(router.bandit.dim)
                            transfer_b = n_effective * theta_neighbor
                            expert.add_model(NEW_MODEL, transfer_A, transfer_b, 0.5)
        
        # Route and track
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        # Track Corralling weights
        if router.corralling_router:
            weights = router.corralling_router.weights.copy()
            weight_history.append(weights)
        else:
            weight_history.append([1.0, 0.0])  # No corralling
        
        reward_history.append(reward)
    
    return weight_history, reward_history

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Diagnosing Corralling Expert Weights Across Seeds")
    print("="*70)
    
    # Test multiple seeds
    seeds = [42, 43, 44]
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print("="*70)
        
        # Run with different n_eff values
        configs = {
            "n_eff=1.0 (Best Transfer)": 1.0,
            "n_eff=20.0 (Strong Prior)": 20.0
        }
        
        for name, n_eff in configs.items():
            print(f"\n{name}")
            print("-" * 70)
            
            weights, rewards = run_with_weight_tracking(n_eff, seed=seed)
            
            # Analyze pre-release (0-300)
            pre_weights = np.array(weights[:RELEASE_STEP])
            pre_warmup = pre_weights[:, 0].mean()
            pre_tabula = pre_weights[:, 1].mean()
            
            # Analyze post-release (300-1000)
            post_weights = np.array(weights[RELEASE_STEP:])
            post_warmup = post_weights[:, 0].mean()
            post_tabula = post_weights[:, 1].mean()
            
            # Analyze rewards
            post_rewards = rewards[RELEASE_STEP:]
            mean_reward = np.mean(post_rewards)
            
            print(f"  Pre-Release  (t<300):  Warmup={pre_warmup:.1%}, Tabula Rasa={pre_tabula:.1%}")
            print(f"  Post-Release (t>300):  Warmup={post_warmup:.1%}, Tabula Rasa={post_tabula:.1%}")
            print(f"  Mean Reward (t>300):   {mean_reward:.4f}")
            
            if post_tabula > 0.5:
                print(f"  ⚠️  Tabula Rasa DOMINATES (>{post_tabula:.0%} weight)")
                print(f"      → n_eff only affects {post_warmup:.0%} of decisions!")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print("- If Warmup weight varies across seeds → explains seed sensitivity")
    print("- If Tabula Rasa weight >50% → n_eff effect is diluted") 
    print("- If weights are stable but rewards vary → intrinsic data variance")
    print("="*70 + "\n")
