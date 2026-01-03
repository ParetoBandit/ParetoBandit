#!/usr/bin/env python3
"""
Priors Only Ablation: Testing prior beliefs without covariance structure.

Configuration:
- prior_n_effective = 20.0 (CSR prior beliefs)
- prior_structure_n_effective = 0.0 (NO covariance, just ridge regularization)

This isolates the b vector (prior means) from the A matrix (covariance structure).
"""

import sys
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from banditgpt.bandit import BanditRouter

def load_test_data():
    """Load test rewards and prompts"""
    data_dir = Path(__file__).parent.parent.parent / "data"
    test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
    
    rewards_data = []
    with open(test_rewards_path) as f:
        for line in f:
            rewards_data.append(json.loads(line))
    
    prompt_to_rewards = defaultdict(dict)
    for entry in rewards_data:
        if entry.get("ok"):
            prompt = entry["prompt"]
            model_id = entry["model_id"]
            score = entry["raw_score"]
            prompt_to_rewards[prompt][model_id] = score
    
    prompts = list(prompt_to_rewards.keys())
    ground_truth = {p: prompt_to_rewards[p] for p in prompts}
    
    return prompts, ground_truth

def simulate_bandit(router, prompts, ground_truth):
    """Run bandit simulation and return cumulative regret"""
    cumulative_regret = 0.0
    
    for prompt in prompts:
        selected_model_id, log = router.route(prompt, profile="balanced")
        
        true_rewards = ground_truth[prompt]
        best_reward = max(true_rewards.values())
        selected_reward = true_rewards.get(selected_model_id, 0.0)
        
        regret = best_reward - selected_reward
        cumulative_regret += regret
        
        router.process_feedback(log.request_id, selected_reward)
    
    return cumulative_regret

def run_priors_only_ablation(num_trials=20):
    """Test priors without covariance structure"""
    print("=" * 70)
    print("PRIORS ONLY ABLATION: No Covariance Structure")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading test data...")
    prompts, ground_truth = load_test_data()
    print(f"  Prompts: {len(prompts)}")
    
    # Load registry
    models_path = Path(__file__).parent.parent.parent / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    print("\n[2/3] Running ablation...")
    print(f"  Configuration:")
    print(f"    - prior_n_effective = 20.0 (CSR beliefs)")
    print(f"    - prior_structure_n_effective = 0.0 (NO covariance)")
    print(f"    - A matrix = ridge regularization only (identity)")
    print(f"    - b vector = CSR prior means")
    print(f"  Trials: {num_trials}")
    
    results = []
    
    for trial in range(num_trials):
        print(f"\n  Trial {trial + 1}/{num_trials}:")
        
        shuffled = prompts.copy()
        random.seed(trial)
        random.shuffle(shuffled)
        
        # Create router with priors but NO covariance structure
        router = BanditRouter.create(
            model_registry=registry,
            context_model="sentence-transformers/all-MiniLM-L6-v2",
            priors="csr",
            prior_n_effective=20.0,  # Use CSR prior beliefs
            prior_structure_n_effective=0.0,  # NO covariance structure
            exploration="safe",
            ridge_lambda=1.0,
            forgetting_factor=1.0
        )
        
        regret = simulate_bandit(router, shuffled, ground_truth)
        results.append(regret)
        print(f"    Regret: {regret:.1f}")
    
    # Aggregate
    print(f"\n  Computing statistics...")
    mean_regret = np.mean(results)
    std_regret = np.std(results)
    
    return {
        "mean": mean_regret,
        "std": std_regret,
        "trials": results
    }

def main():
    """Main execution"""
    results = run_priors_only_ablation(num_trials=20)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nPriors Only (b=20, A=0): {results['mean']:.1f} ± {results['std']:.1f}")
    
    print("\n📊 Comparison Context:")
    print("  Structure Only + Full:     208.1 ± 4.3  (b=0, A=20, Full Σ)")
    print("  Structure Only + Diagonal: 203.2 ± 5.5  (b=0, A=20, Diag Σ)")
    print(f"  Priors Only:               {results['mean']:.1f} ± {results['std']:.1f}  (b=20, A=0, No Σ)")
    
    # Save results
    output_path = Path(__file__).parent / "priors_only_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n✅ ABLATION COMPLETE!")

if __name__ == "__main__":
    main()
