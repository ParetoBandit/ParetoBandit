#!/usr/bin/env python3
"""
Diagnose Prior Strength: Test different λ_boost values

The held-out results show warm-start WORSE than cold-start.
This might be due to over-confident priors (λ_boost=50).

Let's test multiple prior strengths: 1, 5, 10, 20, 50, 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_rq1 import (
    ExperimentConfig,
    run_experiment,
)
from banditgpt._resources import get_priors_path


def main():
    # Test different prior strengths
    strengths = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    print("=" * 70)
    print("Diagnosing Prior Strength on Held-Out Data")
    print("=" * 70)
    print()
    
    results = []
    
    for strength in strengths:
        print(f"\n[Test] λ_boost = {strength}")
        print("-" * 70)
        
        config = ExperimentConfig(
            priors_path=get_priors_path("expert_priors_train.npz"),
            prompts_path=get_priors_path("test_archetypes.jsonl"),
            rewards_path=get_priors_path("test_rewards.jsonl"),
            embeddings_cache=get_priors_path("test_prompt_embeddings.npy"),
            n_test=2000,
            alpha=0.5,
            prior_strength=strength,  # Testing different values
            seed=42,
            output_dir=Path(f"results/rq1_strength_{int(strength)}"),
        )
        
        result = run_experiment(config)
        
        reduction = result.regret_reduction_pct
        symbol = "✓" if reduction > 0 else "✗"
        
        print(f"   {symbol} Cold: {result.final_regret_cold:.1f}, "
              f"Warm: {result.final_regret_warm:.1f}, "
              f"Reduction: {reduction:+.1f}%")
        
        results.append({
            "strength": strength,
            "cold_regret": result.final_regret_cold,
            "warm_regret": result.final_regret_warm,
            "reduction_pct": reduction,
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY: Prior Strength vs. Performance")
    print("=" * 70)
    print(f"{'λ_boost':<10} {'Cold Regret':<15} {'Warm Regret':<15} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    for r in results:
        symbol = "✓" if r["reduction_pct"] > 0 else "✗"
        print(f"{r['strength']:<10.1f} "
              f"{r['cold_regret']:<15.1f} "
              f"{r['warm_regret']:<15.1f} "
              f"{r['reduction_pct']:>+10.1f}%  "
              f"{symbol}")
    
    print("=" * 70)
    
    # Find best
    best = max(results, key=lambda x: x["reduction_pct"])
    if best["reduction_pct"] > 0:
        print(f"\n✓ Best setting: λ_boost = {best['strength']} "
              f"({best['reduction_pct']:+.1f}% reduction)")
    else:
        print(f"\n✗ NO POSITIVE SETTING FOUND - priors consistently hurt performance!")
    
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

