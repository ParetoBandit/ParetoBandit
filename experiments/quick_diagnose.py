#!/usr/bin/env python3
"""
Quick Diagnosis: Test a few key prior strengths on held-out data

Focuses on λ_boost values: 0, 1, 5, 10 (not 50 which is too high)
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
    # Test lower prior strengths (hypothesis: 50 is too high for held-out)
    strengths = [1.0, 5.0, 10.0, 20.0]
    
    print("=" * 70)
    print("Quick Diagnosis: Prior Strength on Held-Out Data")
    print("=" * 70)
    print("Hypothesis: λ_boost=50 is too high, causes over-confidence")
    print("=" * 70)
    print()
    
    results = []
    
    for strength in strengths:
        print(f"\n[Test] λ_boost = {strength}")
        
        config = ExperimentConfig(
            priors_path=get_priors_path("expert_priors_train.npz"),
            prompts_path=get_priors_path("test_archetypes.jsonl"),
            rewards_path=get_priors_path("test_rewards.jsonl"),
            embeddings_cache=get_priors_path("test_prompt_embeddings.npy"),
            n_test=1000,  # Reduced for speed
            alpha=0.5,
            prior_strength=strength,
            seed=42,
            output_dir=Path(f"results/quick_diag/strength_{int(strength)}"),
        )
        
        result = run_experiment(config)
        
        reduction = result.regret_reduction_pct
        symbol = "✓" if reduction > 0 else "✗"
        
        print(f"   {symbol} Cold: {result.final_regret_cold:.1f}, "
              f"Warm: {result.final_regret_warm:.1f}, "
              f"Reduction: {reduction:+.1f}%")
        
        results.append({
            "strength": strength,
            "cold": result.final_regret_cold,
            "warm": result.final_regret_warm,
            "reduction": reduction,
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'λ':<8} {'Cold':<12} {'Warm':<12} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    for r in results:
        symbol = "✓ HELPS" if r["reduction"] > 0 else "✗ HURTS"
        print(f"{r['strength']:<8.1f} "
              f"{r['cold']:<12.1f} "
              f"{r['warm']:<12.1f} "
              f"{r['reduction']:>+10.1f}%  "
              f"{symbol}")
    
    print("=" * 70)
    
    # Analysis
    best = max(results, key=lambda x: x["reduction"])
    if best["reduction"] > 0:
        print(f"\n✓ Best: λ={best['strength']} gives {best['reduction']:+.1f}% reduction")
    else:
        print(f"\n✗ ALL TESTED VALUES HURT PERFORMANCE!")
        print(f"   This suggests distribution shift between train and test.")
    
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

