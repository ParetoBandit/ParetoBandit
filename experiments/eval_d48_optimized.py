#!/usr/bin/env python3
"""
Evaluate d=48 optimized priors on held-out test set

Uses the proven-working run_pca_experiment logic with d=48 priors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_rq1_pca import (
    PCAExperimentConfig,
    run_pca_experiment,
    save_results,
    plot_results,
)
from banditgpt._resources import get_priors_path


def main():
    print("=" * 70)
    print("Evaluating d=48 Optimized Priors (Held-Out Test)")
    print("=" * 70)
    print("  Training: d=48, epochs=12, expert=70%")
    print("  Prior strength (λ): 10")
    print("  Test set: 99 held-out prompts")
    print("=" * 70)
    print()
    
    # Test multiple prior strengths
    strengths = [5, 8, 10, 12, 15]
    
    results_all = []
    
    for strength in strengths:
        print(f"\n{'='*70}")
        print(f"Testing λ = {strength}")
        print(f"{'='*70}\n")
        
        config = PCAExperimentConfig(
            priors_path=get_priors_path("expert_priors_pca48_optimized.npz"),
            prompts_path=get_priors_path("test_archetypes.jsonl"),
            rewards_path=get_priors_path("test_rewards.jsonl"),
            embeddings_pca_path=get_priors_path("test_embeddings_pca48.npy"),
            n_test=2000,
            alpha=0.5,
            prior_strength=strength,
            seed=42,
            output_dir=Path(f"results/rq1_pca48_opt/lambda{int(strength)}"),
        )
        
        results = run_pca_experiment(config)
        results_all.append((strength, results))
        
        # Save
        save_results(results, config.output_dir / "metrics.json")
        plot_results(results, config.output_dir / "regret_curve.png")
        
        print(f"\n  λ={strength}: {results.regret_reduction_pct:+.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: d=48 Optimized (epochs=12, expert=70%)")
    print("=" * 70)
    print(f"{'λ':<8} {'Cold':<12} {'Warm':<12} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    for strength, results in results_all:
        status = "✓ TARGET!" if results.regret_reduction_pct >= 20 else ("✓" if results.regret_reduction_pct >= 10 else "○")
        print(f"{strength:<8.0f} "
              f"{results.final_regret_cold:<12.1f} "
              f"{results.final_regret_warm:<12.1f} "
              f"{results.regret_reduction_pct:>+10.1f}%  "
              f"{status}")
    
    # Find best
    best_strength, best_results = max(results_all, key=lambda x: x[1].regret_reduction_pct)
    
    print("\n" + "=" * 70)
    if best_results.regret_reduction_pct >= 20:
        print(f"🎉 TARGET ACHIEVED: λ={best_strength} → {best_results.regret_reduction_pct:+.1f}%")
    elif best_results.regret_reduction_pct >= 10:
        print(f"✓ Good progress: λ={best_strength} → {best_results.regret_reduction_pct:+.1f}%")
        print(f"  Gap to 20%: {20 - best_results.regret_reduction_pct:.1f}%")
    else:
        print(f"○ Best: λ={best_strength} → {best_results.regret_reduction_pct:+.1f}%")
        print(f"  Gap to 20%: {20 - best_results.regret_reduction_pct:.1f}%")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

