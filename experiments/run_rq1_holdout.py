#!/usr/bin/env python3
"""
RQ1 with Held-Out Test Set (Rigorous Evaluation)

This script runs RQ1 using:
- Priors trained on train_archetypes.jsonl (398 prompts)
- Evaluation on test_archetypes.jsonl (99 held-out prompts)

This addresses the data leakage issue in the original RQ1.

Usage:
    python experiments/run_rq1_holdout.py
"""

import sys
from pathlib import Path

# Import the original RQ1 script
sys.path.insert(0, str(Path(__file__).parent))
from run_rq1 import (
    ExperimentConfig,
    run_experiment,
    plot_results,
    save_results,
)
from banditgpt._resources import get_priors_path


def main():
    # Configuration for held-out evaluation
    config = ExperimentConfig(
        # Priors trained on TRAIN data only
        priors_path=get_priors_path("expert_priors_train.npz"),
        
        # Evaluate on HELD-OUT TEST data
        prompts_path=get_priors_path("test_archetypes.jsonl"),
        rewards_path=get_priors_path("test_rewards.jsonl"),
        
        # Cache embeddings separately for test set
        embeddings_cache=get_priors_path("test_prompt_embeddings.npy"),
        
        # Standard config
        n_test=2000,  # Will cycle through 99 test prompts ~20 times
        alpha=0.5,
        prior_strength=50.0,
        seed=42,
        output_dir=Path("results/rq1_holdout"),
    )
    
    print("=" * 70)
    print("RQ1: Held-Out Test Set Evaluation (RIGOROUS)")
    print("=" * 70)
    print("Priors trained on: train_archetypes.jsonl (398 prompts)")
    print("Testing on:        test_archetypes.jsonl (99 HELD-OUT prompts)")
    print("=" * 70)
    print()
    
    # Run experiment
    results = run_experiment(config)
    
    # Save results
    save_results(results, config.output_dir / "metrics.json")
    plot_results(results, config.output_dir / "regret_curve.png")
    
    print("\n" + "=" * 70)
    print("Held-Out Evaluation Complete!")
    print("=" * 70)
    print(f"  Test Prompts: {results.n_prompts} (held-out)")
    print(f"  Models: {results.n_models}")
    print(f"  Cold Start Regret: {results.final_regret_cold:.1f}")
    print(f"  Warm Start Regret: {results.final_regret_warm:.1f}")
    print(f"  Regret Reduction: {results.regret_reduction_pct:.1f}%")
    print(f"  Saved to: {config.output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

