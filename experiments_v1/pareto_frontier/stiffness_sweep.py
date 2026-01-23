#!/usr/bin/env python3
"""
Stiffness Sensitivity Sweep (N-Sweep)

Hypothesis: There is a 'Goldilocks' stiffness that balances Prior Trust vs. Online Adaptation.

Key findings expected:
- Low N (<10): Poor performance (too much variance, "Dip of Death")
- Optimal N (~40-80): Peak performance ("Synergy Zone")
- High N (>200): Declining performance ("Inertia Zone" - priors may not match test set)

This produces Figure: "The Stiffness Goldilocks Curve"
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import logging

logging.getLogger("banditgpt").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from banditgpt.bandit import BanditRouter, DEFAULT_CONTEXT_MODEL


# =============================================================================
# DATA LOADING (from pareto_frontier_plot.py)
# =============================================================================

def load_data_with_zscores():
    """Load rewards and z-scores from models.json."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
    train_rewards_path = data_dir / "train_rewards_1k.jsonl"
    models_path = Path(__file__).parent.parent.parent / "models.json"
    
    # Verify paths exist
    assert test_rewards_path.exists(), f"Test rewards not found: {test_rewards_path}"
    assert train_rewards_path.exists(), f"Train rewards not found: {train_rewards_path}"
    assert models_path.exists(), f"Models not found: {models_path}"
    
    # Load registry with z-scores
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Build z-score lookup: (model_id, cluster_id) -> z_score
    zscore_lookup = {}
    for model_id, model in registry.items():
        if "cluster_success_rates" in model:
            for cluster_id_str, cluster_data in model["cluster_success_rates"].items():
                if isinstance(cluster_data, dict) and "z_score" in cluster_data:
                    zscore_lookup[(model_id, int(cluster_id_str))] = cluster_data["z_score"]
    
    def load_rewards(path, label):
        prompt_data = defaultdict(lambda: {"cluster_id": None, "rewards": {}, "zscores": {}})
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    prompt = entry["prompt"]
                    model_id = entry["model_id"]
                    cluster_id = entry.get("cluster_id", 0)
                    
                    prompt_data[prompt]["cluster_id"] = cluster_id
                    prompt_data[prompt]["rewards"][model_id] = entry["raw_score"]
                    
                    # Look up z-score from registry
                    zscore = zscore_lookup.get((model_id, cluster_id), 0.0)
                    prompt_data[prompt]["zscores"][model_id] = zscore
        
        print(f"  {label}: {len(prompt_data)} prompts")
        return dict(prompt_data)
    
    train_data = load_rewards(train_rewards_path, "Training")
    test_data = load_rewards(test_rewards_path, "Test")
    
    return train_data, test_data, registry, zscore_lookup


# =============================================================================
# STIFFNESS SWEEP
# =============================================================================

def run_stiffness_sweep(train_data, test_data, registry, n_values, encoder, n_trials=3):
    """
    Sweep over prior_n_effective (and prior_structure_n_effective) values.
    Uses greedy evaluation to measure what the bandit learned.
    """
    print(f"\n{'='*70}")
    print(f"STIFFNESS SWEEP: N = {n_values}")
    print(f"{'='*70}")
    
    results = {n: [] for n in n_values}
    
    # Fixed profile for sweep - isolate stiffness effect
    # lambda_cost=0.0 to maximize quality (ignore cost)
    profile = {"lambda_cost": 0.0, "lambda_latency": 0.001}
    
    for n in n_values:
        print(f"\nTesting Stiffness N={n}...")
        
        for trial in range(n_trials):
            # Initialize Router with specific Stiffness
            # NOTE: Both knobs set to same value for this sweep
            router = BanditRouter.create(
                registry,
                exploration="safe",
                priors="csr",
                prior_n_effective=float(n),  # Knob 2: b vector
                prior_structure_n_effective=float(n),  # Knob 1: A matrix
                context_encoder=encoder
            )
            
            # Phase 1: Burn-in (WITH exploration)
            train_prompts = list(train_data.keys())
            random.seed(42 + trial)
            random.shuffle(train_prompts)
            
            for prompt in train_prompts:
                data = train_data[prompt]
                selected, log = router.route(prompt, profile=profile, input_tokens=100)
                if selected in data["rewards"]:
                    router.process_feedback(log.request_id, data["rewards"][selected])
            
            # Phase 2: Evaluate (GREEDY - no exploration tax)
            test_prompts = list(test_data.keys())
            random.shuffle(test_prompts)
            
            zscores = []
            
            # Force greedy by setting alpha=0
            original_alpha = router.bandit.alpha
            router.bandit.alpha = 0.0
            
            for prompt in test_prompts:
                data = test_data[prompt]
                selected, log = router.route(prompt, profile=profile, input_tokens=100)
                
                if selected in data["zscores"]:
                    zscores.append(data["zscores"][selected])
            
            router.bandit.alpha = original_alpha
            
            avg_z = np.mean(zscores) if zscores else 0.0
            results[n].append(avg_z)
            print(f"  Trial {trial+1}: Z={avg_z:+.4f}σ")
    
    return results


def plot_stiffness_curve(results, output_path):
    """Create the Goldilocks Curve visualization."""
    n_values = sorted(results.keys())
    means = [np.mean(results[n]) for n in n_values]
    stds = [np.std(results[n]) for n in n_values]
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Main curve
    plt.errorbar(n_values, means, yerr=stds, fmt='-o', linewidth=2.5, 
                 capsize=5, color='#0055A4', markersize=8, label='BanditRouter')
    
    # Zero line (Random baseline)
    plt.axhline(y=0, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Random Baseline (Y=0)')
    
    # Find and annotate peak
    max_y = max(means)
    max_idx = means.index(max_y)
    max_x = n_values[max_idx]
    
    plt.annotate(f'Peak\nN={max_x}\nZ={max_y:+.3f}σ', 
                 xy=(max_x, max_y), 
                 xytext=(max_x * 2, max_y + 0.05),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                 fontsize=10, fontweight='bold', color='green',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Formatting
    plt.xscale('log')
    plt.xlabel('Stiffness ($N_{effective}$)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Z-Score (Quality)', fontsize=12, fontweight='bold')
    plt.title('The Stiffness "Goldilocks" Curve\nBalancing Prior Trust vs Online Adaptation', 
              fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='lower right', fontsize=10)
    
    # Add interpretation zones
    ax = plt.gca()
    xlim = ax.get_xlim()
    
    # Shade zones (subtle)
    plt.axvspan(xlim[0], 15, alpha=0.1, color='red', label='_nolegend_')  # Dip of Death
    plt.axvspan(30, 100, alpha=0.1, color='green', label='_nolegend_')    # Goldilocks
    plt.axvspan(300, xlim[1], alpha=0.1, color='orange', label='_nolegend_')  # Inertia
    
    plt.text(5, min(means) + 0.02, 'Dip of\nDeath', fontsize=9, ha='center', alpha=0.7)
    plt.text(55, max(means) - 0.05, 'Goldilocks\nZone', fontsize=9, ha='center', alpha=0.7, color='green')
    plt.text(600, max(means) - 0.1, 'Inertia\nZone', fontsize=9, ha='center', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved Goldilocks Curve to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("STIFFNESS SENSITIVITY SWEEP")
    print("Finding the Goldilocks N_effective")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data with z-scores...")
    train_data, test_data, registry, zscore_lookup = load_data_with_zscores()
    print(f"  Z-score lookup: {len(zscore_lookup)} entries")
    print(f"  Models in registry: {len(registry)}")
    
    # Initialize encoder once (expensive)
    print("\n[2/3] Initializing encoder...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    
    # Run sweep
    print("\n[3/3] Running Stiffness Sweep...")
    
    # Sweep values: 1 (cold start) to 1000 (frozen)
    n_values = [1, 10, 20, 50, 100, 200, 500, 1000]
    
    results = run_stiffness_sweep(
        train_data, test_data, registry, 
        n_values=n_values, 
        encoder=encoder,
        n_trials=3
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("STIFFNESS SWEEP RESULTS")
    print("=" * 70)
    print(f"\n{'N':<10} {'Mean Z-Score':<15} {'Std':<10}")
    print("-" * 35)
    
    for n in sorted(results.keys()):
        mean = np.mean(results[n])
        std = np.std(results[n])
        marker = "← PEAK" if mean == max(np.mean(results[nv]) for nv in n_values) else ""
        print(f"{n:<10} {mean:+.4f}σ        ±{std:.4f}   {marker}")
    
    # Plot
    output_path = Path(__file__).parent / "stiffness_goldilocks_curve.png"
    plot_stiffness_curve(results, output_path)
    
    # Find optimal
    best_n = max(results.keys(), key=lambda n: np.mean(results[n]))
    best_z = np.mean(results[best_n])
    
    print("\n" + "=" * 70)
    print(f"OPTIMAL STIFFNESS: N = {best_n}")
    print(f"Best Z-Score: {best_z:+.4f}σ")
    print("=" * 70)
    
    print("\n✅ STIFFNESS SWEEP COMPLETE!")


if __name__ == "__main__":
    main()
