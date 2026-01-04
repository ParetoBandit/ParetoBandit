#!/usr/bin/env python3
"""
2D Synergy Grid Search: N_structure vs N_prior

Hypothesis: There is a "Synergy Ridge" where optimal performance requires
tuning BOTH knobs independently.

Zones Expected:
- Zone A (Low Boost, High Stiffness): "Inertia" - no signal but rigid
- Zone B (High Boost, Low Stiffness): "Volatile" - signal washed away by noise
- Zone C (Balanced): "Synergy Peak" - stiffness protects the strong prior

This produces: "The Synergy Landscape" Heatmap
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
# DATA LOADING
# =============================================================================

def load_data_with_zscores():
    """Load rewards and z-scores from models.json."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
    train_rewards_path = data_dir / "train_rewards_1k.jsonl"
    models_path = Path(__file__).parent.parent.parent / "models.json"
    
    # Verify paths exist - NO FALLBACKS
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
                    
                    # Look up z-score from registry - NO FALLBACK
                    zscore = zscore_lookup.get((model_id, cluster_id))
                    if zscore is not None:
                        prompt_data[prompt]["zscores"][model_id] = zscore
        
        print(f"  {label}: {len(prompt_data)} prompts")
        return dict(prompt_data)
    
    train_data = load_rewards(train_rewards_path, "Training")
    test_data = load_rewards(test_rewards_path, "Test")
    
    return train_data, test_data, registry, zscore_lookup


# =============================================================================
# 2D GRID SEARCH
# =============================================================================

def run_2d_synergy_sweep(train_data, test_data, registry, encoder, 
                          n_struct_values, n_prior_values, n_trials=3):
    """
    2D Grid Search: N_structure (A matrix) vs N_prior (b vector)
    Returns a matrix of Z-scores for heatmap visualization.
    """
    print(f"\n{'='*70}")
    print(f"2D SYNERGY GRID SEARCH")
    print(f"N_structure (X): {n_struct_values}")
    print(f"N_prior (Y): {n_prior_values}")
    print(f"{'='*70}")
    
    # Results matrix: rows=N_prior, cols=N_structure
    heatmap_matrix = np.zeros((len(n_prior_values), len(n_struct_values)))
    std_matrix = np.zeros((len(n_prior_values), len(n_struct_values)))
    
    # Fixed profile for evaluation - lambda_cost=0 for max quality
    profile = {"lambda_cost": 0.0, "lambda_latency": 0.001}
    
    total_points = len(n_prior_values) * len(n_struct_values)
    current_point = 0
    
    for i, n_prior in enumerate(n_prior_values):
        for j, n_struct in enumerate(n_struct_values):
            current_point += 1
            print(f"\n[{current_point}/{total_points}] Grid Point: N_prior={n_prior}, N_struct={n_struct}")
            
            trial_zscores = []
            
            for trial in range(n_trials):
                # THE TWO KNOBS
                router = BanditRouter.create(
                    registry,
                    exploration="safe",
                    priors="csr",
                    prior_n_effective=float(n_prior),           # Knob 1: Belief Strength (b vector)
                    prior_structure_n_effective=float(n_struct), # Knob 2: Stiffness (A matrix)
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
                trial_zscores.append(avg_z)
                print(f"  Trial {trial+1}: Z={avg_z:+.4f}σ")
            
            heatmap_matrix[i, j] = np.mean(trial_zscores)
            std_matrix[i, j] = np.std(trial_zscores)
            print(f"  → Mean: Z={heatmap_matrix[i, j]:+.4f}σ ± {std_matrix[i, j]:.4f}")
    
    return heatmap_matrix, std_matrix


def plot_synergy_heatmap(matrix, x_labels, y_labels, output_path):
    """Create the Synergy Landscape heatmap visualization."""
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Use a diverging colormap centered at 0
    vmin = min(matrix.min(), -0.1)
    vmax = max(matrix.max(), 0.4)
    
    im = plt.imshow(matrix, cmap='RdYlGn', aspect='auto', origin='lower',
                    vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, label='Average Z-Score (Quality)', shrink=0.8)
    cbar.ax.axhline(y=0, color='black', linewidth=2)  # Mark zero line
    
    # Axis labels
    plt.xticks(range(len(x_labels)), x_labels, fontsize=11)
    plt.yticks(range(len(y_labels)), y_labels, fontsize=11)
    
    plt.xlabel('Stiffness ($N_{structure}$) - A Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Prior Boost ($N_{prior}$) - b Vector', fontsize=12, fontweight='bold')
    plt.title('The Synergy Landscape:\nInteraction of Trust (b) & Rigidity (A)', 
              fontsize=14, fontweight='bold')
    
    # Annotate each cell with value
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            value = matrix[i, j]
            # Choose text color based on background
            text_color = 'white' if value < 0.1 else 'black'
            plt.text(j, i, f"{value:+.2f}",
                     ha="center", va="center", 
                     color=text_color, fontweight='bold', fontsize=10)
    
    # Find and mark the peak
    max_val = matrix.max()
    max_idx = np.unravel_index(matrix.argmax(), matrix.shape)
    plt.plot(max_idx[1], max_idx[0], 'k*', markersize=20, markeredgewidth=2)
    plt.annotate(f'PEAK\n{max_val:+.3f}σ', 
                 xy=(max_idx[1], max_idx[0]),
                 xytext=(max_idx[1] + 0.5, max_idx[0] + 0.5),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved Synergy Heatmap to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("2D SYNERGY GRID SEARCH")
    print("Finding the optimal N_structure × N_prior combination")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data with z-scores...")
    train_data, test_data, registry, zscore_lookup = load_data_with_zscores()
    print(f"  Z-score lookup: {len(zscore_lookup)} entries")
    print(f"  Models in registry: {len(registry)}")
    
    # Initialize encoder once
    print("\n[2/3] Initializing encoder...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    
    # Define grid
    # X-Axis: Stiffness (A matrix scaling)
    n_struct_values = [0, 10, 50, 200, 1000]
    # Y-Axis: Prior Boost (b vector scaling)  
    n_prior_values = [1, 5, 20, 50, 200]
    
    # Run 2D sweep
    print("\n[3/3] Running 2D Grid Search...")
    heatmap_matrix, std_matrix = run_2d_synergy_sweep(
        train_data, test_data, registry, encoder,
        n_struct_values=n_struct_values,
        n_prior_values=n_prior_values,
        n_trials=3
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("2D GRID SEARCH RESULTS")
    print("=" * 70)
    
    # Print as table
    print(f"\n{'N_prior \\ N_struct':<18}", end="")
    for n_s in n_struct_values:
        print(f"{n_s:<10}", end="")
    print()
    print("-" * (18 + 10 * len(n_struct_values)))
    
    for i, n_p in enumerate(n_prior_values):
        print(f"{n_p:<18}", end="")
        for j, n_s in enumerate(n_struct_values):
            val = heatmap_matrix[i, j]
            marker = "★" if val == heatmap_matrix.max() else " "
            print(f"{val:+.3f}{marker}   ", end="")
        print()
    
    # Find optimal
    max_idx = np.unravel_index(heatmap_matrix.argmax(), heatmap_matrix.shape)
    optimal_n_prior = n_prior_values[max_idx[0]]
    optimal_n_struct = n_struct_values[max_idx[1]]
    optimal_z = heatmap_matrix.max()
    
    print(f"\n{'='*70}")
    print(f"OPTIMAL COMBINATION:")
    print(f"  N_prior (b vector):    {optimal_n_prior}")
    print(f"  N_structure (A matrix): {optimal_n_struct}")
    print(f"  Best Z-Score:          {optimal_z:+.4f}σ")
    print(f"{'='*70}")
    
    # Plot heatmap
    output_path = Path(__file__).parent / "synergy_heatmap.png"
    plot_synergy_heatmap(heatmap_matrix, n_struct_values, n_prior_values, output_path)
    
    # Save raw results
    results_path = Path(__file__).parent / "synergy_grid_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "n_struct_values": n_struct_values,
            "n_prior_values": n_prior_values,
            "heatmap_matrix": heatmap_matrix.tolist(),
            "std_matrix": std_matrix.tolist(),
            "optimal": {
                "n_prior": optimal_n_prior,
                "n_struct": optimal_n_struct,
                "z_score": optimal_z
            }
        }, f, indent=2)
    print(f"✓ Saved raw results to: {results_path}")
    
    print("\n✅ 2D SYNERGY GRID SEARCH COMPLETE!")


if __name__ == "__main__":
    main()
