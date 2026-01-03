#!/usr/bin/env python3
"""
Covariance Structure Ablation: PCA-Aware CSR Implementation

Tests the impact of off-diagonal correlations in PCA-reduced space (45D: 32 PCA + 13 explicit).
Uses CSR default parameters and properly configures PCA in the router.

Key Experimental Design:
- Condition A (Full): Complete Σ_CSR with off-diagonal correlations
- Condition B (Diagonal): diag(Σ_CSR) - variances only, no correlations
- Uses prior_structure_n_effective to scale covariance (default 20.0 for CSR)
- Uses prior_n_effective=0.0 to isolate structure contribution (no prior means)
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from banditgpt.bandit import BanditRouter

def load_test_data():
    """Load test rewards and prompts"""
    data_dir = Path(__file__).parent.parent.parent / "data"
    test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
    
    if not test_rewards_path.exists():
        raise FileNotFoundError(f"Test rewards not found: {test_rewards_path}")
    
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

def load_csr_covariance():
    """Load CSR covariance matrix from priors_meta_pca.npz"""
    priors_path = Path(__file__).parent.parent.parent / "priors" / "priors_meta_pca.npz"
    
    if not priors_path.exists():
        raise FileNotFoundError(f"Priors not found: {priors_path}")
    
    # Load the priors
    priors_data = np.load(priors_path)
    cov_matrix = priors_data['cov_matrix']  # CSR covariance matrix (45x45)
    
    print(f"  Loaded CSR covariance: {cov_matrix.shape}")
    print(f"  Mean diagonal: {np.mean(np.diag(cov_matrix)):.4f}")
    
    # Calculate off-diagonal statistics
    off_diag_mask = ~np.eye(cov_matrix.shape[0], dtype=bool)
    mean_off_diag = np.mean(np.abs(cov_matrix[off_diag_mask]))
    print(f"  Mean |off-diagonal|: {mean_off_diag:.4f}")
    
    # Get total samples for scaling
    if 'cluster_counts' in priors_data and np.sum(priors_data['cluster_counts']) > 0:
        n_offline = float(np.sum(priors_data['cluster_counts']))
        print(f"  N_offline (from cluster_counts): {n_offline:.0f}")
    else:
        # Fallback
        n_offline = 21000.0
        print(f"  N_offline (approximated): {n_offline:.0f}")
    
    return cov_matrix, n_offline

def create_router_with_custom_covariance(cov_type="full", cov_matrix=None, n_offline=None):
    """
    Create router with custom covariance matrix for ablation.
    
    Args:
        cov_type: "full" or "diagonal"
        cov_matrix: CSR covariance matrix (45x45)
        n_offline: Total offline samples
    
    Returns:
        BanditRouter instance
    """
    # Prepare custom covariance based on type
    if cov_type == "full":
        # Full covariance with all correlations
        custom_cov = cov_matrix.copy()
        
    elif cov_type == "diagonal":
        # Diagonal only - extract variances, zero out off-diagonals
        custom_cov = np.diag(np.diag(cov_matrix))
        
    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")
    
    # Scale the covariance matrix
    # CSR default: prior_structure_n_effective = 20.0
    # Scaling: gamma_structure = N_target / N_offline
    N_target = 20.0  # CSR default
    gamma_structure = N_target / n_offline
    custom_cov_scaled = custom_cov * gamma_structure
    
    # Create router using BanditRouter.create() with CSR defaults
    # CRITICAL: Must pass custom_covariance parameter
    # Note: BanditRouter.create() doesn't have custom_covariance parameter!
    # We need to use load_from_benchmark() which does accept it
    
    # Load registry
    models_path = Path(__file__).parent.parent.parent / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Get paths
    base_dir = Path(__file__).parent.parent.parent
    pca_path = base_dir / "data" / "pca_32.joblib"
    priors_meta_path = base_dir / "priors" / "priors_meta_pca.npz"
    
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA model not found: {pca_path}")
    
    # Create router using load_from_benchmark with custom covariance
    # We'll need to modify the approach since custom_covariance isn't a standard parameter
    
    # WORKAROUND: Create router, then manually inject custom covariance
    # Note: Don't pass context_model="pca" - that makes it look for a model called "pca"
    # Instead, the router will use the default SBERT model and load PCA internally
    router = BanditRouter.create(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",  # Default SBERT model
        priors="csr",
        prior_n_effective=0.0,  # No prior means (isolate structure)
        prior_structure_n_effective=20.0,  # CSR default structure strength
        exploration="safe",  # alpha=0.1
        ridge_lambda=1.0,
        forgetting_factor=1.0
    )
    
    # Manually override the A matrices with our custom covariance
    # The router already initialized with priors, so we need to replace them
    dim = router.bandit.dim
    
    # Pad custom covariance to match bandit dimension (45 -> 46 with bias)
    cov_padded = np.eye(dim)
    cov_padded[:custom_cov_scaled.shape[0], :custom_cov_scaled.shape[1]] = custom_cov_scaled
    
    # Replace A matrices for all models
    for model in router.bandit.models:
        # Reset to ridge regularization
        router.bandit.A[model] = np.eye(dim) * router.bandit.ridge_lambda
        # Add custom covariance
        router.bandit.A[model] += cov_padded
        # Reset b vector to zero (no prior means)
        router.bandit.b[model] = np.zeros(dim)
        # Recompute inverse
        from banditgpt.bandit import safe_inv
        router.bandit.A_inv[model] = safe_inv(router.bandit.A[model])
    
    return router

def simulate_bandit(router, prompts, ground_truth):
    """Run bandit simulation and return cumulative regret"""
    cumulative_regret = 0.0
    regret_history = []
    
    for i, prompt in enumerate(prompts):
        # Route and get prediction
        selected_model_id, log = router.route(prompt, profile="balanced")
        
        # Calculate regret
        true_rewards = ground_truth[prompt]
        best_reward = max(true_rewards.values())
        selected_reward = true_rewards.get(selected_model_id, 0.0)
        
        regret = best_reward - selected_reward
        cumulative_regret += regret
        regret_history.append(cumulative_regret)
        
        # Update router
        router.process_feedback(log.request_id, selected_reward)
        
        # Debug first few decisions
        if i < 5:
            print(f"    [Step {i}] Model: {selected_model_id}, "
                  f"Pred: {log.predicted_utility:.4f}, Regret: {regret:.4f}")
    
    return cumulative_regret

def run_covariance_ablation(num_trials=20):
    """
    Run covariance ablation experiment.
    
    Tests:
    - Full Covariance: Complete Σ_CSR with off-diagonal correlations
    - Diagonal Only: diag(Σ_CSR) - variances only
    """
    print("=" * 70)
    print("COVARIANCE STRUCTURE ABLATION: PCA-Aware CSR")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading test data...")
    prompts, ground_truth = load_test_data()
    print(f"  Prompts: {len(prompts)}")
    print(f"  Models: {len(next(iter(ground_truth.values())))}")
    
    # Load CSR covariance
    print("\n[2/4] Loading CSR covariance matrix...")
    cov_matrix, n_offline = load_csr_covariance()
    
    # Run ablation
    print(f"\n[3/4] Running ablation experiment...")
    print(f"  Configuration:")
    print(f"    - PCA: 384 → 32 dimensions")
    print(f"    - Features: 32 PCA + 8 explicit + 5 cluster + 1 bias = 46D")
    print(f"    - Prior strength: N_eff = 20.0 (CSR default)")
    print(f"    - Prior means: 0.0 (isolate structure)")
    print(f"    - Exploration: safe (alpha=0.1)")
    print(f"  Conditions:")
    print(f"    - Full: Complete Σ_CSR (with correlations)")
    print(f"    - Diagonal: diag(Σ_CSR) (variances only)")
    print(f"  Trials: {num_trials}")
    
    results = {
        "Full Covariance": [],
        "Diagonal Only": []
    }
    
    for trial in range(num_trials):
        print(f"\n  Trial {trial + 1}/{num_trials}:")
        
        # Shuffle prompts with fixed seed
        shuffled = prompts.copy()
        random.seed(trial)
        random.shuffle(shuffled)
        
        # Full covariance
        print(f"    Testing Full Covariance...")
        full_router = create_router_with_custom_covariance("full", cov_matrix, n_offline)
        full_regret = simulate_bandit(full_router, shuffled, ground_truth)
        results["Full Covariance"].append(full_regret)
        print(f"    → Full regret: {full_regret:.1f}")
        
        # Diagonal only
        print(f"    Testing Diagonal Only...")
        diag_router = create_router_with_custom_covariance("diagonal", cov_matrix, n_offline)
        diag_regret = simulate_bandit(diag_router, shuffled, ground_truth)
        results["Diagonal Only"].append(diag_regret)
        print(f"    → Diagonal regret: {diag_regret:.1f}")
        
        # Calculate benefit
        if diag_regret > full_regret:
            benefit = (diag_regret - full_regret) / diag_regret * 100
            print(f"    → Correlation benefit: {benefit:.1f}%")
        else:
            print(f"    ⚠ Warning: Diagonal performed better!")
    
    # Aggregate results
    print(f"\n  Computing statistics...")
    aggregated = {}
    for config in results:
        aggregated[config] = {
            "mean": np.mean(results[config]),
            "std": np.std(results[config]),
            "trials": results[config]
        }
    
    return aggregated

def plot_results(results, output_path):
    """Create bar plot comparing Full vs Diagonal covariance"""
    print("\n[4/4] Generating plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    configs = ["Full\nCovariance", "Diagonal\nOnly"]
    means = [results["Full Covariance"]["mean"], 
             results["Diagonal Only"]["mean"]]
    stds = [results["Full Covariance"]["std"],
            results["Diagonal Only"]["std"]]
    
    colors = ["#27ae60", "#f39c12"]
    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8, 
                   capsize=10, width=0.5)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mean:.1f}\n±{std:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Cumulative Regret", fontsize=13, fontweight='bold')
    ax.set_title("Covariance Ablation: Off-Diagonal Correlations in PCA Space",
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Calculate and display benefit
    full_mean = results["Full Covariance"]["mean"]
    diag_mean = results["Diagonal Only"]["mean"]
    corr_benefit = (diag_mean - full_mean) / diag_mean * 100 if diag_mean > 0 else 0
    
    # Annotation
    annotation_text = (
        f"Correlation Benefit: {corr_benefit:.1f}%\n\n"
        f"Configuration:\n"
        f"• PCA: 384 → 32 dims\n"
        f"• Structure strength: N_eff = 20.0\n"
        f"• Prior means: 0.0 (isolated)\n\n"
        f"{'✓ Off-diagonals are critical!' if corr_benefit > 20 else '⚠ Weak correlation effect'}"
    )
    
    ax.text(0.98, 0.98, annotation_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved to: {output_path}")

def main():
    """Main execution"""
    # Run ablation
    results = run_covariance_ablation(num_trials=20)
    
    # Plot results
    output_path = Path(__file__).parent / "covariance_ablation_csr.png"
    plot_results(results, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    full_mean = results["Full Covariance"]["mean"]
    full_std = results["Full Covariance"]["std"]
    diag_mean = results["Diagonal Only"]["mean"]
    diag_std = results["Diagonal Only"]["std"]
    
    print(f"\nFull Covariance:  {full_mean:.1f} ± {full_std:.1f}")
    print(f"Diagonal Only:    {diag_mean:.1f} ± {diag_std:.1f}")
    
    corr_benefit = (diag_mean - full_mean) / diag_mean * 100 if diag_mean > 0 else 0
    print(f"\n→ Correlation Benefit: {corr_benefit:.1f}%")
    
    if corr_benefit > 30:
        print(f"\n✓ Off-diagonal correlations are CRITICAL ({corr_benefit:.0f}% improvement)")
    elif corr_benefit > 10:
        print(f"\n~ Correlations provide moderate benefit ({corr_benefit:.0f}%)")
    else:
        print(f"\n⚠ Weak correlation benefit ({corr_benefit:.0f}%)")
    
    print("\n✅ ABLATION COMPLETE!")

if __name__ == "__main__":
    main()
