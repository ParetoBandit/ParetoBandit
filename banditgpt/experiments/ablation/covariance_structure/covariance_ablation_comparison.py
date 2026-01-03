#!/usr/bin/env python3
"""
Comprehensive Covariance Ablation: Isolating Off-Diagonal Impact

Tests diagonal vs. full covariance under TWO prior configurations:
1. Structure Only: prior_n_effective=0, prior_structure_n_effective=20 (isolate A matrix)
2. Full CSR: prior_n_effective=20, prior_structure_n_effective=20 (realistic deployment)

This comparison reveals whether off-diagonal correlations provide value:
- Independently (useful even without prior means)
- Synergistically (only useful when combined with prior means)
- Not at all (diagonal is sufficient)
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from banditgpt.bandit import BanditRouter, safe_inv

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

def create_router_with_custom_config(
    cov_type="full", 
    prior_mode="structure_only",
    cov_matrix=None, 
    n_offline=None
):
    """
    Create router with specific covariance and prior configuration.
    
    Args:
        cov_type: "full" or "diagonal"
        prior_mode: "structure_only" (b=0, A=20) or "full_csr" (b=20, A=20)
        cov_matrix: CSR covariance matrix (45x45)
        n_offline: Total offline samples
    """
    # Determine prior strengths based on mode
    if prior_mode == "structure_only":
        prior_n_eff = 0.0   # No prior means
        prior_struct_n_eff = 20.0  # CSR default structure
    elif prior_mode == "full_csr":
        prior_n_eff = 20.0  # CSR default means
        prior_struct_n_eff = 20.0  # CSR default structure
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")
    
    # Prepare covariance based on type
    if cov_type == "full":
        custom_cov = cov_matrix.copy()
    elif cov_type == "diagonal":
        custom_cov = np.diag(np.diag(cov_matrix))
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")
    
    # Scale covariance
    gamma_structure = prior_struct_n_eff / n_offline
    custom_cov_scaled = custom_cov * gamma_structure
    
    # Load registry
    models_path = Path(__file__).parent.parent.parent / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Create router with standard CSR priors
    router = BanditRouter.create(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        priors="csr",
        prior_n_effective=prior_n_eff,
        prior_structure_n_effective=prior_struct_n_eff,
        exploration="safe",
        ridge_lambda=1.0,
        forgetting_factor=1.0
    )
    
    # Override A matrices with custom covariance
    dim = router.bandit.dim
    cov_padded = np.eye(dim)
    cov_padded[:custom_cov_scaled.shape[0], :custom_cov_scaled.shape[1]] = custom_cov_scaled
    
    for model in router.bandit.models:
        # Subtract the default prior structure that was already added
        # Router already initialized with priors, we need to replace just the covariance part
        
        if prior_mode == "structure_only":
            # Reset A to ridge + custom covariance, reset b to zero
            router.bandit.A[model] = np.eye(dim) * router.bandit.ridge_lambda
            router.bandit.A[model] += cov_padded
            router.bandit.b[model] = np.zeros(dim)
        else:
            # full_csr: Keep the b vector (prior means), replace A matrix
            # Save current b vector
            saved_b = router.bandit.b[model].copy()
            # Reset A to ridge + custom covariance
            router.bandit.A[model] = np.eye(dim) * router.bandit.ridge_lambda
            router.bandit.A[model] += cov_padded
            # Restore b vector
            router.bandit.b[model] = saved_b
        
        # Recompute inverse
        router.bandit.A_inv[model] = safe_inv(router.bandit.A[model])
    
    return router

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

def run_comparison_ablation(num_trials=20):
    """
    Run comprehensive comparison across 4 conditions.
    """
    print("=" * 80)
    print("COMPREHENSIVE COVARIANCE ABLATION: Structure vs. Means")
    print("=" * 80)
    
    # Load data
    print("\n[1/4] Loading test data...")
    prompts, ground_truth = load_test_data()
    print(f"  Prompts: {len(prompts)}")
    
    # Load CSR covariance
    print("\n[2/4] Loading CSR covariance matrix...")
    priors_path = Path(__file__).parent.parent.parent / "priors" / "priors_meta_pca.npz"
    priors_data = np.load(priors_path)
    cov_matrix = priors_data['cov_matrix']
    n_offline = float(np.sum(priors_data['cluster_counts']))
    
    print(f"  Covariance shape: {cov_matrix.shape}")
    print(f"  N_offline: {n_offline:.0f}")
    print(f"  Mean diagonal: {np.mean(np.diag(cov_matrix)):.1f}")
    off_diag_mask = ~np.eye(cov_matrix.shape[0], dtype=bool)
    print(f"  Mean |off-diagonal|: {np.mean(np.abs(cov_matrix[off_diag_mask])):.1f}")
    
    # Run ablation
    print(f"\n[3/4] Running ablation ({num_trials} trials)...")
    print(f"  Configurations:")
    print(f"    1. Structure Only + Full Cov     (b=0,  A=20, Full Σ)")
    print(f"    2. Structure Only + Diagonal     (b=0,  A=20, Diag Σ)")
    print(f"    3. Full CSR + Full Cov           (b=20, A=20, Full Σ)")
    print(f"    4. Full CSR + Diagonal           (b=20, A=20, Diag Σ)")
    
    results = {
        "Structure Only + Full": [],
        "Structure Only + Diagonal": [],
        "Full CSR + Full": [],
        "Full CSR + Diagonal": []
    }
    
    for trial in range(num_trials):
        print(f"\n  Trial {trial + 1}/{num_trials}:")
        
        shuffled = prompts.copy()
        random.seed(trial)
        random.shuffle(shuffled)
        
        # Structure Only experiments
        print(f"    Structure Only (b=0)...")
        
        full_router = create_router_with_custom_config("full", "structure_only", cov_matrix, n_offline)
        full_regret = simulate_bandit(full_router, shuffled, ground_truth)
        results["Structure Only + Full"].append(full_regret)
        print(f"      Full Cov: {full_regret:.1f}")
        
        diag_router = create_router_with_custom_config("diagonal", "structure_only", cov_matrix, n_offline)
        diag_regret = simulate_bandit(diag_router, shuffled, ground_truth)
        results["Structure Only + Diagonal"].append(diag_regret)
        print(f"      Diagonal: {diag_regret:.1f}")
        
        benefit_struct = (diag_regret - full_regret) / diag_regret * 100 if diag_regret > 0 else 0
        print(f"      → Correlation benefit: {benefit_struct:.1f}%")
        
        # Full CSR experiments
        print(f"    Full CSR (b=20, A=20)...")
        
        full_csr_router = create_router_with_custom_config("full", "full_csr", cov_matrix, n_offline)
        full_csr_regret = simulate_bandit(full_csr_router, shuffled, ground_truth)
        results["Full CSR + Full"].append(full_csr_regret)
        print(f"      Full Cov: {full_csr_regret:.1f}")
        
        diag_csr_router = create_router_with_custom_config("diagonal", "full_csr", cov_matrix, n_offline)
        diag_csr_regret = simulate_bandit(diag_csr_router, shuffled, ground_truth)
        results["Full CSR + Diagonal"].append(diag_csr_regret)
        print(f"      Diagonal: {diag_csr_regret:.1f}")
        
        benefit_csr = (diag_csr_regret - full_csr_regret) / diag_csr_regret * 100 if diag_csr_regret > 0 else 0
        print(f"      → Correlation benefit: {benefit_csr:.1f}%")
    
    # Aggregate
    print(f"\n  Computing statistics...")
    aggregated = {}
    for config in results:
        aggregated[config] = {
            "mean": np.mean(results[config]),
            "std": np.std(results[config]),
            "trials": results[config]
        }
    
    return aggregated

def plot_comparison_results(results, output_path):
    """Create comparison plot"""
    print("\n[4/4] Generating comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Structure Only
    ax1 = axes[0]
    configs1 = ["Full\nCov", "Diagonal"]
    means1 = [results["Structure Only + Full"]["mean"],
              results["Structure Only + Diagonal"]["mean"]]
    stds1 = [results["Structure Only + Full"]["std"],
             results["Structure Only + Diagonal"]["std"]]
    
    colors1 = ["#27ae60", "#f39c12"]
    x_pos1 = np.arange(len(configs1))
    bars1 = ax1.bar(x_pos1, means1, yerr=stds1, color=colors1, alpha=0.8, capsize=8, width=0.5)
    
    for bar, mean, std in zip(bars1, means1, stds1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mean:.1f}\n±{std:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel("Cumulative Regret", fontsize=12, fontweight='bold')
    ax1.set_title("Structure Only (b=0, A=20)", fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos1)
    ax1.set_xticklabels(configs1, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    struct_benefit = (means1[1] - means1[0]) / means1[1] * 100 if means1[1] > 0 else 0
    ax1.text(0.95, 0.95, f"Benefit: {struct_benefit:.1f}%",
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 2: Full CSR
    ax2 = axes[1]
    configs2 = ["Full\nCov", "Diagonal"]
    means2 = [results["Full CSR + Full"]["mean"],
              results["Full CSR + Diagonal"]["mean"]]
    stds2 = [results["Full CSR + Full"]["std"],
             results["Full CSR + Diagonal"]["std"]]
    
    colors2 = ["#3498db", "#e74c3c"]
    x_pos2 = np.arange(len(configs2))
    bars2 = ax2.bar(x_pos2, means2, yerr=stds2, color=colors2, alpha=0.8, capsize=8, width=0.5)
    
    for bar, mean, std in zip(bars2, means2, stds2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mean:.1f}\n±{std:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel("Cumulative Regret", fontsize=12, fontweight='bold')
    ax2.set_title("Full CSR (b=20, A=20)", fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(configs2, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    csr_benefit = (means2[1] - means2[0]) / means2[1] * 100 if means2[1] > 0 else 0
    ax2.text(0.95, 0.95, f"Benefit: {csr_benefit:.1f}%",
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle("Covariance Ablation: Off-Diagonal Impact Analysis", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")

def main():
    """Main execution"""
    results = run_comparison_ablation(num_trials=20)
    
    output_path = Path(__file__).parent / "covariance_ablation_comparison.png"
    plot_comparison_results(results, output_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n📊 Structure Only (b=0, A=20):")
    struct_full = results["Structure Only + Full"]["mean"]
    struct_diag = results["Structure Only + Diagonal"]["mean"]
    struct_benefit = (struct_diag - struct_full) / struct_diag * 100 if struct_diag > 0 else 0
    print(f"  Full Covariance:  {struct_full:.1f} ± {results['Structure Only + Full']['std']:.1f}")
    print(f"  Diagonal Only:    {struct_diag:.1f} ± {results['Structure Only + Diagonal']['std']:.1f}")
    print(f"  → Correlation Benefit: {struct_benefit:.1f}%")
    
    print("\n📊 Full CSR (b=20, A=20):")
    csr_full = results["Full CSR + Full"]["mean"]
    csr_diag = results["Full CSR + Diagonal"]["mean"]
    csr_benefit = (csr_diag - csr_full) / csr_diag * 100 if csr_diag > 0 else 0
    print(f"  Full Covariance:  {csr_full:.1f} ± {results['Full CSR + Full']['std']:.1f}")
    print(f"  Diagonal Only:    {csr_diag:.1f} ± {results['Full CSR + Diagonal']['std']:.1f}")
    print(f"  → Correlation Benefit: {csr_benefit:.1f}%")
    
    print("\n🔬 Interpretation:")
    if struct_benefit > 10 and csr_benefit > 10:
        print("  ✓ Off-diagonal correlations are valuable INDEPENDENTLY of prior means")
        print("    → Structure itself encodes useful transfer learning")
    elif csr_benefit > 10 and struct_benefit < 10:
        print("  ✓ Off-diagonal correlations require prior means to be useful")
        print("    → Correlations and beliefs work SYNERGISTICALLY")
    elif struct_benefit < 10 and csr_benefit < 10:
        print("  ⚠ Off-diagonal correlations provide minimal benefit")
        print("    → Diagonal variances may be sufficient")
    
    print("\n✅ COMPARISON COMPLETE!")

if __name__ == "__main__":
    main()
