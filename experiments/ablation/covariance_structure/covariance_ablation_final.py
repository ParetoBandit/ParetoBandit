#!/usr/bin/env python3
"""
Covariance Structure Ablation: CORRECT Implementation

Load CSR covariance matrix directly from priors file and inject it into routers.
Test with N_eff=0 (no prior means) to isolate covariance contribution.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from banditgpt.bandit import BanditRouter

def load_test_data():
    """Load test rewards and prompts"""
    data_dir = Path(__file__).parent.parent.parent.parent / "banditgpt" / "data"
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

def load_csr_covariance(registry):
    """Load CSR covariance matrix directly from priors file"""
    priors_path = Path(__file__).parent.parent.parent.parent / "banditgpt" / "priors" / "priors_meta_pca.npz"
    
    # Load the priors
    priors_data = np.load(priors_path)
    cov_matrix = priors_data['cov_matrix']  # CSR covariance matrix
    
    print(f"  Loaded CSR covariance: {cov_matrix.shape}")
    print(f"  Mean diagonal: {np.mean(np.diag(cov_matrix)):.4f}")
    print(f"  Mean off-diagonal: {np.mean(cov_matrix[np.triu_indices_from(cov_matrix, k=1)]):.4f}")

    
    # Calculate exact total samples (N_offline)
    if 'cluster_counts' in priors_data and np.sum(priors_data['cluster_counts']) > 0:
        n_offline = float(np.sum(priors_data['cluster_counts']))
        print(f"  Exact N_offline (from cluster_counts): {n_offline}")
    elif 'global_sum' in priors_data:
        # Fallback: global_sum is Sum(feature_vectors). The last element is the Bias term (1.0).
        # So global_sum[-1] = Sum(1.0) = N_total.
        n_offline = float(priors_data['global_sum'][-1])
        print(f"  Exact N_offline (from global_sum[-1]): {n_offline}")
    else:
        # Fallback if specific counts not found
        n_offline = 21000.0
        print(f"  Approximated N_offline: {n_offline} (cluster_counts/global_sum missing)")
    
    if n_offline <= 0:
        print(f"  WARNING: Derived N_offline is {n_offline}. Forcing 21000.0.")
        n_offline = 21000.0
    
    return cov_matrix, n_offline

def create_router_with_injected_covariance(registry, cov_matrix, n_offline, cov_type="full"):
    """
    Create router with custom covariance matrix for ablation study.
    
    Args:
        registry: Model registry
        cov_matrix: CSR covariance matrix
        n_offline: Total samples in offline dataset
        cov_type: "full", "diagonal", or "identity"
    """
    # Prepare the appropriate covariance structure
    # SCALING FIX: The raw covariance matrix has magnitude ~7300 (equivalent to N samples).
    # Combined with N_eff=0, this creates "Infinite Stiffness" (frozen bandit).
    # We must scale the matrix to a reasonable prior strength (N_target) to allow learning
    # while preserving the GEOMETRY (correlations) of the prior.
    
    N_target = 20.0      # The strength we WANT (allows learning)
    gamma_structure = N_target / n_offline
    
    if cov_type == "full":
        # Use complete CSR covariance with all correlations, scaled by gamma_structure
        custom_cov = cov_matrix.copy() * gamma_structure
        
    elif cov_type == "diagonal":
        # Use only diagonal of CSR covariance (variances only), scaled
        custom_cov = np.diag(np.diag(cov_matrix)) * gamma_structure
        
    elif cov_type == "identity":
        # Use Identity matrix (no correlations), but SCALED to match the average variance of CSR.
        # This ensures fair comparison of "Structure" vs "No Structure" at equal prior strength (Iso-Energy).
        mean_variance = np.mean(np.diag(cov_matrix))
        custom_cov = np.eye(cov_matrix.shape[0]) * mean_variance * gamma_structure
        
    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")
    
    # Create router with custom covariance
    # N_eff=0 ensures b vector stays at 0 (no prior means)
    router = BanditRouter.create(
        registry,
        exploration="safe",
        priors="benchmark",  # Use benchmark infrastructure
        prior_n_effective=0.0,  # No prior means (b=0)
        custom_covariance=custom_cov  # But use our custom covariance matrix
    )

    return router

def simulate_bandit(router, prompts, ground_truth, model_ids):
    """Run bandit simulation and return final cumulative regret"""
    cumulative_regret = 0.0
    
    for prompt in prompts:
        # Ensure we are passing raw text (router will compute embeddings internally)
        assert isinstance(prompt, str), "Prompt must be a raw string; embeddings should be computed inside the router"
        selected_model_id, log = router.route(
            prompt,
            profile="balanced",
            # input_tokens=100  <-- Removed hardcoded value to allow dynamic estimation from prompt length
        )
        
        # DEBUG: Print first 5 decisions
        if len(prompts) > 0 and prompts.index(prompt) < 5:
            print(f"    [Step {prompts.index(prompt)}] Model: {selected_model_id}, Pred: {log.predicted_utility:.4f}")

        
        true_rewards = ground_truth[prompt]
        best_reward = max(true_rewards.values())
        selected_reward = true_rewards.get(selected_model_id, 0.0)
        
        regret = best_reward - selected_reward
        cumulative_regret += regret
        router.process_feedback(log.request_id, selected_reward)
    
    return cumulative_regret

def run_covariance_ablation(num_trials=5):
    """Run the covariance ablation with a reduced number of trials (default 5).
    This speeds up debugging while still exercising all three covariance configurations.
    """
    """Test full vs diagonal vs identity covariance with N_eff=0"""
    print("=" * 70)
    print("COVARIANCE STRUCTURE ABLATION: Direct Injection")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading test data...")
    prompts, ground_truth = load_test_data()
    model_ids = list(next(iter(ground_truth.values())).keys())
    print(f"  Prompts: {len(prompts)}, Models: {len(model_ids)}")
    
    # Load registry
    models_path = Path(__file__).parent.parent.parent.parent / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Load CSR covariance
    print("\n[2/4] Loading CSR covariance matrix...")
    cov_matrix, n_offline = load_csr_covariance(registry)
    
    # Run ablation
    print(f"\n[3/4] Running ablation (N_eff=0, varying covariance)...")
    print(f"  Configurations:")
    print(f"    - Full: Complete Σ_CSR (with correlations)")
    print(f"    - Diagonal: diag(Σ_CSR) (variances only)")
    print(f"    - Identity: I (no structure)")
    print(f"  Trials: {num_trials}")
    
    results = {
        "Full Covariance": [],
        "Diagonal Only": [],
        "Identity": []
    }
    
    for trial in range(num_trials):
        print(f"\n  Trial {trial + 1}/{num_trials}:")
        
        shuffled = prompts.copy()
        random.seed(trial)
        random.shuffle(shuffled)
        
        # Full covariance
        print(f"    Full Covariance...", end=" ")
        full_router = create_router_with_injected_covariance(registry, cov_matrix, n_offline, "full")
        full_regret = simulate_bandit(full_router, shuffled, ground_truth, model_ids)
        results["Full Covariance"].append(full_regret)
        print(f"{full_regret:.1f}")
        
        # Diagonal only
        print(f"    Diagonal Only...", end=" ")
        diag_router = create_router_with_injected_covariance(registry, cov_matrix, n_offline, "diagonal")
        diag_regret = simulate_bandit(diag_router, shuffled, ground_truth, model_ids)
        results["Diagonal Only"].append(diag_regret)
        print(f"{diag_regret:.1f}")
        
        # Identity
        print(f"    Identity...", end=" ")
        identity_router = create_router_with_injected_covariance(registry, cov_matrix, n_offline, "identity")
        identity_regret = simulate_bandit(identity_router, shuffled, ground_truth, model_ids)
        results["Identity"].append(identity_regret)
        print(f"{identity_regret:.1f}")
        
        if diag_regret > full_regret:
            benefit = (diag_regret - full_regret) / diag_regret * 100
            print(f"    → Correlation benefit: {benefit:.1f}%")
    
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

def plot_results(results, output_path):
    """Create bar plot"""
    print("\n[4/4] Generating plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    configs = ["Full\nCovariance", "Diagonal\nOnly", "Identity"]
    means = [results["Full Covariance"]["mean"],
             results["Diagonal Only"]["mean"],
             results["Identity"]["mean"]]
    stds = [results["Full Covariance"]["std"],
            results["Diagonal Only"]["std"],
            results["Identity"]["std"]]
    
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8, capsize=10, width=0.6)
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mean:.1f}\n±{std:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel("Cumulative Regret", fontsize=13, fontweight='bold')
    ax.set_title("Covariance Ablation: Isolating Off-Diagonal Correlations", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotation
    full_mean = results["Full Covariance"]["mean"]
    diag_mean = results["Diagonal Only"]["mean"]
    id_mean = results["Identity"]["mean"]
    
    corr_benefit = (diag_mean - full_mean) / diag_mean * 100 if diag_mean > 0 else 0
    struct_benefit = (id_mean - full_mean) / id_mean * 100 if id_mean > 0 else 0
    
    ax.text(0.98, 0.98,
            f"Correlation benefit: {corr_benefit:.1f}%\n"
            f"Structure benefit: {struct_benefit:.1f}%\n\n"
            f"{'✓ Correlations matter!' if corr_benefit > 20 else '⚠ Weak correlation effect'}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved to: {output_path}")

def main():
    results = run_covariance_ablation(num_trials=30)
    
    output_path = Path(__file__).parent / "covariance_ablation_final.png"
    plot_results(results, output_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    full = results["Full Covariance"]["mean"]
    diag = results["Diagonal Only"]["mean"]
    identity = results["Identity"]["mean"]
    
    print(f"\nFull Covariance:  {full:.1f} ± {results['Full Covariance']['std']:.1f}")
    print(f"Diagonal Only:    {diag:.1f} ± {results['Diagonal Only']['std']:.1f}")
    print(f"Identity:         {identity:.1f} ± {results['Identity']['std']:.1f}")
    
    corr_benefit = (diag - full) / diag * 100 if diag > 0 else 0
    print(f"\n→ Correlation benefit: {corr_benefit:.1f}%")
    print(f"→ Architecture benefit: {(identity - full) / identity * 100:.1f}%")
    
    if corr_benefit > 30:
        print(f"\n✓ Off-diagonal correlations are CRITICAL ({corr_benefit:.0f}% improvement)")
    elif corr_benefit > 10:
        print(f"\n~ Correlations provide moderate benefit ({corr_benefit:.0f}%)")
    else:
        print(f"\n⚠ Weak correlation benefit ({corr_benefit:.0f}%)")
    
    print("\n✅ ABLATION COMPLETE!")

if __name__ == "__main__":
    main()
