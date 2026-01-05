#!/usr/bin/env python3
"""
Synergy Proof Ablation: Structure vs. Bias

This experiment compares 5 regimes to quantify the "Generalization Dividend" 
and the benefit of the Two-Knob architecture.

Regimes:
1. Cold Start: No priors (b=0, A=I)
2. Compass Only: CSR Means without Structure (b=CSR, A=I)
3. Terrain Only: Diagonal Variance without Correlations (b=0, A=diag(CSR))
4. Map Only: Full Covariance Structure without Means (b=0, A=full(CSR))
5. Full Synergy: Complete Navigation (b=CSR, A=full(CSR))
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
from scipy import stats

# Add the project root to sys.path so we can import banditgpt
# synergy_proof_experiment.py is in banditgpt/experiments/ablation/stiffness_calibration/
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root.parent))

from banditgpt.bandit import BanditRouter, safe_inv

def load_test_data():
    """Load test rewards and prompts"""
    data_dir = project_root / "data" / "offline_dataset"
    test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
    
    if not test_rewards_path.exists():
        # Fallback to banditgpt/data
        test_rewards_path = project_root / "data" / "test_rewards_pareto_dedup.jsonl"
        
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

def create_router_custom(config_name, cov_matrix, n_offline):
    """
    Create router with specific ablation configuration.
    """
    # Load registry
    models_path = project_root / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Defaults
    n_prior = 0.0
    n_struct = 0.0
    p_type = "csr"
    c_type = "full"
    
    if config_name == "Cold Start":
        n_prior = 0.0
        n_struct = 0.0
    elif config_name == "Compass Only":
        n_prior = 20.0
        n_struct = 0.0
        p_type = "csr" # Task-specific success rates
    elif config_name == "Terrain Only":
        n_prior = 0.0
        n_struct = 20.0
        c_type = "diagonal"
    elif config_name == "Map Only":
        n_prior = 0.0
        n_struct = 20.0
        c_type = "full"
    elif config_name == "Full Synergy":
        n_prior = 20.0
        n_struct = 20.0
        c_type = "full"
        
    router = BanditRouter.create(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        priors=p_type,
        prior_n_effective=n_prior,
        prior_structure_n_effective=n_struct,
        exploration="safe",
        ridge_lambda=1.0,
        forgetting_factor=1.0
    )
    
    # Apply special covariance override for "Variance Only" (Diagonal)
    if c_type == "diagonal":
        dim = router.bandit.dim
        # Extract diagonal elements from meta cov
        diag_cov = np.diag(np.diag(cov_matrix))
        gamma = n_struct / n_offline
        scaled_diag = diag_cov * gamma
        
        cov_padded = np.eye(dim)
        cov_padded[:scaled_diag.shape[0], :scaled_diag.shape[1]] = scaled_diag
        
        for model in router.bandit.models:
            # Re-initialize A to Ridge Floor + Scaled Diagonal
            router.bandit.A[model] = np.eye(dim) * 1.0 # Ridge floor lambda=1.0
            router.bandit.A[model] += cov_padded
            router.bandit.A_inv[model] = safe_inv(router.bandit.A[model])
            
    return router

def simulate_bandit(router, prompts, ground_truth, milestones):
    """Run bandit simulation and capture regret at milestones"""
    cumulative_regret = 0.0
    results = {}
    
    for i, prompt in enumerate(prompts):
        t = i + 1
        selected_model_id, log = router.route(prompt, profile="balanced")
        
        true_rewards = ground_truth[prompt]
        best_reward = max(true_rewards.values())
        
        # STRICT: No fallback - ensure we only use real data
        if selected_model_id not in true_rewards:
            raise ValueError(
                f"Model {selected_model_id} selected but has no ground truth reward for prompt. "
                f"This indicates incomplete data. Available models: {list(true_rewards.keys())}"
            )
        selected_reward = true_rewards[selected_model_id]
        
        regret = best_reward - selected_reward
        cumulative_regret += regret
        
        router.process_feedback(log.request_id, selected_reward)
        
        if t in milestones:
            results[t] = cumulative_regret
            
    return results

def run_synergy_ablation(num_trials=10):
    """Run the 5-way ablation study"""
    print("=" * 80)
    print("THE SYNERGY PROOF: Structure vs. Bias Ablation")
    print("=" * 80)
    
    prompts, ground_truth = load_test_data()
    
    priors_path = project_root / "priors" / "priors_meta_pca.npz"
    meta = np.load(priors_path)
    cov_matrix = meta['cov_matrix']
    n_offline = float(np.sum(meta['cluster_counts']))
    
    configs = [
        "Cold Start",
        "Compass Only",
        "Terrain Only",
        "Map Only",
        "Full Synergy"
    ]
    
    milestones = [100, 250, 500, 981]
    all_results = {c: {m: [] for m in milestones} for c in configs}
    
    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}:")
        shuffled = prompts.copy()
        random.seed(trial)
        random.shuffle(shuffled)
        
        for config in configs:
            router = create_router_custom(config, cov_matrix, n_offline)
            res = simulate_bandit(router, shuffled, ground_truth, milestones)
            for m in milestones:
                all_results[config][m].append(res[m])
            print(f"  {config:20}: R@500 = {res[500]:.1f}")
            
    # Aggregate stats
    aggregated = {}
    for config in configs:
        aggregated[config] = {}
        for m in milestones:
            aggregated[config][m] = {
                "mean": np.mean(all_results[config][m]),
                "std": np.std(all_results[config][m])
            }
    
    # Statistical significance testing
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (Paired t-test at T=500)")
    print("=" * 80)
    for i in range(len(configs)-1):
        c1, c2 = configs[i], configs[i+1]
        vals1 = all_results[c1][500]
        vals2 = all_results[c2][500]
        t_stat, p_val = stats.ttest_rel(vals1, vals2)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        delta = np.mean(vals1) - np.mean(vals2)
        print(f"  {c1:20} vs {c2:20}: Δ={delta:+6.1f}, p={p_val:.4f} {sig}")
            
    return aggregated, milestones

def plot_synergy_results(results, milestones, output_path):
    """
    Create professional KDD visualization of the synergy proof.
    """
    import matplotlib.style as style
    style.use('seaborn-v0_8-whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Bar Chart (T=500 Summary)
    configs = list(results.keys())
    m500 = [results[c][500]["mean"] for c in configs]
    s500 = [results[c][500]["std"] for c in configs]
    
    colors = ['#95a5a6', '#f39c12', '#e67e22', '#2980b9', '#27ae60']
    x_pos = np.arange(len(configs))
    
    bars = ax1.bar(x_pos, m500, yerr=s500, color=colors, alpha=0.9, capsize=7, edgecolor='black', linewidth=1)
    
    ax1.set_title("A. Impact Comparison (T=500)", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Cumulative Regret", fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs, rotation=30, ha='right', fontsize=12)
    
    # Annotate heights
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 2, f'{h:.1f}', ha='center', fontsize=11, fontweight='bold')

    # 2. Time Trajectories
    ax2.set_title("B. The Synergy Effect Over Time", fontsize=16, fontweight='bold', pad=15)
    styles = [('--', 'o'), ('--', 's'), (':', '^'), ('-', 'v'), ('-', 'P')]
    
    for (config, color, (ls, marker)) in zip(configs, colors, styles):
        means = [results[config][m]["mean"] for m in milestones]
        ax2.plot(milestones, means, label=config, color=color, linestyle=ls, marker=marker, linewidth=2.5, markersize=8)

    ax2.set_xlabel("Requests", fontsize=14)
    ax2.set_ylabel("Cumulative Regret", fontsize=14)
    ax2.legend(fontsize=12, frameon=True, shadow=True, loc='upper left')
    
    # Add Dividend Callouts
    dividend_off = results["Terrain Only"][500]["mean"] - results["Map Only"][500]["mean"]
    ax1.annotate(f'Generalization\nDividend: -{dividend_off:.1f}', 
                xy=(3.4, results["Map Only"][500]["mean"]),
                xytext=(3.5, results["Map Only"][500]["mean"] + 15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10, ha='center', bbox=dict(boxstyle='round', fc='white', ec='#2980b9'))
    
    synergy_jump = results["Map Only"][500]["mean"] - results["Full Synergy"][500]["mean"]
    ax1.annotate(f'Synergy Bonus\n(Means+Cov): -{synergy_jump:.1f}', 
                xy=(4.3, results["Full Synergy"][500]["mean"]),
                xytext=(4.5, results["Full Synergy"][500]["mean"] + 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10, ha='center', bbox=dict(boxstyle='round', fc='white', ec='#27ae60'))

    plt.suptitle("The Synergy Proof: Off-Diagonal Covariance & Informed Priors", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Professional Synergy Plot saved to: {output_path}")

def main():
    start_time = time.time()
    results, milestones = run_synergy_ablation(num_trials=10)
    
    output_path = Path(__file__).parent / "synergy_proof_ablation.png"
    plot_synergy_results(results, milestones, output_path)
    
    # Save results to JSON
    with open(Path(__file__).parent / "synergy_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nTotal Time: {time.time() - start_time:.2f}s")
    print("=" * 80)
    print("SYNERGY PROOF COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
