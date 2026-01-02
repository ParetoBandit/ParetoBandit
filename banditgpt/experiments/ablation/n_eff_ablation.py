#!/usr/bin/env python3
"""
Prior Strength (γ_prior) Ablation Study

Compares CSR vs. HLE priors while varying the strength of the prior means (b vector).
This tests the incremental value of prior means on top of structural priors.

Setup:
- γ_structure = 20 (fixed covariance structure strength, N_target = 20)
- γ_prior varies from 0 to 50 (controls b vector strength)

Hypothesis:
- CSR: Superior prior means → lower regret as γ_prior increases
- HLE: Generic prior means → less benefit from increased γ_prior

This demonstrates the value of task-specific cluster success rates over generic benchmarks.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from multiprocessing import Pool, cpu_count

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from banditgpt import BanditRouter

def load_test_data():
    """Load test rewards and prompts"""
    # Script is in banditgpt/experiments/ablation/
    # parent.parent.parent.parent -> repo root
    repo_root = Path(__file__).parent.parent.parent.parent
    test_rewards_path = repo_root / "banditgpt" / "data" / "test_rewards_pareto_dedup.jsonl"
    
    rewards_data = []
    with open(test_rewards_path) as f:
        for line in f:
            rewards_data.append(json.loads(line))
    
    # Extract prompts and ground truth
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

def simulate_bandit(router, prompts, ground_truth, model_ids):
    """Run bandit simulation and return final cumulative regret"""
    cumulative_regret = 0.0
    
    for prompt in prompts:
        # Get router's choice
        selected_model_id, log = router.route(prompt, profile="balanced", input_tokens=100)
        
        # Get ground truth
        true_rewards = ground_truth[prompt]
        best_reward = max(true_rewards.values())
        selected_reward = true_rewards.get(selected_model_id, 0.0)
        
        # Calculate regret
        regret = best_reward - selected_reward
        cumulative_regret += regret
        
        # Update router with feedback
        router.process_feedback(log.request_id, selected_reward)
    
    return cumulative_regret

def load_priors_metadata():
    """Load priors metadata (N_offline, Covariance)"""
    # Script is in banditgpt/experiments/ablation/
    # parent.parent.parent.parent -> repo root
    repo_root = Path(__file__).parent.parent.parent.parent
    priors_path = repo_root / "banditgpt" / "priors" / "priors_meta_pca.npz"
    
    if not priors_path.exists():
        print(f"Warning: {priors_path} not found. Using defaults.")
        return 21000.0, None
        
    data = np.load(priors_path)
    cov_matrix = data["cov_matrix"]
    
    # Calculate N_offline EXACTLY (matching Run 16 logic)
    if 'global_sum' in data:
         # Bias term sum = N
         n_offline = float(data['global_sum'][-1])
    elif 'cluster_counts' in data and np.sum(data['cluster_counts']) > 0:
         n_offline = float(np.sum(data['cluster_counts']))
    else:
         n_offline = 21000.0
         
    return n_offline, cov_matrix

def run_single_trial(args):
    """Helper function to run a single trial for one strategy (for multiprocessing)"""
    strategy, gamma_prior, trial, prompts, ground_truth, model_ids, registry = args
    
    # Shuffle prompts with consistent seed
    shuffled = prompts.copy()
    random.seed(int(gamma_prior * 1000) + trial)
    random.shuffle(shuffled)
    
    # Create router based on strategy
    if strategy == "CSR":
        router = BanditRouter.create(
            registry,
            exploration="safe",
            priors="benchmark",
            prior_n_effective=float(gamma_prior),
            prior_structure_n_effective=20.0
        )
    else:  # HLE
        router = BanditRouter.create(
            registry,
            exploration="safe",
            priors="hle",
            prior_n_effective=float(gamma_prior),
            prior_structure_n_effective=20.0
        )
    
    regret = simulate_bandit(router, shuffled, ground_truth, model_ids)
    return strategy, gamma_prior, regret


def run_ablation(gamma_prior_values, num_trials=20):
    """Run ablation study across gamma_prior (prior_n_effective) values"""
    print("=" * 70)
    print("PRIOR STRENGTH (γ_prior) ABLATION STUDY")
    print("=" * 70)
    print("\nConfiguration:")
    print("  γ_structure = 20 (moderate stiffness, allows beliefs to influence routing)")
    print("  γ_prior varies from 0 to 50 (controls b vector strength)")
    
    # Load data
    print("\n[1/3] Loading test data...")
    prompts, ground_truth = load_test_data()
    model_ids = list(next(iter(ground_truth.values())).keys())
    
    print(f"  Prompts: {len(prompts)}")
    print(f"  Models: {len(model_ids)}")
    
    # Load registry
    repo_root = Path(__file__).parent.parent.parent.parent
    models_path = repo_root / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Run ablation
    print(f"\n[2/3] Running ablation study...")
    print(f"  γ_prior values: {gamma_prior_values}")
    print(f"  Trials per value: {num_trials}")
    print(f"  Total simulations: {len(gamma_prior_values) * num_trials * 2} (CSR + HLE)")
    
    results = {
        "CSR Priors": {n: [] for n in gamma_prior_values},
        "HLE Priors": {n: [] for n in gamma_prior_values}
    }
    
    for gamma_prior in gamma_prior_values:
        print(f"\n  Testing γ_prior = {gamma_prior}:")
        
        for trial in range(num_trials):
            if trial % 5 == 0:
                print(f"    Trial {trial + 1}/{num_trials}...", end='\r')
            
            # Shuffle prompts
            shuffled = prompts.copy()
            random.seed(int(gamma_prior * 1000) + trial)  # Different seed per gamma_prior
            random.shuffle(shuffled)
            
            # Test CSR (Cluster Success Rates)
            # γ_structure = 20 (fixed covariance structure)
            # γ_prior varies (tests quality of prior means)
            # Ridge is automatically scaled to 10x init_scale
            csr_router = BanditRouter.create(
                registry,
                exploration="safe",
                priors="csr",  # Cluster Success Rates (task-specific)
                prior_n_effective=float(gamma_prior),  # γ_prior: controls b vector
                prior_structure_n_effective=20.0  # γ_structure: controls A matrix (ridge auto-scales)
            )
            csr_regret = simulate_bandit(csr_router, shuffled, ground_truth, model_ids)
            results["CSR Priors"][gamma_prior].append(csr_regret)
            
            # Test HLE
            # Ridge is automatically scaled to 10x init_scale
            hle_router = BanditRouter.create(
                registry,
                exploration="safe",
                priors="hle",  # Generic HLE benchmark scores
                prior_n_effective=float(gamma_prior),  # γ_prior: controls b vector
                prior_structure_n_effective=20.0  # γ_structure: controls A matrix (ridge auto-scales)
            )
            hle_regret = simulate_bandit(hle_router, shuffled, ground_truth, model_ids)
            results["HLE Priors"][gamma_prior].append(hle_regret)
        
        csr_mean = np.mean(results["CSR Priors"][gamma_prior])
        hle_mean = np.mean(results["HLE Priors"][gamma_prior])
        print(f"    CSR={csr_mean:.1f}, HLE={hle_mean:.1f}                ")
    
    # Aggregate statistics
    print(f"\n  Computing statistics...")
    aggregated = {}
    for strategy in ["CSR Priors", "HLE Priors"]:
        aggregated[strategy] = {
            "gamma_prior_values": gamma_prior_values,
            "means": [np.mean(results[strategy][n]) for n in gamma_prior_values],
            "stds": [np.std(results[strategy][n]) for n in gamma_prior_values]
        }
    
    return aggregated

def plot_ablation(results, output_path):
    """Create ablation plot"""
    print("\n[3/3] Generating plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot each strategy
    colors = {
        "CSR Priors": "#27ae60",  # Green - task-specific
        "HLE Priors": "#f39c12"   # Orange - generic
    }
    
    markers = {
        "CSR Priors": "o",
        "HLE Priors": "s"
    }
    
    for strategy in ["CSR Priors", "HLE Priors"]:
        data = results[strategy]
        x = data["gamma_prior_values"]
        means = data["means"]
        stds = data["stds"]
        
        ax.plot(x, means, label=strategy, color=colors[strategy], 
                linewidth=2.5, marker=markers[strategy], markersize=8)
        ax.fill_between(x, 
                        np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds), 
                        alpha=0.2, color=colors[strategy])
    
    ax.set_xlabel("Prior Strength ($\\gamma_{prior}$)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Final Cumulative Regret", fontsize=13, fontweight='bold')
    ax.set_title("CSR vs. HLE: Quality of Prior Means", fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation
    ax.text(0.02, 0.98,
            "Configuration:\n"
            "γ_structure = 20 (fixed)\n"
            "γ_prior varies (0→50)\n\n"
            "Expected: CSR outperforms HLE\n"
            "→ Task-specific priors matter",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

def main():
    # Configuration
    GAMMA_PRIOR_VALUES = [0, 1, 5, 10, 20, 40, 50]
    NUM_TRIALS = 1  # Quick test after bug fix
    
    # Run ablation
    results = run_ablation(GAMMA_PRIOR_VALUES, NUM_TRIALS)
    
    # Plot
    output_path = Path(__file__).parent / "prior_strength_ablation.png"
    plot_ablation(results, output_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Prior Strength (γ_prior) Impact")
    print("=" * 70)
    
    csr_means = results["CSR Priors"]["means"]
    hle_means = results["HLE Priors"]["means"]
    
    print(f"\nCSR Priors (Task-Specific):")
    for n, mean in zip(GAMMA_PRIOR_VALUES, csr_means):
        print(f"  γ_prior={n:2d}: {mean:.1f}")
    
    csr_range = max(csr_means) - min(csr_means)
    csr_baseline = csr_means[0]  # γ_prior=0 (cold start)
    csr_improvement = ((csr_baseline - min(csr_means)) / csr_baseline) * 100 if csr_baseline > 0 else 0
    
    print(f"\n  Best regret: {min(csr_means):.1f} (at γ_prior={GAMMA_PRIOR_VALUES[csr_means.index(min(csr_means))]})")
    print(f"  Improvement over cold start: {csr_improvement:.1f}%")
    
    print(f"\nHLE Priors (Generic):")
    for n, mean in zip(GAMMA_PRIOR_VALUES, hle_means):
        print(f"  γ_prior={n:2d}: {mean:.1f}")
    
    hle_range = max(hle_means) - min(hle_means)
    hle_baseline = hle_means[0]  # γ_prior=0 (cold start)
    hle_improvement = ((hle_baseline - min(hle_means)) / hle_baseline) * 100 if hle_baseline > 0 else 0
    
    print(f"\n  Best regret: {min(hle_means):.1f} (at γ_prior={GAMMA_PRIOR_VALUES[hle_means.index(min(hle_means))]})")
    print(f"  Improvement over cold start: {hle_improvement:.1f}%")
    
    print(f"\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"CSR improvement: {csr_improvement:.1f}%")
    print(f"HLE improvement: {hle_improvement:.1f}%")
    
    if csr_improvement > hle_improvement:
        ratio = csr_improvement / hle_improvement if hle_improvement > 0 else float('inf')
        print(f"\n✓ CSR priors are {ratio:.1f}x more effective than HLE priors")
        print(f"  → Task-specific cluster success rates provide superior initialization")
    
    # Check if CSR consistently beats HLE
    csr_wins = sum(1 for c, h in zip(csr_means, hle_means) if c < h)
    print(f"\nCSR outperforms HLE in {csr_wins}/{len(GAMMA_PRIOR_VALUES)} configurations")
    
    print("\n✅ ABLATION COMPLETE!")

if __name__ == "__main__":
    main()
