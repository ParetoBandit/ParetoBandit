#!/usr/bin/env python3
"""
Figure 1: Value of Benchmark Initialization

Uses the ACTUAL BanditRouter from the library to test if benchmark initialization helps.

Compares:
1. With Benchmarks (Metadata-Guided): Initialized with 3-benchmark averages
2. Without Benchmarks (Pure Cold Start): Uniform initialization (no metadata)

Shows whether public benchmark scores provide a meaningful advantage in cold start.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

# Add library to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from banditgpt import load_default_registry
from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path

# Publication-quality plot settings
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})


def load_data():
    """Load the full dataset and pre-compute embeddings once."""
    print("Loading dataset...")
    
    prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
    prompts = []
    cluster_ids = []
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            cluster_ids.append(data["cluster_id"])
    
    print(f"  Loaded {len(prompts)} prompts")
    
    # Pre-compute embeddings ONCE
    print("  Computing embeddings (this takes ~30 seconds)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = encoder.encode(prompts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    rewards_path = get_priors_path("archetype_grid_dense_run.jsonl")
    rewards = {}
    models = set()
    
    with open(rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model = data["model_id"]
                cluster = data["cluster_id"]
                logit = data.get("reward_logit", 0.0)
                reward = 1.0 / (1.0 + np.exp(-logit))
                
                if cluster not in rewards:
                    rewards[cluster] = {}
                rewards[cluster][model] = reward
                models.add(model)
    
    return embeddings, cluster_ids, rewards, sorted(models)


def evaluate_policy(policy, test_embeddings, test_clusters, test_rewards, 
                    model_names, n_rounds=2000, seed=42):
    """Evaluate a LinUCB policy and return regret curve."""
    rng = np.random.default_rng(seed)
    rng_env = np.random.default_rng(seed + 1000)
    
    cum_regret = 0.0
    regret_curve = []
    
    n_test = len(test_embeddings)
    test_emb_copy = test_embeddings.copy()
    test_clusters_copy = test_clusters.copy()
    
    for t in range(n_rounds):
        idx = t % n_test
        
        # Reshuffle after each epoch
        if t % n_test == 0 and t > 0:
            perm = rng_env.permutation(n_test)
            test_clusters_copy = [test_clusters[i] for i in perm]
            test_emb_copy = test_embeddings[perm]
        
        embedding = test_emb_copy[idx]
        cluster = test_clusters_copy[idx]
        
        # Get optimal reward for this cluster
        optimal = max([test_rewards.get(cluster, {}).get(m, 0.0) for m in model_names])
        
        # Select model using LinUCB
        selected_model, _, _ = policy.select_arm(embedding, rng=rng)
        
        # Get ground truth reward (with small noise)
        reward = test_rewards.get(cluster, {}).get(selected_model, 0.5)
        reward += rng.standard_normal() * 0.02
        reward = np.clip(reward, 0.0, 1.0)
        
        # Update policy
        policy.update(selected_model, embedding, reward)
        
        # Calculate regret (using expected reward, not noisy)
        expected = test_rewards.get(cluster, {}).get(selected_model, 0.5)
        cum_regret += optimal - expected
        regret_curve.append(cum_regret)
    
    return regret_curve


def run_single_fold(all_embeddings, all_clusters, all_rewards, model_names, 
                   fold_idx, n_folds=5):
    """Run a single fold of cross-validation comparing initialization strategies."""
    n_samples = len(all_embeddings)
    fold_size = n_samples // n_folds
    
    test_start = fold_idx * fold_size
    test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples
    
    test_indices = list(range(test_start, test_end))
    train_indices = [i for i in range(n_samples) if i not in test_indices]
    
    train_embeddings = all_embeddings[train_indices]
    train_clusters = [all_clusters[i] for i in train_indices]
    test_embeddings = all_embeddings[test_indices]
    test_clusters = [all_clusters[i] for i in test_indices]
    
    print(f"\nFold {fold_idx + 1}/{n_folds}")
    print(f"  Train: {len(train_embeddings)} prompts")
    print(f"  Test:  {len(test_embeddings)} prompts")
    
    # Load registry with benchmark scores
    registry = load_default_registry()
    
    # Filter to only models in our dataset
    filtered_registry = {k: v for k, v in registry.items() if k in model_names}
    
    print(f"  Models: {len(filtered_registry)}")
    
    dim = train_embeddings.shape[1]
    
    # =========================================================================
    # Condition 1: WITH Benchmarks (Metadata-Guided)
    # =========================================================================
    print("  Running: WITH Benchmarks (Metadata-Guided)...")
    policy_with_benchmarks = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=0.5
    )
    
    # Initialize b vectors with benchmark averages
    for model in model_names:
        if model in filtered_registry:
            bench_avg = filtered_registry[model]['benchmarks']['average']
            # Initialize b = benchmark_avg * mean_embedding
            mean_emb = train_embeddings.mean(axis=0)
            policy_with_benchmarks.b[model] = bench_avg * mean_emb
    
    regret_with_benchmarks = evaluate_policy(
        policy_with_benchmarks,
        test_embeddings,
        test_clusters,
        all_rewards,
        model_names,
        n_rounds=2000,
        seed=fold_idx * 1000
    )
    
    # =========================================================================
    # Condition 2: WITHOUT Benchmarks (Pure Cold Start)
    # =========================================================================
    print("  Running: WITHOUT Benchmarks (Pure Cold Start)...")
    policy_no_benchmarks = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=0.5
    )
    # Don't initialize b vectors - they start at zero (uniform prior)
    
    regret_no_benchmarks = evaluate_policy(
        policy_no_benchmarks,
        test_embeddings,
        test_clusters,
        all_rewards,
        model_names,
        n_rounds=2000,
        seed=fold_idx * 1000
    )
    
    # Calculate benefit of benchmarks
    final_with = regret_with_benchmarks[-1]
    final_without = regret_no_benchmarks[-1]
    regret_reduction = 100 * (final_without - final_with) / final_without
    
    print(f"  Final regret: With Benchmarks={final_with:.1f}, "
          f"Without Benchmarks={final_without:.1f} "
          f"(Reduction: {regret_reduction:.1f}%)")
    
    return {
        'regret_with_benchmarks': regret_with_benchmarks,
        'regret_no_benchmarks': regret_no_benchmarks,
        'regret_reduction_pct': regret_reduction
    }


def main():
    """Run 5-fold cross-validation and generate figure."""
    print("=" * 80)
    print("Figure 1: Value of Benchmark Initialization")
    print("=" * 80)
    
    # Load data (embeddings pre-computed once)
    embeddings, cluster_ids, rewards, model_names = load_data()
    
    print(f"\nDataset: {len(embeddings)} prompts, {len(model_names)} models")
    
    # Run 5-fold CV
    n_folds = 5
    fold_results = []
    
    for fold_idx in range(n_folds):
        result = run_single_fold(embeddings, cluster_ids, rewards, model_names, fold_idx, n_folds)
        fold_results.append(result)
    
    # =========================================================================
    # Aggregate results across folds
    # =========================================================================
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    
    all_with_benchmarks = np.array([r['regret_with_benchmarks'] for r in fold_results])
    all_no_benchmarks = np.array([r['regret_no_benchmarks'] for r in fold_results])
    all_reductions = np.array([r['regret_reduction_pct'] for r in fold_results])
    
    mean_with = all_with_benchmarks.mean(axis=0)
    mean_without = all_no_benchmarks.mean(axis=0)
    ci_with = 1.96 * all_with_benchmarks.std(axis=0) / np.sqrt(n_folds)
    ci_without = 1.96 * all_no_benchmarks.std(axis=0) / np.sqrt(n_folds)
    
    mean_reduction = all_reductions.mean()
    ci_reduction = 1.96 * all_reductions.std() / np.sqrt(n_folds)
    
    print(f"\nRegret Reduction from Benchmark Initialization:")
    print(f"  Mean: {mean_reduction:.1f}%")
    print(f"  95% CI: [{mean_reduction - ci_reduction:.1f}%, {mean_reduction + ci_reduction:.1f}%]")
    
    # Statistical test (is reduction > 0?)
    t_stat, p_value = stats.ttest_1samp(all_reductions, 0.0, alternative='greater')
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value (one-sided): {p_value:.3f}")
    print(f"  Directional consistency: {sum(all_reductions > 0)}/{n_folds} folds show benefit")
    
    # =========================================================================
    # Create figure
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Regret curves
    steps = np.arange(len(mean_with))
    
    ax1.plot(steps, mean_with, 'g-', linewidth=2.5, label='With Benchmarks (Metadata-Guided)')
    ax1.fill_between(steps, mean_with - ci_with, mean_with + ci_with,
                     alpha=0.2, color='green')
    
    ax1.plot(steps, mean_without, 'b--', linewidth=2.5, label='Without Benchmarks (Pure Cold Start)')
    ax1.fill_between(steps, mean_without - ci_without, mean_without + ci_without,
                     alpha=0.2, color='blue')
    
    ax1.set_xlabel('Routing Decisions', fontweight='bold')
    ax1.set_ylabel('Cumulative Regret (Mean ± 95% CI)', fontweight='bold')
    ax1.set_title('A) Regret Curves: Benchmark Initialization Reduces Regret', 
                  fontweight='bold', pad=15)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Strip plot showing regret reduction
    rng = np.random.default_rng(42)
    jitter = 0.05
    
    x_reductions = np.ones(n_folds) * 1 + rng.uniform(-jitter, jitter, n_folds)
    
    # Plot individual dots
    ax2.scatter(x_reductions, all_reductions, s=150, alpha=0.6, color='green', 
                edgecolors='darkgreen', linewidths=2, zorder=3)
    
    # Plot mean and CI
    ax2.errorbar([1], [mean_reduction], yerr=[[ci_reduction], [ci_reduction]],
                fmt='D', markersize=12, color='darkgreen', markeredgecolor='black',
                linewidth=3, capsize=10, capthick=3, zorder=4,
                label=f'Mean: {mean_reduction:.1f}%')
    
    # Reference line at y=0
    ax2.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5,
                label='No Benefit')
    
    # Annotations
    if sum(all_reductions > 0) == n_folds:
        color = 'green'
        msg = f'{sum(all_reductions > 0)}/{n_folds} folds\nshow benefit'
    else:
        color = 'orange'
        msg = f'{sum(all_reductions > 0)}/{n_folds} folds\nshow benefit'
    
    ax2.text(1.15, max(all_reductions) * 0.9, 
             msg,
             fontsize=11, fontweight='bold', color=color,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
    
    ax2.text(1.15, mean_reduction,
             f'p={p_value:.3f}',
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax2.set_xlim([0.7, 1.4])
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Benchmark\nInitialization'])
    ax2.set_ylabel('Regret Reduction (%)', fontweight='bold')
    ax2.set_title('B) Consistency: Benchmarks Provide Value', 
                  fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent
    output_path = output_dir / "figure1_library_version.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_path}")
    
    # Save statistics
    stats_data = {
        'mean_regret_reduction_pct': float(mean_reduction),
        'ci_95_pct': float(ci_reduction),
        'p_value_one_sided': float(p_value),
        't_statistic': float(t_stat),
        'folds_showing_benefit': int(sum(all_reductions > 0)),
        'total_folds': int(n_folds),
        'fold_results': [float(x) for x in all_reductions],
        'interpretation': 'Positive values mean benchmarks reduce regret'
    }
    
    stats_path = output_dir / "figure1_library_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    print(f"✅ Statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()

