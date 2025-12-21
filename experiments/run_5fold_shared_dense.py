#!/usr/bin/env python3
"""
5-Fold Cross-Validation with Shared Covariance + Dense Training

Mathematical advantages:
- Parameters: 1,552 (vs 12M for disjoint)
- Training samples: ~32K per fold (vs ~400)
- Samples/param: ~21 (vs 0.00003)

This should demonstrate robust warm-start benefits.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from shared_covariance_policy import SharedCovarianceLinUCBPolicy
from banditgpt._resources import get_priors_path


def load_full_dataset():
    """Load complete dataset with dense evaluations."""
    # Load prompts
    prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
    cluster_ids = []
    with open(prompts_path) as f:
        for line in f:
            cluster_ids.append(json.loads(line)["cluster_id"])
    
    # Load embeddings
    embeddings = np.load(get_priors_path("full_embeddings_384.npy"))
    
    # Load dense rewards
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
    
    model_names = sorted(models)
    
    return embeddings, cluster_ids, rewards, model_names


def train_shared_dense(embeddings_pca, cluster_ids, rewards, model_names, epochs=3, seed=42):
    """Train shared policy on dense data."""
    dim = embeddings_pca.shape[1]
    policy = SharedCovarianceLinUCBPolicy(model_names, dim, alpha=0.5)
    
    rng = np.random.default_rng(seed)
    n_prompts = len(cluster_ids)
    
    for epoch in range(epochs):
        perm = rng.permutation(n_prompts)
        
        for idx in perm:
            embedding = embeddings_pca[idx]
            cluster = cluster_ids[idx]
            
            if cluster not in rewards:
                continue
            
            # Dense update: ALL models
            for model in model_names:
                if model in rewards[cluster]:
                    reward = rewards[cluster][model]
                    policy.update(model, embedding, reward)
    
    return policy


def evaluate_fold(policy_warm, test_embeddings_pca, test_clusters, test_rewards, model_names, strength, seed):
    """Evaluate on held-out fold."""
    dim = policy_warm.dim
    
    # Apply prior strength
    policy_warm.apply_strength(strength)
    
    # Cold start policy
    policy_cold = SharedCovarianceLinUCBPolicy(model_names, dim, alpha=0.5)
    
    cum_c, cum_w = 0.0, 0.0
    rng_c = np.random.default_rng(seed)
    rng_w = np.random.default_rng(seed + 1000)
    rng_env = np.random.default_rng(seed + 2000)
    
    n_test = len(test_clusters)
    n_rounds = 2000
    
    test_emb_copy = test_embeddings_pca.copy()
    test_clusters_copy = test_clusters.copy()
    
    for t in range(n_rounds):
        idx = t % n_test
        
        # Shuffle after each pass
        if t % n_test == 0 and t > 0:
            perm = rng_env.permutation(n_test)
            test_clusters_copy = [test_clusters[i] for i in perm]
            test_emb_copy = test_embeddings_pca[perm]
        
        ctx = test_emb_copy[idx]
        cluster = test_clusters_copy[idx]
        optimal = max([test_rewards.get(cluster, {}).get(m, 0.0) for m in model_names])
        
        # Cold start
        model_c = policy_cold.select(ctx, rng_c)
        reward_c = test_rewards.get(cluster, {}).get(model_c, 0.5)
        reward_c += rng_c.standard_normal() * 0.02
        reward_c = np.clip(reward_c, 0.0, 1.0)
        policy_cold.update(model_c, ctx, reward_c)
        
        expected_c = test_rewards.get(cluster, {}).get(model_c, 0.5)
        cum_c += optimal - expected_c
        
        # Warm start
        model_w = policy_warm.select(ctx, rng_w)
        reward_w = test_rewards.get(cluster, {}).get(model_w, 0.5)
        reward_w += rng_w.standard_normal() * 0.02
        reward_w = np.clip(reward_w, 0.0, 1.0)
        policy_warm.update(model_w, ctx, reward_w)
        
        expected_w = test_rewards.get(cluster, {}).get(model_w, 0.5)
        cum_w += optimal - expected_w
    
    reduction = 100.0 * (cum_c - cum_w) / cum_c if cum_c > 0 else 0.0
    return cum_c, cum_w, reduction


def main():
    print("=" * 70)
    print("5-Fold CV: Shared Covariance + Dense Training")
    print("=" * 70)
    print("Configuration:")
    print("  PCA: d=16")
    print("  Policy: Shared Covariance LinUCB")
    print("  Training: Dense (all 81 models per prompt)")
    print("  Epochs: 3")
    print("  Prior strength: λ=5 (lighter for shared)")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading full dataset...")
    all_embeddings, all_clusters, all_rewards, model_names = load_full_dataset()
    n_samples = len(all_clusters)
    print(f"  Samples: {n_samples}")
    print(f"  Models: {len(model_names)}")
    print()
    
    # Shuffle
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(n_samples)
    all_embeddings = all_embeddings[perm]
    all_clusters = [all_clusters[i] for i in perm]
    
    # Hyperparameters
    n_folds = 5
    pca_dim = 16
    epochs = 3
    prior_strength = 5.0  # Lighter for shared (less overfitting risk)
    
    results = []
    fold_size = n_samples // n_folds
    
    for fold in range(n_folds):
        print(f"{'='*70}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*70}")
        
        # Split
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        test_indices = list(range(test_start, test_end))
        train_indices = [i for i in range(n_samples) if i not in test_indices]
        
        train_embeddings = all_embeddings[train_indices]
        train_clusters = [all_clusters[i] for i in train_indices]
        
        test_embeddings = all_embeddings[test_indices]
        test_clusters = [all_clusters[i] for i in test_indices]
        
        print(f"  Train: {len(train_indices)} prompts")
        print(f"  Test: {len(test_indices)} prompts")
        
        # Fit PCA on training only
        print(f"  Fitting PCA (d={pca_dim})...")
        pca = PCA(n_components=pca_dim, random_state=42)
        train_pca = pca.fit_transform(train_embeddings)
        test_pca = pca.transform(test_embeddings)
        
        # Count dense training samples
        n_dense_train = sum(
            len(all_rewards.get(train_clusters[i], {}))
            for i in range(len(train_clusters))
        )
        print(f"  Dense training samples: {n_dense_train * epochs} ({n_dense_train} × {epochs} epochs)")
        
        # Train
        print(f"  Training shared policy...")
        policy = train_shared_dense(
            train_pca,
            train_clusters,
            all_rewards,
            model_names,
            epochs=epochs,
            seed=42 + fold * 1000,
        )
        
        # Evaluate
        print(f"  Evaluating (λ={prior_strength})...")
        cold, warm, reduction = evaluate_fold(
            policy,
            test_pca,
            test_clusters,
            all_rewards,
            model_names,
            prior_strength,
            seed=42 + fold * 10000,
        )
        
        results.append({
            "fold": fold + 1,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "n_dense_train": n_dense_train * epochs,
            "cold": cold,
            "warm": warm,
            "reduction": reduction,
        })
        
        status = "✓" if reduction > 0 else "✗"
        print(f"  {status} Cold: {cold:.1f}, Warm: {warm:.1f}")
        print(f"  {status} Regret Reduction: {reduction:+.1f}%")
        print()
    
    # Statistics
    reductions = [r["reduction"] for r in results]
    mean_reduction = np.mean(reductions)
    std_reduction = np.std(reductions, ddof=1)
    se_reduction = std_reduction / np.sqrt(n_folds)
    
    t_critical = stats.t.ppf(0.975, n_folds - 1)
    ci_lower = mean_reduction - t_critical * se_reduction
    ci_upper = mean_reduction + t_critical * se_reduction
    
    t_stat, p_value = stats.ttest_1samp(reductions, 0)
    
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Fold':<6} {'Train':<8} {'Test':<8} {'Dense':<10} {'Cold':<10} {'Warm':<10} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "✓" if r["reduction"] > 0 else "✗"
        print(f"{r['fold']:<6} {r['n_train']:<8} {r['n_test']:<8} {r['n_dense_train']:<10} "
              f"{r['cold']:<10.1f} {r['warm']:<10.1f} {r['reduction']:>+10.1f}%  {status}")
    
    print("=" * 70)
    print(f"Mean:           {mean_reduction:+.2f}%")
    print(f"Std Dev:        {std_reduction:.2f}%")
    print(f"Std Err:        {se_reduction:.2f}%")
    print(f"95% CI:         [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")
    print(f"t-statistic:    {t_stat:.3f}")
    print(f"p-value:        {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"Significance:   ✓ SIGNIFICANT (p < 0.05)")
    else:
        print(f"Significance:   ✗ Not significant (p ≥ 0.05)")
    
    print("=" * 70)
    
    # Paper-ready
    print()
    print("=" * 70)
    print("PAPER-READY RESULT:")
    print("=" * 70)
    
    if p_value < 0.05 and mean_reduction > 0:
        print(f"✓ Warm-start reduces regret by {mean_reduction:.1f}% ± {se_reduction:.1f}% (SE)")
        print(f"  95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        print(f"  Statistically significant (t={t_stat:.2f}, p={p_value:.4f})")
        print(f"  Method: Shared covariance + dense training ({results[0]['n_dense_train']} samples/fold)")
    elif mean_reduction > 0:
        print(f"○ Warm-start shows {mean_reduction:.1f}% ± {se_reduction:.1f}% reduction")
        print(f"  95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        print(f"  Trending positive but not significant (p={p_value:.4f})")
    else:
        print(f"✗ No warm-start benefit: {mean_reduction:.1f}% ± {se_reduction:.1f}%")
        print(f"  95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
    
    print("=" * 70)
    
    # Save
    output_dir = Path("results/5fold_shared_dense")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump({
            "config": {
                "pca_dim": pca_dim,
                "policy": "SharedCovarianceLinUCB",
                "training": "dense",
                "epochs": epochs,
                "prior_strength": prior_strength,
                "n_folds": n_folds,
            },
            "folds": results,
            "statistics": {
                "mean": float(mean_reduction),
                "std": float(std_reduction),
                "se": float(se_reduction),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            },
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'cv_results.json'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

