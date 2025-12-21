#!/usr/bin/env python3
"""
5-Fold Cross-Validation on FULL Dataset (497 prompts)

Uses the complete archetype_grid dataset with proper 80/20 splits per fold.
Each fold: ~398 train, ~99 test (matches our target split ratio).

Config: d=32, epochs=8, expert=65%, λ=9
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path
from run_rq1_pca import select_arm


def load_full_dataset():
    """Load the complete dataset (497 prompts)."""
    # Load full embeddings (or compute them)
    full_prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
    
    cluster_ids = []
    with open(full_prompts_path) as f:
        for line in f:
            data = json.loads(line)
            cluster_ids.append(data["cluster_id"])
    
    n_prompts = len(cluster_ids)
    print(f"  Loaded {n_prompts} prompts")
    
    # Load or compute full embeddings
    full_emb_cache = get_priors_path("full_embeddings_384.npy")
    
    if full_emb_cache.exists():
        print(f"  Loading cached full embeddings...")
        full_embeddings = np.load(full_emb_cache)
    else:
        print(f"  Computing embeddings for all {n_prompts} prompts...")
        from sentence_transformers import SentenceTransformer
        from banditgpt.core.bandit_router import DEFAULT_CONTEXT_MODEL
        
        prompts = []
        with open(full_prompts_path) as f:
            for line in f:
                prompts.append(json.loads(line)["prompt"])
        
        encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
        full_embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True)
        full_embeddings = np.asarray(full_embeddings, dtype=np.float64)
        
        np.save(full_emb_cache, full_embeddings)
        print(f"  Cached to {full_emb_cache.name}")
    
    print(f"  Embeddings shape: {full_embeddings.shape}")
    
    # Load full rewards
    full_rewards_path = get_priors_path("archetype_grid_dense_run.jsonl")
    
    model_set = set()
    truth = {}
    with open(full_rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model = data["model_id"]
                cluster = data["cluster_id"]
                logit = data.get("reward_logit", 0.0)
                reward = 1.0 / (1.0 + np.exp(-logit))
                if cluster not in truth:
                    truth[cluster] = {}
                truth[cluster][model] = reward
                model_set.add(model)
    
    model_names = sorted(model_set)
    print(f"  Models: {len(model_names)}")
    print(f"  Reward entries: {sum(len(v) for v in truth.values())}")
    
    return full_embeddings, cluster_ids, truth, model_names


def fit_pca_on_fold(train_embeddings, n_components=32):
    """Fit PCA on training fold."""
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_embeddings)
    return pca, train_pca


def train_priors(embeddings_pca, cluster_ids, truth, model_names, epochs, expert_rate, seed):
    """Train priors on PCA-reduced embeddings."""
    dim = embeddings_pca.shape[1]
    policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    rng = np.random.default_rng(seed)
    
    n_samples = len(cluster_ids)
    
    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        for idx in perm:
            x = embeddings_pca[idx]
            cluster = cluster_ids[idx]
            
            if rng.random() < expert_rate:
                # Get optimal model
                best_model = model_names[0]
                best_reward = truth.get(cluster, {}).get(best_model, 0.0)
                for m in model_names:
                    r = truth.get(cluster, {}).get(m, 0.0)
                    if r > best_reward:
                        best_reward, best_model = r, m
                model, reward = best_model, best_reward
            else:
                model = rng.choice(model_names)
                reward = truth.get(cluster, {}).get(model, 0.5)
            
            policy.update(model, x, reward)
    
    return policy


def evaluate_fold(policy, test_embeddings_pca, test_clusters, test_truth, model_names, strength, seed):
    """Evaluate on held-out fold."""
    dim = policy.dim
    
    # Apply prior strength
    for m in model_names:
        policy.A[m] = policy.A[m] * strength
        policy.b[m] = policy.b[m] * strength
        policy.A_inv[m] = np.linalg.inv(policy.A[m])
    
    # Run simulation
    policy_cold = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    policy_warm = policy
    
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
        optimal = max([test_truth.get(cluster, {}).get(m, 0.0) for m in model_names])
        
        # Cold start agent
        model_c = select_arm(policy_cold, ctx, rng_c)
        reward_c = test_truth.get(cluster, {}).get(model_c, 0.5)
        reward_c += rng_c.standard_normal() * 0.02
        reward_c = np.clip(reward_c, 0.0, 1.0)
        policy_cold.update(model_c, ctx, reward_c)
        
        expected_c = test_truth.get(cluster, {}).get(model_c, 0.5)
        cum_c += optimal - expected_c
        
        # Warm start agent
        model_w = select_arm(policy_warm, ctx, rng_w)
        reward_w = test_truth.get(cluster, {}).get(model_w, 0.5)
        reward_w += rng_w.standard_normal() * 0.02
        reward_w = np.clip(reward_w, 0.0, 1.0)
        policy_warm.update(model_w, ctx, reward_w)
        
        expected_w = test_truth.get(cluster, {}).get(model_w, 0.5)
        cum_w += optimal - expected_w
    
    reduction = 100.0 * (cum_c - cum_w) / cum_c if cum_c > 0 else 0.0
    return cum_c, cum_w, reduction


def main():
    print("=" * 70)
    print("5-Fold Cross-Validation on FULL Dataset")
    print("=" * 70)
    print("Configuration:")
    print("  Dataset: All 497 prompts (archetype_grid)")
    print("  Dimensions: d=32 (PCA per fold)")
    print("  Training: epochs=8, expert=65%")
    print("  Prior strength: λ=9")
    print("  Folds: 5 (80/20 split each)")
    print("=" * 70)
    print()
    
    # Load all data
    print("Loading full dataset...")
    all_embeddings, all_clusters, all_truth, model_names = load_full_dataset()
    n_samples = len(all_clusters)
    print(f"  Total samples: {n_samples}")
    print()
    
    # Shuffle data with fixed seed for reproducibility
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(n_samples)
    
    all_embeddings = all_embeddings[perm]
    all_clusters = [all_clusters[i] for i in perm]
    
    # Hyperparameters
    n_folds = 5
    pca_dim = 32
    epochs = 8
    expert_rate = 0.65
    prior_strength = 9.0
    
    results = []
    fold_size = n_samples // n_folds
    
    for fold in range(n_folds):
        print(f"{'='*70}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*70}")
        
        # Split data (80/20)
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        test_indices = list(range(test_start, test_end))
        train_indices = [i for i in range(n_samples) if i not in test_indices]
        
        train_embeddings = all_embeddings[train_indices]
        train_clusters = [all_clusters[i] for i in train_indices]
        
        test_embeddings = all_embeddings[test_indices]
        test_clusters = [all_clusters[i] for i in test_indices]
        
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")
        
        # Fit PCA on training data only
        print(f"  Fitting PCA (d={pca_dim}) on training fold...")
        pca, train_pca = fit_pca_on_fold(train_embeddings, n_components=pca_dim)
        test_pca = pca.transform(test_embeddings)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"    Explained variance: {explained_var:.1%}")
        
        # Train priors
        print(f"  Training priors (epochs={epochs}, expert={expert_rate:.0%})...")
        policy = train_priors(
            train_pca,
            train_clusters,
            all_truth,
            model_names,
            epochs,
            expert_rate,
            seed=42 + fold * 1000,
        )
        
        # Evaluate
        print(f"  Evaluating (λ={prior_strength})...")
        cold, warm, reduction = evaluate_fold(
            policy,
            test_pca,
            test_clusters,
            all_truth,
            model_names,
            prior_strength,
            seed=42 + fold * 10000,
        )
        
        results.append({
            "fold": fold + 1,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "explained_var": explained_var,
            "cold": cold,
            "warm": warm,
            "reduction": reduction,
        })
        
        status = "✓" if reduction > 0 else "✗"
        print(f"  {status} Cold: {cold:.1f}, Warm: {warm:.1f}")
        print(f"  {status} Regret Reduction: {reduction:+.1f}%")
        print()
    
    # Compute statistics
    reductions = [r["reduction"] for r in results]
    mean_reduction = np.mean(reductions)
    std_reduction = np.std(reductions, ddof=1)
    se_reduction = std_reduction / np.sqrt(n_folds)
    
    # 95% confidence interval
    t_critical = stats.t.ppf(0.975, n_folds - 1)
    ci_lower = mean_reduction - t_critical * se_reduction
    ci_upper = mean_reduction + t_critical * se_reduction
    
    # One-sample t-test against null hypothesis (mean = 0)
    t_stat, p_value = stats.ttest_1samp(reductions, 0)
    
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Fold':<6} {'Train':<8} {'Test':<8} {'Cold':<10} {'Warm':<10} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "✓" if r["reduction"] > 0 else "✗"
        print(f"{r['fold']:<6} {r['n_train']:<8} {r['n_test']:<8} "
              f"{r['cold']:<10.1f} {r['warm']:<10.1f} "
              f"{r['reduction']:>+10.1f}%  {status}")
    
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
    
    # Paper-ready format
    print()
    print("=" * 70)
    print("PAPER-READY RESULT:")
    print("=" * 70)
    
    if p_value < 0.05 and mean_reduction > 0:
        print(f"✓ Warm-start reduces regret by {mean_reduction:.1f}% ± {se_reduction:.1f}% (SE)")
        print(f"  95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        print(f"  Statistically significant (t={t_stat:.2f}, p={p_value:.3f}, 5-fold CV)")
    elif mean_reduction > 0:
        print(f"○ Warm-start shows {mean_reduction:.1f}% ± {se_reduction:.1f}% reduction (SE)")
        print(f"  95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        print(f"  Not statistically significant (p={p_value:.3f})")
    else:
        print(f"✗ No warm-start benefit: {mean_reduction:.1f}% ± {se_reduction:.1f}% (SE)")
        print(f"  95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        print(f"  p={p_value:.3f}")
    
    print("=" * 70)
    
    # Save results
    output_dir = Path("results/5fold_full_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "cv_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "dataset": "archetype_grid (497 prompts)",
                "pca_dim": pca_dim,
                "epochs": epochs,
                "expert_rate": expert_rate,
                "prior_strength": prior_strength,
                "n_folds": n_folds,
            },
            "folds": results,
            "statistics": {
                "mean": mean_reduction,
                "std": std_reduction,
                "se": se_reduction,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_level": 0.95,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            },
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

