#!/usr/bin/env python3
"""
5-Fold Cross-Validation for Best Configuration

Config: d=32, epochs=8, expert=65%, λ=9

Splits the data into 5 folds, trains on 4, tests on 1, rotates.
Reports mean ± 95% confidence interval.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path
from run_rq1_pca import select_arm


def load_all_data():
    """Load all training data."""
    # Load embeddings
    train_emb = np.load(get_priors_path("train_embeddings_pca32.npy"))
    
    # Load prompts
    prompts_path = get_priors_path("train_archetypes.jsonl")
    cluster_ids = []
    with open(prompts_path) as f:
        for line in f:
            cluster_ids.append(json.loads(line)["cluster_id"])
    
    # Load rewards
    rewards_path = get_priors_path("train_rewards.jsonl")
    model_set = set()
    truth = {}
    with open(rewards_path) as f:
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
    
    return train_emb, cluster_ids, truth, model_names


def train_priors(embeddings, cluster_ids, truth, model_names, epochs, expert_rate, seed):
    """Train priors on given data."""
    dim = embeddings.shape[1]
    policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    rng = np.random.default_rng(seed)
    
    n_samples = len(cluster_ids)
    
    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        for idx in perm:
            x = embeddings[idx]
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


def evaluate_fold(policy, test_embeddings, test_clusters, test_truth, model_names, strength, seed):
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
    n_rounds = 2000  # Total rounds to simulate
    
    for t in range(n_rounds):
        idx = t % n_test
        
        # Shuffle after each pass
        if t % n_test == 0 and t > 0:
            perm = rng_env.permutation(n_test)
            test_clusters = [test_clusters[i] for i in perm]
            test_embeddings = test_embeddings[perm]
        
        ctx = test_embeddings[idx]
        cluster = test_clusters[idx]
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
    print("5-Fold Cross-Validation")
    print("=" * 70)
    print("Configuration:")
    print("  Dimensions: d=32 (PCA)")
    print("  Training: epochs=8, expert=65%")
    print("  Prior strength: λ=9")
    print("  Folds: 5")
    print("=" * 70)
    print()
    
    # Load all data
    print("Loading data...")
    all_embeddings, all_clusters, all_truth, model_names = load_all_data()
    n_samples = len(all_clusters)
    print(f"  Total samples: {n_samples}")
    print(f"  Models: {len(model_names)}")
    print()
    
    # Create 5 folds
    n_folds = 5
    fold_size = n_samples // n_folds
    
    # Shuffle data with fixed seed for reproducibility
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(n_samples)
    
    all_embeddings = all_embeddings[perm]
    all_clusters = [all_clusters[i] for i in perm]
    
    # Hyperparameters
    epochs = 8
    expert_rate = 0.65
    prior_strength = 9.0
    
    results = []
    
    for fold in range(n_folds):
        print(f"{'='*70}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*70}")
        
        # Split data
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
        
        # Train priors
        print(f"  Training priors (epochs={epochs}, expert={expert_rate:.0%})...")
        policy = train_priors(
            train_embeddings,
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
            test_embeddings,
            test_clusters,
            all_truth,
            model_names,
            prior_strength,
            seed=42 + fold * 10000,
        )
        
        results.append({
            "fold": fold + 1,
            "cold": cold,
            "warm": warm,
            "reduction": reduction,
        })
        
        print(f"  Cold: {cold:.1f}, Warm: {warm:.1f}")
        print(f"  Regret Reduction: {reduction:+.1f}%")
        print()
    
    # Compute statistics
    reductions = [r["reduction"] for r in results]
    mean_reduction = np.mean(reductions)
    std_reduction = np.std(reductions, ddof=1)  # Sample std
    se_reduction = std_reduction / np.sqrt(n_folds)
    
    # 95% confidence interval (t-distribution for small n)
    t_critical = stats.t.ppf(0.975, n_folds - 1)  # 2-tailed, 95%
    ci_lower = mean_reduction - t_critical * se_reduction
    ci_upper = mean_reduction + t_critical * se_reduction
    
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Fold':<8} {'Cold':<12} {'Warm':<12} {'Reduction':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['fold']:<8} {r['cold']:<12.1f} {r['warm']:<12.1f} {r['reduction']:>+10.1f}%")
    
    print("=" * 70)
    print(f"Mean:     {mean_reduction:+.2f}%")
    print(f"Std Dev:  {std_reduction:.2f}%")
    print(f"Std Err:  {se_reduction:.2f}%")
    print(f"95% CI:   [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")
    print("=" * 70)
    
    # Paper-ready format
    print()
    print("=" * 70)
    print("PAPER-READY RESULT:")
    print("=" * 70)
    print(f"Warm-start regret reduction: {mean_reduction:.1f}% ± {se_reduction:.1f}%")
    print(f"(mean ± SE over 5-fold CV, 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%])")
    print("=" * 70)
    
    # Save results
    output_dir = Path("results/5fold_cv")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "cv_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "dim": 32,
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
            },
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

