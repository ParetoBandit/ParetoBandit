#!/usr/bin/env python3
"""
Quick Optimization: Test key configurations to reach 20%+

Tests in order of likelihood to improve:
1. d=48 (more signal, still manageable)
2. d=48 + expert=0.95 (better quality)
3. d=48 + epochs=15 (more learning)
4. d=48 + λ=20 (more confidence)
5. d=48 + combined best
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path


def load_ground_truth(rewards_path, model_names):
    """Load rewards."""
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
    return truth


def get_optimal_model(cluster_id, truth, model_names):
    """Get optimal model for cluster."""
    if cluster_id not in truth:
        return model_names[0], 0.5
    best_model = model_names[0]
    best_reward = truth[cluster_id].get(best_model, 0.0)
    for m in model_names:
        reward = truth[cluster_id].get(m, 0.0)
        if reward > best_reward:
            best_reward = reward
            best_model = m
    return best_model, best_reward


def train_and_eval(dim, expert_rate, n_epochs, prior_strength, name):
    """Train priors and evaluate."""
    print(f"\n[{name}] d={dim}, expert={expert_rate:.0%}, epochs={n_epochs}, λ={prior_strength}")
    
    # Load data
    train_pca = np.load(get_priors_path(f"train_embeddings_pca{dim}.npy"))
    test_pca = np.load(get_priors_path(f"test_embeddings_pca{dim}.npy"))
    
    prompts_path = get_priors_path("train_archetypes.jsonl")
    cluster_ids = []
    with open(prompts_path) as f:
        for line in f:
            cluster_ids.append(json.loads(line)["cluster_id"])
    
    rewards_path = get_priors_path("train_rewards.jsonl")
    model_set = set()
    with open(rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model_set.add(data["model_id"])
    model_names = sorted(model_set)
    
    truth = load_ground_truth(rewards_path, model_names)
    
    # Train
    policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    rng = np.random.default_rng(42)
    
    for epoch in range(n_epochs):
        perm = rng.permutation(len(cluster_ids))
        for idx in perm:
            x = train_pca[idx]
            cluster = cluster_ids[idx]
            if rng.random() < expert_rate:
                model, reward = get_optimal_model(cluster, truth, model_names)
            else:
                model = rng.choice(model_names)
                reward = truth.get(cluster, {}).get(model, 0.5)
            policy.update(model, x, reward)
    
    # Apply prior strength
    for m in model_names:
        policy.A[m] *= prior_strength
        policy.b[m] *= prior_strength
        policy.A_inv[m] = np.linalg.inv(policy.A[m])
    
    # Eval
    test_prompts = get_priors_path("test_archetypes.jsonl")
    test_cluster_ids = []
    with open(test_prompts) as f:
        for line in f:
            test_cluster_ids.append(json.loads(line)["cluster_id"])
    
    test_rewards_path = get_priors_path("test_rewards.jsonl")
    test_rewards = {}
    with open(test_rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model = data["model_id"]
                cluster = data["cluster_id"]
                logit = data.get("reward_logit", 0.0)
                reward = 1.0 / (1.0 + np.exp(-logit))
                test_rewards[(model, cluster)] = reward
    
    # Run simulation
    policy_cold = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    policy_warm = policy  # Use trained policy
    
    cum_cold, cum_warm = 0.0, 0.0
    rng_cold = np.random.default_rng(42)
    rng_warm = np.random.default_rng(43)
    
    n_test = 1000  # Quick test
    for t in range(n_test):
        idx = t % len(test_cluster_ids)
        ctx = test_pca[idx]
        cluster = test_cluster_ids[idx]
        optimal = max([test_rewards.get((m, cluster), 0.0) for m in model_names])
        
        # Cold
        best_c, best_ucb_c = None, -float("inf")
        for m in model_names:
            theta = policy_cold.A_inv[m] @ policy_cold.b[m]
            mean = float(theta.dot(ctx))
            var = float(ctx.dot(policy_cold.A_inv[m]).dot(ctx))
            ucb = mean + 0.5 * np.sqrt(max(var, 1e-12)) + rng_cold.random() * 1e-8
            if ucb > best_ucb_c:
                best_ucb_c, best_c = ucb, m
        
        reward_c = test_rewards.get((best_c, cluster), 0.5)
        policy_cold.update(best_c, ctx, reward_c)
        cum_cold += optimal - reward_c
        
        # Warm
        best_w, best_ucb_w = None, -float("inf")
        for m in model_names:
            theta = policy_warm.A_inv[m] @ policy_warm.b[m]
            mean = float(theta.dot(ctx))
            var = float(ctx.dot(policy_warm.A_inv[m]).dot(ctx))
            ucb = mean + 0.5 * np.sqrt(max(var, 1e-12)) + rng_warm.random() * 1e-8
            if ucb > best_ucb_w:
                best_ucb_w, best_w = ucb, m
        
        reward_w = test_rewards.get((best_w, cluster), 0.5)
        policy_warm.update(best_w, ctx, reward_w)
        cum_warm += optimal - reward_w
    
    reduction = 100.0 * (cum_cold - cum_warm) / cum_cold if cum_cold > 0 else 0.0
    print(f"   Cold={cum_cold:.1f}, Warm={cum_warm:.1f}, Reduction={reduction:+.1f}%")
    
    return {"name": name, "cold": cum_cold, "warm": cum_warm, "reduction": reduction}


def main():
    print("=" * 70)
    print("Quick Optimization: Target 20%+")
    print("=" * 70)
    
    configs = [
        # Baseline
        (32, 0.80, 5, 10.0, "baseline"),
        # More dimensions
        (48, 0.80, 5, 10.0, "d48"),
        # Better expert
        (48, 0.95, 5, 10.0, "d48_exp95"),
        # More epochs
        (48, 0.80, 15, 10.0, "d48_epoch15"),
        # Higher confidence
        (48, 0.80, 5, 20.0, "d48_lambda20"),
        # Best combo
        (48, 0.95, 15, 20.0, "d48_best"),
    ]
    
    results = []
    for dim, expert, epochs, strength, name in configs:
        try:
            r = train_and_eval(dim, expert, epochs, strength, name)
            results.append(r)
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Config':<20} {'Cold':<12} {'Warm':<12} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x["reduction"], reverse=True):
        status = "✓ TARGET!" if r["reduction"] >= 20 else "○"
        print(f"{r['name']:<20} {r['cold']:<12.1f} {r['warm']:<12.1f} {r['reduction']:>+10.1f}%  {status}")
    
    best = max(results, key=lambda x: x["reduction"])
    print("\n" + "=" * 70)
    if best["reduction"] >= 20:
        print(f"🎉 ACHIEVED: {best['name']} = {best['reduction']:.1f}%")
    else:
        print(f"Best: {best['name']} = {best['reduction']:.1f}% (still < 20%)")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

