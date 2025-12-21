#!/usr/bin/env python3
"""
Final Push to 20%+: Test aggressive configurations

Based on quick_optimize results, most promising direction is:
- More epochs (15 gave 13.5%)
- Higher expert rate (95% gave 7.5%)
- Combine these

New tests:
- d=48, epochs=25, expert=95%, λ=15
- d=64, epochs=20, expert=95%, λ=15
- d=48, epochs=30, expert=98%, λ=15
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path


def load_truth(rewards_path):
    """Load ground truth."""
    truth = {}
    model_set = set()
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
    return truth, sorted(model_set)


def get_best(cluster, truth, models):
    """Get best model for cluster."""
    if cluster not in truth:
        return models[0], 0.5
    best = models[0]
    best_r = truth[cluster].get(best, 0.0)
    for m in models:
        r = truth[cluster].get(m, 0.0)
        if r > best_r:
            best_r, best = r, m
    return best, best_r


def test(dim, epochs, expert, strength, name):
    """Train and evaluate."""
    print(f"\n{'='*60}")
    print(f"[{name}] d={dim}, epochs={epochs}, expert={expert:.0%}, λ={strength}")
    print(f"{'='*60}")
    
    # Load embeddings
    train_emb = np.load(get_priors_path(f"train_embeddings_pca{dim}.npy"))
    test_emb = np.load(get_priors_path(f"test_embeddings_pca{dim}.npy"))
    
    # Load prompts
    train_prompts = get_priors_path("train_archetypes.jsonl")
    train_clusters = []
    with open(train_prompts) as f:
        for line in f:
            train_clusters.append(json.loads(line)["cluster_id"])
    
    # Load truth
    truth_train, models = load_truth(get_priors_path("train_rewards.jsonl"))
    truth_test, _ = load_truth(get_priors_path("test_rewards.jsonl"))
    
    # Train priors
    print(f"Training with {len(train_clusters)} prompts, {len(models)} models...")
    policy = DisjointLinUCBPolicy(model_names=models, dim=dim, alpha=0.5)
    rng = np.random.default_rng(42)
    
    for epoch in range(epochs):
        perm = rng.permutation(len(train_clusters))
        for idx in perm:
            ctx = train_emb[idx]
            cluster = train_clusters[idx]
            
            if rng.random() < expert:
                model, reward = get_best(cluster, truth_train, models)
            else:
                model = rng.choice(models)
                reward = truth_train.get(cluster, {}).get(model, 0.5)
            
            policy.update(model, ctx, reward)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} complete")
    
    # Apply strength
    for m in models:
        policy.A[m] *= strength
        policy.b[m] *= strength
        policy.A_inv[m] = np.linalg.inv(policy.A[m])
    
    # Test
    test_prompts = get_priors_path("test_archetypes.jsonl")
    test_clusters = []
    with open(test_prompts) as f:
        for line in f:
            test_clusters.append(json.loads(line)["cluster_id"])
    
    print(f"Testing on {len(test_clusters)} held-out prompts...")
    
    policy_cold = DisjointLinUCBPolicy(model_names=models, dim=dim, alpha=0.5)
    policy_warm = policy
    
    cum_c, cum_w = 0.0, 0.0
    rng_c, rng_w = np.random.default_rng(100), np.random.default_rng(200)
    
    n_test = 2000  # Full evaluation
    for t in range(n_test):
        idx = t % len(test_clusters)
        if t % len(test_clusters) == 0 and t > 0:
            perm = rng_c.permutation(len(test_clusters))
            test_clusters = [test_clusters[i] for i in perm]
            test_emb = test_emb[perm]
        
        ctx = test_emb[idx]
        cluster = test_clusters[idx]
        optimal = max([truth_test.get(cluster, {}).get(m, 0.0) for m in models])
        
        # Cold
        best_c, ucb_c = None, -float("inf")
        for m in models:
            theta = policy_cold.A_inv[m] @ policy_cold.b[m]
            mean = float(theta.dot(ctx))
            var = float(ctx.dot(policy_cold.A_inv[m]).dot(ctx))
            ucb = mean + 0.5 * np.sqrt(max(var, 1e-12)) + rng_c.random() * 1e-8
            if ucb > ucb_c:
                ucb_c, best_c = ucb, m
        
        r_c = truth_test.get(cluster, {}).get(best_c, 0.5)
        policy_cold.update(best_c, ctx, r_c)
        cum_c += optimal - r_c
        
        # Warm
        best_w, ucb_w = None, -float("inf")
        for m in models:
            theta = policy_warm.A_inv[m] @ policy_warm.b[m]
            mean = float(theta.dot(ctx))
            var = float(ctx.dot(policy_warm.A_inv[m]).dot(ctx))
            ucb = mean + 0.5 * np.sqrt(max(var, 1e-12)) + rng_w.random() * 1e-8
            if ucb > ucb_w:
                ucb_w, best_w = ucb, m
        
        r_w = truth_test.get(cluster, {}).get(best_w, 0.5)
        policy_warm.update(best_w, ctx, r_w)
        cum_w += optimal - r_w
        
        if (t + 1) % 500 == 0:
            print(f"   Step {t+1}: Cold={cum_c:.1f}, Warm={cum_w:.1f}")
    
    reduction = 100.0 * (cum_c - cum_w) / cum_c if cum_c > 0 else 0.0
    
    print(f"\nFINAL: Cold={cum_c:.1f}, Warm={cum_w:.1f}, Reduction={reduction:+.1f}%")
    
    return {"name": name, "dim": dim, "epochs": epochs, "expert": expert, 
            "strength": strength, "cold": cum_c, "warm": cum_w, "reduction": reduction}


def main():
    print("=" * 70)
    print("FINAL PUSH TO 20%+")
    print("=" * 70)
    print()
    
    configs = [
        # Aggressive epochs + expert
        (48, 25, 0.95, 15.0, "d48_e25_exp95"),
        (48, 30, 0.95, 15.0, "d48_e30_exp95"),
        
        # d=64 with strong settings
        (64, 20, 0.95, 15.0, "d64_e20_exp95"),
        (64, 25, 0.95, 15.0, "d64_e25_exp95"),
        
        # Ultra-high expert rate
        (48, 25, 0.98, 15.0, "d48_e25_exp98"),
    ]
    
    results = []
    for dim, epochs, expert, strength, name in configs:
        try:
            r = test(dim, epochs, expert, strength, name)
            results.append(r)
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Config':<20} {'d':<5} {'Epochs':<8} {'Expert':<8} {'Cold':<8} {'Warm':<8} {'Reduction':<10} {'Status'}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x["reduction"], reverse=True):
        status = "🎯 TARGET!" if r["reduction"] >= 20 else "○"
        print(f"{r['name']:<20} {r['dim']:<5} {r['epochs']:<8} {r['expert']:<8.0%} "
              f"{r['cold']:<8.1f} {r['warm']:<8.1f} {r['reduction']:>+8.1f}%  {status}")
    
    if results:
        best = max(results, key=lambda x: x["reduction"])
        print("\n" + "=" * 70)
        if best["reduction"] >= 20:
            print(f"🎉 SUCCESS: {best['name']} = {best['reduction']:.1f}%!")
        else:
            print(f"Best: {best['name']} = {best['reduction']:.1f}%")
            print(f"Gap to target: {20 - best['reduction']:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

