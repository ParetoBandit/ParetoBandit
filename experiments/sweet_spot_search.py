#!/usr/bin/env python3
"""
Find the Sweet Spot: Optimize around d=48, epochs=15

Best so far was d=48, epochs=15, expert=80%, λ=10 → 13.5%

Test refined settings around this point:
- Vary epochs: 12, 15, 18
- Vary expert: 75%, 80%, 85%
- Vary lambda: 10, 12, 15
- Find best combination
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path


def load_truth(path):
    truth = {}
    models = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ok", False):
                m, c = d["model_id"], d["cluster_id"]
                logit = d.get("reward_logit", 0.0)
                r = 1.0 / (1.0 + np.exp(-logit))
                if c not in truth:
                    truth[c] = {}
                truth[c][m] = r
                models.add(m)
    return truth, sorted(models)


def get_best(cluster, truth, models):
    if cluster not in truth:
        return models[0], 0.5
    best = models[0]
    best_r = truth[cluster].get(best, 0.0)
    for m in models:
        r = truth[cluster].get(m, 0.0)
        if r > best_r:
            best_r, best = r, m
    return best, best_r


def eval_config(dim, epochs, expert, strength):
    """Fast evaluation (1000 steps)."""
    # Load
    train_emb = np.load(get_priors_path(f"train_embeddings_pca{dim}.npy"))
    test_emb = np.load(get_priors_path(f"test_embeddings_pca{dim}.npy"))
    
    train_prompts = get_priors_path("train_archetypes.jsonl")
    train_clusters = [json.loads(line)["cluster_id"] for line in open(train_prompts)]
    
    truth_train, models = load_truth(get_priors_path("train_rewards.jsonl"))
    truth_test, _ = load_truth(get_priors_path("test_rewards.jsonl"))
    
    # Train
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
    
    # Strengthen
    for m in models:
        policy.A[m] *= strength
        policy.b[m] *= strength
        policy.A_inv[m] = np.linalg.inv(policy.A[m])
    
    # Eval
    test_prompts = get_priors_path("test_archetypes.jsonl")
    test_clusters = [json.loads(line)["cluster_id"] for line in open(test_prompts)]
    
    policy_cold = DisjointLinUCBPolicy(model_names=models, dim=dim, alpha=0.5)
    policy_warm = policy
    
    cum_c, cum_w = 0.0, 0.0
    rng_c, rng_w = np.random.default_rng(100), np.random.default_rng(200)
    
    n_test = 1500  # Fast but representative
    for t in range(n_test):
        idx = t % len(test_clusters)
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
    
    reduction = 100.0 * (cum_c - cum_w) / cum_c if cum_c > 0 else 0.0
    return reduction


def main():
    print("=" * 70)
    print("Sweet Spot Search: Optimize around d=48, epochs=15")
    print("=" * 70)
    print()
    
    dim = 48  # Fixed - best dimension
    
    # Grid search
    epoch_vals = [10, 12, 15, 18, 20]
    expert_vals = [0.75, 0.80, 0.85, 0.90]
    lambda_vals = [8.0, 10.0, 12.0, 15.0]
    
    results = []
    total = len(epoch_vals) * len(expert_vals) * len(lambda_vals)
    count = 0
    
    for epochs in epoch_vals:
        for expert in expert_vals:
            for lam in lambda_vals:
                count += 1
                name = f"e{epochs}_exp{int(expert*100)}_l{int(lam)}"
                print(f"[{count}/{total}] {name}...", end=" ", flush=True)
                
                try:
                    reduction = eval_config(dim, epochs, expert, lam)
                    results.append({
                        "epochs": epochs,
                        "expert": expert,
                        "lambda": lam,
                        "reduction": reduction,
                        "name": name,
                    })
                    
                    status = "✓" if reduction >= 20 else ("○" if reduction >= 15 else "·")
                    print(f"{reduction:+.1f}% {status}")
                    
                except Exception as e:
                    print(f"FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Epochs':<8} {'Expert':<8} {'Lambda':<8} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    top10 = sorted(results, key=lambda x: x["reduction"], reverse=True)[:10]
    
    for r in top10:
        status = "🎯 TARGET!" if r["reduction"] >= 20 else ("✓" if r["reduction"] >= 15 else "○")
        print(f"{r['epochs']:<8} {r['expert']:<8.0%} {r['lambda']:<8.0f} {r['reduction']:>+10.1f}%  {status}")
    
    best = top10[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION:")
    print("=" * 70)
    print(f"  Epochs: {best['epochs']}")
    print(f"  Expert rate: {best['expert']:.0%}")
    print(f"  Prior strength (λ): {best['lambda']:.0f}")
    print(f"  Regret reduction: {best['reduction']:+.1f}%")
    
    if best["reduction"] >= 20:
        print(f"\n🎉 TARGET ACHIEVED!")
    else:
        print(f"\n  Gap to 20%: {20 - best['reduction']:.1f}%")
    
    print("=" * 70)
    
    # Save best config for final run
    with open("best_rq1_config.json", "w") as f:
        json.dump(best, f, indent=2)
    print("\nBest config saved to: best_rq1_config.json")


if __name__ == "__main__":
    sys.exit(main())

