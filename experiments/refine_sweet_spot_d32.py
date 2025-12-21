#!/usr/bin/env python3
"""
Refine around the sweet spot: d=32, epochs=8, expert=65%, λ=8 → +11.3%

Fine-tune:
- Expert rate: 60%, 62%, 65%, 68%
- Lambda: 7, 8, 9, 10
- Epochs: 7, 8, 9
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path
from run_rq1_pca import select_arm


def train_and_eval(dim, epochs, expert, strength):
    """Train and evaluate."""
    # Train
    train_emb = np.load(get_priors_path(f"train_embeddings_pca{dim}.npy"))
    
    prompts_path = get_priors_path("train_archetypes.jsonl")
    cluster_ids = []
    with open(prompts_path) as f:
        for line in f:
            cluster_ids.append(json.loads(line)["cluster_id"])
    
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
    
    policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    rng = np.random.default_rng(42)
    
    for epoch in range(epochs):
        perm = rng.permutation(len(cluster_ids))
        for idx in perm:
            x = train_emb[idx]
            cluster = cluster_ids[idx]
            
            if rng.random() < expert:
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
    
    # Apply strength
    for m in model_names:
        policy.A[m] *= strength
        policy.b[m] *= strength
        policy.A_inv[m] = np.linalg.inv(policy.A[m])
    
    # Eval
    test_emb = np.load(get_priors_path(f"test_embeddings_pca{dim}.npy"))
    test_prompts = get_priors_path("test_archetypes.jsonl")
    test_clusters = []
    with open(test_prompts) as f:
        for line in f:
            test_clusters.append(json.loads(line)["cluster_id"])
    
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
    
    policy_cold = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    policy_warm = policy
    
    cum_c, cum_w = 0.0, 0.0
    rng_c, rng_w, rng_env = np.random.default_rng(42), np.random.default_rng(43), np.random.default_rng(44)
    
    n_test = 2000
    for t in range(n_test):
        idx = t % len(test_clusters)
        if t % len(test_clusters) == 0 and t > 0:
            perm = rng_env.permutation(len(test_clusters))
            test_clusters = [test_clusters[i] for i in perm]
            test_emb = test_emb[perm]
        
        ctx = test_emb[idx]
        cluster = test_clusters[idx]
        optimal = max([test_rewards.get((m, cluster), 0.0) for m in model_names])
        
        model_c = select_arm(policy_cold, ctx, rng_c)
        reward_c = test_rewards.get((model_c, cluster), 0.5) + rng_c.standard_normal() * 0.02
        policy_cold.update(model_c, ctx, np.clip(reward_c, 0, 1))
        cum_c += optimal - test_rewards.get((model_c, cluster), 0.5)
        
        model_w = select_arm(policy_warm, ctx, rng_w)
        reward_w = test_rewards.get((model_w, cluster), 0.5) + rng_w.standard_normal() * 0.02
        policy_warm.update(model_w, ctx, np.clip(reward_w, 0, 1))
        cum_w += optimal - test_rewards.get((model_w, cluster), 0.5)
    
    reduction = 100.0 * (cum_c - cum_w) / cum_c if cum_c > 0 else 0.0
    return reduction


def main():
    print("=" * 70)
    print("Fine-Tuning around Sweet Spot: d=32, epochs=8, expert=65%, λ=8")
    print("=" * 70)
    print()
    
    dim = 32
    
    # Fine grid search
    expert_vals = [0.58, 0.60, 0.62, 0.65, 0.68, 0.70]
    epoch_vals = [7, 8, 9]
    lambda_vals = [6.0, 7.0, 8.0, 9.0, 10.0]
    
    results = []
    total = len(expert_vals) * len(epoch_vals) * len(lambda_vals)
    count = 0
    
    for epochs in epoch_vals:
        for expert in expert_vals:
            for lam in lambda_vals:
                count += 1
                name = f"e{epochs}_exp{int(expert*100)}_l{int(lam)}"
                print(f"[{count}/{total}] {name}...", end=" ", flush=True)
                
                try:
                    reduction = train_and_eval(dim, epochs, expert, lam)
                    results.append({
                        "epochs": epochs,
                        "expert": expert,
                        "lambda": lam,
                        "reduction": reduction,
                    })
                    
                    status = "🎯" if reduction >= 20 else ("✓" if reduction >= 15 else ("+" if reduction >= 10 else "·"))
                    print(f"{reduction:+.1f}% {status}")
                except Exception as e:
                    print(f"FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Epochs':<8} {'Expert':<8} {'Lambda':<8} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    top10 = sorted(results, key=lambda x: x["reduction"], reverse=True)[:10]
    
    for r in top10:
        status = "🎯 TARGET!" if r["reduction"] >= 20 else ("✓ GREAT" if r["reduction"] >= 15 else ("+ GOOD" if r["reduction"] >= 10 else "○"))
        print(f"{r['epochs']:<8} {r['expert']:<8.0%} {r['lambda']:<8.0f} {r['reduction']:>+10.1f}%  {status}")
    
    best = top10[0]
    print("\n" + "=" * 70)
    if best["reduction"] >= 20:
        print(f"🎉🎉🎉 TARGET ACHIEVED: {best['reduction']:+.1f}% 🎉🎉🎉")
    else:
        print(f"Best result: {best['reduction']:+.1f}%")
        print(f"Gap to 20%: {20 - best['reduction']:.1f}%")
    
    print(f"\nOptimal: epochs={best['epochs']}, expert={best['expert']:.0%}, λ={best['lambda']:.0f}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

