#!/usr/bin/env python3
"""
Fair Regret Comparison: Cold Start vs. Warm Start (Dense Priors)

This script performs a rigorous evaluation of the benefit of dense priors
by strictly separating training and testing data to avoid leakage.

Methodology:
1.  **Train Split**: Used to generate dense priors (offline learning).
2.  **Test Split**: Used to evaluate cumulative regret (online learning).
3.  **Agents**:
    -   Cold Start: Standard DisjointLinUCB (alpha=0.5).
    -   Warm Start: DisjointLinUCB initialized with dense priors (prior_strength=50.0).

Usage:
    python -m experiments.compare_regret_fair
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path
from experiments.generate_expert_priors import generate_dense_priors

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_regret")


def load_data_split(archetypes_path: Path, rewards_path: Path) -> Tuple[List[str], List[int], Dict[int, Dict[str, float]], List[str]]:
    """Load prompts, cluster IDs, and rewards for a data split."""
    prompts = []
    cluster_ids = []
    with open(archetypes_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            cluster_ids.append(data["cluster_id"])
    
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
    
    return prompts, cluster_ids, truth, sorted(model_set)


def run_simulation(
    policy: DisjointLinUCBPolicy,
    embeddings: np.ndarray,
    cluster_ids: List[int],
    truth: Dict[int, Dict[str, float]],
    model_names: List[str],
    desc: str,
) -> List[float]:
    """Run a trace-driven simulation and return cumulative regret."""
    cumulative_regret = []
    total_regret = 0.0
    rng = np.random.default_rng(42)
    
    # Shuffle for simulation
    perm = rng.permutation(len(cluster_ids))
    
    for i, idx in enumerate(tqdm(perm, desc=desc)):
        x = embeddings[idx]
        cluster = cluster_ids[idx]
        cluster_rewards = truth.get(cluster, {})
        
        # 1. Oracle Best Reward
        best_reward = 0.0
        for m in model_names:
            r = cluster_rewards.get(m, 0.0)
            if r > best_reward:
                best_reward = r
        
        # 2. Agent Selection
        chosen_model, _, _ = policy.select_arm(x)
        
        # 3. Observed Reward
        observed_reward = cluster_rewards.get(chosen_model, 0.0)
        
        # 4. Update Agent
        policy.update(chosen_model, x, observed_reward)
        
        # 5. Track Regret
        regret = best_reward - observed_reward
        total_regret += regret
        cumulative_regret.append(total_regret)
        
    return cumulative_regret


def main():
    print("=" * 60)
    print("Fair Regret Comparison: Cold vs. Warm Start")
    print("=" * 60)
    
    # Paths
    train_prompts = get_priors_path("train_archetypes.jsonl")
    train_rewards = get_priors_path("train_rewards.jsonl")
    test_prompts = get_priors_path("test_archetypes.jsonl")
    test_rewards = get_priors_path("test_rewards.jsonl")
    
    train_priors_path = get_priors_path("expert_priors_train.npz")
    
    # 1. Generate Priors from TRAIN split
    print("\n[1/4] Generating Dense Priors from TRAIN split...")
    if not train_priors_path.exists():
        generate_dense_priors(
            prompts_path=train_prompts,
            rewards_path=train_rewards,
            output_path=train_priors_path,
            alpha=0.5,
            seed=42,
        )
    else:
        print(f"      Using existing priors: {train_priors_path.name}")
    
    # 2. Load TEST Data
    print("\n[2/4] Loading TEST Data...")
    prompts, cluster_ids, truth, model_names = load_data_split(test_prompts, test_rewards)
    print(f"      Test Prompts: {len(prompts)}")
    print(f"      Test Clusters: {len(truth)}")
    print(f"      Models: {len(model_names)}")
    
    # Load Embeddings (using sentence-transformers)
    print("      Embedding test prompts...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    dim = embeddings.shape[1]
    
    # 3. Initialize Agents
    print("\n[3/4] Initializing Agents...")
    
    # Cold Start
    cold_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    
    # Warm Start
    print("      Loading priors for Warm Start...")
    priors = np.load(train_priors_path, allow_pickle=True)
    warm_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    
    # Inflate priors
    prior_strength = 50.0
    A_stack = priors["A_stack"]
    b_stack = priors["b_stack"]
    prior_models = priors["model_names"]
    
    # Map prior indices to current model indices
    model_to_idx = {m: i for i, m in enumerate(prior_models)}
    
    for m in model_names:
        if m in model_to_idx:
            idx = model_to_idx[m]
            # Apply Prior Strength: A <- lambda * A, b <- lambda * b
            warm_policy.A[m] = A_stack[idx].astype(np.float64) * prior_strength
            warm_policy.b[m] = b_stack[idx].astype(np.float64) * prior_strength
            # Recompute inverse
            warm_policy.A_inv[m] = np.linalg.inv(warm_policy.A[m])
    
    # 4. Run Evaluation
    print("\n[4/4] Running Evaluation on TEST split...")
    cold_regret = run_simulation(cold_policy, embeddings, cluster_ids, truth, model_names, "Cold Start")
    warm_regret = run_simulation(warm_policy, embeddings, cluster_ids, truth, model_names, "Warm Start")
    
    # Results
    final_cold = cold_regret[-1]
    final_warm = warm_regret[-1]
    reduction = (final_cold - final_warm) / final_cold * 100
    
    print("\n" + "=" * 60)
    print("Final Results (Test Set)")
    print("=" * 60)
    print(f"Cold Start Regret: {final_cold:.1f}")
    print(f"Warm Start Regret: {final_warm:.1f}")
    print(f"Regret Reduction:  {reduction:.1f}%")
    print("=" * 60)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cold_regret, label=f"Cold Start (Final: {final_cold:.0f})", linewidth=2)
    plt.plot(warm_regret, label=f"Warm Start (Final: {final_warm:.0f})", linewidth=2)
    plt.xlabel("Requests")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Fair Comparison: Cold vs. Warm Start (Reduction: {reduction:.1f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = Path("results/fair_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "regret_curve.png"
    plt.savefig(plot_path)
    print(f"\nSaved plot to: {plot_path}")


if __name__ == "__main__":
    main()
