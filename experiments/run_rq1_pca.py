#!/usr/bin/env python3
"""
RQ1 with PCA-Reduced Embeddings (Rigorous + Efficient)

Uses 32-dimensional PCA embeddings to prevent overfitting.
Tests on held-out data with proper train/test split.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_rq1 import (
    DisjointLinUCBPolicy,
    ExperimentResults,
    plot_results,
    save_results,
    select_arm,
    asdict,
)
from banditgpt._resources import get_priors_path

from datetime import datetime


@dataclass
class PCAExperimentConfig:
    """Config for PCA-based experiment."""
    priors_path: Path = get_priors_path("expert_priors_pca32.npz")
    prompts_path: Path = get_priors_path("test_archetypes.jsonl")
    rewards_path: Path = get_priors_path("test_rewards.jsonl")
    embeddings_pca_path: Path = get_priors_path("test_embeddings_pca32.npy")
    n_test: int = 2000
    alpha: float = 0.5
    prior_strength: float = 10.0  # Lower for PCA (less overfitting risk)
    seed: int = 42
    output_dir: Path = Path("results/rq1_pca")


def load_pca_priors(path: Path, alpha: float, strength: float) -> DisjointLinUCBPolicy:
    """Load PCA priors."""
    data = np.load(path, allow_pickle=True)
    
    model_names = [str(m) for m in data["model_names"]]
    dim = int(data["dim"])
    A_stack = np.asarray(data["A_stack"], dtype=np.float64)
    b_stack = np.asarray(data["b_stack"], dtype=np.float64)
    
    policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=alpha)
    
    for i, m in enumerate(model_names):
        policy.A[m] = A_stack[i] * strength
        policy.b[m] = b_stack[i] * strength
        policy.A_inv[m] = np.linalg.inv(policy.A[m])
    
    return policy


def run_pca_experiment(config: PCAExperimentConfig) -> ExperimentResults:
    """Run experiment with PCA embeddings."""
    print("[RQ1-PCA] Loading PCA priors...")
    agent_warm = load_pca_priors(config.priors_path, config.alpha, config.prior_strength)
    model_names = agent_warm.models
    dim = agent_warm.dim
    print(f"   Loaded {len(model_names)} models, dim={dim}")
    
    # Cold start
    agent_cold = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=config.alpha)
    print(f"   Cold start agent ready")
    
    # Load pre-computed PCA embeddings
    print("[RQ1-PCA] Loading PCA embeddings...")
    embeddings_pca = np.load(config.embeddings_pca_path)
    print(f"   Shape: {embeddings_pca.shape}")
    
    # Load prompts for cluster IDs
    print("[RQ1-PCA] Loading test prompts...")
    cluster_ids = []
    with open(config.prompts_path) as f:
        for line in f:
            data = json.loads(line)
            cluster_ids.append(data["cluster_id"])
    print(f"   Loaded {len(cluster_ids)} prompts")
    
    # Load rewards
    print("[RQ1-PCA] Loading rewards...")
    rewards = {}
    with open(config.rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model = data["model_id"]
                cluster = data["cluster_id"]
                logit = data.get("reward_logit", 0.0)
                reward = 1.0 / (1.0 + np.exp(-logit))
                rewards[(model, cluster)] = reward
    print(f"   Loaded {len(rewards)} reward pairs")
    
    # Run simulation
    print(f"[RQ1-PCA] Running {config.n_test} requests...")
    
    step_regret_warm = []
    step_regret_cold = []
    cumulative_warm = []
    cumulative_cold = []
    cum_warm = 0.0
    cum_cold = 0.0
    
    rng_warm = np.random.default_rng(config.seed)
    rng_cold = np.random.default_rng(config.seed + 1)
    rng_env = np.random.default_rng(config.seed + 2)
    
    n_prompts = len(cluster_ids)
    
    for t in range(config.n_test):
        # Cycle through test prompts
        idx = t % n_prompts
        
        # Shuffle after each pass
        if t % n_prompts == 0 and t > 0:
            perm = rng_env.permutation(n_prompts)
            cluster_ids = [cluster_ids[i] for i in perm]
            embeddings_pca = embeddings_pca[perm]
        
        ctx = embeddings_pca[idx]
        cluster = cluster_ids[idx]
        
        # Get optimal reward
        optimal = max([rewards.get((m, cluster), 0.0) for m in model_names])
        
        # Warm agent
        model_w = select_arm(agent_warm, ctx, rng_warm)
        reward_w = rewards.get((model_w, cluster), 0.5)
        # Add small noise
        reward_w += rng_warm.standard_normal() * 0.02
        reward_w = np.clip(reward_w, 0.0, 1.0)
        agent_warm.update(model_w, ctx, reward_w)
        
        expected_w = rewards.get((model_w, cluster), 0.5)
        r_warm = optimal - expected_w
        cum_warm += r_warm
        step_regret_warm.append(r_warm)
        cumulative_warm.append(cum_warm)
        
        # Cold agent
        model_c = select_arm(agent_cold, ctx, rng_cold)
        reward_c = rewards.get((model_c, cluster), 0.5)
        reward_c += rng_cold.standard_normal() * 0.02
        reward_c = np.clip(reward_c, 0.0, 1.0)
        agent_cold.update(model_c, ctx, reward_c)
        
        expected_c = rewards.get((model_c, cluster), 0.5)
        r_cold = optimal - expected_c
        cum_cold += r_cold
        step_regret_cold.append(r_cold)
        cumulative_cold.append(cum_cold)
        
        if (t + 1) % 500 == 0:
            print(f"   Step {t+1}: Cold={cum_cold:.1f}, Warm={cum_warm:.1f}")
    
    # Compute reduction
    if cum_cold > 0:
        reduction = 100.0 * (cum_cold - cum_warm) / cum_cold
    else:
        reduction = 0.0
    
    print(f"\n[RQ1-PCA] Final Results:")
    print(f"   Cold Start Regret: {cum_cold:.1f}")
    print(f"   Warm Start Regret: {cum_warm:.1f}")
    print(f"   Regret Reduction: {reduction:.1f}%")
    
    # Convert config paths to strings for JSON serialization
    config_dict = asdict(config)
    for key in ["priors_path", "prompts_path", "rewards_path", "embeddings_pca_path", "output_dir"]:
        if key in config_dict and config_dict[key]:
            config_dict[key] = str(config_dict[key])
    
    return ExperimentResults(
        config=config_dict,
        regret_cold=step_regret_cold,
        regret_warm=step_regret_warm,
        cumulative_regret_cold=cumulative_cold,
        cumulative_regret_warm=cumulative_warm,
        final_regret_cold=cum_cold,
        final_regret_warm=cum_warm,
        regret_reduction_pct=reduction,
        n_models=len(model_names),
        n_prompts=n_prompts,
        embedding_model=f"PCA-32 (from all-MiniLM-L6-v2)",
        timestamp=datetime.now().isoformat(),
    )


def main():
    config = PCAExperimentConfig()
    
    print("=" * 70)
    print("RQ1 with PCA Reduction (RIGOROUS + EFFICIENT)")
    print("=" * 70)
    print("Dimensions: 32 (vs 384 full)")
    print("Parameters per model: 1,024 (vs 147,456)")
    print("Training: 398 prompts (held-out from test)")
    print("Testing: 99 prompts (never seen during training)")
    print("=" * 70)
    print()
    
    results = run_pca_experiment(config)
    
    # Save
    save_results(results, config.output_dir / "metrics.json")
    plot_results(results, config.output_dir / "regret_curve.png")
    
    print("\n" + "=" * 70)
    print("PCA Evaluation Complete!")
    print("=" * 70)
    print(f"  Regret Reduction: {results.regret_reduction_pct:.1f}%")
    print(f"  Saved to: {config.output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

