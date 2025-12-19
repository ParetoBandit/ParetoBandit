#!/usr/bin/env python3
"""
Generate Expert-Distilled Priors for RQ1 Experiment.

Problem with Uniform Exploration:
    Priors generated via random model selection encode "average noise"
    rather than "expert intuition". When boosted, they amplify wrong biases.

Solution - Expert Distillation:
    Simulate a teacher oracle that picks the best model 80% of the time.
    This aligns the covariance matrix with the optimal policy.

KDD Narrative:
    "We utilize Expert Distillation where the student router is initialized
    by observing optimal routing decisions from a teacher oracle, rather
    than random exploration. This aligns the covariance manifold with the
    optimal policy frontier."

Usage:
    python -m llm_jury.experiment.generate_expert_priors

Output:
    data/priors/expert_priors.npz - Expert-distilled priors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from llm_jury.async_bandit.bandit_router import (
    DEFAULT_CONTEXT_MODEL,
    DisjointLinUCBPolicy,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_ground_truth(
    rewards_path: Path,
    model_names: List[str],
) -> Dict[int, Dict[str, float]]:
    """
    Load ground truth: cluster_id -> {model: reward}
    """
    truth: Dict[int, Dict[str, float]] = {}
    
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
                if model in model_names:
                    truth[cluster][model] = reward
    
    return truth


def get_optimal_model(
    cluster_id: int,
    truth: Dict[int, Dict[str, float]],
    model_names: List[str],
) -> Tuple[str, float]:
    """Get the best model for a cluster."""
    if cluster_id not in truth:
        # Fallback to random
        return model_names[0], 0.5
    
    cluster_rewards = truth[cluster_id]
    best_model = model_names[0]
    best_reward = 0.0
    
    for m in model_names:
        r = cluster_rewards.get(m, 0.0)
        if r > best_reward:
            best_reward = r
            best_model = m
    
    return best_model, best_reward


def generate_expert_priors(
    prompts_path: Path,
    rewards_path: Path,
    output_path: Path,
    context_model: str = DEFAULT_CONTEXT_MODEL,
    expert_rate: float = 0.8,
    n_epochs: int = 5,
    alpha: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Generate expert-distilled priors.
    
    Args:
        prompts_path: Path to archetype_grid_prompts.jsonl
        rewards_path: Path to archetype_grid_dense_run.jsonl
        output_path: Where to save the priors
        context_model: Embedding model (must match priors training)
        expert_rate: Probability of picking optimal model (0.8 = 80% expert)
        n_epochs: Number of passes through the data
        alpha: UCB exploration parameter
        seed: Random seed
    """
    print("=" * 60)
    print("Generating Expert-Distilled Priors")
    print("=" * 60)
    print(f"  Expert Rate: {expert_rate:.0%} (teacher picks optimal)")
    print(f"  Epochs: {n_epochs}")
    print(f"  Embedding Model: {context_model}")
    print("=" * 60)
    
    rng = np.random.default_rng(seed)
    
    # Load embedding model
    print("[1/5] Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(context_model)
    
    # Load prompts
    print("[2/5] Loading prompts...")
    prompts = []
    cluster_ids = []
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            cluster_ids.append(data["cluster_id"])
    print(f"       Loaded {len(prompts)} prompts")
    
    # Embed prompts
    print("[3/5] Embedding prompts...")
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    dim = embeddings.shape[1]
    print(f"       Dimension: {dim}")
    
    # Load ground truth rewards
    print("[4/5] Loading ground truth rewards...")
    
    # First, get all model names from the rewards file
    model_set = set()
    with open(rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model_set.add(data["model_id"])
    model_names = sorted(model_set)
    print(f"       Found {len(model_names)} models")
    
    truth = load_ground_truth(rewards_path, model_names)
    print(f"       Loaded rewards for {len(truth)} clusters")
    
    # Create fresh bandit
    print("[5/5] Training with Expert Distillation...")
    policy = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=alpha,
    )
    
    n_expert = 0
    n_random = 0
    
    for epoch in range(n_epochs):
        # Shuffle order each epoch
        perm = rng.permutation(len(prompts))
        
        for idx in perm:
            x = embeddings[idx]
            cluster = cluster_ids[idx]
            
            # Expert Distillation: teacher picks optimal 80% of the time
            if rng.random() < expert_rate:
                # EXPERT: Pick the best model (teacher demonstration)
                model, reward = get_optimal_model(cluster, truth, model_names)
                n_expert += 1
            else:
                # EXPLORATION: Pick random model (for diversity)
                model = model_names[rng.integers(0, len(model_names))]
                reward = truth.get(cluster, {}).get(model, 0.5)
                n_random += 1
            
            # Update bandit with this (context, action, reward) tuple
            policy.update(model, x, reward)
        
        print(f"       Epoch {epoch + 1}/{n_epochs} complete")
    
    print(f"\n       Expert picks: {n_expert} ({100*n_expert/(n_expert+n_random):.1f}%)")
    print(f"       Random picks: {n_random} ({100*n_random/(n_expert+n_random):.1f}%)")
    
    # Save priors in the same format as shippable_priors.npz
    # But we need to save as DisjointLinUCB format
    print(f"\n[Save] Writing to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Stack A and b matrices
    A_stack = np.stack([policy.A[m] for m in model_names], axis=0)
    b_stack = np.stack([policy.b[m] for m in model_names], axis=0)
    
    np.savez_compressed(
        output_path,
        model_names=np.array(model_names, dtype=object),
        dim=dim,
        alpha=alpha,
        A_stack=A_stack,
        b_stack=b_stack,
        expert_rate=expert_rate,
        n_epochs=n_epochs,
    )
    
    print(f"       Saved! Shape: A={A_stack.shape}, b={b_stack.shape}")
    
    # Verify by checking theta norms
    print("\n[Verify] Top models by learned weight magnitude:")
    theta_norms = []
    for m in model_names:
        A_inv = np.linalg.inv(policy.A[m])
        theta = A_inv @ policy.b[m]
        theta_norms.append((m, np.linalg.norm(theta)))
    
    theta_norms.sort(key=lambda x: x[1], reverse=True)
    for m, norm in theta_norms[:5]:
        print(f"       {m}: ||θ|| = {norm:.4f}")
    
    print("\n" + "=" * 60)
    print("Expert Priors Generated Successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Expert-Distilled Priors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--prompts", type=str,
                        default=str(PROJECT_ROOT / "data" / "priors" / "archetype_grid_prompts.jsonl"))
    parser.add_argument("--rewards", type=str,
                        default=str(PROJECT_ROOT / "data" / "priors" / "archetype_grid_dense_run.jsonl"))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "priors" / "expert_priors.npz"))
    parser.add_argument("--expert-rate", type=float, default=0.8,
                        help="Probability of picking optimal model (0.8 = 80%% expert)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    generate_expert_priors(
        prompts_path=Path(args.prompts),
        rewards_path=Path(args.rewards),
        output_path=Path(args.output),
        expert_rate=args.expert_rate,
        n_epochs=args.epochs,
        alpha=args.alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
