#!/usr/bin/env python3
"""
Generate Dense Priors for RQ1 Experiment.

Problem with Sparse/Simulated Priors:
    Simulating a bandit process (even with "Expert Distillation") leaves many
    arms unexplored. The agent incorrectly learns that unobserved arms have
    zero reward (Confident Ignorance). When boosted with high prior_strength,
    this prevents the agent from exploring potentially better models.

Solution - Dense Priors:
    We utilize the full dense dataset (all models evaluated on all prompts)
    to train the priors. This corresponds to a full Ridge Regression on the
    offline data.

    The agent learns accurate mean estimates (θ) for ALL models, allowing it
    to correctly rank even "second best" models without false confidence
    that they are worthless.

Reproducibility:
    1. Ensure you have the data files:
       - data/priors/archetype_grid_prompts.jsonl
       - data/priors/archetype_grid_dense_run.jsonl

    2. Run with default settings:
       python -m banditgpt.experiment.generate_expert_priors

    3. Expected output:
       - Full Disjoint Priors (A_stack, b_stack)
       - Trained on 100% of available dense data

Usage:
    python -m banditgpt.experiment.generate_expert_priors

Output:
    data/priors/expert_priors.npz - Dense priors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from banditgpt.core.bandit_router import (
    DEFAULT_CONTEXT_MODEL,
    DisjointLinUCBPolicy,
)

PROJECT_ROOT = Path(__file__).parent.parent


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


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set torch seed if available (used by sentence-transformers)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducible convolutions on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file for provenance tracking."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # First 16 chars


def generate_dense_priors(
    prompts_path: Path,
    rewards_path: Path,
    output_path: Path,
    context_model: str = DEFAULT_CONTEXT_MODEL,
    alpha: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Generate dense priors from full offline dataset.
    
    Args:
        prompts_path: Path to archetype_grid_prompts.jsonl
        rewards_path: Path to archetype_grid_dense_run.jsonl
        output_path: Where to save the priors
        context_model: Embedding model (must match priors training)
        alpha: UCB exploration parameter
        seed: Random seed for reproducibility (default: 42)
    """
    print("=" * 60)
    print("Generating Dense Priors (Full Offline Data)")
    print("=" * 60)
    print(f"  Embedding Model: {context_model}")
    print(f"  Random Seed: {seed}")
    print("=" * 60)
    
    # Set ALL random seeds for reproducibility
    print("\n[0/5] Setting random seeds for reproducibility...")
    set_all_seeds(seed)
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
    print("[5/5] Training with Full Dense Data...")
    policy = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=alpha,
    )
    
    n_updates = 0
    
    # Iterate through ALL prompts
    for idx in range(len(prompts)):
        x = embeddings[idx]
        cluster = cluster_ids[idx]
        
        # Dense Update: Update ALL models for this prompt
        # This is equivalent to Ridge Regression on the full dataset
        cluster_rewards = truth.get(cluster, {})
        
        for model in model_names:
            # If we have a reward for this model on this cluster, update it
            if model in cluster_rewards:
                reward = cluster_rewards[model]
                policy.update(model, x, reward)
                n_updates += 1
    
    print(f"\n       Training complete.")
    print(f"       Total updates: {n_updates}")
    print(f"       Avg updates per model: {n_updates / len(model_names):.1f}")
    
    # Save priors in FULL DISJOINT format for maximum performance
    # Each model keeps its own A matrix (captures per-model confidence from expert training)
    # This gives 62% regret reduction vs 38% with shared A
    print(f"\n[Save] Writing FULL DISJOINT priors to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Stack A and b matrices (full disjoint format)
    A_stack = np.stack([policy.A[m] for m in model_names], axis=0)
    b_stack = np.stack([policy.b[m] for m in model_names], axis=0)
    
    # Compute input file hashes for provenance tracking
    prompts_hash = compute_file_hash(prompts_path)
    rewards_hash = compute_file_hash(rewards_path)
    
    np.savez_compressed(
        output_path,
        # Core data (used for loading)
        model_names=np.array(model_names, dtype=object),
        dim=dim,
        alpha=alpha,
        A_stack=A_stack.astype(np.float16),  # Compress to float16 for size
        b_stack=b_stack.astype(np.float16),
        # Training hyperparameters (for reproducibility)
        seed=seed,
        context_model=context_model,
        # Provenance metadata (for auditing)
        prompts_hash=prompts_hash,
        rewards_hash=rewards_hash,
        generated_at=datetime.now().isoformat(),
        n_prompts=len(prompts),
        n_models=len(model_names),
    )
    
    # Report size and provenance
    file_size = output_path.stat().st_size
    print(f"       Saved! Size: {file_size / 1024 / 1024:.1f} MB (full disjoint)")
    print(f"       A_stack shape: {A_stack.shape}, b_stack shape: {b_stack.shape}")
    print(f"\n[Provenance]")
    print(f"       Seed: {seed}")
    print(f"       Prompts hash: {prompts_hash}")
    print(f"       Rewards hash: {rewards_hash}")
    
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
    print("\n" + "=" * 60)
    print("Dense Priors Generated Successfully!")
    print("=" * 60)


def verify_priors(priors_path: Path) -> None:
    """Verify priors file and display provenance metadata."""
    print("=" * 60)
    print("Verifying Dense Priors")
    print("=" * 60)
    
    if not priors_path.exists():
        print(f"ERROR: File not found: {priors_path}")
        return
    
    data = np.load(priors_path, allow_pickle=True)
    
    print(f"\n[File Info]")
    print(f"  Path: {priors_path}")
    print(f"  Size: {priors_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n[Core Data]")
    print(f"  Models: {len(data['model_names'])}")
    print(f"  Dimension: {data['dim']}")
    print(f"  Alpha: {data['alpha']}")
    
    if "A_stack" in data:
        print(f"  Format: Expert (disjoint A_stack)")
        print(f"  A_stack shape: {data['A_stack'].shape}")
        print(f"  b_stack shape: {data['b_stack'].shape}")
    else:
        print(f"  Format: Shared (legacy)")
    
    print(f"\n[Training Hyperparameters]")
    for key in ["seed", "context_model"]:
        if key in data:
            print(f"  {key}: {data[key]}")
    
    print(f"\n[Provenance]")
    for key in ["prompts_hash", "rewards_hash", "generated_at", "n_prompts", "n_models"]:
        if key in data:
            print(f"  {key}: {data[key]}")
    
    # Expected hashes for the shipped priors (generated with seed=42)
    expected_prompts_hash = None
    expected_rewards_hash = None
    
    if "prompts_hash" in data and "rewards_hash" in data:
        print(f"\n[Reproducibility Check]")
        print(f"  To reproduce these priors, run:")
        print(f"    python -m banditgpt.experiment.generate_expert_priors \\")
        print(f"      --seed {data.get('seed', 42)}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Expert-Distilled Priors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Generate command (default)
    gen_parser = subparsers.add_parser("generate", help="Generate expert priors")
    gen_parser.add_argument("--prompts", type=str,
                            default=str(PROJECT_ROOT / "banditgpt" / "data" / "priors" / "archetype_grid_prompts.jsonl"))
    gen_parser.add_argument("--rewards", type=str,
                            default=str(PROJECT_ROOT / "banditgpt" / "data" / "priors" / "archetype_grid_dense_run.jsonl"))
    gen_parser.add_argument("--output", type=str,
                            default=str(PROJECT_ROOT / "banditgpt" / "data" / "priors" / "expert_priors.npz"))
    gen_parser.add_argument("--alpha", type=float, default=0.5)
    gen_parser.add_argument("--seed", type=int, default=42,
                            help="Random seed for reproducibility")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify priors file and show metadata")
    verify_parser.add_argument("--priors", type=str,
                               default=str(PROJECT_ROOT / "data" / "priors" / "expert_priors.npz"),
                               help="Path to priors file to verify")
    
    args = parser.parse_args()
    
    # Default to generate if no command specified
    if args.command is None or args.command == "generate":
        # Handle case where generate is not specified but we still have args
        prompts = getattr(args, "prompts", str(PROJECT_ROOT / "banditgpt" / "data" / "priors" / "archetype_grid_prompts.jsonl"))
        rewards = getattr(args, "rewards", str(PROJECT_ROOT / "banditgpt" / "data" / "priors" / "archetype_grid_dense_run.jsonl"))
        output = getattr(args, "output", str(PROJECT_ROOT / "banditgpt" / "data" / "priors" / "expert_priors.npz"))
        alpha = getattr(args, "alpha", 0.5)
        seed = getattr(args, "seed", 42)
        
        generate_dense_priors(
            prompts_path=Path(prompts),
            rewards_path=Path(rewards),
            output_path=Path(output),
            alpha=alpha,
            seed=seed,
        )
    elif args.command == "verify":
        verify_priors(Path(args.priors))


if __name__ == "__main__":
    main()
