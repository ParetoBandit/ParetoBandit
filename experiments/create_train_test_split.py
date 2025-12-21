#!/usr/bin/env python3
"""
Create Train/Test Split for Rigorous RQ1 Evaluation

This script addresses the data leakage issue in the original RQ1:
- Splits archetype_grid_prompts.jsonl into train (80%) and test (20%)
- Ensures test set is truly held-out
- Creates corresponding reward files for each split

Usage:
    python experiments/create_train_test_split.py

Output:
    - banditgpt/data/priors/train_archetypes.jsonl (397 prompts, ~80%)
    - banditgpt/data/priors/test_archetypes.jsonl (100 prompts, ~20%)
    - banditgpt/data/priors/train_rewards.jsonl (rewards for train set)
    - banditgpt/data/priors/test_rewards.jsonl (rewards for test set)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt._resources import get_priors_path


def load_prompts(path: Path) -> List[Dict]:
    """Load prompts with cluster IDs."""
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_rewards(path: Path) -> List[Dict]:
    """Load all reward records."""
    rewards = []
    with open(path) as f:
        for line in f:
            rewards.append(json.loads(line))
    return rewards


def create_split(
    prompts: List[Dict],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict], Set[int], Set[int]]:
    """
    Split prompts into train/test.
    
    Returns:
        (train_prompts, test_prompts, train_cluster_ids, test_cluster_ids)
    """
    rng = np.random.default_rng(seed)
    
    n_total = len(prompts)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_test
    
    # Shuffle and split
    indices = rng.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_prompts = [prompts[i] for i in train_indices]
    test_prompts = [prompts[i] for i in test_indices]
    
    # Get cluster IDs for each split
    train_clusters = {p["cluster_id"] for p in train_prompts}
    test_clusters = {p["cluster_id"] for p in test_prompts}
    
    return train_prompts, test_prompts, train_clusters, test_clusters


def filter_rewards(
    rewards: List[Dict],
    cluster_ids: Set[int],
) -> List[Dict]:
    """Filter rewards to only include specified clusters."""
    return [r for r in rewards if r.get("cluster_id") in cluster_ids]


def main():
    print("=" * 70)
    print("Creating Train/Test Split for Rigorous RQ1 Evaluation")
    print("=" * 70)
    print()
    
    # Paths
    prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
    rewards_path = get_priors_path("archetype_grid_dense_run.jsonl")
    
    train_prompts_path = get_priors_path("train_archetypes.jsonl")
    test_prompts_path = get_priors_path("test_archetypes.jsonl")
    train_rewards_path = get_priors_path("train_rewards.jsonl")
    test_rewards_path = get_priors_path("test_rewards.jsonl")
    
    # Load data
    print("[1/4] Loading original data...")
    prompts = load_prompts(prompts_path)
    rewards = load_rewards(rewards_path)
    print(f"   Loaded {len(prompts)} prompts")
    print(f"   Loaded {len(rewards)} reward records")
    
    # Create split
    print("\n[2/4] Creating 80/20 train/test split...")
    train_prompts, test_prompts, train_clusters, test_clusters = create_split(
        prompts, test_fraction=0.2, seed=42
    )
    
    print(f"   Train: {len(train_prompts)} prompts ({len(train_clusters)} clusters)")
    print(f"   Test:  {len(test_prompts)} prompts ({len(test_clusters)} clusters)")
    
    # Check for overlap (should be zero)
    overlap = train_clusters & test_clusters
    if overlap:
        print(f"   ⚠️  WARNING: {len(overlap)} clusters appear in both splits!")
    else:
        print(f"   ✓ No cluster overlap (clean split)")
    
    # Filter rewards
    print("\n[3/4] Filtering rewards by split...")
    train_rewards_filtered = filter_rewards(rewards, train_clusters)
    test_rewards_filtered = filter_rewards(rewards, test_clusters)
    
    print(f"   Train rewards: {len(train_rewards_filtered)} records")
    print(f"   Test rewards:  {len(test_rewards_filtered)} records")
    
    # Save splits
    print("\n[4/4] Saving split files...")
    
    with open(train_prompts_path, "w") as f:
        for p in train_prompts:
            f.write(json.dumps(p) + "\n")
    print(f"   ✓ Saved {train_prompts_path.name}")
    
    with open(test_prompts_path, "w") as f:
        for p in test_prompts:
            f.write(json.dumps(p) + "\n")
    print(f"   ✓ Saved {test_prompts_path.name}")
    
    with open(train_rewards_path, "w") as f:
        for r in train_rewards_filtered:
            f.write(json.dumps(r) + "\n")
    print(f"   ✓ Saved {train_rewards_path.name}")
    
    with open(test_rewards_path, "w") as f:
        for r in test_rewards_filtered:
            f.write(json.dumps(r) + "\n")
    print(f"   ✓ Saved {test_rewards_path.name}")
    
    print("\n" + "=" * 70)
    print("Split Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Regenerate priors using train data:")
    print("   python experiments/generate_expert_priors.py generate \\")
    print("     --prompts banditgpt/data/priors/train_archetypes.jsonl \\")
    print("     --rewards banditgpt/data/priors/train_rewards.jsonl \\")
    print("     --output banditgpt/data/priors/expert_priors_train.npz")
    print()
    print("2. Run RQ1 on held-out test data:")
    print("   python experiments/run_rq1.py \\")
    print("     --priors banditgpt/data/priors/expert_priors_train.npz \\")
    print("     --prompts banditgpt/data/priors/test_archetypes.jsonl \\")
    print("     --rewards banditgpt/data/priors/test_rewards.jsonl \\")
    print("     --n-test 2000")
    print("=" * 70)


if __name__ == "__main__":
    main()

