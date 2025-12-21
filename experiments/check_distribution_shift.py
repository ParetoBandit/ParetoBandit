#!/usr/bin/env python3
"""
Check for Distribution Shift between Train and Test Sets

Analyzes whether the train/test split has different characteristics
that might explain why priors don't transfer well.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt._resources import get_priors_path


def load_rewards_by_cluster(rewards_path: Path):
    """Load average reward per cluster."""
    cluster_rewards = {}
    cluster_counts = {}
    
    with open(rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                cluster = data["cluster_id"]
                logit = data.get("reward_logit", 0.0)
                reward = 1.0 / (1.0 + np.exp(-logit))
                
                if cluster not in cluster_rewards:
                    cluster_rewards[cluster] = []
                cluster_rewards[cluster].append(reward)
    
    # Average across models for each cluster
    avg_rewards = {c: np.mean(rewards) for c, rewards in cluster_rewards.items()}
    
    return avg_rewards


def main():
    print("=" * 70)
    print("Checking for Distribution Shift")
    print("=" * 70)
    print()
    
    # Load train and test rewards
    train_rewards = load_rewards_by_cluster(get_priors_path("train_rewards.jsonl"))
    test_rewards = load_rewards_by_cluster(get_priors_path("test_rewards.jsonl"))
    
    train_vals = list(train_rewards.values())
    test_vals = list(test_rewards.values())
    
    print(f"[Train Set]")
    print(f"  Clusters: {len(train_vals)}")
    print(f"  Avg Reward: {np.mean(train_vals):.4f} ± {np.std(train_vals):.4f}")
    print(f"  Min/Max: {np.min(train_vals):.4f} / {np.max(train_vals):.4f}")
    print()
    
    print(f"[Test Set]")
    print(f"  Clusters: {len(test_vals)}")
    print(f"  Avg Reward: {np.mean(test_vals):.4f} ± {np.std(test_vals):.4f}")
    print(f"  Min/Max: {np.min(test_vals):.4f} / {np.max(test_vals):.4f}")
    print()
    
    # Statistical comparison
    print("[Distribution Comparison]")
    diff_mean = np.mean(test_vals) - np.mean(train_vals)
    diff_std = np.std(test_vals) - np.std(train_vals)
    
    print(f"  Mean difference: {diff_mean:+.4f}")
    print(f"  Std difference: {diff_std:+.4f}")
    
    if abs(diff_mean) > 0.05:
        print(f"  ⚠️  SIGNIFICANT MEAN SHIFT: Test set is "
              f"{'harder' if diff_mean < 0 else 'easier'}")
    else:
        print(f"  ✓ Mean shift is small")
    
    if abs(diff_std) > 0.05:
        print(f"  ⚠️  SIGNIFICANT VARIANCE SHIFT")
    else:
        print(f"  ✓ Variance shift is small")
    
    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print("If test set has substantially different reward distribution,")
    print("priors learned from train may not generalize well.")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

