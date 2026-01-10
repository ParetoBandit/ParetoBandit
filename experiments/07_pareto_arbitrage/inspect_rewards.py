#!/usr/bin/env python3
"""
Check if oracle rewards are binary (0/1) or continuous (logits).
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils.data_loader import load_oracle_rewards

def main():
    print("="*70)
    print("INSPECTING ORACLE REWARD VALUES")
    print("="*70)
    
    # Load data
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    # Collect all reward values
    all_values = []
    for prompt, rewards in all_rewards.items():
        all_values.extend(rewards.values())
    
    print(f"\nTotal reward samples: {len(all_values)}")
    
    # Check unique values
    unique_vals = sorted(set(all_values))
    print(f"\nUnique values (first 20): {unique_vals[:20]}")
    print(f"Total unique values: {len(unique_vals)}")
    
    # Distribution
    print(f"\nMin: {min(all_values):.4f}")
    print(f"Max: {max(all_values):.4f}")
    print(f"Mean: {np.mean(all_values):.4f}")
    print(f"Std: {np.std(all_values):.4f}")
    
    # Histogram
    print("\nValue distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(all_values, bins=bins)
    for i, count in enumerate(hist):
        pct = count / len(all_values) * 100
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {count:6d} ({pct:5.1f}%)")
    
    # Check if binary
    binary_vals = [v for v in all_values if v in [0.0, 1.0]]
    print(f"\nBinary (0.0 or 1.0): {len(binary_vals)} / {len(all_values)} ({len(binary_vals)/len(all_values)*100:.1f}%)")
    
    # Sample some intermediate values
    intermediate = [v for v in all_values if 0.0 < v < 1.0]
    if intermediate:
        print(f"\nIntermediate values (sample of 20): {sorted(set(intermediate))[:20]}")

if __name__ == "__main__":
    main()
