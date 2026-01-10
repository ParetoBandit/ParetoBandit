#!/usr/bin/env python3
"""
Test Script: Verify Data Split Complexity Distribution
Objective: Ensure Train/Val/Test splits are reasonably stratified by difficulty.
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.utils.data_loader import load_oracle_rewards

def get_variance(prompt_rewards):
    if not prompt_rewards:
        return 0.0
    values = list(prompt_rewards.values())
    return np.var(values)

def check_distribution():
    print("="*70)
    print("📊 CHECKING SPLIT DISTRIBUTION (Hard vs Easy)")
    print("="*70)
    
    # 1. Load All Data
    print("📦 Loading corpus...")
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    
    full_corpus = {**train_rewards, **test_rewards}
    all_prompts = list(full_corpus.keys())
    
    # 2. Replicate Split Logic (40/20/40)
    # Using same random_state=42 as the main script
    train_pool, temp_pool = train_test_split(all_prompts, test_size=0.6, random_state=42)
    val_pool, test_pool = train_test_split(temp_pool, test_size=0.666, random_state=42)
    
    # 3. Analyze
    def analyze_pool(name, pool):
        count = len(pool)
        hard_prompts = [p for p in pool if get_variance(full_corpus[p]) > 0.05]
        hard_count = len(hard_prompts)
        pct = (hard_count / count) * 100 if count else 0
        
        print(f"{name:<10} | {count:<5} prompts | {hard_count:<4} Hard ({pct:.1f}%) | {count-hard_count:<4} Easy")

    print("-" * 60)
    print(f"{'Split':<10} | {'Total':<5}         | {'Complexity Distribution'}")
    print("-" * 60)
    
    analyze_pool("Train", train_pool)
    analyze_pool("Valid", val_pool)
    analyze_pool("Test", test_pool)
    print("-" * 60)
    
    # Check consistency
    print("\n✅ Verification:")
    print("   - Are splits roughly 28% Hard (Global Average)?")
    print("   - Is Validation representative of Test?")

if __name__ == "__main__":
    check_distribution()
