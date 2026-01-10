#!/usr/bin/env python3
"""
Deep dive into why gpt-oss-120B survives the 'failures >= 3' filter.
We analysis the specific correlation of failures.
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

def main():
    print("="*70)
    print("DEEP DIVE: WHO IS FAILING?")
    print("="*70)
    
    # Load data
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    # Load registry to get model list
    registry = load_model_registry()
    model_ids = list(registry.keys())
    
    # Load splits
    splits_path = Path(__file__).parent.parent / "01_effectiveness" / "results" / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    test_prompts = splits["holdout_pool"]
    
    print(f"Total Test Prompts: {len(test_prompts)}")
    
    # Analyze the filter: failures >= 3 and solvable
    hard_prompts = []
    
    for prompt in test_prompts:
        oracle_scores = list(all_rewards.get(prompt, {}).values())
        if oracle_scores:
            failures = sum(1 for r in oracle_scores if r == 0.0)
            is_solvable = max(oracle_scores) == 1.0
            
            if failures >= 3 and is_solvable:
                hard_prompts.append(prompt)
                
    print(f"\nHard Prompts (Failures >= 3): {len(hard_prompts)}")
    
    # Who is failing in this set?
    failure_counts = defaultdict(int)
    gpt_oss_id = "openai/gpt-oss-120b"
    co_occurrence = defaultdict(int) # Who fails together with gpt-oss?
    
    gpt_oss_failures = 0
    gpt_oss_successes = 0
    
    for prompt in hard_prompts:
        rewards = all_rewards.get(prompt, {})
        
        # Check gpt-oss status
        gpt_oss_reward = rewards.get(gpt_oss_id)
        if gpt_oss_reward == 0.0:
            gpt_oss_failures += 1
        elif gpt_oss_reward == 1.0:
            gpt_oss_successes += 1
            
        # Count failures for all models
        current_failures = []
        for mid in model_ids:
            if rewards.get(mid) == 0.0:
                failure_counts[mid] += 1
                current_failures.append(mid)
        
        # If gpt-oss fails, who else fails?
        if gpt_oss_reward == 0.0:
            for mid in current_failures:
                if mid != gpt_oss_id:
                    co_occurrence[mid] += 1

    print(f"\nGPT-OSS-120B Performance on Hard Set:")
    print(f"  Success: {gpt_oss_successes}")
    print(f"  Failure: {gpt_oss_failures}")
    print(f"  Success Rate: {gpt_oss_successes / len(hard_prompts) * 100:.1f}%")
    
    print("\nWho contributes to the '>= 3 failures' count?")
    sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (mid, count) in enumerate(sorted_failures):
        print(f"  {i+1}. {mid[:25]:25}: {count} failures ({count/len(hard_prompts)*100:.1f}%)")

if __name__ == "__main__":
    main()
