#!/usr/bin/env python3
"""
Debug script to investigate why gpt-oss-120B scores so high on hard prompts.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

def main():
    print("="*70)
    print("DEBUGGING ORACLE DATA")
    print("="*70)
    
    # Load data
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    # Load splits.json directly
    splits_path = Path(__file__).parent.parent / "01_effectiveness" / "results" / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    test_prompts = splits["holdout_pool"]
    
    print(f"\nTotal test prompts from splits.json: {len(test_prompts)}")
    
    # Load registry
    registry = load_model_registry()
    
    # Analyze filtering
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        hard_prompts = []
        for prompt in test_prompts:
            oracle_scores = list(all_rewards.get(prompt, {}).values())
            if oracle_scores:
                complexity = max(oracle_scores) - min(oracle_scores)
                if complexity >= threshold:
                    hard_prompts.append(prompt)
        
        print(f"\nThreshold >= {threshold}: {len(hard_prompts)} prompts ({len(hard_prompts)/len(test_prompts)*100:.1f}%)")
        
        if hard_prompts:
            # Check gpt-oss-120B performance
            gpt_oss_id = "openai/gpt-oss-120b"
            gpt_oss_rewards = []
            
            for prompt in hard_prompts:
                r = all_rewards.get(prompt, {}).get(gpt_oss_id)
                if r is not None:
                    gpt_oss_rewards.append(r)
            
            if gpt_oss_rewards:
                avg = np.mean(gpt_oss_rewards)
                std = np.std(gpt_oss_rewards)
                print(f"  gpt-oss-120B: {avg*100:.1f}% ± {std*100:.1f}% (n={len(gpt_oss_rewards)})")
                
                # Show distribution
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                hist, _ = np.histogram(gpt_oss_rewards, bins=bins)
                print(f"  Distribution: 0-20%: {hist[0]}, 20-40%: {hist[1]}, 40-60%: {hist[2]}, 60-80%: {hist[3]}, 80-100%: {hist[4]}")

if __name__ == "__main__":
    main()
