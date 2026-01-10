#!/usr/bin/env python3
"""
Diagnose the oracle/registry mismatch: Are we counting failures from models
we don't even use in the experiment?
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

def main():
    print("="*70)
    print("ORACLE VS REGISTRY MISMATCH ANALYSIS")
    print("="*70)
    
    # Load oracle data
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    # Get all model IDs in oracle
    oracle_models = set()
    for prompt, rewards in all_rewards.items():
        oracle_models.update(rewards.keys())
    
    # Get registry models
    registry = load_model_registry()
    registry_models = set(registry.keys())
    
    print(f"\nOracle Models: {len(oracle_models)}")
    print(f"Registry Models: {len(registry_models)}")
    
    # Find overlap
    in_both = oracle_models & registry_models
    only_oracle = oracle_models - registry_models
    only_registry = registry_models - oracle_models
    
    print(f"\nIn Both: {len(in_both)}")
    print(f"Only in Oracle: {len(only_oracle)}")
    print(f"Only in Registry: {len(only_registry)}")
    
    if only_oracle:
        print(f"\nModels in Oracle but NOT in Registry (these failures are counted!):")
        for i, mid in enumerate(sorted(only_oracle)[:15], 1):
            print(f"  {i}. {mid}")
    
    if only_registry:
        print(f"\nModels in Registry but NOT in Oracle:")
        for mid in sorted(only_registry):
            print(f"  - {mid}")
    
    # Load test split and analyze the "hard" set
    splits_path = Path(__file__).parent.parent / "01_effectiveness" / "results" / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    test_prompts = splits["holdout_pool"]
    
    # Analyze hard prompts using ONLY registry models
    hard_prompts_all_models = []
    hard_prompts_registry_only = []
    
    for prompt in test_prompts:
        full_rewards = all_rewards.get(prompt, {})
        
        # Method 1: Count failures across ALL oracle models
        all_scores = list(full_rewards.values())
        if all_scores:
            all_failures = sum(1 for r in all_scores if r == 0.0)
            all_solvable = max(all_scores) == 1.0
            if all_failures >= 3 and all_solvable:
                hard_prompts_all_models.append(prompt)
        
        # Method 2: Count failures ONLY in registry models
        registry_rewards = {k: v for k, v in full_rewards.items() if k in registry_models}
        if registry_rewards:
            reg_scores = list(registry_rewards.values())
            reg_failures = sum(1 for r in reg_scores if r == 0.0)
            reg_solvable = max(reg_scores) == 1.0
            if reg_failures >= 3 and reg_solvable:
                hard_prompts_registry_only.append(prompt)
    
    print(f"\n🚨 THE SMOKING GUN:")
    print(f"Hard Prompts (counting ALL {len(oracle_models)} oracle models): {len(hard_prompts_all_models)}")
    print(f"Hard Prompts (counting ONLY {len(registry_models)} registry models): {len(hard_prompts_registry_only)}")
    
    if len(hard_prompts_all_models) != len(hard_prompts_registry_only):
        print(f"\n⚠️  MISMATCH DETECTED!")
        print(f"You are filtering based on failures from {len(oracle_models)} models,")
        print(f"but only evaluating {len(registry_models)} models.")
        print(f"This inflates the 'hard' set with prompts that are easy for your registry!")

if __name__ == "__main__":
    main()
