#!/usr/bin/env python3
"""
Verify data leakage between train, test, and unused LMSYS prompts.

This script checks that the three datasets are completely disjoint:
- train_prompts.jsonl (4,000 prompts)
- test_prompts.jsonl (1,000 prompts)
- lmsys_unused_enriched.jsonl (16,522 unique prompts)

KDD Requirement: Zero overlap to prevent data leakage.
"""

import json
from pathlib import Path
from typing import Set

def load_prompt_set(filepath: Path, key: str = 'prompt') -> Set[str]:
    """Load prompts from JSONL file into a set."""
    prompts = set()
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data.get(key, '')
            if prompt:
                prompts.add(prompt)
    return prompts

def main():
    print("="*70)
    print("Data Leakage Verification")
    print("="*70)
    print()
    
    data_dir = Path('src/bandit_gpt/data')
    
    # Load all three datasets
    print("📂 Loading datasets...")
    
    train_prompts = load_prompt_set(data_dir / 'train_prompts.jsonl')
    print(f"  ✓ Training set: {len(train_prompts)} prompts")
    
    test_prompts = load_prompt_set(data_dir / 'test_prompts.jsonl')
    print(f"  ✓ Test set: {len(test_prompts)} prompts")
    
    # For unused enriched, we need to extract first-turn prompts
    # to match what tune_n_prior.py actually uses
    unused_prompts = set()
    with open(data_dir / 'lmsys_unused_enriched.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # Only first turn
            if data.get('turn', 1) != 1:
                continue
            # Extract first user message
            conversation = data.get('conversation', [])
            for message in conversation:
                if message.get('role') == 'user':
                    prompt = message.get('content', '')
                    if prompt:
                        unused_prompts.add(prompt)
                    break
    
    print(f"  ✓ Unused set (first-turn only): {len(unused_prompts)} prompts")
    print()
    
    # Check for overlaps
    print("🔍 Checking for data leakage...")
    print()
    
    train_test_overlap = train_prompts & test_prompts
    train_unused_overlap = train_prompts & unused_prompts
    test_unused_overlap = test_prompts & unused_prompts
    
    # Results
    has_leakage = False
    
    print("📊 Overlap Analysis:")
    print(f"  Train ∩ Test: {len(train_test_overlap)} prompts")
    if train_test_overlap:
        has_leakage = True
        print(f"    ❌ CRITICAL: Train/Test overlap detected!")
        for i, prompt in enumerate(list(train_test_overlap)[:3], 1):
            preview = prompt[:80] + '...' if len(prompt) > 80 else prompt
            print(f"       {i}. {preview}")
    else:
        print(f"    ✅ No overlap (PASS)")
    
    print()
    print(f"  Train ∩ Unused: {len(train_unused_overlap)} prompts")
    if train_unused_overlap:
        has_leakage = True
        print(f"    ❌ CRITICAL: Train/Unused overlap detected!")
        for i, prompt in enumerate(list(train_unused_overlap)[:3], 1):
            preview = prompt[:80] + '...' if len(prompt) > 80 else prompt
            print(f"       {i}. {preview}")
    else:
        print(f"    ✅ No overlap (PASS)")
    
    print()
    print(f"  Test ∩ Unused: {len(test_unused_overlap)} prompts")
    if test_unused_overlap:
        has_leakage = True
        print(f"    ❌ CRITICAL: Test/Unused overlap detected!")
        for i, prompt in enumerate(list(test_unused_overlap)[:3], 1):
            preview = prompt[:80] + '...' if len(prompt) > 80 else prompt
            print(f"       {i}. {preview}")
    else:
        print(f"    ✅ No overlap (PASS)")
    
    print()
    print("="*70)
    
    if has_leakage:
        print("❌ DATA LEAKAGE DETECTED - KDD VIOLATION")
        print("   Action required: Remove overlapping prompts")
        return 1
    else:
        print("✅ NO DATA LEAKAGE - All sets are disjoint")
        print()
        print("Summary:")
        print(f"  Training set: {len(train_prompts)} prompts")
        print(f"  Test set: {len(test_prompts)} prompts")
        print(f"  Unused set: {len(unused_prompts)} prompts (first-turn only)")
        print(f"  Total unique: {len(train_prompts | test_prompts | unused_prompts)} prompts")
        print()
        print("✓ KDD Data Hygiene: PASS")
        return 0

if __name__ == '__main__':
    exit(main())
