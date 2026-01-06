#!/usr/bin/env python3
"""
Verify that train and test sets have no overlapping prompts.
"""
import json
from pathlib import Path

def load_prompts(path):
    """Load unique prompts from a JSONL file."""
    prompts = set()
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if 'prompt' in entry:
                prompts.add(entry['prompt'])
    return prompts

def main():
    data_dir = Path(__file__).parent.parent / "src" / "bandit_gpt" / "data" / "offline_dataset"
    
    train_path = data_dir / "train_rewards_hle_models.jsonl"
    test_path = data_dir / "test_rewards_hle_models.jsonl"
    
    print("Loading train prompts...")
    train_prompts = load_prompts(train_path)
    print(f"  ✓ Found {len(train_prompts)} unique training prompts")
    
    print("\nLoading test prompts...")
    test_prompts = load_prompts(test_path)
    print(f"  ✓ Found {len(test_prompts)} unique test prompts")
    
    print("\nChecking for overlap...")
    overlap = train_prompts & test_prompts
    
    if overlap:
        print(f"  ❌ FAIL: Found {len(overlap)} overlapping prompts!")
        print("\nFirst 5 overlapping prompts:")
        for i, prompt in enumerate(list(overlap)[:5]):
            print(f"  {i+1}. {prompt[:100]}...")
    else:
        print(f"  ✅ PASS: No overlap detected!")
        print(f"\nSummary:")
        print(f"  - Training set: {len(train_prompts)} prompts")
        print(f"  - Test set: {len(test_prompts)} prompts")
        print(f"  - Total unique: {len(train_prompts) + len(test_prompts)} prompts")

if __name__ == "__main__":
    main()
