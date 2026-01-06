#!/usr/bin/env python3
"""
Find LMSYS prompts that are not in training or test datasets.

This extracts prompts from lmsys_all_prompts.jsonl that haven't been
used in train_prompts.jsonl or test_prompts.jsonl.
"""

import json
from pathlib import Path

def load_prompts(filepath):
    """Load prompts from a JSONL file."""
    prompts = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
    return set(prompts)

def main():
    data_dir = Path('src/bandit_gpt/data')
    
    # Load all LMSYS prompts
    print("Loading LMSYS prompts...")
    all_lmsys = load_prompts(data_dir / 'lmsys_all_prompts.jsonl')
    print(f"  Total LMSYS prompts: {len(all_lmsys)}")
    
    # Load training prompts
    print("Loading training prompts...")
    train_prompts = load_prompts(data_dir / 'train_prompts.jsonl')
    print(f"  Training prompts: {len(train_prompts)}")
    
    # Load test prompts
    print("Loading test prompts...")
    test_prompts = load_prompts(data_dir / 'test_prompts.jsonl')
    print(f"  Test prompts: {len(test_prompts)}")
    
    # Find unused prompts
    used_prompts = train_prompts | test_prompts
    unused_prompts = all_lmsys - used_prompts
    
    print(f"\n📊 Summary:")
    print(f"  Total LMSYS prompts: {len(all_lmsys)}")
    print(f"  Used in train/test: {len(used_prompts)}")
    print(f"  Unused prompts: {len(unused_prompts)}")
    print(f"  Percentage unused: {len(unused_prompts)/len(all_lmsys)*100:.1f}%")
    
    # Save unused prompts
    output_file = data_dir / 'lmsys_unused_prompts.jsonl'
    print(f"\n💾 Saving unused prompts to {output_file}...")
    
    with open(output_file, 'w') as f:
        for prompt in sorted(unused_prompts):  # Sort for consistency
            json.dump({'prompt': prompt}, f)
            f.write('\n')
    
    print(f"✅ Saved {len(unused_prompts)} unused prompts")
    
    # Show a few examples
    print(f"\n📝 Sample unused prompts (first 5):")
    for i, prompt in enumerate(sorted(unused_prompts)[:5], 1):
        preview = prompt[:100] + '...' if len(prompt) > 100 else prompt
        print(f"  {i}. {preview}")

if __name__ == '__main__':
    main()
