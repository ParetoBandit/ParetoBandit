#!/usr/bin/env python3
"""
Expand unused LMSYS dataset to 20K unique prompts.

Strategy:
1. Load all 33K prompts from lmsys_all_prompts.jsonl
2. Exclude prompts already in train/test sets (5K)
3. Select 20K from the remaining ~28K
4. Save as lmsys_unused_20k.jsonl for enrichment
"""

import json
import random
from pathlib import Path
from typing import Set

def load_prompt_set(filepath: Path) -> Set[str]:
    """Load prompts into a set."""
    prompts = set()
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data.get('prompt', '')
            if prompt:
                prompts.add(prompt)
    return prompts

def main():
    print("="*70)
    print("Expanding Unused Dataset to 20K Prompts")
    print("="*70)
    print()
    
    data_dir = Path('src/bandit_gpt/data')
    
    # Load train and test sets to exclude
    print("📂 Loading train/test sets for exclusion...")
    train_prompts = load_prompt_set(data_dir / 'train_prompts.jsonl')
    test_prompts = load_prompt_set(data_dir / 'test_prompts.jsonl')
    used_prompts = train_prompts | test_prompts
    print(f"  ✓ Excluding {len(used_prompts)} prompts (train + test)")
    print()
    
    # Load all LMSYS prompts
    print("📂 Loading all LMSYS prompts...")
    all_prompts = load_prompt_set(data_dir / 'lmsys_all_prompts.jsonl')
    print(f"  ✓ Total LMSYS prompts: {len(all_prompts)}")
    print()
    
    # Find available prompts (not in train/test)
    available_prompts = all_prompts - used_prompts
    print(f"📊 Available prompts: {len(available_prompts)}")
    print(f"   (After excluding train/test)")
    print()
    
    # Target: 20K prompts
    target_size = 20000
    
    if len(available_prompts) < target_size:
        print(f"⚠️  Warning: Only {len(available_prompts)} available prompts")
        print(f"   Cannot reach target of {target_size}")
        print(f"   Using all {len(available_prompts)} available prompts")
        selected_prompts = list(available_prompts)
    else:
        print(f"🎯 Selecting {target_size} prompts from {len(available_prompts)} available...")
        # Set seed for reproducibility
        random.seed(42)
        selected_prompts = random.sample(list(available_prompts), target_size)
    
    print()
    
    # Save selected prompts
    output_file = data_dir / 'lmsys_unused_20k.jsonl'
    print(f"💾 Saving {len(selected_prompts)} prompts to {output_file}...")
    
    with open(output_file, 'w') as f:
        for prompt in sorted(selected_prompts):  # Sort for consistency
            json.dump({'prompt': prompt}, f)
            f.write('\n')
    
    print(f"✅ Saved {len(selected_prompts)} prompts")
    print()
    
    # Verification
    print("🔍 Verification:")
    overlap_train = set(selected_prompts) & train_prompts
    overlap_test = set(selected_prompts) & test_prompts
    
    print(f"  Train overlap: {len(overlap_train)} ({'✅ PASS' if len(overlap_train) == 0 else '❌ FAIL'})")
    print(f"  Test overlap: {len(overlap_test)} ({'✅ PASS' if len(overlap_test) == 0 else '❌ FAIL'})")
    print()
    
    print("="*70)
    print(f"✅ Created unused dataset with {len(selected_prompts)} unique prompts")
    print(f"   Next step: Run enrichment to get HuggingFace metadata")
    print(f"   Command: python scripts/enrich_unused_lmsys.py --input lmsys_unused_20k.jsonl")
    print("="*70)

if __name__ == '__main__':
    main()
