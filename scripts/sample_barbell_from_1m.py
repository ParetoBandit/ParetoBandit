#!/usr/bin/env python3
"""
Sample barbell distribution from full 1M LMSYS dataset.

Strategy:
1. Load train/test sets to exclude (5K prompts)
2. Stream through full 1M HF dataset
3. Classify and collect prompts for each category
4. Stop when we hit targets for all 6 categories
5. Save balanced 20K dataset
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

# Import classification functions from create_barbell_dataset
import sys
sys.path.insert(0, str(Path(__file__).parent))
from create_barbell_dataset import classify_prompt

def load_exclusion_set(data_dir: Path) -> Set[str]:
    """Load train and test prompts to exclude."""
    excluded = set()
    
    for filename in ['train_prompts.jsonl', 'test_prompts.jsonl']:
        filepath = data_dir / filename
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                excluded.add(data['prompt'])
    
    return excluded

def main():
    print("="*70)
    print("Barbell Sampling from Full 1M LMSYS Dataset")
    print("="*70)
    print()
    
    data_dir = Path('src/bandit_gpt/data')
    
    # Load exclusion set
    print("📂 Loading train/test sets for exclusion...")
    excluded_prompts = load_exclusion_set(data_dir)
    print(f"  ✓ Excluding {len(excluded_prompts)} prompts")
    print()
    
    # Target counts per category (3,330 each for 6 categories = ~20K total)
    targets = {
        'deep_calculus': 3330,
        'arithmetic_trick': 3330,
        'kernel_debugging': 3330,
        'html_boilerplate': 3330,
        'email_draft': 3330,
        'nuanced_haiku': 3330,
    }
    
    # Collection buckets
    collected = defaultdict(list)
    
    # Download LMSYS dataset
    print("📦 Loading LMSYS Chat-1M from HuggingFace...")
    try:
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
        print(f"✅ Loaded {len(dataset)} conversations")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print()
    print("🔍 Searching for prompts across all categories...")
    print("   Targets per category: 3,330")
    print()
    
    processed = 0
    first_turn_count = 0
    excluded_count = 0
    
    # Track which categories are complete
    complete_categories = set()
    
    for record in tqdm(dataset, desc="Processing"):
        processed += 1
        
        # Only process first turn
        turn = record.get('turn')
        if turn != 1:
            continue
        
        first_turn_count += 1
        
        # Extract first user message
        conversation = record.get('conversation', [])
        if not conversation:
            continue
        
        prompt = None
        for message in conversation:
            if message.get('role') == 'user':
                prompt = message.get('content', '')
                break
        
        if not prompt:
            continue
        
        # Check if prompt is in exclusion set
        if prompt in excluded_prompts:
            excluded_count += 1
            continue
        
        # Classify prompt
        category, subcategory = classify_prompt(prompt)
        
        # Skip if not classifiable or category already full
        if not subcategory or subcategory in complete_categories:
            continue
        
        # Add to collection if under target
        if len(collected[subcategory]) < targets[subcategory]:
            # Store full record with metadata
            collected[subcategory].append({
                'prompt': prompt,
                'conversation_id': record.get('conversation_id'),
                'model': record.get('model'),
                'conversation': conversation,
                'turn': turn,
                'language': record.get('language'),
                'openai_moderation': record.get('openai_moderation'),
                'redacted': record.get('redacted', False),
                'timestamp': record.get('tstamp'),
                'subcategory': subcategory  # Tag for verification
            })
            
            # Check if category is complete
            if len(collected[subcategory]) >= targets[subcategory]:
                complete_categories.add(subcategory)
                print(f"\n✓ {subcategory}: Target reached ({targets[subcategory]} prompts)")
        
        # Early exit if all categories complete
        if len(complete_categories) == 6:
            print(f"\n🎉 All categories complete! Processed {processed}/{len(dataset)} records")
            break
        
        # Progress update every 50K
        if processed % 50000 == 0:
            print(f"\n📊 Progress at {processed}/{len(dataset)}:")
            for subcat, target in targets.items():
                count = len(collected[subcat])
                pct = count / target * 100
                status = "✓" if count >= target else f"{pct:.1f}%"
                print(f"  {subcat}: {count}/{target} ({status})")
    
    print()
    print("="*70)
    print("📊 Final Collection Results:")
    print()
    
    total_collected = 0
    for subcat in ['deep_calculus', 'arithmetic_trick', 'kernel_debugging',
                   'html_boilerplate', 'email_draft', 'nuanced_haiku']:
        count = len(collected[subcat])
        target = targets[subcat]
        total_collected += count
        status = "✅" if count >= target else "⚠️"
        print(f"  {status} {subcat}: {count}/{target}")
    
    print()
    print(f"Total collected: {total_collected} prompts")
    print(f"First-turn records processed: {first_turn_count}")
    print(f"Excluded (in train/test): {excluded_count}")
    print()
    
    # Save results
    output_file = data_dir / 'lmsys_barbell_20k.jsonl'
    print(f"💾 Saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        for subcat in ['deep_calculus', 'arithmetic_trick', 'kernel_debugging',
                       'html_boilerplate', 'email_draft', 'nuanced_haiku']:
            for record in collected[subcat]:
                json.dump(record, f)
                f.write('\n')
    
    print(f"✅ Saved {total_collected} prompts")
    print()
    print("="*70)
    print("Next steps:")
    print("  1. Verify data leakage: python scripts/verify_data_leakage.py")
    print("  2. Update tune_n_prior.py to use lmsys_barbell_20k.jsonl")
    print("="*70)

if __name__ == '__main__':
    main()
