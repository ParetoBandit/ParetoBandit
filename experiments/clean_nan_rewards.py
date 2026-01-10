#!/usr/bin/env python3
"""
Clean NaN rewards from LMSYS dataset files.

This script removes any entries with NaN raw_score values from the dataset.
"""

import gzip
import json
from pathlib import Path

def clean_dataset(input_path: Path, output_path: Path):
    """Remove entries with NaN rewards from a dataset file."""
    
    kept_count = 0
    removed_count = 0
    removed_prompts = set()
    
    with gzip.open(input_path, 'rt') as fin:
        with gzip.open(output_path, 'wt') as fout:
            for line in fin:
                entry = json.loads(line)
                raw_score = entry.get('raw_score')
                
                # Check for NaN (Python NaN != NaN)
                is_nan = raw_score is None or (isinstance(raw_score, float) and raw_score != raw_score)
                
                if is_nan:
                    removed_count += 1
                    prompt_preview = entry.get('prompt', '')[:100]
                    removed_prompts.add(prompt_preview)
                    print(f"  Removing: {entry.get('model_id')} - prompt: {prompt_preview}...")
                else:
                    fout.write(line)
                    kept_count += 1
    
    return kept_count, removed_count, removed_prompts


def main():
    data_dir = Path(__file__).parent.parent / "src" / "bandit_gpt" / "data" / "offline_dataset"
    
    # Process test data
    print("=" * 70)
    print("Cleaning TEST dataset...")
    print("=" * 70)
    test_input = data_dir / "lmsys_test_final_rewards_1k.jsonl.gz"
    test_output = data_dir / "lmsys_test_final_rewards_1k_clean.jsonl.gz"
    
    kept, removed, prompts = clean_dataset(test_input, test_output)
    print(f"\nTest: Kept {kept}, Removed {removed}")
    print(f"Unique prompts removed: {len(prompts)}")
    
    # Process train data
    print("\n" + "=" * 70)
    print("Cleaning TRAIN dataset...")
    print("=" * 70)
    train_input = data_dir / "lmsys_train_final_rewards_1k.jsonl.gz"
    train_output = data_dir / "lmsys_train_final_rewards_1k_clean.jsonl.gz"
    
    kept, removed, prompts = clean_dataset(train_input, train_output)
    print(f"\nTrain: Kept {kept}, Removed {removed}")
    print(f"Unique prompts removed: {len(prompts)}")
    
    print("\n" + "=" * 70)
    print("✅ Cleaning complete!")
    print(f"New files created:")
    print(f"  - {test_output}")
    print(f"  - {train_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
