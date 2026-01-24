#!/usr/bin/env python3
"""
Merge rejudged GPT-4-Turbo rewards into dev/holdout complete files.
Ensures format matches existing Mixtral and GPT-4o entries.
"""

import json
import gzip
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent

def verify_format(entry, expected_fields):
    """Verify entry has all expected fields."""
    missing = [f for f in expected_fields if f not in entry]
    if missing:
        print(f"  ⚠️  Missing fields: {missing}")
        return False
    return True

def merge_rejudged_data(complete_file, rejudged_file, output_file):
    """Merge rejudged GPT-4-Turbo data into complete file."""
    
    print(f"\n{'='*70}")
    print(f"Merging: {rejudged_file.name} → {complete_file.name}")
    print(f"{'='*70}")
    
    # Load existing complete data (excluding old GPT-4-Turbo)
    print("\n1️⃣  Loading existing data (excluding old GPT-4-Turbo)...")
    existing_entries = []
    gpt4turbo_count = 0
    
    with gzip.open(complete_file, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry['model_id'] == 'openai/gpt-4-turbo':
                gpt4turbo_count += 1
                continue  # Skip old GPT-4-Turbo entries
            existing_entries.append(entry)
    
    print(f"   ✅ Loaded {len(existing_entries)} entries (removed {gpt4turbo_count} old GPT-4-Turbo)")
    
    # Load rejudged GPT-4-Turbo data
    print("\n2️⃣  Loading rejudged GPT-4-Turbo data...")
    rejudged_entries = []
    
    if not rejudged_file.exists():
        print(f"   ❌ Rejudged file not found: {rejudged_file}")
        return False
    
    with open(rejudged_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('ok', False):
                rejudged_entries.append(entry)
    
    print(f"   ✅ Loaded {len(rejudged_entries)} rejudged GPT-4-Turbo entries")
    
    # Verify format matches
    print("\n3️⃣  Verifying format consistency...")
    
    # Check a sample existing entry (Mixtral or GPT-4o)
    sample_existing = next(e for e in existing_entries if e.get('ok', True))
    sample_rejudged = rejudged_entries[0] if rejudged_entries else None
    
    if not sample_rejudged:
        print("   ❌ No valid rejudged entries found")
        return False
    
    print(f"\n   Sample existing entry (Mixtral/GPT-4o):")
    print(f"     Fields: {list(sample_existing.keys())}")
    print(f"     raw_score: {sample_existing.get('raw_score')}")
    print(f"     Has judge_details: {'judge_details' in sample_existing}")
    
    print(f"\n   Sample rejudged entry (GPT-4-Turbo):")
    print(f"     Fields: {list(sample_rejudged.keys())}")
    print(f"     raw_score: {sample_rejudged.get('raw_score')}")
    print(f"     Has judge_details: {'judge_details' in sample_rejudged}")
    
    # Check required fields
    required_fields = ['model_id', 'prompt', 'response', 'ok', 'raw_score']
    
    print(f"\n   Checking required fields: {required_fields}")
    existing_ok = verify_format(sample_existing, required_fields)
    rejudged_ok = verify_format(sample_rejudged, required_fields)
    
    if not (existing_ok and rejudged_ok):
        print("   ❌ Format mismatch detected!")
        return False
    
    print("   ✅ Format matches!")
    
    # Check score distribution
    print("\n4️⃣  Checking score distributions...")
    
    from collections import Counter
    
    existing_scores = [e['raw_score'] for e in existing_entries if e.get('ok', True)]
    rejudged_scores = [e['raw_score'] for e in rejudged_entries]
    
    print(f"\n   Existing scores (Mixtral + GPT-4o):")
    for score, count in sorted(Counter(existing_scores).items()):
        print(f"     {score:.2f}: {count}")
    
    print(f"\n   Rejudged GPT-4-Turbo scores:")
    for score, count in sorted(Counter(rejudged_scores).items()):
        print(f"     {score:.2f}: {count}")
    
    # Check if rejudged scores are binary (0.0 or 1.0)
    unique_scores = set(rejudged_scores)
    if unique_scores - {0.0, 1.0}:
        print(f"   ⚠️  WARNING: Rejudged scores are not binary: {unique_scores}")
        print(f"   Expected only 0.0 and 1.0")
    else:
        print(f"   ✅ Rejudged scores are binary (0.0, 1.0)")
    
    # Merge
    print("\n5️⃣  Merging data...")
    all_entries = existing_entries + rejudged_entries
    print(f"   Total entries: {len(all_entries)}")
    
    # Count by model
    model_counts = Counter(e['model_id'] for e in all_entries if e.get('ok', True))
    print(f"\n   Entries by model:")
    for model_id, count in sorted(model_counts.items()):
        print(f"     {model_id}: {count}")
    
    # Save
    print(f"\n6️⃣  Saving to: {output_file}")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with gzip.open(output_file, 'wt') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"   ✅ Saved {len(all_entries)} entries")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "holdout", "both"], default="both")
    args = parser.parse_args()
    
    success = True
    
    if args.split in ["dev", "both"]:
        success &= merge_rejudged_data(
            complete_file=PROJECT_ROOT / "src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz",
            rejudged_file=PROJECT_ROOT / "data/dev_rewards_gpt4turbo_rejudged.jsonl",
            output_file=PROJECT_ROOT / "src/bandit_gpt/data/offline_dataset/dev_rewards_complete_NEW.jsonl.gz"
        )
    
    if args.split in ["holdout", "both"]:
        success &= merge_rejudged_data(
            complete_file=PROJECT_ROOT / "src/bandit_gpt/data/offline_dataset/holdout_rewards_complete.jsonl.gz",
            rejudged_file=PROJECT_ROOT / "data/holdout_rewards_gpt4turbo_rejudged.jsonl",
            output_file=PROJECT_ROOT / "src/bandit_gpt/data/offline_dataset/holdout_rewards_complete_NEW.jsonl.gz"
        )
    
    if success:
        print("\n" + "="*70)
        print("✅ MERGE COMPLETE!")
        print("="*70)
        print("\nTo use the new files:")
        if args.split in ["dev", "both"]:
            print("  mv src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz \\")
            print("     src/bandit_gpt/data/offline_dataset/dev_rewards_complete_BEFORE_REJUDGE.jsonl.gz")
            print("  mv src/bandit_gpt/data/offline_dataset/dev_rewards_complete_NEW.jsonl.gz \\")
            print("     src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz")
        if args.split in ["holdout", "both"]:
            print("  mv src/bandit_gpt/data/offline_dataset/holdout_rewards_complete.jsonl.gz \\")
            print("     src/bandit_gpt/data/offline_dataset/holdout_rewards_complete_BEFORE_REJUDGE.jsonl.gz")
            print("  mv src/bandit_gpt/data/offline_dataset/holdout_rewards_complete_NEW.jsonl.gz \\")
            print("     src/bandit_gpt/data/offline_dataset/holdout_rewards_complete.jsonl.gz")
    else:
        print("\n❌ Merge failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

