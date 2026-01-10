#!/usr/bin/env python3
"""
Create split-specific reward files from canonical splits.json

This script reads the canonical splits.json and creates separate reward files
for dev_pool and holdout_pool, preventing confusion and accidental data leakage.

Output files:
  - dev_rewards.jsonl.gz (development set rewards)
  - holdout_rewards.jsonl.gz (test set rewards)
"""

import json
import gzip
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from scripts/ to project root
DATA_DIR = PROJECT_ROOT / "src" / "bandit_gpt" / "data" / "offline_dataset"
SPLITS_PATH = PROJECT_ROOT / "experiments" / "01_effectiveness" / "results" / "splits.json"

# Input files (combined train + test rewards)
TRAIN_REWARDS = DATA_DIR / "lmsys_train_final_rewards_1k_clean.jsonl.gz"
TEST_REWARDS = DATA_DIR / "lmsys_test_final_rewards_1k_clean.jsonl.gz"

# Output files (split by dev/holdout)
DEV_REWARDS_OUT = DATA_DIR / "dev_rewards.jsonl.gz"
HOLDOUT_REWARDS_OUT = DATA_DIR / "holdout_rewards.jsonl.gz"


def load_splits():
    """Load canonical splits from splits.json."""
    print(f"📂 Loading splits from {SPLITS_PATH}")
    
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"❌ splits.json not found at {SPLITS_PATH}\n"
            f"   Run experiments/01_effectiveness/run_budget_experiment.py first "
            f"to generate canonical splits."
        )
    
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    
    dev_pool = set(splits["dev_pool"])
    holdout_pool = set(splits["holdout_pool"])
    
    # Verify disjointness
    overlap = dev_pool.intersection(holdout_pool)
    if overlap:
        raise ValueError(f"❌ Data leakage detected! {len(overlap)} overlapping prompts.")
    
    print(f"  ✓ Dev pool: {len(dev_pool)} prompts")
    print(f"  ✓ Holdout pool: {len(holdout_pool)} prompts")
    print(f"  ✓ Verified disjoint (0 overlaps)")
    
    return dev_pool, holdout_pool


def load_all_rewards():
    """Load all rewards from train and test files."""
    print(f"\n📦 Loading reward files...")
    
    all_rewards = []
    
    # Load train rewards
    if TRAIN_REWARDS.exists():
        print(f"  - Reading {TRAIN_REWARDS.name}")
        with gzip.open(TRAIN_REWARDS, 'rt') as f:
            for line in f:
                all_rewards.append(json.loads(line))
    else:
        print(f"  ⚠️  {TRAIN_REWARDS.name} not found, skipping")
    
    # Load test rewards
    if TEST_REWARDS.exists():
        print(f"  - Reading {TEST_REWARDS.name}")
        with gzip.open(TEST_REWARDS, 'rt') as f:
            for line in f:
                all_rewards.append(json.loads(line))
    else:
        print(f"  ⚠️  {TEST_REWARDS.name} not found, skipping")
    
    print(f"  ✓ Loaded {len(all_rewards)} total reward entries")
    return all_rewards


def split_rewards(all_rewards, dev_pool, holdout_pool):
    """Split rewards by dev/holdout membership."""
    print(f"\n✂️  Splitting rewards by canonical splits...")
    
    dev_rewards = []
    holdout_rewards = []
    unassigned = []
    
    for entry in all_rewards:
        prompt = entry.get("prompt")
        
        if prompt in dev_pool:
            dev_rewards.append(entry)
        elif prompt in holdout_pool:
            holdout_rewards.append(entry)
        else:
            unassigned.append(entry)
    
    print(f"  ✓ Dev rewards: {len(dev_rewards)}")
    print(f"  ✓ Holdout rewards: {len(holdout_rewards)}")
    
    if unassigned:
        print(f"  ⚠️  Unassigned: {len(unassigned)} (prompts not in splits.json)")
    
    return dev_rewards, holdout_rewards


def write_reward_file(rewards, output_path):
    """Write rewards to compressed JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with gzip.open(output_path, 'wt') as f:
        for entry in rewards:
            f.write(json.dumps(entry) + '\n')
    
    print(f"  ✓ Wrote {len(rewards)} entries to {output_path.name}")


def verify_splits():
    """Verify the split reward files are disjoint."""
    print(f"\n🔍 Verifying split integrity...")
    
    dev_prompts = set()
    holdout_prompts = set()
    
    with gzip.open(DEV_REWARDS_OUT, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            dev_prompts.add(entry["prompt"])
    
    with gzip.open(HOLDOUT_REWARDS_OUT, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            holdout_prompts.add(entry["prompt"])
    
    overlap = dev_prompts.intersection(holdout_prompts)
    
    if overlap:
        raise ValueError(
            f"❌ CRITICAL: Data leakage detected!\n"
            f"   {len(overlap)} prompts appear in both dev_rewards.jsonl.gz "
            f"and holdout_rewards.jsonl.gz"
        )
    
    print(f"  ✓ Dev unique prompts: {len(dev_prompts)}")
    print(f"  ✓ Holdout unique prompts: {len(holdout_prompts)}")
    print(f"  ✓ Verified disjoint: 0 overlaps")


def generate_summary():
    """Generate summary statistics."""
    print(f"\n📊 Summary Statistics:")
    
    # Count model coverage
    dev_coverage = defaultdict(int)
    holdout_coverage = defaultdict(int)
    
    with gzip.open(DEV_REWARDS_OUT, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                dev_coverage[entry["model_id"]] += 1
    
    with gzip.open(HOLDOUT_REWARDS_OUT, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                holdout_coverage[entry["model_id"]] += 1
    
    print(f"\n  Dev Set:")
    print(f"    - Models: {len(dev_coverage)}")
    print(f"    - Total successful entries: {sum(dev_coverage.values())}")
    
    print(f"\n  Holdout Set:")
    print(f"    - Models: {len(holdout_coverage)}")
    print(f"    - Total successful entries: {sum(holdout_coverage.values())}")
    
    print(f"\n📁 Output Files:")
    print(f"  - {DEV_REWARDS_OUT}")
    print(f"  - {HOLDOUT_REWARDS_OUT}")


def main():
    """Main execution."""
    print("="*70)
    print("CREATING SPLIT-SPECIFIC REWARD FILES")
    print("="*70)
    
    # 1. Load canonical splits
    dev_pool, holdout_pool = load_splits()
    
    # 2. Load all rewards
    all_rewards = load_all_rewards()
    
    # 3. Split rewards by membership
    dev_rewards, holdout_rewards = split_rewards(all_rewards, dev_pool, holdout_pool)
    
    # 4. Write split-specific files
    print(f"\n💾 Writing split-specific reward files...")
    write_reward_file(dev_rewards, DEV_REWARDS_OUT)
    write_reward_file(holdout_rewards, HOLDOUT_REWARDS_OUT)
    
    # 5. Verify integrity
    verify_splits()
    
    # 6. Generate summary
    generate_summary()
    
    print("\n" + "="*70)
    print("✅ SUCCESS! Split-specific reward files created.")
    print("="*70)


if __name__ == "__main__":
    main()
