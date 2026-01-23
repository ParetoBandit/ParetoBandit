#!/usr/bin/env python3
"""
Prepare Dev Data for Calibration

Converts the multi-model dev evaluation data into the format needed for
gamma calibration (2-model rewards).

Input: dev_rewards_complete.jsonl.gz (1,354 prompts × 35-42 models)
Output: canonical_dev_calibration.jsonl (dev prompts × 2 models)
"""

import json
import gzip
import argparse
from pathlib import Path
from typing import Dict, List
import random

# Target models for calibration
WEAK_MODEL = "mistralai/mixtral-8x7b-instruct"
STRONG_MODEL = "openai/gpt-4o"


def load_dev_data(dev_file: Path) -> Dict[str, List[Dict]]:
    """Load dev data grouped by prompt."""
    print(f"📥 Loading dev data from: {dev_file.name}")
    
    prompts_data = {}
    total_entries = 0
    
    # Handle both gzipped and plain JSONL
    if dev_file.suffix == '.gz':
        open_func = lambda f: gzip.open(f, 'rt')
    else:
        open_func = lambda f: open(f, 'r')
    
    with open_func(dev_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '')
                model_id = entry.get('model_id', '')
                
                if not prompt or not model_id:
                    continue
                
                # Group by prompt
                if prompt not in prompts_data:
                    prompts_data[prompt] = []
                
                prompts_data[prompt].append(entry)
                total_entries += 1
                
            except Exception as e:
                continue
    
    print(f"   ✅ Loaded {total_entries:,} entries")
    print(f"   ✅ Unique prompts: {len(prompts_data):,}")
    
    return prompts_data


def extract_target_models(prompts_data: Dict[str, List[Dict]]) -> List[Dict]:
    """Extract calibration data for target model pair."""
    print(f"\n🎯 Extracting target models:")
    print(f"   Weak:   {WEAK_MODEL}")
    print(f"   Strong: {STRONG_MODEL}")
    
    calibration_data = []
    missing_weak = 0
    missing_strong = 0
    
    for prompt, entries in prompts_data.items():
        # Find rewards for target models
        rewards = {}
        
        for entry in entries:
            model_id = entry.get('model_id', '')
            
            if model_id == WEAK_MODEL:
                # Get reward (try different field names)
                reward = entry.get('raw_score')
                if reward is None:
                    reward = entry.get('reward')
                if reward is None:
                    reward = entry.get('score')
                
                if reward is not None:
                    rewards[WEAK_MODEL] = float(reward)
            
            elif model_id == STRONG_MODEL:
                reward = entry.get('raw_score')
                if reward is None:
                    reward = entry.get('reward')
                if reward is None:
                    reward = entry.get('score')
                
                if reward is not None:
                    rewards[STRONG_MODEL] = float(reward)
        
        # Only include prompts with both models
        if WEAK_MODEL in rewards and STRONG_MODEL in rewards:
            calibration_data.append({
                'prompt': prompt,
                'rewards': rewards
            })
        else:
            if WEAK_MODEL not in rewards:
                missing_weak += 1
            if STRONG_MODEL not in rewards:
                missing_strong += 1
    
    print(f"\n   ✅ Prompts with both models: {len(calibration_data):,}")
    if missing_weak > 0 or missing_strong > 0:
        print(f"   ⚠️  Missing weak model: {missing_weak:,}")
        print(f"   ⚠️  Missing strong model: {missing_strong:,}")
    
    return calibration_data


def analyze_data(calibration_data: List[Dict]):
    """Analyze the calibration data distribution."""
    print(f"\n📊 Data Analysis:")
    
    # Reward statistics
    weak_rewards = [d['rewards'][WEAK_MODEL] for d in calibration_data]
    strong_rewards = [d['rewards'][STRONG_MODEL] for d in calibration_data]
    
    import numpy as np
    
    print(f"\n   Weak model ({WEAK_MODEL.split('/')[-1]}):")
    print(f"      Mean reward: {np.mean(weak_rewards):.4f}")
    print(f"      Std reward:  {np.std(weak_rewards):.4f}")
    print(f"      Min/Max:     {np.min(weak_rewards):.4f} / {np.max(weak_rewards):.4f}")
    
    print(f"\n   Strong model ({STRONG_MODEL.split('/')[-1]}):")
    print(f"      Mean reward: {np.mean(strong_rewards):.4f}")
    print(f"      Std reward:  {np.std(strong_rewards):.4f}")
    print(f"      Min/Max:     {np.min(strong_rewards):.4f} / {np.max(strong_rewards):.4f}")
    
    # Performance gap
    gaps = [strong_rewards[i] - weak_rewards[i] for i in range(len(calibration_data))]
    print(f"\n   Performance Gap (Strong - Weak):")
    print(f"      Mean gap: {np.mean(gaps):.4f}")
    print(f"      Std gap:  {np.std(gaps):.4f}")
    print(f"      Min/Max:  {np.min(gaps):.4f} / {np.max(gaps):.4f}")
    
    # Difficulty distribution
    easy = sum(1 for g in gaps if g < 0.2)
    moderate = sum(1 for g in gaps if 0.2 <= g <= 0.6)
    hard = sum(1 for g in gaps if g > 0.6)
    
    print(f"\n   Difficulty Distribution:")
    print(f"      Easy (<0.2 gap):     {easy:,} ({easy/len(gaps)*100:.1f}%)")
    print(f"      Moderate (0.2-0.6):  {moderate:,} ({moderate/len(gaps)*100:.1f}%)")
    print(f"      Hard (>0.6 gap):     {hard:,} ({hard/len(gaps)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Prepare dev data for calibration")
    parser.add_argument(
        "--dev-file", type=str,
        default="../../../src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz",
        help="Path to dev rewards file"
    )
    parser.add_argument(
        "--output", type=str,
        default="../data/canonical_dev_calibration.jsonl",
        help="Output file for calibration data"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N prompts (for faster experiments)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PREPARE DEV DATA FOR CALIBRATION")
    print("="*80)
    
    # Load dev data
    dev_file = Path(args.dev_file)
    if not dev_file.exists():
        print(f"❌ Dev file not found: {dev_file}")
        return
    
    prompts_data = load_dev_data(dev_file)
    
    # Extract target models
    calibration_data = extract_target_models(prompts_data)
    
    if not calibration_data:
        print("\n❌ No valid calibration data found!")
        print("   Check that dev file contains both target models.")
        return
    
    # Sample if requested
    if args.sample and args.sample < len(calibration_data):
        print(f"\n🎲 Randomly sampling {args.sample} prompts (seed={args.seed})...")
        random.seed(args.seed)
        calibration_data = random.sample(calibration_data, args.sample)
        print(f"   ✅ Sampled {len(calibration_data):,} prompts")
    
    # Limit if requested
    if args.limit:
        calibration_data = calibration_data[:args.limit]
        print(f"\n✂️  Limited to {len(calibration_data):,} prompts")
    
    # Analyze
    analyze_data(calibration_data)
    
    # Save
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving calibration data to: {output_file}")
    with open(output_file, 'w') as f:
        for item in calibration_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"   ✅ Saved {len(calibration_data):,} prompts")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*80)
    print("✅ DEV DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"\n📋 Next steps:")
    print(f"   1. Find optimal gamma:")
    print(f"      python3 find_gamma.py \\")
    print(f"        --calibration-data {output_file} \\")
    print(f"        --output dev_gamma_results/")
    print(f"\n   2. Calibrate router:")
    print(f"      python3 calibrate_router.py \\")
    print(f"        --calibration-data {output_file} \\")
    print(f"        --gamma 0.002 \\")
    print(f"        --output dev_calibrated_router.joblib")
    print("="*80)


if __name__ == "__main__":
    main()

