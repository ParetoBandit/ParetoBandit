#!/usr/bin/env python3
"""
Prepare Canonical Dev Data for Calibration (Mixtral vs GPT-4o)

Uses the complete canonical dev set with existing Mixtral + GPT-4o data.
No generation needed - all data already exists!
"""

import json
import gzip
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# Target models
WEAK_MODEL = "mistralai/mixtral-8x7b-instruct"
STRONG_MODEL = "openai/gpt-4o"  # Using GPT-4o instead of GPT-4-turbo


def load_canonical_data(canonical_file: Path) -> dict:
    """Load canonical dev data and extract Mixtral + GPT-4o pairs."""
    print(f"📥 Loading canonical data from: {canonical_file.name}")
    
    prompt_data = defaultdict(lambda: {"rewards": {}, "responses": {}})
    
    with gzip.open(canonical_file, 'rt') as f:
        for line in f:
            try:
                entry = json.loads(line)
                
                if not entry.get("ok"):
                    continue
                
                prompt = entry.get("prompt", "")
                model_id = entry.get("model_id", "")
                
                # Only process our target models
                if model_id not in [WEAK_MODEL, STRONG_MODEL]:
                    continue
                
                # Get reward (try multiple field names)
                reward = entry.get("raw_score")
                if reward is None:
                    reward = entry.get("reward")
                if reward is None:
                    reward = entry.get("score")
                
                if reward is not None and prompt:
                    prompt_data[prompt]["rewards"][model_id] = float(reward)
                    prompt_data[prompt]["responses"][model_id] = entry.get("response", "")
                    
            except Exception as e:
                continue
    
    # Filter to prompts with BOTH models
    calibration_data = []
    for prompt, data in prompt_data.items():
        if len(data["rewards"]) == 2:  # Has both models
            calibration_data.append({
                "prompt": prompt,
                "rewards": data["rewards"]
            })
    
    print(f"   ✅ Loaded {len(prompt_data):,} prompts")
    print(f"   ✅ With both models: {len(calibration_data):,}")
    
    return calibration_data


def analyze_data(calibration_data: list):
    """Analyze the calibration data distribution."""
    print(f"\n📊 Data Analysis:")
    
    weak_rewards = [d['rewards'][WEAK_MODEL] for d in calibration_data]
    strong_rewards = [d['rewards'][STRONG_MODEL] for d in calibration_data]
    gaps = [strong_rewards[i] - weak_rewards[i] for i in range(len(calibration_data))]
    
    print(f"\n   Weak model ({WEAK_MODEL.split('/')[-1]}):")
    print(f"      Mean reward: {np.mean(weak_rewards):.4f}")
    print(f"      Std reward:  {np.std(weak_rewards):.4f}")
    print(f"      Min/Max:     {np.min(weak_rewards):.4f} / {np.max(weak_rewards):.4f}")
    
    print(f"\n   Strong model ({STRONG_MODEL.split('/')[-1]}):")
    print(f"      Mean reward: {np.mean(strong_rewards):.4f}")
    print(f"      Std reward:  {np.std(strong_rewards):.4f}")
    print(f"      Min/Max:     {np.min(strong_rewards):.4f} / {np.max(strong_rewards):.4f}")
    
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
    parser = argparse.ArgumentParser(description="Prepare canonical dev data for calibration")
    parser.add_argument(
        "--canonical-file", type=str,
        default="../../../src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz",
        help="Path to canonical dev file"
    )
    parser.add_argument(
        "--output", type=str,
        default="../data/canonical_dev_calibration.jsonl",
        help="Output file for calibration data"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PREPARE CANONICAL DEV FOR CALIBRATION (MIXTRAL vs GPT-4o)")
    print("="*80)
    
    # Load data
    canonical_file = Path(args.canonical_file)
    if not canonical_file.exists():
        print(f"❌ File not found: {canonical_file}")
        return
    
    calibration_data = load_canonical_data(canonical_file)
    
    if not calibration_data:
        print("\n❌ No valid calibration data found!")
        return
    
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
    print("✅ CANONICAL DEV DATA READY!")
    print("="*80)
    print(f"\n📋 Next steps:")
    print(f"   1. Find optimal gamma:")
    print(f"      python3 find_gamma.py \\")
    print(f"        --calibration-data {output_file} \\")
    print(f"        --output canonical_gamma_results/")
    print(f"\n   2. Calibrate router:")
    print(f"      python3 calibrate_router.py \\")
    print(f"        --calibration-data {output_file} \\")
    print(f"        --gamma 0.002 \\")
    print(f"        --output canonical_calibrated_router.joblib")
    print("="*80)


if __name__ == "__main__":
    main()

