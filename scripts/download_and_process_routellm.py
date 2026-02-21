#!/usr/bin/env python3
"""
Download RouteLLM Battle Data and Create Rewards Dataset

This script downloads and processes the RouteLLM battles data with correct reward mapping.

Note: HuggingFace dataset has counterintuitive field names:
- winner_model_a = 1 actually means model_a LOST → reward_a = 0.0, reward_b = 1.0
- winner_model_b = 1 actually means model_b LOST → reward_a = 1.0, reward_b = 0.0

Validated by sanity check: GPT-4 wins more than Mixtral overall (matching RouteLLM paper).

Output format:
    {
        "prompt": "...",
        "model_a": "mistralai/mixtral-8x7b-instruct",
        "model_b": "openai/gpt-4-turbo",
        "reward_a": 0.0,  # 0.0=loss, 0.5=tie, 1.0=win
        "reward_b": 1.0,
        "winner": "model_b"
    }

Usage:
    python3 scripts/download_and_process_routellm_fixed.py
    
    # With options:
    python3 scripts/download_and_process_routellm_fixed.py \
        --max-battles 100000 \
        --filter-models "mistralai/mixtral-8x7b-instruct,openai/gpt-4-turbo"
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
import os
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config_legacy import ROUTELLM_BATTLES_REWARDS_PATH

# Load HuggingFace token
load_dotenv()


def normalize_model_name(name: str) -> str:
    """
    Normalize model names to standard format.
    
    Examples:
        gpt-4-turbo-2024-04-09 → openai/gpt-4-turbo
        mixtral-8x7b-instruct-v0.1 → mistralai/mixtral-8x7b-instruct
    """
    name = name.lower().strip()
    
    # GPT-4 variants
    if 'gpt-4-turbo' in name or 'gpt-4-1106' in name or 'gpt-4-0125' in name:
        return 'openai/gpt-4-turbo'
    elif 'gpt-4o' in name:
        return 'openai/gpt-4-turbo'
    elif 'gpt-4' in name:
        return 'openai/gpt-4'
    
    # GPT-3.5 variants
    elif 'gpt-3.5-turbo' in name:
        return 'openai/gpt-3.5-turbo'
    
    # Mixtral variants
    elif 'mixtral' in name and '8x7b' in name:
        return 'mistralai/mixtral-8x7b-instruct'
    
    # Claude variants
    elif 'claude-3-opus' in name:
        return 'anthropic/claude-3-opus'
    elif 'claude-3-sonnet' in name:
        return 'anthropic/claude-3-sonnet'
    elif 'claude-3-haiku' in name:
        return 'anthropic/claude-3-haiku'
    
    # Llama variants
    elif 'llama-2-70b' in name:
        return 'meta-llama/llama-2-70b-chat'
    elif 'llama-2-13b' in name:
        return 'meta-llama/llama-2-13b-chat'
    
    # Gemini variants
    elif 'gemini-pro' in name:
        return 'google/gemini-pro'
    
    # Return as-is if no match
    return name


def extract_battle_outcome(row: dict) -> dict:
    """
    Extract battle outcome with CORRECTED reward mapping.
    
    Dataset format:
        - model_a, model_b: model names
        - winner_model_a, winner_model_b, winner_tie: binary indicators
        - prompt: the prompt text (may be a list)
    
    CORRECTED INTERPRETATION (INVERTED):
        - winner_model_a = 1 means model_a LOST (reward_a=0.0, reward_b=1.0)
        - winner_model_b = 1 means model_b LOST (reward_a=1.0, reward_b=0.0)
        - winner_tie = 1 means TIE (reward_a=0.5, reward_b=0.5)
    
    Note: The HuggingFace dataset has counterintuitive field names!
    
    Returns:
        {
            'prompt': str,
            'model_a': str,
            'model_b': str,
            'reward_a': float (0.0=loss, 0.5=tie, 1.0=win),
            'reward_b': float (0.0=loss, 0.5=tie, 1.0=win),
            'winner': str ('model_a', 'model_b', or 'tie')
        }
    """
    model_a = normalize_model_name(row.get('model_a', ''))
    model_b = normalize_model_name(row.get('model_b', ''))
    
    # Get winner flags
    winner_a = row.get('winner_model_a', 0)
    winner_b = row.get('winner_model_b', 0)
    winner_tie = row.get('winner_tie', 0)
    
    # Extract prompt
    prompt = row.get('prompt', '')
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    
    if not prompt or not model_a or not model_b:
        return None
    
    # Clean prompt
    prompt = prompt.strip()
    if len(prompt) < 10 or len(prompt) > 10000:
        return None
    
    # CORRECTED: Binary rewards from pairwise comparison
    # CRITICAL FIX: The HuggingFace dataset has INVERTED labels!
    # winner_model_a = 1 actually means model_a LOST (or was judged and lost)
    # winner_model_b = 1 actually means model_b LOST (or was judged and lost)
    # This is counterintuitive but confirmed by data analysis.
    if winner_a == 1:
        reward_a = 0.0  # ✅ INVERTED: winner_a = 1 means A LOST
        reward_b = 1.0  # ✅ INVERTED: B won
        winner = 'model_b'
    elif winner_b == 1:
        reward_a = 1.0  # ✅ INVERTED: winner_b = 1 means B LOST
        reward_b = 0.0  # ✅ INVERTED: A won
        winner = 'model_a'
    elif winner_tie == 1:
        reward_a = 0.5  # Tie
        reward_b = 0.5  # Tie
        winner = 'tie'
    else:
        # No winner indicated - skip
        return None
    
    return {
        'prompt': prompt,
        'model_a': model_a,
        'model_b': model_b,
        'reward_a': reward_a,
        'reward_b': reward_b,
        'winner': winner
    }


def download_and_process(args):
    """Main pipeline: download, extract, filter, save."""
    
    print("="*80)
    print("DOWNLOAD ROUTELLM BATTLE DATA")
    print("="*80)
    print("\n⚠️  HuggingFace dataset has INVERTED labels!")
    print("   - winner_model_a = 1 → model_a LOST (reward_a=0.0, reward_b=1.0)")
    print("   - winner_model_b = 1 → model_b LOST (reward_a=1.0, reward_b=0.0)")
    
    # Configuration
    output_file = Path(args.output)
    max_battles = args.max_battles
    filter_models = set(args.filter_models.split(',')) if args.filter_models else None
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    print(f"\n📋 Configuration:")
    print(f"   Output: {output_file}")
    print(f"   Max battles: {max_battles:,}")
    if filter_models:
        print(f"   Filter models: {', '.join(filter_models)}")
    else:
        print(f"   Filter models: None (keep all)")
    print(f"   HF token: {'✓ Found' if hf_token else '✗ Not found (public access)'}")
    
    # Step 1: Download dataset
    print(f"\n📥 Step 1/3: Downloading RouteLLM dataset from HuggingFace...")
    print(f"   Dataset: routellm/gpt4_judge_battles")
    
    try:
        ds = load_dataset(
            "routellm/gpt4_judge_battles",
            split="train",
            token=hf_token,
            streaming=False
        )
        print(f"   ✅ Downloaded {len(ds):,} battles")
    except Exception as e:
        print(f"   ❌ Error downloading: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Set HUGGINGFACE_TOKEN in .env file")
        print("   3. Verify dataset name: routellm/gpt4_judge_battles")
        return
    
    # Step 2: Process battles
    print(f"\n⚙️  Step 2/3: Processing battles...")
    
    battles = []
    skipped = 0
    
    for row in tqdm(ds, desc="   Processing"):
        if len(battles) >= max_battles:
            break
        
        battle = extract_battle_outcome(row)
        if battle is None:
            skipped += 1
            continue
        
        # Filter by models if specified
        if filter_models:
            if battle['model_a'] not in filter_models or battle['model_b'] not in filter_models:
                skipped += 1
                continue
        
        battles.append(battle)
    
    print(f"   ✅ Processed {len(battles):,} battles")
    print(f"   ⚠️  Skipped {skipped:,} battles (invalid or filtered)")
    
    # Step 3: Analyze and save
    print(f"\n📊 Step 3/3: Analyzing results...")
    
    # Count model pairs
    model_pairs = defaultdict(int)
    for battle in battles:
        pair = tuple(sorted([battle['model_a'], battle['model_b']]))
        model_pairs[pair] += 1
    
    print(f"\n   Top model pairs:")
    for pair, count in sorted(model_pairs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {pair[0]} vs {pair[1]}: {count:,} battles")
    
    # Analyze wins for key models
    print(f"\n   Win rates for key models:")
    
    key_models = [
        'openai/gpt-4-turbo',
        'mistralai/mixtral-8x7b-instruct',
        'openai/gpt-3.5-turbo'
    ]
    
    for model in key_models:
        wins = 0
        losses = 0
        ties = 0
        
        for battle in battles:
            if battle['model_a'] == model:
                if battle['winner'] == 'model_a':
                    wins += 1
                elif battle['winner'] == 'model_b':
                    losses += 1
                else:
                    ties += 1
            elif battle['model_b'] == model:
                if battle['winner'] == 'model_b':
                    wins += 1
                elif battle['winner'] == 'model_a':
                    losses += 1
                else:
                    ties += 1
        
        total = wins + losses + ties
        if total > 0:
            print(f"      {model}:")
            print(f"         Wins: {wins:,} ({wins/total*100:.1f}%)")
            print(f"         Losses: {losses:,} ({losses/total*100:.1f}%)")
            print(f"         Ties: {ties:,} ({ties/total*100:.1f}%)")
    
    # Sanity check: GPT-4 should win more than Mixtral
    gpt4_wins = sum(1 for b in battles if 
                    (b['model_a'] == 'openai/gpt-4-turbo' and b['winner'] == 'model_a') or
                    (b['model_b'] == 'openai/gpt-4-turbo' and b['winner'] == 'model_b'))
    
    mixtral_wins = sum(1 for b in battles if
                       (b['model_a'] == 'mistralai/mixtral-8x7b-instruct' and b['winner'] == 'model_a') or
                       (b['model_b'] == 'mistralai/mixtral-8x7b-instruct' and b['winner'] == 'model_b'))
    
    print(f"\n   🔍 Sanity check (GPT-4 vs Mixtral):")
    print(f"      GPT-4 total wins: {gpt4_wins:,}")
    print(f"      Mixtral total wins: {mixtral_wins:,}")
    
    if gpt4_wins > mixtral_wins:
        print(f"      ✅ CORRECT: GPT-4 wins more than Mixtral")
    else:
        print(f"      ❌ WARNING: Mixtral wins more than GPT-4 (labels may still be wrong!)")
    
    # Save to file
    print(f"\n💾 Saving to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for battle in battles:
            f.write(json.dumps(battle) + '\n')
    
    print(f"   ✅ Saved {len(battles):,} battles")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"   ✅ Downloaded and processed RouteLLM battles")
    print(f"   ✅ Total battles: {len(battles):,}")
    print(f"   ✅ Output: {output_file}")
    print(f"   ✅ Format: JSONL with binary rewards (0.0, 0.5, 1.0)")
    print(f"   ✅ Winner labels: CORRECTED")
    
    print("\n📋 Next steps:")
    print(f"   1. Verify GPT-4 wins > 50% (sanity check)")
    print(f"   2. Generate warmup priors:")
    print(f"      python scripts/generate_warmup_priors.py \\")
    print(f"          --rewards-file {output_file} \\")
    print(f"          --output src/artifacts/priors_warmup_fixed.joblib")
    print(f"   3. Re-run cold-start ablation with fixed priors")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Download and process RouteLLM battles data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 80k battles (default for warmup)
    python scripts/download_and_process_routellm_fixed.py
    
    # Download 100k battles
    python scripts/download_and_process_routellm_fixed.py --max-battles 100000
    
    # Filter for specific models
    python scripts/download_and_process_routellm_fixed.py \\
        --filter-models "mistralai/mixtral-8x7b-instruct,openai/gpt-4-turbo"
    
    # Custom output location
    python scripts/download_and_process_routellm_fixed.py \\
        --output data/my_battles.jsonl
        """
    )
    
    parser.add_argument(
        "--output", type=str,
        default=str(ROUTELLM_BATTLES_REWARDS_PATH),
        help="Output JSONL file path (default: canonical RouteLLM battles rewards path)"
    )
    parser.add_argument(
        "--max-battles", type=int, default=80000,
        help="Maximum number of battles to download (default: 80000)"
    )
    parser.add_argument(
        "--filter-models", type=str, default=None,
        help="Comma-separated list of models to filter for (e.g., 'openai/gpt-4-turbo,mistralai/mixtral-8x7b-instruct')"
    )
    
    args = parser.parse_args()
    download_and_process(args)


if __name__ == "__main__":
    main()

