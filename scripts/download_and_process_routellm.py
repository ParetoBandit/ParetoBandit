#!/usr/bin/env python3
"""
Download RouteLLM Battle Data and Create Rewards Dataset

This script uses RouteLLM's technique:
1. Downloads the routellm/gpt4_judge_battles dataset from HuggingFace
2. Extracts pairwise battle outcomes (model A vs model B)
3. Each battle provides one training example with binary rewards:
   - Winner: 1.0
   - Loser: 0.0
   - Tie: 0.5 for both

This matches how RouteLLM trains their router on preference data.

Output format (RouteLLM style):
    {
        "prompt": "...",
        "model_a": "mistralai/mixtral-8x7b-instruct",
        "model_b": "openai/gpt-4-turbo",
        "reward_a": 0.0,  # 0.0=loss, 0.5=tie, 1.0=win
        "reward_b": 1.0
    }

Usage:
    python3 scripts/download_and_process_routellm.py
    
    # With options:
    python3 scripts/download_and_process_routellm.py \
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
        mixtral-8x7b-instruct-v0.1 → mistralai/mixtral-8x7b-instruct
        gpt-4-1106-preview → openai/gpt-4-turbo
        gpt-4o → openai/gpt-4o
    """
    if not name:
        return None
    
    name = name.lower().strip()
    
    # Mixtral variants
    if 'mixtral' in name and '8x7b' in name:
        return "mistralai/mixtral-8x7b-instruct"
    
    # GPT-4-turbo variants
    if 'gpt-4-turbo' in name or 'gpt-4-1106' in name:
        return "openai/gpt-4-turbo"
    
    # GPT-4o variants
    if 'gpt-4o' in name and 'mini' not in name:
        return "openai/gpt-4o"
    
    # GPT-4o-mini
    if 'gpt-4o-mini' in name:
        return "openai/gpt-4o-mini"
    
    # Claude variants
    if 'claude-3-opus' in name:
        return "anthropic/claude-3-opus"
    if 'claude-3-sonnet' in name:
        return "anthropic/claude-3-sonnet"
    if 'claude-3-haiku' in name:
        return "anthropic/claude-3-haiku"
    
    # Llama variants
    if 'llama-2-70b' in name or 'llama-2-70b-chat' in name:
        return "meta-llama/Llama-2-70b-chat-hf"
    if 'llama-2-7b' in name or 'llama-2-7b-chat' in name:
        return "meta-llama/Llama-2-7b-chat-hf"
    
    # Gemini variants
    if 'gemini-1.5-pro' in name:
        return "google/gemini-1.5-pro"
    if 'gemini-pro' in name:
        return "google/gemini-pro"
    
    return name


def extract_battle_outcome(row: dict) -> dict:
    """
    Extract battle outcome using RouteLLM's technique.
    
    Dataset format:
        - model_a, model_b: model names
        - winner_model_a, winner_model_b, winner_tie: binary indicators
        - prompt: the prompt text (may be a list)
    
    Returns RouteLLM format:
        {
            'prompt': str,
            'model_a': str,
            'model_b': str,
            'reward_a': float (0.0=loss, 0.5=tie, 1.0=win),
            'reward_b': float (0.0=loss, 0.5=tie, 1.0=win)
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
    
    # RouteLLM technique: Binary rewards from pairwise comparison
    if winner_a == 1:
        reward_a = 1.0
        reward_b = 0.0
    elif winner_b == 1:
        reward_a = 0.0
        reward_b = 1.0
    elif winner_tie == 1:
        reward_a = 0.5
        reward_b = 0.5
    else:
        # No winner indicated - skip
        return None
    
    return {
        'prompt': prompt,
        'model_a': model_a,
        'model_b': model_b,
        'reward_a': reward_a,
        'reward_b': reward_b
    }


def download_and_process(args):
    """Main pipeline: download, extract, filter, save."""
    
    print("="*80)
    print("DOWNLOAD ROUTELLM BATTLE DATA (RouteLLM Technique)")
    print("="*80)
    
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
    print(f"   Technique: Pairwise battle outcomes (RouteLLM method)")
    
    try:
        ds = load_dataset(
            "routellm/gpt4_judge_battles",
            split="train",
            streaming=True,
            token=hf_token
        )
        print(f"   ✅ Dataset loaded (streaming mode)")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Install datasets: pip install datasets")
        print(f"   2. Set HF token in .env: HF_TOKEN=your_token_here (if gated)")
        return
    
    # Step 2: Extract battle outcomes (RouteLLM technique)
    print(f"\n🔍 Step 2/3: Extracting pairwise battle outcomes...")
    battles = []
    skipped = 0
    filtered_out = 0
    
    for i, row in enumerate(tqdm(ds, total=max_battles, desc="   Processing")):
        if i >= max_battles:
            break
        
        battle = extract_battle_outcome(row)
        if battle is None:
            skipped += 1
            continue
        
        # Filter by models if specified
        if filter_models:
            if battle['model_a'] not in filter_models or battle['model_b'] not in filter_models:
                filtered_out += 1
                continue
        
        battles.append(battle)
        
        # Print sample
        if len(battles) == 1:
            print(f"\n   📋 Sample battle (RouteLLM format):")
            print(f"      Prompt: {battle['prompt'][:100]}...")
            print(f"      Model A: {battle['model_a']} → reward: {battle['reward_a']}")
            print(f"      Model B: {battle['model_b']} → reward: {battle['reward_b']}")
            print(f"      Interpretation: {'A wins' if battle['reward_a'] > battle['reward_b'] else 'B wins' if battle['reward_b'] > battle['reward_a'] else 'Tie'}\n")
    
    print(f"\n   ✅ Extracted {len(battles):,} battles")
    print(f"   ⏩ Skipped {skipped:,} (invalid/missing data)")
    if filter_models:
        print(f"   🔎 Filtered out {filtered_out:,} (models not in filter)")
    
    if len(battles) == 0:
        print(f"\n   ❌ No valid battles extracted!")
        if filter_models:
            print(f"   💡 Try removing --filter-models or check model names")
        return
    
    # Step 3: Statistics
    print(f"\n📊 Step 3/3: Computing statistics...")
    
    # Model participation
    model_counts = defaultdict(int)
    for battle in battles:
        model_counts[battle['model_a']] += 1
        model_counts[battle['model_b']] += 1
    
    print(f"\n   Top 10 models by battle participation:")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / len(battles) * 100
        print(f"      {model}: {count:,} battles ({pct:.1f}%)")
    
    # Outcome distribution
    wins_a = sum(1 for b in battles if b['reward_a'] > b['reward_b'])
    wins_b = sum(1 for b in battles if b['reward_b'] > b['reward_a'])
    ties = sum(1 for b in battles if b['reward_a'] == b['reward_b'])
    
    print(f"\n   Battle outcomes:")
    print(f"      Model A wins: {wins_a:,} ({wins_a/len(battles)*100:.1f}%)")
    print(f"      Model B wins: {wins_b:,} ({wins_b/len(battles)*100:.1f}%)")
    print(f"      Ties: {ties:,} ({ties/len(battles)*100:.1f}%)")
    
    # Reward distribution
    all_rewards = []
    for battle in battles:
        all_rewards.extend([battle['reward_a'], battle['reward_b']])
    
    print(f"\n   Reward values (RouteLLM binary rewards):")
    print(f"      0.0 (loss): {all_rewards.count(0.0):,}")
    print(f"      0.5 (tie): {all_rewards.count(0.5):,}")
    print(f"      1.0 (win): {all_rewards.count(1.0):,}")
    
    # Step 4: Save
    print(f"\n💾 Saving to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for battle in battles:
            f.write(json.dumps(battle) + '\n')
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved {len(battles):,} battles ({size_mb:.1f} MB)")
    
    # Step 5: Usage instructions
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE (RouteLLM Technique)")
    print("="*80)
    
    print(f"\n📊 Output Summary:")
    print(f"   File: {output_file}")
    print(f"   Battles (training examples): {len(battles):,}")
    print(f"   Models: {len(model_counts)}")
    print(f"   Format: RouteLLM pairwise battles")
    
    print(f"\n🔬 RouteLLM Technique:")
    print(f"   • Each battle = 1 training example")
    print(f"   • Binary rewards: 0.0 (loss), 0.5 (tie), 1.0 (win)")
    print(f"   • Pairwise preference data from GPT-4 judge")
    print(f"   • No aggregation needed - use battles directly")
    
    print(f"\n🚀 Next Steps:")
    print(f"\n   1. Generate warmup priors:")
    print(f"      python3 scripts/generate_warmup_priors.py \\")
    print(f"          --rewards-file {output_file} \\")
    print(f"          --pca artifacts/pca_23.joblib \\")
    print(f"          --output artifacts/priors_warmup.joblib")
    
    print(f"\n   2. Inspect data:")
    print(f"      head -5 {output_file} | jq")
    print(f"      # Look for: prompt, model_a, model_b, reward_a, reward_b")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Download RouteLLM battle data using their pairwise comparison technique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all battles (default: 100K)
    python3 scripts/download_and_process_routellm.py
    
    # Filter to specific model pair
    python3 scripts/download_and_process_routellm.py \\
        --filter-models "mistralai/mixtral-8x7b-instruct,openai/gpt-4-turbo"
    
    # Download more battles
    python3 scripts/download_and_process_routellm.py \\
        --max-battles 150000

RouteLLM Technique:
    Each battle provides binary preference data:
    - reward_a = 1.0 if model_a wins, 0.0 if loses, 0.5 if tie
    - reward_b = 1.0 if model_b wins, 0.0 if loses, 0.5 if tie
    
    This matches how RouteLLM trains their router on GPT-4 judge battles.

Environment:
    Set HUGGINGFACE_TOKEN or HF_TOKEN in .env file for dataset access.
        """
    )
    
    parser.add_argument(
        "--output", type=str,
        default=str(ROUTELLM_BATTLES_REWARDS_PATH),
        help="Output JSONL file path (default: canonical RouteLLM battles rewards path)"
    )
    parser.add_argument(
        "--max-battles", type=int, default=100000,
        help="Maximum number of battles to process (default: 100000)"
    )
    parser.add_argument(
        "--filter-models", type=str, default=None,
        help="Comma-separated list of models to keep (e.g., 'model1,model2'). If not set, keeps all."
    )
    
    args = parser.parse_args()
    
    download_and_process(args)


if __name__ == "__main__":
    main()
