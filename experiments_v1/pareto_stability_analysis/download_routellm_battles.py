#!/usr/bin/env python3
"""
Download the actual routellm/gpt4_judge_battles dataset from HuggingFace
and extract real battle outcomes (rewards) for our models.

This gives us REAL data that RouteLLM was trained on, not synthetic IRT simulations.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

# Load HF token from .env
load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "routellm" / "data"
OUTPUT_FILE = DATA_DIR / "routellm_battles_raw.jsonl"

# Models we care about
TARGET_MODELS = {
    "mixtral-8x7b-instruct-v0.1",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistralai/mixtral-8x7b-instruct",
    "gpt-4-1106-preview",
    "gpt-4-turbo-preview",
    "openai/gpt-4-turbo",
    "gpt-4-turbo",
    "gpt-4o",
    "openai/gpt-4o"
}


def normalize_model_name(name: str) -> str:
    """Normalize model names to standard format."""
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
    if 'gpt-4o' in name:
        return "openai/gpt-4o"
    
    return name


def extract_battle_outcome(row: dict) -> dict:
    """
    Extract battle outcome from a row in the dataset.
    
    Actual format:
    - model_a, model_b: model names
    - winner_model_a, winner_model_b, winner_tie: binary indicators
    - prompt: the prompt text (may be a list)
    """
    model_a = normalize_model_name(row.get('model_a', ''))
    model_b = normalize_model_name(row.get('model_b', ''))
    
    # Get winner flags
    winner_a = row.get('winner_model_a', 0)
    winner_b = row.get('winner_model_b', 0)
    winner_tie = row.get('winner_tie', 0)
    
    prompt = row.get('prompt', '')
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    
    if not prompt or not model_a or not model_b:
        return None
    
    # Determine rewards based on winner flags
    if winner_a == 1:
        reward_a = 1.0
        reward_b = 0.0
        winner = "model_a"
    elif winner_b == 1:
        reward_a = 0.0
        reward_b = 1.0
        winner = "model_b"
    elif winner_tie == 1:
        reward_a = 0.5
        reward_b = 0.5
        winner = "tie"
    else:
        return None
    
    return {
        'prompt': prompt,
        'model_a': model_a,
        'model_b': model_b,
        'reward_a': reward_a,
        'reward_b': reward_b,
        'winner': winner
    }


def download_and_extract():
    """Download dataset and extract battle outcomes."""
    print("="*80)
    print("Downloading RouteLLM GPT-4 Judge Battles Dataset")
    print("="*80)
    
    print(f"\n📥 Loading dataset from HuggingFace...")
    print(f"   Dataset: routellm/gpt4_judge_battles")
    
    try:
        # Load dataset (streaming for memory efficiency)
        ds = load_dataset(
            "routellm/gpt4_judge_battles",
            split="train",
            streaming=True,
            token=hf_token
        )
        
        print(f"   ✅ Dataset loaded successfully")
        
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        print(f"\n💡 Make sure you have:")
        print(f"   1. Installed datasets: pip install datasets")
        print(f"   2. HuggingFace token in .env: HF_TOKEN=...")
        return
    
    # Extract battles
    print(f"\n🔍 Extracting battle outcomes...")
    battles = []
    target_models_count = 0
    
    for i, row in enumerate(tqdm(ds, desc="   Processing")):
        if i >= 100000:  # Limit to first 100K for safety
            break
        
        battle = extract_battle_outcome(row)
        if battle is None:
            continue
        
        battles.append(battle)
        
        # Check if involves our target models
        if (battle['model_a'] in TARGET_MODELS or 
            battle['model_b'] in TARGET_MODELS):
            target_models_count += 1
        
        # Print sample
        if i == 0:
            print(f"\n   📋 Sample battle:")
            print(f"      Model A: {battle['model_a']}")
            print(f"      Model B: {battle['model_b']}")
            print(f"      Winner: {battle['winner']}")
            print(f"      Prompt: {battle['prompt'][:100]}...")
    
    print(f"\n   ✅ Extracted {len(battles):,} battles")
    print(f"   🎯 Involving target models: {target_models_count:,}")
    
    if len(battles) == 0:
        print(f"\n   ❌ No battles extracted! Check dataset format.")
        return
    
    # Save to file
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for battle in battles:
            f.write(json.dumps(battle) + '\n')
    
    print(f"   ✅ Saved {len(battles):,} battles")
    
    # Statistics
    print(f"\n📊 Dataset Statistics:")
    
    # Count models
    model_counts = {}
    for battle in battles:
        for model in [battle['model_a'], battle['model_b']]:
            model_counts[model] = model_counts.get(model, 0) + 1
    
    print(f"\n   Top 10 models by battle count:")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        in_target = "✓" if model in TARGET_MODELS else " "
        print(f"      [{in_target}] {model}: {count:,}")
    
    # Outcome distribution
    winner_counts = {'model_a': 0, 'model_b': 0, 'tie': 0}
    for battle in battles:
        winner_counts[battle['winner']] = winner_counts.get(battle['winner'], 0) + 1
    
    print(f"\n   Winner distribution:")
    for winner, count in winner_counts.items():
        pct = count / len(battles) * 100
        print(f"      {winner}: {count:,} ({pct:.1f}%)")
    
    # Filter to target models
    target_battles = [
        b for b in battles 
        if b['model_a'] in TARGET_MODELS or b['model_b'] in TARGET_MODELS
    ]
    
    if target_battles:
        target_file = DATA_DIR / "routellm_battles_target_models.jsonl"
        with open(target_file, 'w') as f:
            for battle in target_battles:
                f.write(json.dumps(battle) + '\n')
        
        print(f"\n   ✅ Saved {len(target_battles):,} battles involving target models to:")
        print(f"      {target_file}")
    
    print(f"\n{'='*80}")
    print("✅ Download Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    download_and_extract()

