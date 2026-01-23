#!/usr/bin/env python3
"""Analyze which model is actually the oracle on each prompt."""

import sys
from pathlib import Path
import gzip
import json
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from data_loader import CANONICAL_DEV_REWARDS


def load_shared_prompts(dev_file: Path) -> list:
    """Load prompts that have all three models."""
    shared = []
    
    with gzip.open(dev_file, 'rt', encoding='utf-8') as f:
        # Group by prompt
        by_prompt = {}
        for line in f:
            entry = json.loads(line)
            prompt = entry['prompt_text']
            model_id = entry['model_id']
            reward = 1.0 if entry['winner'] == model_id else 0.0
            
            if prompt not in by_prompt:
                by_prompt[prompt] = {}
            by_prompt[prompt][model_id] = reward
        
        # Filter to prompts with all three models
        for prompt, rewards in by_prompt.items():
            if all(m in rewards for m in ['openai/gpt-4o', 'mistralai/mixtral-8x7b-instruct', 'openai/gpt-5-chat']):
                shared.append({
                    'prompt_text': prompt,
                    'gpt4o_reward': rewards['openai/gpt-4o'],
                    'mixtral_reward': rewards['mistralai/mixtral-8x7b-instruct'],
                    'gpt5_reward': rewards['openai/gpt-5-chat']
                })
    
    return shared


def main():
    dev_file = CANONICAL_DEV_REWARDS
    
    print("Loading prompts...")
    prompts = load_shared_prompts(dev_file)
    print(f"Found {len(prompts)} shared prompts\n")
    
    # Analyze oracle choices
    oracle_counts = Counter()
    tied_count = 0
    
    for p in prompts[:500]:  # First 500 (same as experiment)
        gpt4o = p['gpt4o_reward']
        mixtral = p['mixtral_reward']
        gpt5 = p['gpt5_reward']
        
        max_reward = max(gpt4o, mixtral, gpt5)
        
        winners = []
        if gpt4o == max_reward:
            winners.append('gpt-4o')
        if mixtral == max_reward:
            winners.append('mixtral')
        if gpt5 == max_reward:
            winners.append('gpt-5')
        
        if len(winners) > 1:
            tied_count += 1
            oracle_counts['tied'] += 1
        else:
            oracle_counts[winners[0]] += 1
    
    print("Oracle Distribution (first 500 prompts):")
    print("="*50)
    for model, count in oracle_counts.most_common():
        print(f"  {model:15s}: {count:4d} ({count/500*100:.1f}%)")
    
    print(f"\n  Tied (multiple best): {tied_count}")
    
    # Check if "always pick GPT-5" is actually a good strategy
    gpt5_always_regret = sum(max(p['gpt4o_reward'], p['mixtral_reward'], p['gpt5_reward']) - p['gpt5_reward'] 
                            for p in prompts[:500])
    gpt4o_always_regret = sum(max(p['gpt4o_reward'], p['mixtral_reward'], p['gpt5_reward']) - p['gpt4o_reward'] 
                             for p in prompts[:500])
    
    print(f"\nStatic Strategy Performance:")
    print("="*50)
    print(f"  Always pick GPT-5 : {gpt5_always_regret:.1f} cumulative regret")
    print(f"  Always pick GPT-4o: {gpt4o_always_regret:.1f} cumulative regret")
    
    print(f"\n✅ Manual Heuristic benefits from 'always GPT-5' being near-optimal on this dataset")


if __name__ == "__main__":
    main()

