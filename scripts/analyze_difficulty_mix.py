#!/ death/py3
import sys
import numpy as np
from pathlib import Path
from collections import Counter

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils.data_loader import load_oracle_rewards

def classify_prompt_difficulty(rewards_dict):
    """
    rewards_dict: { 'gpt-4': 1.0, 'llama-2': 0.0, ... }
    """
    if not rewards_dict:
        return "NOISE"
        
    scores = list(rewards_dict.values())
    mean_score = np.mean(scores)
    variance = np.var(scores)
    
    if variance > 0.10:
        return "CONTENTIOUS"  # The Gold Mine for BanditGPT
    elif mean_score > 0.85:
        return "COMMODITY"    # Easy (Use for Cost Savings)
    elif mean_score < 0.50:
        return "FRONTIER"     # Hard (Need SOTA)
    else:
        return "NOISE"

def analyze_set(name, rewards):
    print(f"\nAnalyzing {name} set ({len(rewards)} prompts)...")
    difficulties = [classify_prompt_difficulty(r) for r in rewards.values()]
    counts = Counter(difficulties)
    total = len(difficulties)
    
    for category in ["CONTENTIOUS", "COMMODITY", "FRONTIER", "NOISE"]:
        count = counts.get(category, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {category:12s}: {count:4d} ({percentage:5.1f}%)")
    
    return counts

def main():
    print("=" * 60)
    print("DIFFICULTY MIX ANALYSIS")
    print("=" * 60)
    
    try:
        train_rewards = load_oracle_rewards("train_rewards_hle_models.jsonl")
        test_rewards = load_oracle_rewards("test_rewards_hle_models.jsonl")
        
        analyze_set("Training", train_rewards)
        analyze_set("Test", test_rewards)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
