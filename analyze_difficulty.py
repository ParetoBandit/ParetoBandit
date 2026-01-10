
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from src.bandit_gpt.utils.experiment import ExperimentBurnIn
from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

def main():
    print("🔍 Analyzing Holdout Set Difficulty (Threshold Sensitivity)...")
    
    # 1. Load Data
    full_registry = load_model_registry()
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    splits_path = PROJECT_ROOT / "experiments" / "01_effectiveness" / "results" / "splits.json"
    
    # 2. Identify 9-Model Portfolio
    portfolio = [
        "mistralai/ministral-3b",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "openai/gpt-oss-20b",
        "google/gemma-3-27b-it",
        "x-ai/grok-3-mini",
        "google/gemini-2.5-flash-preview",
        "google/gemini-3-pro-preview",
        "openai/gpt-4.1"
    ]
    
    # 3. Initialize ExperimentBurnIn
    burner = ExperimentBurnIn(full_registry, all_rewards, splits_path)
    
    # 4. Get Holdout Set
    _, test_prompts = burner.get_splits()
    print(f"✅ Loaded {len(test_prompts)} prompts in the holdout set.")
    
    # 5. Define Thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20]
    results = {}
    
    model_a = "openai/gpt-4.1"
    model_b = "google/gemini-3-pro-preview"
    
    for thresh in thresholds:
        hard_count = 0
        rewards_a = []
        rewards_b = []
        
        for p in test_prompts:
            rewards_map = all_rewards.get(p, {})
            if not rewards_map:
                continue
                
            p_rewards = [rewards_map[m] for m in portfolio if m in rewards_map]
            if not p_rewards:
                continue
                
            variance = np.var(p_rewards)
            if variance > thresh:
                hard_count += 1
                if model_a in rewards_map:
                    rewards_a.append(rewards_map[model_a])
                if model_b in rewards_map:
                    rewards_b.append(rewards_map[model_b])
                    
        results[thresh] = {
            "count": hard_count,
            "pct": (hard_count / len(test_prompts)) * 100,
            "gpt4_sr": np.mean(rewards_a) if rewards_a else 0,
            "gemini3_sr": np.mean(rewards_b) if rewards_b else 0
        }

    print("\n" + "="*80)
    print(f"{'Threshold (Var)':<15} | {'Count':<6} | {'% Set':<8} | {'GPT-4.1 SR':<12} | {'Gemini-3 SR':<12} | {'Gap':<6}")
    print("-" * 80)
    for thresh in thresholds:
        res = results[thresh]
        gap = (res['gpt4_sr'] - res['gemini3_sr']) * 100
        print(f"{thresh:<15.2f} | {res['count']:<6} | {res['pct']:<8.1f}% | {res['gpt4_sr']:<12.2%} | {res['gemini3_sr']:<12.2%} | {gap:+.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()
