#!/usr/bin/env python3
"""
Calculate empirical success rates for HLE scores using proper train/test splits.
Uses ExperimentBurnIn to ensure no data leakage.
"""
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
    print("=" * 70)
    print("CALCULATING EMPIRICAL SUCCESS RATES FROM DEV SET")
    print("=" * 70)
    
    # 1. Load data
    full_registry = load_model_registry()
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    splits_path = PROJECT_ROOT / "experiments" / "01_effectiveness" / "results" / "splits.json"
    
    # 2. Initialize ExperimentBurnIn
    burner = ExperimentBurnIn(full_registry, all_rewards, splits_path)
    
    # 3. Get proper splits
    dev_prompts, test_prompts = burner.get_splits()
    print(f"\n✓ Loaded splits:")
    print(f"  Dev (training):   {len(dev_prompts)} prompts")
    print(f"  Holdout (test):   {len(test_prompts)} prompts")
    
    # 4. Calculate success rates from DEV set only
    print(f"\n📊 Calculating success rates from DEV set (avoiding data leakage)...")
    
    dev_success_rates = {}
    test_success_rates = {}
    
    # Our 9 models
    models = [
        'mistralai/ministral-3b',
        'google/gemma-3-4b-it',
        'google/gemma-3-12b-it',
        'openai/gpt-oss-20b',
        'google/gemma-3-27b-it',
        'x-ai/grok-3-mini',
        'google/gemini-2.5-flash-preview',
        'anthropic/claude-opus-4.5',
        'openai/gpt-4.1'
    ]
    
    for model_id in models:
        # DEV set
        dev_rewards = []
        for prompt in dev_prompts:
            if prompt in all_rewards and model_id in all_rewards[prompt]:
                dev_rewards.append(all_rewards[prompt][model_id])
        
        # TEST set (for comparison)
        test_rewards_list = []
        for prompt in test_prompts:
            if prompt in all_rewards and model_id in all_rewards[prompt]:
                test_rewards_list.append(all_rewards[prompt][model_id])
        
        if dev_rewards:
            dev_success_rates[model_id] = np.mean(dev_rewards)
        
        if test_rewards_list:
            test_success_rates[model_id] = np.mean(test_rewards_list)
    
    # 5. Display results
    print("\n" + "=" * 70)
    print("EMPIRICAL SUCCESS RATES")
    print("=" * 70)
    print(f"{'Model ID':<45} | {'Dev (Train)':<12} | {'Holdout (Test)':<12} | {'Δ':<8}")
    print("-" * 70)
    
    for model_id in models:
        dev_sr = dev_success_rates.get(model_id, 0)
        test_sr = test_success_rates.get(model_id, 0)
        delta = test_sr - dev_sr if (dev_sr and test_sr) else 0
        
        if dev_sr:
            print(f"{model_id:<45} | {dev_sr:<12.4f} | {test_sr:<12.4f} | {delta:+.4f}")
        else:
            print(f"{model_id:<45} | {'NO DATA':<12} | {'NO DATA':<12} | {'N/A':<8}")
    
    print("=" * 70)
    
    # 6. Generate update script
    print("\n💾 Generating HLE update script...")
    
    update_code = """import json
from pathlib import Path

# Success rates from DEV set (no data leakage)
empirical_hle = {
"""
    
    for model_id in models:
        if model_id in dev_success_rates:
            # Clip to [0.01, 0.99] as requested
            clipped = np.clip(dev_success_rates[model_id], 0.01, 0.99)
            update_code += f"    '{model_id}': {clipped:.4f},\n"
    
    update_code += """}

# Update models.json
for path in ['src/bandit_gpt/config/models.json', 'src/bandit_gpt/config/models_full.json']:
    with open(path) as f:
        data = json.load(f)
    
    for model in data['models']:
        model_id = model['openrouter_id']
        if model_id in empirical_hle:
            old_hle = model.get('hle')
            model['hle'] = empirical_hle[model_id]
            print(f"Updated {model_id}: {old_hle} → {empirical_hle[model_id]}")
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated {path}")
"""
    
    with open(PROJECT_ROOT / "update_hle_from_dev.py", "w") as f:
        f.write(update_code)
    
    print("✅ Created: update_hle_from_dev.py")
    print("\nRun: python update_hle_from_dev.py")

if __name__ == "__main__":
    main()
