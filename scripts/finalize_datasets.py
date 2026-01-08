#!/usr/bin/env python3
import json
import gzip
from pathlib import Path
from tqdm import tqdm

def load_prompt_set(path):
    prompts = set()
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.add(data['prompt'])
    return prompts

def main():
    data_dir = Path('data')
    offline_dir = Path('src/bandit_gpt/data/offline_dataset')
    
    test_dist_path = data_dir / 'lmsys_test_distribution_1k.jsonl'
    train_dist_path = data_dir / 'lmsys_train_distribution_1k.jsonl'
    new_rewards_path = data_dir / 'lmsys_new_rewards_888.jsonl'
    
    test_reused_source = offline_dir / 'test_rewards_hle_models.jsonl.gz'
    train_reused_source = offline_dir / 'train_rewards_hle_models.jsonl.gz'

    print("🔍 Loading distribution sets...")
    test_prompts = load_prompt_set(test_dist_path)
    train_prompts = load_prompt_set(train_dist_path)
    all_needed = test_prompts.union(train_prompts)
    
    print(f"  - Test: {len(test_prompts)} prompts")
    print(f"  - Train: {len(train_prompts)} prompts")

    # rewards[prompt] = list of reward records
    rewards = {}

    print("\n📦 Loading NEW rewards (888 prompts)...")
    with open(new_rewards_path, 'r') as f:
        for line in tqdm(f, desc="New rewards"):
            try:
                data = json.loads(line)
                p = data['prompt']
                if p in all_needed:
                    if p not in rewards: rewards[p] = []
                    rewards[p].append(data)
            except: continue

    print("\n📂 Loading REUSED rewards from TEST source...")
    with gzip.open(test_reused_source, 'rt') as f:
        for line in tqdm(f, desc="Test source"):
            try:
                data = json.loads(line)
                p = data['prompt']
                if p in all_needed:
                    if p not in rewards: rewards[p] = []
                    # Avoid duplicates if any
                    rewards[p].append(data)
            except: continue

    print("\n📂 Loading REUSED rewards from TRAIN source...")
    with gzip.open(train_reused_source, 'rt') as f:
        for line in tqdm(f, desc="Train source"):
            try:
                data = json.loads(line)
                p = data['prompt']
                if p in all_needed:
                    if p not in rewards: rewards[p] = []
                    rewards[p].append(data)
            except: continue

    # Deduplicate rewards per prompt (by model_id) to be safe
    for p in rewards:
        unique_models = {}
        for rec in rewards[p]:
            unique_models[rec['model_id']] = rec
        rewards[p] = list(unique_models.values())

    print("\n💾 Saving merged datasets...")
    for label, prompt_set in [('test', test_prompts), ('train', train_prompts)]:
        output_path = data_dir / f'lmsys_{label}_final_rewards_1k.jsonl.gz'
        count = 0
        with gzip.open(output_path, 'wt') as f:
            for p in prompt_set:
                if p in rewards:
                    for rec in rewards[p]:
                        f.write(json.dumps(rec) + '\n')
                    count += 1
                else:
                    print(f"⚠️ Warning: No rewards found for prompt in {label} set: {p[:50]}...")
        
        print(f"✅ Saved rewards for {count}/1000 prompts to {output_path}")

    print("\n🎉 Final consolidation complete!")

if __name__ == "__main__":
    main()
