#!/usr/bin/env python3
"""
Generate oracle rewards for Mixtral and GPT-4o only on train and test datasets.
"""
import json
from pathlib import Path
from experiments.rejudge_cot import CoTRewardGenerator

# Target models
TARGET_MODELS = [
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-4o"
]

root = Path(__file__).parent.parent
gen = CoTRewardGenerator(max_workers=20)

# Process both train and test
datasets = [
    {
        "name": "train",
        "input": root / "experiments/01_effectiveness/data/budget_train_800.jsonl",
        "output": root / "data/mixtral_gpt4o_train_rewards.jsonl"
    },
    {
        "name": "test",
        "input": root / "experiments/01_effectiveness/data/budget_test_800.jsonl",
        "output": root / "data/mixtral_gpt4o_test_rewards.jsonl"
    }
]

# Load cache
cache_file = root / "data/test_rewards_cache.jsonl"
gen.load_cache(cache_file)

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Processing {dataset['name'].upper()} dataset")
    print(f"{'='*60}")
    
    # Load prompts
    prompts = []
    with open(dataset["input"]) as f:
        for line in f:
            data = json.loads(line)
            prompts.append({"prompt": data["prompt"]})
    
    print(f"Loaded {len(prompts)} prompts")
    print(f"Processing {len(prompts)} prompts x {len(TARGET_MODELS)} models = {len(prompts)*len(TARGET_MODELS)} tasks")
    
    # Create tasks
    tasks = []
    for p in prompts:
        for model_id in TARGET_MODELS:
            tasks.append((p["prompt"], model_id))
    
    # Ensure output directory exists
    dataset["output"].parent.mkdir(parents=True, exist_ok=True)
    if not dataset["output"].exists():
        with open(dataset["output"], 'w') as f: pass
    
    # Run parallel generation
    with ThreadPoolExecutor(max_workers=gen.max_workers) as executor:
        futures = {executor.submit(gen.process_task, t): t for t in tasks}
        
        with tqdm(total=len(tasks), desc=f"Generating {dataset['name']} rewards") as pbar:
            for f in as_completed(futures):
                res = f.result()
                with open(dataset["output"], 'a') as outfile:
                    outfile.write(json.dumps(res) + "\n")
                pbar.update(1)
    
    print(f"✅ Saved {dataset['name']} rewards to: {dataset['output']}")

print(f"\n{'='*60}")
print("✅ All datasets processed successfully!")
print(f"{'='*60}")
