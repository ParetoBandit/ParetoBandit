#!/usr/bin/env python3
"""
Check reward coverage for all models.
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')
MODELS_FILE = Path('/Users/annette/repostitories/llm_jury/banditgpt/models.json')

def check_coverage():
    # 1. Load Models
    with open(MODELS_FILE) as f:
        models_data = json.load(f)
        # Filter for models that look like they are part of the study (e.g. have specific IDs)
        # Or just checking all unique model_ids found in the rewards file + known targets
        registry_models = {m['openrouter_id'] for m in models_data['models']}

    # 2. Key Targets from task.md (ensure we check these even if 0 rewards)
    targets = [
        "openai/gpt-5",
        "google/gemini-3-pro-preview",
        "anthropic/claude-4.5-sonnet",
        "openai/o3",
        "openai/gpt-5.1",
        "openai/gpt-oss-20b"
    ]
    
    # 3. Load Expected Counts
    train_count = 0
    with open(DATA_DIR / 'train_prompts.jsonl') as f:
        for _ in f: train_count += 1
        
    test_count = 0
    with open(DATA_DIR / 'test_prompts.jsonl') as f:
        for _ in f: test_count += 1
        
    print(f"Expected: Train={train_count}, Test={test_count}")
    print("-" * 60)
    print(f"{'Model ID':<35} | {'Train':<10} | {'Test':<10} | {'Status'}")
    print("-" * 60)

    # 4. Count Rewards
    # Map: model_id -> {'train': set(prompts), 'test': set(prompts)}
    coverage = defaultdict(lambda: {'train': set(), 'test': set()})
    
    for dataset, expected in [('train', train_count), ('test', test_count)]:
        filename = DATA_DIR / f'{dataset}_rewards.jsonl'
        if not filename.exists(): continue
        
        with open(filename) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get('ok'):
                        coverage[r['model_id']][dataset].add(r['prompt'])
                except: pass

    # 5. Report
    # Union of registry models and found models to be safe
    all_models =  set(targets) | set(coverage.keys())
    
    # Filter to only interesting ones for clarity? 
    # Let's show all that have non-zero or are in verification list
    
    sorted_models = sorted(list(all_models))
    
    for model in sorted_models:
        # Skip if not in registry AND not in targets (likely old/renamed models)
        if model not in registry_models and model not in targets:
             continue
             
        n_train = len(coverage[model]['train'])
        n_test = len(coverage[model]['test'])
        
        status = []
        if n_train < train_count: status.append(f"Missing {train_count - n_train} Train")
        if n_test < test_count: status.append(f"Missing {test_count - n_test} Test")
        
        if not status:
            status_str = "✅ Complete"
        else:
            status_str = "❌ " + ", ".join(status)
            
        # Only print valid study models or ones with data
        if n_train > 0 or n_test > 0 or model in targets:
             print(f"{model:<35} | {n_train:<5}/{train_count} | {n_test:<5}/{test_count} | {status_str}")

if __name__ == "__main__":
    check_coverage()
