import requests
import json
import time
import numpy as np
from pathlib import Path
from threading import Lock

API_KEY = "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"
DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')
TARGET_MODELS = ["openai/gpt-5", "openai/gpt-5.1", "openai/o3"]

def process_failures():
    # 1. Identify failures
    failures = []
    seen_prompts = set()
    
    for dataset in ['train', 'test']:
        filename = DATA_DIR / f'{dataset}_rewards.jsonl'
        if not filename.exists(): continue
        
        with open(filename) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r['model_id'] in TARGET_MODELS and not r['ok']:
                        key = (r['model_id'], r['prompt'])
                        if key not in seen_prompts:
                            failures.append((r, dataset))
                            seen_prompts.add(key)
                except: pass
            
    print(f"Found {len(failures)} unique failures for OpenAI targets.")
    
    # Save to separate JSON for analysis
    refusal_encounters = []
    for fail, dataset in failures:
        refusal_encounters.append({
            "prompt": fail['prompt'],
            "cluster_id": fail['cluster_id'],
            "model_id": fail['model_id'],
            "dataset": dataset,
            "reason": "Safety Refusal (Empty Response)"
        })
    
    with open(DATA_DIR / 'safety_refusal_prompts_openai.json', 'w') as f:
        json.dump(refusal_encounters, f, indent=2)
    print(f"Saved {len(refusal_encounters)} prompts to safety_refusal_prompts_openai.json")

    # 2. Process each as FORCED REFUSAL
    for fail, dataset in failures:
        prompt = fail['prompt']
        cluster = fail['cluster_id']
        model_id = fail['model_id']
        # print(f"Resolving as Refusal (0.0): {model_id} - {prompt[:30]}...")
        
        # Force 0.0 score for safety refusal
        score = 0.0
        logit = -5.0 # Low logit for 0.0
        
        record = {
            "prompt": prompt,
            "cluster_id": cluster,
            "model_id": model_id,
            "ok": True,
            "reward_logit": float(logit),
            "raw_score": score,
            "ts": time.time(),
            "note": "Forced Safety Refusal"
        }
        
        with open(DATA_DIR / f'{dataset}_rewards.jsonl', 'a') as f:
            f.write(json.dumps(record) + "\n")
            
    print("✅ All OpenAI refusals resolved.")

if __name__ == "__main__":
    process_failures()
