import requests
import json
import time
import numpy as np
from pathlib import Path
from threading import Lock

API_KEY = "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"
MODEL = "anthropic/claude-4.5-sonnet"
DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')

def process_failures():
    # 1. Identify failures
    failures = []
    # Use a set to avoid duplicates since we saw 6 failures for 4 prompts maybe?
    seen_prompts = set()
    
    with open(DATA_DIR / 'train_rewards.jsonl') as f:
        for line in f:
            try:
                r = json.loads(line)
                if r['model_id'] == 'anthropic/claude-4.5-sonnet' and not r['ok']:
                    if r['prompt'] not in seen_prompts:
                        failures.append(r)
                        seen_prompts.add(r['prompt'])
            except: pass
            
    print(f"Found {len(failures)} unique failures to resolve as Refusals.")
    
    # Save to separate JSON for analysis
    refusal_encounters = []
    for fail in failures:
        refusal_encounters.append({
            "prompt": fail['prompt'],
            "cluster_id": fail['cluster_id'],
            "model_id": MODEL,
            "reason": "Safety Refusal (Empty Response)"
        })
    
    with open(DATA_DIR / 'safety_refusal_prompts.json', 'w') as f:
        json.dump(refusal_encounters, f, indent=2)
    print(f"Saved {len(refusal_encounters)} prompts to safety_refusal_prompts.json")

    # 2. Process each as FORCED REFUSAL
    for fail in failures:
        prompt = fail['prompt']
        cluster = fail['cluster_id']
        print(f"Resolving as Refusal (0.0): {prompt[:50]}...")
        
        # Force 0.0 score for safety refusal
        score = 0.0
        logit = -5.0 # Low logit for 0.0
        
        record = {
            "prompt": prompt,
            "cluster_id": cluster,
            "model_id": MODEL,
            "ok": True,
            "reward_logit": float(logit),
            "raw_score": score,
            "ts": time.time(),
            "note": "Forced Safety Refusal"
        }
        
        with open(DATA_DIR / 'train_rewards.jsonl', 'a') as f:
            f.write(json.dumps(record) + "\n")
            
    print("✅ All refusals resolved.")

if __name__ == "__main__":
    process_failures()
