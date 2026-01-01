#!/usr/bin/env python3
"""Transparent single-model backfill with immediate validation."""

import json
import os
import time
import requests
import numpy as np
from pathlib import Path

# Confirmed API key from user
API_KEY = "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"
DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')

class BackfillValidator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.judge = "google/gemini-3-flash-preview"
        
    def get_response(self, model, prompt):
        """Get model response with error handling and retries."""
        print(f"DEBUG: Calling API for model='{model}'", flush=True)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
            "X-Title": "llm_jury",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 8000,
            "stream": False
        }
        
        retries = 5
        for attempt in range(retries):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=180)
                
                if resp.status_code in [429, 500, 502, 503, 529]:
                    wait_time = 2 ** attempt * 2
                    print(f" [Status {resp.status_code}, retrying in {wait_time}s] ", end="", flush=True)
                    if attempt < retries - 1:
                        time.sleep(wait_time)
                        continue
                
                resp.raise_for_status()
                data = resp.json()
                
                if "choices" not in data:
                    print(f"\nERROR: Response missing 'choices'. Full response: {json.dumps(data, indent=2)}", flush=True)
                    raise ValueError(f"API response missing 'choices'. Keys: {list(data.keys())}")
                
                msg = data["choices"][0]["message"]
                content = msg.get("content", "")
                reasoning = msg.get("reasoning", "")
                
                print(f" [content:{len(content)} reasoning:{len(reasoning)}]", end="", flush=True)
                
                if not content:
                    raise ValueError(f"Model returned empty content (reasoning: {len(reasoning)} chars)")
                
                return content
                
            except Exception as e:
                if attempt < retries - 1:
                    print(f" [Error: {str(e)[:20]}..., retrying] ", end="", flush=True)
                    time.sleep(2 ** attempt)
                    continue
                raise e

    def judge_response(self, prompt, response):
        """Judge with simple 0-1 scoring."""
        system = "Rate 0.0-1.0 only. No other text."
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        payload = {
            "model": self.judge,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}"}
            ],
            "temperature": 0.0,
            "max_tokens": 10
        }
        
        # Simple retry for judge too
        for attempt in range(3):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                if "choices" not in data:
                    raise ValueError(f"Judge API response missing 'choices': {list(data.keys())}")
                
                content = data["choices"][0]["message"]["content"].strip()
                
                import re
                match = re.search(r"(\d+(\.\d+)?)", content)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                raise ValueError(f"Judge returned invalid score: '{content}' (no number found)")
            except Exception:
                if attempt < 2: time.sleep(1); continue
                raise

    def logit(self, score):
        score = np.clip(score, 0.01, 0.99)
        return np.log(score / (1 - score))


def backfill_single_model(model_id, validator):
    """Backfill ONE model and validate."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_id}")
    print(f"{'='*70}")
    
    # Get C36/C80 prompts
    prompts = []
    for dataset in ['train', 'test']:
        with open(DATA_DIR / f'{dataset}_prompts.jsonl') as f:
            for line in f:
                p = json.loads(line)
                if p['cluster_id'] in [36, 80]:
                    prompts.append((p, dataset))
    
    # Get existing
    existing = set()
    for dataset in ['train', 'test']:
        with open(DATA_DIR / f'{dataset}_rewards.jsonl') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get('ok') and r['model_id'] == model_id and r['cluster_id'] in [36, 80]:
                        existing.add((r['prompt'], r['cluster_id']))
                except: pass
    
    # Find missing
    missing = [(p, ds) for (p, ds) in prompts if (p['prompt'], p['cluster_id']) not in existing]
    
    print(f"Existing: {len(existing)}, Missing: {len(missing)}")
    
    if not missing:
        print("✅ COMPLETE - No missing data")
        return True
    
    # Process missing
    success_count = 0
    for i, (p_data, dataset) in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] C{p_data['cluster_id']} ({dataset})...", end=" ", flush=True)
        
        try:
            response = validator.get_response(model_id, p_data['prompt'])
            score = validator.judge_response(p_data['prompt'], response)
            
            record = {
                "prompt": p_data['prompt'],
                "cluster_id": p_data['cluster_id'],
                "model_id": model_id,
                "ok": True,
                "reward_logit": validator.logit(score),
                "raw_score": score,
                "ts": time.time()
            }
            
            # Write immediately
            output_file = DATA_DIR / f'{dataset}_rewards.jsonl'
            with open(output_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
            
            print(f"✓ {score:.2f}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ {str(e)[:50]}")
            # Write failure record
            record = {
                "prompt": p_data['prompt'],
                "cluster_id": p_data['cluster_id'],
                "model_id": model_id,
                "ok": False,
                "error": str(e)[:100],
                "ts": time.time()
            }
            output_file = DATA_DIR / f'{dataset}_rewards.jsonl'
            with open(output_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
        
        time.sleep(0.5)
    
    print(f"\nCompleted: {success_count}/{len(missing)} successful")
    
    if success_count == 0:
        raise RuntimeError(f"FAILED: No successful API calls for {model_id}")
    
    return success_count == len(missing)

if __name__ == "__main__":
    import sys
    validator = BackfillValidator(API_KEY)
    
    # Process ONE model from args
    if len(sys.argv) > 1:
        MODEL = sys.argv[1]
    else:
        MODEL = "openai/gpt-5"
    
    try:
        complete = backfill_single_model(MODEL, validator)
        if complete:
            print(f"\n✅ {MODEL} is now 100% complete!")
        else:
            print(f"\n⚠️  {MODEL} has partial coverage")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        raise
