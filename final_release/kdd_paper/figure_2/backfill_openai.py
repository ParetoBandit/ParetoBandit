#!/usr/bin/env python3
"""Backfill for O3 and GPT-5.1 with Gemini-3-Flash as judge."""

import json
import os
import time
import requests
import numpy as np
from pathlib import Path
import sys

API_KEY = "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"
DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')

class OpenAIBackfiller:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.judge = "google/gemini-3-flash-preview"
        
    def get_response(self, model, prompt):
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
            "max_tokens": 8000,  # Increased for O3 reasoning model
            "stream": False
        }
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        if "choices" not in data:
            raise ValueError(f"API response missing 'choices'")
        
        content = data["choices"][0]["message"].get("content", "")
        if not content:
            raise ValueError(f"Model returned empty content")
        
        return content
    
    def judge_response(self, prompt, response):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        payload = {
            "model": self.judge,
            "messages": [
                {"role": "system", "content": "Rate 0.0-1.0 only. No other text."},
                {"role": "user", "content": f"PROMPT: {prompt}\\n\\nRESPONSE: {response}"}
            ],
            "temperature": 0.0,
            "max_tokens": 10
        }
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30)
        data = resp.json()
        
        if "choices" not in data:
            raise ValueError(f"Judge missing 'choices'")
        
        content = data["choices"][0]["message"]["content"].strip()
        
        import re
        match = re.search(r"(\d+(\.\d+)?)", content)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        raise ValueError(f"Judge invalid score: '{content}'")
    
    def logit(self, score):
        score = np.clip(score, 0.01, 0.99)
        return np.log(score / (1 - score))

def backfill_model(model_id, backfiller):
    print(f"\\n{'='*70}")
    print(f"MODEL: {model_id}")
    print(f"{'='*70}")
    
    prompts = []
    for dataset in ['train', 'test']:
        with open(DATA_DIR / f'{dataset}_prompts.jsonl') as f:
            for line in f:
                p = json.loads(line)
                if p['cluster_id'] in [36, 80]:
                    prompts.append((p, dataset))
    
    existing = set()
    for dataset in ['train', 'test']:
        with open(DATA_DIR / f'{dataset}_rewards.jsonl') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get('ok') and r['model_id'] == model_id and r['cluster_id'] in [36, 80]:
                        existing.add((r['prompt'], r['cluster_id']))
                except: pass
    
    missing = [(p, ds) for (p, ds) in prompts if (p['prompt'], p['cluster_id']) not in existing]
    
    print(f"Existing: {len(existing)}, Missing: {len(missing)}")
    
    if not missing:
        print("✅ COMPLETE")
        return True
    
    success_count = 0
    for i, (p_data, dataset) in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] C{p_data['cluster_id']} ({dataset})...", end=" ", flush=True)
        
        try:
            response = backfiller.get_response(model_id, p_data['prompt'])
            score = backfiller.judge_response(p_data['prompt'], response)
            
            record = {
                "prompt": p_data['prompt'],
                "cluster_id": p_data['cluster_id'],
                "model_id": model_id,
                "ok": True,
                "reward_logit": backfiller.logit(score),
                "raw_score": score,
                "ts": time.time()
            }
            
            output_file = DATA_DIR / f'{dataset}_rewards.jsonl'
            with open(output_file, 'a') as f:
                f.write(json.dumps(record) + "\\n")
                f.flush()
            
            print(f"✓ {score:.2f}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ {str(e)[:50]}")
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
                f.write(json.dumps(record) + "\\n")
                f.flush()
        
        time.sleep(0.5)
    
    print(f"\\nCompleted: {success_count}/{len(missing)}")
    return success_count == len(missing)

if __name__ == "__main__":
    backfiller = OpenAIBackfiller(API_KEY)
    
    # Get model from command line
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/o3"
    
    try:
        complete = backfill_model(MODEL, backfiller)
        if complete:
            print(f"\\n✅ {MODEL} complete!")
        else:
            print(f"\\n⚠️  {MODEL} partial")
    except Exception as e:
        print(f"\\n❌ ERROR: {e}")
        raise
