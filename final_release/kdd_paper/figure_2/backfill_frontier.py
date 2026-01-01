#!/usr/bin/env python3
"""Targeted backfill for frontier models only (C36 & C80)."""

import json
import os
import time
import requests
import numpy as np
from pathlib import Path

# Hardcoded Key
API_KEY = "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"

DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')

# FRONTIER MODELS ONLY
FRONTIER_MODELS = [
    'anthropic/claude-4.5-sonnet',
    'openai/gpt-5',
    'google/gemini-3-pro-preview',
    'openai/o3',
    'openai/gpt-5.1'
]

class RewardBackfiller:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_judge = "google/gemini-2.0-flash-exp"
        self.judge_max_tokens = 50
        
    def get_model_response(self, model_id, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
            "X-Title": "llm_jury",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4096
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers, 
                json=payload, 
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error: {e}", flush=True)
            return None

    def judge_response(self, prompt, response):
        system_prompt = (
            "You are an impartial judge. Rate the quality of the response.\\n"
            "Output ONLY a single float between 0.0 and 1.0.\\n"
            "0.0 = Wrong/harmful, 0.5 = Partial, 1.0 = Perfect.\\n"
            "No other text."
        )
        user_content = f"PROMPT: {prompt}\\n\\nRESPONSE: {response}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        payload = {
            "model": self.default_judge,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "max_tokens": self.judge_max_tokens
        }
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            import re
            match = re.search(r"(\\d+(\\.\\d+)?)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except:
            return 0.5

    def logit_transform(self, score):
        score = np.clip(score, 0.01, 0.99)
        return np.log(score / (1 - score))

def main():
    backfiller = RewardBackfiller(API_KEY)
    
    print("="*60)
    print("TARGETED BACKFILL: Frontier Models for C36 & C80")
    print("="*60)
    
    # Process both train and test
    for dataset_type in ['train', 'test']:
        prompts_file = DATA_DIR / f'{dataset_type}_prompts.jsonl'
        rewards_file = DATA_DIR / f'{dataset_type}_rewards.jsonl'
        
        print(f"\\n[{dataset_type.upper()}]")
        
        # Load C36/C80 prompts
        prompts_data = []
        with open(prompts_file) as f:
            for l in f:
                p = json.loads(l)
                if p['cluster_id'] in [36, 80]:
                    prompts_data.append(p)
        
        # Load existing
        existing_pairs = set()
        with open(rewards_file) as f:
            for l in f:
                try:
                    r = json.loads(l)
                    if r.get('ok') and r['cluster_id'] in [36, 80]:
                        existing_pairs.add((r['prompt'], r['model_id']))
                except: pass
        
        # Build queue for FRONTIER models only
        queue = []
        for p in prompts_data:
            for m in FRONTIER_MODELS:
                if (p['prompt'], m) not in existing_pairs:
                    queue.append((p, m, rewards_file))
        
        print(f"  Missing: {len(queue)} pairs")
        if not queue:
            print(f"  Complete!")
            continue
        
        # Process
        for i, (p_data, model_id, output_file) in enumerate(queue):
            print(f"  [{i+1}/{len(queue)}] {model_id[:30]:30s} C{p_data['cluster_id']}...", end=" ", flush=True)
            
            response = backfiller.get_model_response(model_id, p_data['prompt'])
            
            record = {
                "prompt": p_data['prompt'],
                "cluster_id": p_data['cluster_id'],
                "model_id": model_id,
                "ts": time.time()
            }
            
            if response:
                score = backfiller.judge_response(p_data['prompt'], response)
                record.update({
                    "ok": True,
                    "teacher_used": True,
                    "teacher_model": backfiller.default_judge,
                    "reward_logit": backfiller.logit_transform(score),
                    "raw_score": score,
                    "response": response[:100] + "..."
                })
                print(f"✓ {score:.2f}", flush=True)
            else:
                record.update({
                    "ok": False,
                    "error": "API Error",
                    "reward_logit": 0.0
                })
                print(f"✗", flush=True)
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(record) + "\\n")
                f.flush()
            
            time.sleep(0.5)
    
    print("\\n✅ Frontier model backfill complete!")

if __name__ == "__main__":
    main()
