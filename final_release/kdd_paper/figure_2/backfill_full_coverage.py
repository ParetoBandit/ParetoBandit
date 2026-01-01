#!/usr/bin/env python3
"""
Backfill script for Google Gemini 3 Pro Preview - cleanup pass.
Removes cluster filters to ensure 100% coverage.
Enforces anthropic/claude-sonnet-4.5 as judge.
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys

API_KEY = "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"
DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')
WRITE_LOCK = Lock()

class FullCoverageBackfiller:
    def __init__(self, target_model, api_key, num_workers=10):
        self.target_model = target_model
        # Enforce usage of claude-4.5 as requested by user
        self.judge = "anthropic/claude-sonnet-4.5" 
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.num_workers = num_workers
        
    def call_api_retry(self, payload, timeout=120):
        """Generic API call with exponential backoff."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
            "X-Title": "llm_jury",
        }
        
        retries = 5
        for attempt in range(retries):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
                
                # Handle Rate Limits and Server Errors
                if resp.status_code in [429, 500, 502, 503, 529]:
                    wait_time = 2 ** attempt * 2
                    if attempt < retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        resp.raise_for_status()
                
                resp.raise_for_status()
                data = resp.json()
                
                if "choices" not in data:
                    raise ValueError(f"Missing 'choices': {list(data.keys())}")
                
                return data["choices"][0]["message"].get("content", "")
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise e
        raise Exception("Max retries exceeded")

    def process_pair(self, prompt_text, cluster_id, dataset):
        try:
            # 1. Get Model Response
            payload = {
                "model": self.target_model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.7,
                "max_tokens": 5000, 
                "stream": False
            }
            response = self.call_api_retry(payload, timeout=180)
            if not response: raise ValueError("Empty response")
            
            # 2. Get Judge Response (Claude 4.5 Sonnet)
            judge_prompt = f"""Rate this response on a scale of 0.0 to 1.0.
Respond in markdown format with just the score.

PROMPT: {prompt_text}

RESPONSE: {response}

## Score
**Rating:**"""
            
            judge_payload = {
                "model": self.judge,
                "messages": [{"role": "user", "content": judge_prompt}],
                "temperature": 0.0,
                "max_tokens": 50,
                "stream": False
            }
            judge_resp = self.call_api_retry(judge_payload, timeout=45)
            
            # 3. Parse Score
            import re
            match = re.search(r'(\d+\.?\d*)', judge_resp)
            if match:
                score = float(match.group(1))
                if score > 1: score = score / 100
                score = max(0.0, min(1.0, score))
            else:
                # Fallback heuristics
                positive = ['excellent', 'good', 'great', 'comprehensive', 'accurate']
                score = 0.7 if any(w in judge_resp.lower() for w in positive) else 0.3
            
            # 4. Save
            logit = np.log(np.clip(score, 0.01, 0.99) / (1 - np.clip(score, 0.01, 0.99)))
            record = {
                "prompt": prompt_text,
                "cluster_id": cluster_id,
                "model_id": self.target_model,
                "ok": True,
                "reward_logit": float(logit),
                "raw_score": score,
                "ts": time.time()
            }
            
            output_file = DATA_DIR / f'{dataset}_rewards.jsonl'
            with WRITE_LOCK:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(record) + "\n")
                    
            return True, score
            
        except Exception as e:
            # Log failure
            record = {
                "prompt": prompt_text,
                "cluster_id": cluster_id,
                "model_id": self.target_model,
                "ok": False,
                "error": str(e)[:100],
                "ts": time.time()
            }
            output_file = DATA_DIR / f'{dataset}_rewards.jsonl'
            with WRITE_LOCK:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(record) + "\n")
            return False, str(e)[:50]

    def run(self):
        print(f"Starting cleanup backfill for {self.target_model}")
        print(f"Workers: {self.num_workers}")
        print(f"Judge: {self.judge}")
        
        # Load ALL prompts (removed cluster filter)
        prompts = []
        for dataset in ['train', 'test']:
            with open(DATA_DIR / f'{dataset}_prompts.jsonl') as f:
                for line in f:
                    p = json.loads(line)
                    # No cluster filtering!
                    prompts.append((p['prompt'], p['cluster_id'], dataset))
        
        # Filter existing
        existing = set()
        for dataset in ['train', 'test']:
            with open(DATA_DIR / f'{dataset}_rewards.jsonl') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get('ok') and r['model_id'] == self.target_model:
                             # Key is (prompt, cluster) - ignoring dataset for key uniqueness
                            existing.add((r['prompt'], r['cluster_id']))
                    except: pass
        
        queue = [p for p in prompts if (p[0], p[1]) not in existing]
        total = len(queue)
        print(f"Missing items: {total}")
        
        if total == 0:
            print("✅ Complete!")
            return

        success = 0
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.process_pair, *args): args for args in queue}
            
            for future in as_completed(futures):
                completed += 1
                ok, res = future.result()
                if ok:
                    success += 1
                    status = f"✓ {res:.2f}"
                else:
                    status = f"✗ {res}"
                
                if completed % 10 == 0:
                    print(f"[{completed}/{total}] {status} | Success: {success/completed*100:.0f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "google/gemini-3-pro-preview"
        
    backfiller = FullCoverageBackfiller(model, API_KEY, num_workers=50) # Updated to 50 workers
    backfiller.run()
