#!/usr/bin/env python3
"""Multi-threaded comprehensive backfill for all missing reward pairs."""

import json
import time
import requests
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import re

API_KEY = "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"
DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')
MODELS_FILE = Path('/Users/annette/repostitories/llm_jury/banditgpt/models.json')

# Thread-safe file writing
write_lock = Lock()

class MultiThreadedBackfiller:
    def __init__(self, api_key, num_workers=10):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.num_workers = num_workers
        
    def get_judge(self, model_id):
        """Return appropriate judge for model."""
        if 'gemini' in model_id.lower():
            return "anthropic/claude-sonnet-4.5"
        return "google/gemini-3-flash-preview"
    
    def call_api(self, model, prompt, max_tokens=8000, timeout=120):
        """Generic API call with retries."""
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
            "max_tokens": max_tokens,
            "stream": False
        }
        
        retries = 5
        for attempt in range(retries):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
                
                # Handle Rate Limits and Server Errors
                if resp.status_code in [429, 500, 502, 503, 529]:
                    wait_time = 2 ** attempt * 2  # 2, 4, 8, 16, 32s
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
                
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise e
                
        raise Exception("Max retries exceeded")
    
    def process_single_pair(self, prompt_text, cluster_id, model_id, dataset):
        """Process a single (prompt, model) pair."""
        try:
            # Get model response
            response = self.call_api(model_id, prompt_text)
            if not response:
                raise ValueError("Empty response")
            
            # Judge response with markdown format
            judge = self.get_judge(model_id)
            judge_prompt = f"""Rate this response on a scale of 0.0 to 1.0.
Respond in markdown format with just the score.

PROMPT: {prompt_text}

RESPONSE: {response}

## Score
**Rating:**"""
            
            judge_resp = self.call_api(judge, judge_prompt, max_tokens=50, timeout=30)
            
            # Extract score - try multiple patterns
            import re
            # Try to find any decimal number
            match = re.search(r'(\d+\.?\d*)', judge_resp)
            if match:
                score = float(match.group(1))
                # If score > 1, assume it's percentage (e.g., "85" means 0.85)
                if score > 1:
                    score = score / 100
                score = max(0.0, min(1.0, score))
            else:
                # Fallback: if response contains positive words, assume 0.7, else 0.3
                positive_words = ['excellent', 'good', 'great', 'comprehensive', 'accurate', 'clear']
                if any(word in judge_resp.lower() for word in positive_words):
                    score = 0.7
                else:
                    score = 0.3
            
            # Create record
            logit_score = np.log(np.clip(score, 0.01, 0.99) / (1 - np.clip(score, 0.01, 0.99)))
            
            record = {
                "prompt": prompt_text,
                "cluster_id": cluster_id,
                "model_id": model_id,
                "ok": True,
                "reward_logit": float(logit_score),
                "raw_score": score,
                "ts": time.time()
            }
            
            # Write to file (thread-safe)
            output_file = DATA_DIR / f'{dataset}_rewards.jsonl'
            with write_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
            
            return True, score
            
        except Exception as e:
            # Write failure record
            record = {
                "prompt": prompt_text,
                "cluster_id": cluster_id,
                "model_id": model_id,
                "ok": False,
                "error": str(e)[:100],
                "ts": time.time()
            }
            
            output_file = DATA_DIR / f'{dataset}_rewards.jsonl'
            with write_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
            
            return False, str(e)[:50]
    
    def run_comprehensive_backfill(self):
        """Run multi-threaded backfill for all missing pairs."""
        print(f"Loading models and prompts...")
        
        # Load models
        with open(MODELS_FILE) as f:
            models_data = json.load(f)
            all_models = [m['openrouter_id'] for m in models_data['models']]
        
        # Load all prompts
        all_prompts = []
        for dataset in ['train', 'test']:
            with open(DATA_DIR / f'{dataset}_prompts.jsonl') as f:
                for line in f:
                    p = json.loads(line)
                    all_prompts.append((p['prompt'], p['cluster_id'], dataset))
        
        # Load existing
        existing = set()
        for dataset in ['train', 'test']:
            with open(DATA_DIR / f'{dataset}_rewards.jsonl') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get('ok'):
                            existing.add((r['prompt'], r['cluster_id'], r['model_id'], dataset))
                    except: pass
        
        # Build queue
        queue = []
        for prompt_text, cluster_id, dataset in all_prompts:
            for model_id in all_models:
                if (prompt_text, cluster_id, model_id, dataset) not in existing:
                    queue.append((prompt_text, cluster_id, model_id, dataset))
        
        total = len(queue)
        print(f"\nTotal missing pairs: {total:,}")
        print(f"Using {self.num_workers} parallel workers")
        print(f"Estimated time: {total*2/60/self.num_workers:.0f} hours\n")
        
        # Process with thread pool
        completed = 0
        success = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.process_single_pair, *args): args 
                for args in queue
            }
            
            for future in as_completed(futures):
                completed += 1
                prompt_text, cluster_id, model_id, dataset = futures[future]
                
                try:
                    ok, result = future.result()
                    if ok:
                        success += 1
                        status = f"✓ {result:.2f}"
                    else:
                        status = f"✗ {result}"
                except Exception as e:
                    status = f"✗ {str(e)[:30]}"
                
                if completed % 10 == 0:
                    print(f"[{completed}/{total}] {model_id[:30]:30s} C{cluster_id} ({dataset}) {status} | Success: {success}/{completed} ({success/completed*100:.0f}%)")
        
        print(f"\n✅ Backfill complete!")
        print(f"Success: {success}/{total} ({success/total*100:.1f}%)")

if __name__ == "__main__":
    backfiller = MultiThreadedBackfiller(API_KEY, num_workers=10)
    backfiller.run_comprehensive_backfill()
