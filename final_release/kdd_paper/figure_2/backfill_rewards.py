
import json
import os
import time
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Hardcoded Key provided by user
API_KEY = "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"

DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')
MODELS_FILE = Path('/Users/annette/repostitories/llm_jury/banditgpt/models.json')
PROMPTS_FILE = DATA_DIR / 'test_prompts.jsonl'
REWARDS_FILE = DATA_DIR / 'test_rewards.jsonl'

class RewardBackfiller:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_judge = "google/gemini-3-flash-preview"
        self.gemini_judge = "anthropic/claude-4.5-sonnet"
        self.judge_max_tokens = 50
        
    def get_model_response(self, model_id, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
            "X-Title": "llm_jury", # App Name provided by user
        }
        # INCREASED max_tokens to 8000 to prevent truncation errors
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 8000 
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers, 
                json=payload, 
                timeout=60 # Increased timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            # print(f"Error getting response from {model_id}: {e}")
            return None

    def judge_response(self, prompt, response, model_being_judged):
        if "gemini" in model_being_judged.lower():
            judge_model = self.gemini_judge
        else:
            judge_model = self.default_judge
            
        system_prompt = (
            "You are an impartial judge. Rate the quality of the response to the prompt.\n"
            "Output ONLY a single float number between 0.0 and 1.0.\n"
            "0.0 = Completely wrong, harmful, or unhelpful.\n"
            "0.5 = Partially correct but missing key details.\n"
            "1.0 = Perfectly correct, helpful, and comprehensive.\n"
            "Do not output any other text."
        )
        user_content = f"PROMPT: {prompt}\n\nRESPONSE: {response}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        payload = {
            "model": judge_model,
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
            match = re.search(r"(\d+(\.\d+)?)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception as e:
            print(f"Error judging: {e}")
            return 0.5

    def logit_transform(self, score):
        score = np.clip(score, 0.01, 0.99)
        return np.log(score / (1 - score))

def main():
    backfiller = RewardBackfiller(API_KEY)
    
    # 1. Load Registry Models
    with open(MODELS_FILE) as f:
        registry_models = [m['openrouter_id'] for m in json.load(f)['models']]
    
    # 2. Process BOTH train and test datasets
    for dataset_type in ['train', 'test']:
        prompts_file = DATA_DIR / f'{dataset_type}_prompts.jsonl'
        rewards_file = DATA_DIR / f'{dataset_type}_rewards.jsonl'
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_type.upper()} dataset")
        print(f"{'='*60}")
        
        # Load prompts (all clusters)
        prompts = []
        with open(prompts_file) as f:
            for l in f:
                p = json.loads(l)
                prompts.append(p)
                    
        # Load Existing OK Pairs
        existing_pairs = set()
        with open(rewards_file) as f:
            for l in f:
                try:
                    r = json.loads(l)
                    if r.get('ok'):
                        existing_pairs.add((r['prompt'], r['model_id']))
                except: pass
                
        # Build Work Queue
        queue = []
        for p in prompts:
            for m in registry_models:
                if (p['prompt'], m) not in existing_pairs:
                    queue.append((p, m, rewards_file))
                    
        # Sort by cluster
        queue.sort(key=lambda x: x[0]['cluster_id'])
                    
        print(f"Found {len(queue)} missing pairs in {dataset_type} dataset.")
        if not queue:
            print(f"{dataset_type.upper()} data is complete!")
            continue

        # Process Queue for this dataset
        print(f"Backfilling {len(queue)} missing items...", flush=True)
        
        for i, (p_data, model_id, output_file) in enumerate(queue):
            print(f"[{i+1}/{len(queue)}] {model_id} on C{p_data['cluster_id']}...", end=" ", flush=True)
            
            response = backfiller.get_model_response(model_id, p_data['prompt'])
            
            record = {
                "prompt": p_data['prompt'],
                "cluster_id": p_data['cluster_id'],
                "model_id": model_id,
                "ts": time.time()
            }
            
            if response:
                score = backfiller.judge_response(p_data['prompt'], response, model_id)
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
                    "error": "Failed in Backfill",
                    "reward_logit": 0.0
                })
                print(f"✗", flush=True)
                
            # Write IMMEDIATELY
            with open(output_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
            
            time.sleep(0.5)
                
    print("\nBackfill Complete for all datasets.")

if __name__ == "__main__":
    main()
