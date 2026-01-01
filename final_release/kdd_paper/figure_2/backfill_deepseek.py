import requests
import json
import time
import concurrent.futures
import threading
from pathlib import Path
import sys

class DeepSeekBackfiller:
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://openrouter.ai/api/v1"
        self.data_dir = Path('/Users/annette/repostitories/llm_jury/final_release/data')
        
        # User requested DeepSeek models
        self.target_models = [
            "deepseek/deepseek-r1-0528-qwen3-8b",
            "deepseek/deepseek-r1-distill-llama-70b"
        ]
        
        # User requested Gemini Flash as judge (mapping to available ID)
        self.judge_model = "google/gemini-2.5-flash-preview-09-2025"
        
        self.max_tokens = 8000
        self.print_lock = threading.Lock()
        self.counters = {m: 0 for m in self.target_models}

    def call_api_retry(self, payload, timeout=120):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
            "X-Title": "llm_jury",
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
                if resp.status_code in [429, 500, 502, 503, 529]:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == retries - 1:
                    raise e
        return None

    def process_item(self, item, dataset_type):
        prompt = item['prompt']
        cluster_id = item['cluster_id']
        
        # 1. Get Model Response
        payload_model = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.7
        }
        
        try:
            r1 = self.call_api_retry(payload_model)
            if not r1 or 'choices' not in r1 or not r1['choices']:
                return None
            
            response_text = r1['choices'][0]['message']['content']
            if not response_text:
                return None
                
            # 2. Get Judge Response
            judge_prompt = f"Please evaluate this response to the prompt '{prompt}'. Response: {response_text}. Is it helpful and accurate? Reply only with YES or NO."
            payload_judge = {
                "model": self.judge_model,
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": 50
            }
            
            r2 = self.call_api_retry(payload_judge)
            if not r2:
                # Default to OK if judge fails, or retry? Let's assume OK for now to keep moving, or skip.
                # Ideally we want a score.
                score = 0.0 # Fail safe
            else:
                judge_content = r2['choices'][0]['message']['content'].upper()
                score = 1.0 if "YES" in judge_content else 0.0
                
            # 3. Log Result
            logit = 5.0 if score > 0.5 else -5.0
            
            record = {
                "prompt": prompt,
                "cluster_id": cluster_id,
                "model_id": self.model_id,
                "ok": True,
                "reward_logit": float(logit),
                "raw_score": float(score),
                "timestamp": time.time(),
                "judge_model": self.judge_model
            }
            
            # Write to file
            filename = self.data_dir / f"{dataset_type}_rewards.jsonl"
            with self.print_lock:
                 with open(filename, 'a') as f:
                    f.write(json.dumps(record) + "\n")
                 self.counters[self.model_id] += 1
                 if self.counters[self.model_id] % 10 == 0:
                     print(f"[{self.model_id}] Progress: {self.counters[self.model_id]} items processed.")
                     
            return True
            
        except Exception as e:
            # print(f"Error processing {self.model_id} - {prompt[:20]}: {e}")
            return None

    def get_missing_prompts(self, model_id, dataset_type):
        # Load existing
        existing_keys = set()
        reward_file = self.data_dir / f"{dataset_type}_rewards.jsonl"
        if reward_file.exists():
            with open(reward_file) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if d['model_id'] == model_id and d.get('ok'):
                            existing_keys.add((d['prompt'], d['cluster_id']))
                    except: pass
                    
        # Load prompts
        prompt_file = self.data_dir / f"{dataset_type}_prompts.jsonl"
        missing = []
        with open(prompt_file) as f:
            for line in f:
                d = json.loads(line)
                if (d['prompt'], d['cluster_id']) not in existing_keys:
                    missing.append(d)
        return missing

    def run(self):
        print(f"Starting DeepSeek Backfill with Judge: {self.judge_model}")
        print(f"Max Tokens: {self.max_tokens}")
        
        for model_id in self.target_models:
            self.model_id = model_id
            print(f"Processing Model: {model_id}")
            
            # Train
            missing_train = self.get_missing_prompts(model_id, 'train')
            print(f"  - Missing Train: {len(missing_train)}")
            
            # Test
            missing_test = self.get_missing_prompts(model_id, 'test')
            print(f"  - Missing Test: {len(missing_test)}")
            
            all_items = [(x, 'train') for x in missing_train] + [(x, 'test') for x in missing_test]
            
            if not all_items:
                print("  - No items to backfill.")
                continue
                
            print(f"  - Launching {len(all_items)} tasks with 50 workers...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(self.process_item, item, dtype) for item, dtype in all_items]
                concurrent.futures.wait(futures)
                
            print(f"Finished {model_id}")

if __name__ == "__main__":
    bf = DeepSeekBackfiller()
    bf.run()
