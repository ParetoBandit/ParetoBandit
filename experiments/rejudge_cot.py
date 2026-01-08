import json
import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Any
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the class structure but modify for CoT / Re-judging
class CoTRewardGenerator:
    def __init__(self, api_key: str = None, max_workers: int = 10):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            # Try loading from .env manually if not in env
             try:
                from dotenv import load_dotenv
                # banditgpt/rejudge_cot.py -> parent = banditgpt -> parent = root
                env_path = Path(__file__).parent.parent / '.env'
                if env_path.exists():
                    load_dotenv(env_path)
                self.api_key = os.getenv("OPENROUTER_API_KEY")
             except: pass
             
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Judge Pool
        self.judge_pool = {
            "openai": "openai/gpt-4o",
            "anthropic": "anthropic/claude-3.5-sonnet",
            "meta": "meta-llama/llama-3.1-405b-instruct",
            "google": "google/gemini-2.5-pro-preview-06-05"
        }
        self.family_map = {
            "gpt": "openai", "o1": "openai", "o3": "openai",
            "claude": "anthropic", "llama": "meta",
            "gemini": "google", "gemma": "google"
        }
        
        self.judge_max_tokens = 4000  # Increased for CoT reasoning
        self.max_workers = max_workers
        self.lock = threading.Lock()
        
        # Cache for existing responses: (model_id, prompt) -> response_text
        self.response_cache = {}

    def load_cache(self, cache_file: Path):
        """Load existing responses from a previous run."""
        if not cache_file.exists():
            return
        
        print(f"Loading cache from {cache_file}...")
        count = 0
        with open(cache_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("ok") and data.get("response"):
                        key = (data["model_id"], data["prompt"])
                        self.response_cache[key] = data["response"]
                        count += 1
                except:
                    continue
        print(f"Loaded {count} cached responses.")

    def get_judges_for_model(self, model_id: str) -> List[str]:
        family = None
        lower_id = model_id.lower()
        for key, val in self.family_map.items():
            if key in lower_id:
                family = val
                break
        if not family:
            if "openai/" in lower_id: family = "openai"
            elif "anthropic/" in lower_id: family = "anthropic"
            elif "google/" in lower_id: family = "google"
            elif "meta-llama/" in lower_id: family = "meta"
        
        selected = []
        for org, judge_id in self.judge_pool.items():
            if family == org: continue
            selected.append(judge_id)
        return selected

    def get_model_response(self, model_id: str, prompt: str) -> str:
        # Check cache first
        if (model_id, prompt) in self.response_cache:
            return self.response_cache[(model_id, prompt)]
            
        # Fetch fresh
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return None

    def judge_single_cot(self, judge_model: str, system_prompt: str, user_content: str) -> Tuple[int, float, str]:
        """Query a single judge for Vote, Confidence, and Reasoning."""
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
            "max_tokens": self.judge_max_tokens,
        }
        
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=40)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Robust Parsing
            import re
            
            # Extract Vote (0 or 1)
            vote = 0
            vote_match = re.search(r"## Vote\s*(\d)", content, re.IGNORECASE)
            if vote_match:
                vote = int(vote_match.group(1))
                if vote != 1: vote = 0 # Enforce binary
            
            # Extract Confidence (0-100 or 0.0-1.0)
            confidence = 0.5
            conf_match = re.search(r"## Confidence\s*(\d+(\.\d+)?)", content, re.IGNORECASE)
            if conf_match:
                val = float(conf_match.group(1))
                if val > 1.0: val = val / 100.0 # Normalize 90 -> 0.9
                confidence = max(0.0, min(1.0, val))

            # Extract Reasoning
            reasoning = content
            reason_match = re.search(r"## Reasoning\s*(.*?)(\n## Vote|$)", content, re.DOTALL | re.IGNORECASE)
            if reason_match:
                reasoning = reason_match.group(1).strip()
            
            return vote, confidence, reasoning
                
        except Exception as e:
            return None, None, f"Error: {str(e)}"

    def judge_with_panel_cot(self, prompt: str, response: str, model_id: str) -> Tuple[float, List[Dict]]:
        judges = self.get_judges_for_model(model_id)
        
        system_prompt = (
            "You are an impartial judge evaluating the quality of an LLM response.\n"
            "Please follow this Chain-of-Thought process:\n"
            "1. Provide a concise reasoning explanation.\n"
            "2. Cast a VOTE: 1 (Pass/Good) or 0 (Fail/Bad).\n"
            "3. Assign a CONFIDENCE score (0-100%) reflecting your certainty.\n\n"
            "Format your response exactly as follows:\n\n"
            "## Reasoning\n"
            "<reasoning text>\n\n"
            "## Vote\n"
            "<0 or 1>\n\n"
            "## Confidence\n"
            "<0-100>"
        )
        user_content = f"PROMPT: {prompt}\n\nRESPONSE: {response}"
        
    
        results = []
        score_1_sum = 0.0
        score_0_sum = 0.0
        
        # Parallel judge calls for speedup
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=len(judges)) as executor:
            futures = {executor.submit(self.judge_single_cot, judge, system_prompt, user_content): judge 
                       for judge in judges}
            
            for future in as_completed(futures):
                result = future.result()
                if result[0] is None: continue  # Skip failed judges
                vote, confidence, reasoning = result
                judge = futures[future]
                
                if vote == 1:
                    score_1_sum += confidence
                else:
                    score_0_sum += confidence
                    
                results.append({
                    "judge": judge,
                    "vote": vote,
                    "confidence": confidence,
                    "reasoning": reasoning
                })
            
        # Tie-Breaker Logic
        if score_1_sum >= score_0_sum:
            final_reward = 1.0
        else:
            final_reward = 0.0
            
        if not results: final_reward = float('nan')

        return final_reward, results

    def logit_transform(self, score: float) -> float:
        import numpy as np
        import math
        if math.isnan(score): return float('nan')
        score = np.clip(score, 0.01, 0.99)
        return np.log(score / (1 - score))

    def process_task(self, task):
        prompt_text, model_id = task
        
        # 1. Get Response (Cached or New)
        response = self.get_model_response(model_id, prompt_text)
        
        if not response:
            return {
                "model_id": model_id, "ok": False, "ts": time.time()
            }

        # 2. Judge with CoT Panel
        final_score, judge_details = self.judge_with_panel_cot(prompt_text, response, model_id)
        reward_logit = self.logit_transform(final_score)
        
        return {
            "model_id": model_id,
            "prompt": prompt_text,
            "response": response,
            "ok": True,
            "teacher_used": True,
            "judge_details": judge_details, # Contains individual reasoning/scores
            "reward_logit": reward_logit,
            "raw_score": final_score,
            "ts": time.time()
        }

    def run(self, prompts_file, models_file, output_file, cache_file, is_lmsys=False, limit=None):
        # 1. Load Cache
        self.load_cache(cache_file)
        
        # 2. Load Prompts
        prompts = []
        with open(prompts_file) as f:
            for line in f:
                data = json.loads(line)
                if is_lmsys:
                    # Check for direct 'prompt' key first (cleaned format)
                    if 'prompt' in data:
                        prompts.append(data)
                    else:
                        # Fallback to raw LMSYS format: conversation[0]['content']
                        try:
                            prompt_text = data['conversation'][0]['content']
                            prompts.append({"prompt": prompt_text})
                        except:
                            continue
                else:
                    prompts.append(data)
        
        if limit:
            prompts = prompts[:limit]
            print(f"Limiting to first {limit} prompts.")
        
        # 3. Load Models
        with open(models_file) as f:
            registry = json.load(f)
        models = [m["openrouter_id"] for m in registry["models"]]
        
        print(f"Processing {len(prompts)} prompts x {len(models)} models = {len(prompts)*len(models)} tasks")
        
        # 4. Create Tasks
        tasks = []
        for p in prompts:
            for m in models:
                tasks.append((p["prompt"], m))
                
        # 5. Run Parallel
        print(f"Saving to {output_file} (Appending)")
        # Clear output or create if doesn't exist
        if not output_file.exists():
            with open(output_file, 'w') as f: pass
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_task, t): t for t in tasks}
            
            with tqdm(total=len(tasks), desc="CoT Judging") as pbar:
                for f in as_completed(futures):
                    res = f.result()
                    with open(output_file, 'a') as outfile:
                        outfile.write(json.dumps(res) + "\n")
                    pbar.update(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="pareto", choices=["pareto", "distribution"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to process")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    gen = CoTRewardGenerator(max_workers=64)
    
    models_file = root / "src/bandit_gpt/config/models.json"
    
    if args.mode == "distribution":
        # Process the 888 new prompts for the distribution warmup
        gen.run(
            prompts_file=root / "data/lmsys_needs_rewards_combined.jsonl",
            models_file=models_file,
            output_file=root / "data/lmsys_new_rewards_888.jsonl",
            cache_file=root / "data/lmsys_rewards_cache.jsonl",
            is_lmsys=True,
            limit=args.limit
        )
    else:
        # Original Pareto logic
        gen.run(
            prompts_file=root / "src/bandit_gpt/data/test_prompts.jsonl",
            models_file=models_file,
            output_file=root / "src/bandit_gpt/data/test_rewards_pareto.jsonl",
            cache_file=root / "src/bandit_gpt/data/test_rewards_cache.jsonl",
            limit=args.limit
        )
