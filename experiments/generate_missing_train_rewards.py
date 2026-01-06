import json
import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Any
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    # Try to load .env from project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

class MissingRewardGenerator:
    def __init__(self, api_key: str = None, max_workers: int = 10):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment or .env file")
        
        self.base_url = "https://openrouter.ai/api/v1"
        
        # The 7 Missing Models identified earlier
        self.target_models = [
            "amazon/nova-lite-v1",
            "amazon/nova-pro-v1",
            "google/gemini-2.5-flash-lite",
            "google/gemini-2.5-flash-preview-09-2025",
            "google/gemini-2.5-pro-preview-06-05",
            "google/gemini-3-pro-preview",
            "meta-llama/llama-3.1-405b-instruct"
        ]

        # Judge Pool (same as rejudge_cot.py)
        self.judge_pool = {
            "openai": "openai/gpt-4o",
            "anthropic": "anthropic/claude-3.5-sonnet",
            "meta": "meta-llama/llama-3.1-405b-instruct",
            "google": "google/gemini-2.5-pro-preview-06-05"
        }
        self.family_map = {
            "gpt": "openai", "o1": "openai", "o3": "openai",
            "claude": "anthropic", "llama": "meta",
            "gemini": "google", "gemma": "google",
            "nova": "amazon" # Added amazon/nova
        }
        
        self.judge_max_tokens = 8000
        self.max_workers = max_workers
        self.response_cache = {}

    def get_judges_for_model(self, model_id: str) -> List[str]:
        family = None
        lower_id = model_id.lower()
        for key, val in self.family_map.items():
            if key in lower_id:
                family = val
                break
        
        # Fallback detection
        if not family:
            if "openai/" in lower_id: family = "openai"
            elif "anthropic/" in lower_id: family = "anthropic"
            elif "google/" in lower_id: family = "google"
            elif "meta-llama/" in lower_id: family = "meta"
            elif "amazon/" in lower_id: family = "amazon"
        
        selected = []
        for org, judge_id in self.judge_pool.items():
            if family == org: continue
            selected.append(judge_id)
        
        # Ensure we have at least 3 judges if possible, or all available non-conflicting
        if len(selected) < 3 and "amazon" in lower_id:
             # Amazon has no judge in pool, so it gets all 4? 
             # Wait, judge_pool has 4 keys. Amazon != any of them. So it gets all 4.
             # but we usually want 3. Let's just return all valid non-conflicting ones.
             pass

        return selected

    def get_model_response(self, model_id: str, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 8000
        }
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=300)
            # Handle rate limits minimally
            if resp.status_code == 429:
                time.sleep(2)
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=300)
            
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error getting response from {model_id}: {e}")
            return None

    def judge_single_cot(self, judge_model: str, system_prompt: str, user_content: str) -> Tuple[int, float, str]:
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
            
            vote = 0
            vote_match = re.search(r"## Vote\s*(\d)", content, re.IGNORECASE)
            if vote_match:
                vote = int(vote_match.group(1))
                if vote != 1: vote = 0 
            
            confidence = 0.5
            conf_match = re.search(r"## Confidence\s*(\d+(\.\d+)?)", content, re.IGNORECASE)
            if conf_match:
                val = float(conf_match.group(1))
                if val > 1.0: val = val / 100.0
                confidence = max(0.0, min(1.0, val))

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
        
        with ThreadPoolExecutor(max_workers=len(judges)) as executor:
            futures = {executor.submit(self.judge_single_cot, judge, system_prompt, user_content): judge 
                       for judge in judges}
            
            for future in as_completed(futures):
                result = future.result()
                if result[0] is None: continue
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
        cluster_id, prompt_text, model_id = task
        
        # 1. Get Model Response
        response = self.get_model_response(model_id, prompt_text)
        
        if not response:
            return {
                "cluster_id": cluster_id, "model_id": model_id, "ok": False, "ts": time.time()
            }

        # 2. Judge with Panel
        final_score, judge_details = self.judge_with_panel_cot(prompt_text, response, model_id)
        reward_logit = self.logit_transform(final_score)
        
        return {
            "cluster_id": cluster_id,
            "model_id": model_id,
            "prompt": prompt_text,
            "response": response,
            "ok": True,
            "teacher_used": True,
            "judge_details": judge_details,
            "reward_logit": reward_logit,
            "raw_score": final_score,
            "ts": time.time()
        }

    def run(self, prompts_file: Path, output_file: Path, limit: int = None):
        print(f"🚀 Starting generation for 7 missing models")
        print(f"📂 Loading prompts from {prompts_file}")
        
        prompts = []
        with open(prompts_file, 'r') as f:
            for line in f:
                prompts.append(json.loads(line))
        
        if limit:
            print(f"⚠️ Limit set to {limit} prompts for testing")
            prompts = prompts[:limit]

        tasks = []
        for p in prompts:
            # ONLY for the 7 missing models
            for m in self.target_models:
                tasks.append((p.get("cluster_id", 0), p.get("prompt", ""), m))
        
        print(f"Total tasks: {len(tasks)} ({len(prompts)} prompts x {len(self.target_models)} models)")
        
        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Output will be saved to {output_file}")
        
        # Append mode
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_task, t): t for t in tasks}
            
            with tqdm(total=len(tasks), desc="Generating Rewards") as pbar:
                for f in as_completed(futures):
                    res = f.result()
                    with open(output_file, 'a') as outfile:
                        outfile.write(json.dumps(res) + "\n")
                    pbar.update(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of prompts for testing")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    prompts_path = project_root / "src/bandit_gpt/data/train_prompts_sampled_1k.jsonl"
    output_path = project_root / "src/bandit_gpt/data/offline_dataset/train_rewards_missing_7models.jsonl"
    
    generator = MissingRewardGenerator(max_workers=50)
    generator.run(prompts_path, output_path, limit=args.limit)
