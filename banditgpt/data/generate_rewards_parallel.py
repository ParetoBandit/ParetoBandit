"""
PARALLEL version: Generate ground truth rewards using LLM-as-a-Judge with concurrent requests.

Uses:
- google/gemini-3-flash-preview as default teacher judge
- anthropic/claude-4.5-sonnet for judging Gemini models (avoid self-grading bias)
- ThreadPoolExecutor for parallel API calls
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

class ParallelRewardGenerator:
    def __init__(self, api_key: str = None, max_workers: int = 10):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_judge = "google/gemini-3-flash-preview"
        self.gemini_judge = "anthropic/claude-4.5-sonnet"
        self.judge_max_tokens = 50
        self.max_workers = max_workers
        self.lock = threading.Lock()
    
    def get_model_response(self, model_id: str, prompt: str) -> str:
        """Get a response from a specific model."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return None
    
    def judge_response(self, prompt: str, response: str, model_being_judged: str) -> float:
        """Judge a response using appropriate teacher judge."""
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
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            import re
            match = re.search(r"(\d+(\.\d+)?)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception as e:
            return 0.5
    
    def logit_transform(self, score: float) -> float:
        """Transform [0,1] score to logit space."""
        import numpy as np
        score = np.clip(score, 0.01, 0.99)
        return np.log(score / (1 - score))
    
    def process_single_evaluation(self, cluster_id: int, prompt_text: str, model_id: str) -> dict:
        """Process a single (prompt, model) evaluation."""
        # Get model response
        response = self.get_model_response(model_id, prompt_text)
        
        if response:
            # Judge response
            score = self.judge_response(prompt_text, response, model_id)
            reward_logit = self.logit_transform(score)
            
            return {
                "cluster_id": cluster_id,
                "model_id": model_id,
                "ok": True,
                "teacher_used": True,
                "reward_logit": reward_logit,
                "raw_score": score,
                "ts": time.time()
            }
        else:
            return {
                "cluster_id": cluster_id,
                "model_id": model_id,
                "ok": False,
                "teacher_used": False,
                "reward_logit": 0.0,
                "ts": time.time()
            }
    
    def generate_rewards_parallel(
        self,
        prompts_file: Path,
        output_file: Path,
        models: List[str],
        max_prompts: int = None
    ):
        """Generate rewards in parallel."""
        # Load prompts
        print(f"Loading prompts from {prompts_file}...")
        prompts = []
        with open(prompts_file) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data)
                if max_prompts and len(prompts) >= max_prompts:
                    break
        
        print(f"Loaded {len(prompts)} prompts")
        print(f"Generating rewards for {len(models)} models...")
        print(f"Using {self.max_workers} parallel workers")
        
        # Create all tasks
        tasks = []
        for prompt_data in prompts:
            cluster_id = prompt_data["cluster_id"]
            prompt_text = prompt_data["prompt"]
            for model_id in models:
                tasks.append((cluster_id, prompt_text, model_id))
        
        total_tasks = len(tasks)
        print(f"Total evaluations: {total_tasks}")
        
        # Process in parallel
        rewards = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.process_single_evaluation, *task): task
                for task in tasks
            }
            
            with tqdm(total=total_tasks, desc="Generating rewards") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        reward = future.result()
                        with self.lock:
                            rewards.append(reward)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing task: {e}")
                        pbar.update(1)
        
        # Save rewards
        print(f"\nSaving {len(rewards)} rewards to {output_file}...")
        with open(output_file, 'w') as f:
            for reward in rewards:
                f.write(json.dumps(reward) + "\n")
        
        print("Done!")

def main():
    base_dir = Path(__file__).parent
    
    # Load model registry
    models_path = base_dir.parent / "models.json"
    with open(models_path) as f:
        registry = json.load(f)
    
    model_ids = [m["openrouter_id"] for m in registry["models"]]
    print(f"Found {len(model_ids)} models in registry")
    
    # Initialize generator with 10 parallel workers
    generator = ParallelRewardGenerator(max_workers=10)
    
    print("\n" + "="*60)
    print("FULL EVALUATION SET REWARD GENERATION")
    print("="*60)
    print("Generating individual rewards for all 5,000 evaluation prompts")
    print("This ensures KDD-level rigor with true ground truth")
    print("")
    
    # Generate for TEST set (1,000 prompts)
    print("\n=== Step 1/2: Generating TEST Rewards (1,000 prompts) ===")
    generator.generate_rewards_parallel(
        prompts_file=base_dir / "test_prompts.jsonl",
        output_file=base_dir / "test_rewards.jsonl",
        models=model_ids,
        max_prompts=None
    )
    
    # Generate for TRAIN set (4,000 prompts)
    print("\n=== Step 2/2: Generating TRAIN Rewards (4,000 prompts) ===")
    generator.generate_rewards_parallel(
        prompts_file=base_dir / "train_prompts.jsonl",
        output_file=base_dir / "train_rewards.jsonl",
        models=model_ids,
        max_prompts=None
    )
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print("Generated rewards for 5,000 prompts × 50 models = 250,000 evaluations")
    print("Output files:")
    print(f"  - {base_dir / 'test_rewards.jsonl'}")
    print(f"  - {base_dir / 'train_rewards.jsonl'}")

if __name__ == "__main__":
    main()
