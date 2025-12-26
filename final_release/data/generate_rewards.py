"""
Generate ground truth rewards for train/test sets using LLM-as-a-Judge.

Uses:
- google/gemini-3-flash-preview as default teacher judge
- anthropic/claude-4.5-sonnet for judging Gemini models (avoid self-grading bias)
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

class RewardGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_judge = "google/gemini-3-flash-preview"  # Fast, cost-effective
        self.gemini_judge = "anthropic/claude-4.5-sonnet"  # For Gemini models (avoid self-grading)
        self.judge_max_tokens = 50  # Sufficient for score output with safety margin
    
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
            print(f"Error getting response from {model_id}: {e}")
            return None
    
    def judge_response(self, prompt: str, response: str, model_being_judged: str) -> float:
        """
        Judge a response using appropriate teacher judge.
        
        Uses Claude Sonnet 4.5 for Gemini models to avoid self-grading bias.
        Uses Gemini-3-Flash for all other models.
        """
        # Select judge based on model being evaluated
        if "gemini" in model_being_judged.lower():
            judge_model = self.gemini_judge
        else:
            judge_model = self.default_judge
        
        system_prompt = (
            "You are an impartial judge. Rate the quality of the response to the prompt.\\n"
            "Output ONLY a single float number between 0.0 and 1.0.\\n"
            "0.0 = Completely wrong, harmful, or unhelpful.\\n"
            "0.5 = Partially correct but missing key details.\\n"
            "1.0 = Perfectly correct, helpful, and comprehensive.\\n"
            "Do not output any other text."
        )
        
        user_content = f"PROMPT: {prompt}\\n\\nRESPONSE: {response}"
        
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
            
            # Parse score
            import re
            match = re.search(r"(\\d+(\\.\\d+)?)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception as e:
            print(f"Error judging with {judge_model}: {e}")
            return 0.5
    
    def logit_transform(self, score: float) -> float:
        """Transform [0,1] score to logit space."""
        import numpy as np
        # Clip to avoid log(0)
        score = np.clip(score, 0.01, 0.99)
        return np.log(score / (1 - score))
    
    def generate_rewards_for_file(
        self, 
        prompts_file: Path, 
        output_file: Path,
        models: List[str],
        max_prompts: int = None
    ):
        """Generate rewards for a prompts file."""
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
        
        rewards = []
        total_calls = len(prompts) * len(models)
        
        with tqdm(total=total_calls, desc="Generating rewards") as pbar:
            for prompt_data in prompts:
                cluster_id = prompt_data["cluster_id"]
                prompt_text = prompt_data["prompt"]
                
                for model_id in models:
                    # Get model response
                    response = self.get_model_response(model_id, prompt_text)
                    
                    if response:
                        # Judge response
                        score = self.judge_response(prompt_text, response, model_id)
                        reward_logit = self.logit_transform(score)
                        
                        # Store
                        rewards.append({
                            "cluster_id": cluster_id,
                            "model_id": model_id,
                            "ok": True,
                            "teacher_used": True,
                            "reward_logit": reward_logit,
                            "raw_score": score,
                            "ts": time.time()
                        })
                    else:
                        # Failed to get response
                        rewards.append({
                            "cluster_id": cluster_id,
                            "model_id": model_id,
                            "ok": False,
                            "teacher_used": False,
                            "reward_logit": 0.0,
                            "ts": time.time()
                        })
                    
                    pbar.update(1)
                    
                    # Rate limiting
                    time.sleep(0.1)
        
        # Save rewards
        print(f"Saving {len(rewards)} rewards to {output_file}...")
        with open(output_file, 'w') as f:
            for reward in rewards:
                f.write(json.dumps(reward) + "\\n")
        
        print("Done!")

def main():
    base_dir = Path(__file__).parent
    
    # Load model registry
    models_path = base_dir.parent / "models.json"
    with open(models_path) as f:
        registry = json.load(f)
    
    model_ids = [m["openrouter_id"] for m in registry["models"]]
    print(f"Found {len(model_ids)} models in registry")
    
    # Initialize generator
    generator = RewardGenerator()
    
    # Generate for GOLDEN prompts only (100 prompts, one per cluster)
    print("\n=== Generating Rewards for Golden Prompts ===")
    print("Using 100 golden prompts (one per cluster) instead of all prompts")
    print("This reduces cost by ~267x while maintaining cluster-level accuracy")
    
    generator.generate_rewards_for_file(
        prompts_file=base_dir / "golden_prompts.jsonl",
        output_file=base_dir / "golden_rewards.jsonl",
        models=model_ids,
        max_prompts=None  # Process all 500 golden prompts
    )

if __name__ == "__main__":
    main()
