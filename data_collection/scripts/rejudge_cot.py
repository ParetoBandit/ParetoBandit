import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Reuse the class structure but modify for CoT / Re-judging
class CoTRewardGenerator:
    def __init__(self, api_key: str = None, max_workers: int = 10):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            # Try loading from .env manually if not in env
             try:
                from dotenv import load_dotenv
                # banditgpt/rejudge_cot.py -> parent = banditgpt -> parent = root
                env_path = Path(__file__).parent.parent.parent / '.env'
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
            "anthropic": "anthropic/claude-3.5-haiku",
            "meta": "meta-llama/llama-3.3-70b-instruct",
            "google": "google/gemini-2.5-flash",
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

    def judge_single_cot(
        self, judge_model: str, system_prompt: str, user_content: str,
    ) -> Dict[str, Any] | None:
        """Query a single judge and parse the three-factor rubric response.

        Returns
        -------
        dict | None
            Keys: ``logic`` (0 or 1), ``constraint`` (0 or 1),
            ``utility`` (float 0-1), ``reward`` (composite float),
            ``reasoning`` (str).  ``None`` on API failure.
        """
        import re

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }

        payload = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": self.judge_max_tokens,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers, json=payload, timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # --- Parse Logical Integrity (binary: 0 or 1) ---
            logic = 0
            m = re.search(
                r"##\s*Logical Integrity\s*[:\-]?\s*(\d)", content, re.IGNORECASE,
            )
            if m:
                logic = 1 if int(m.group(1)) == 1 else 0

            # --- Parse Constraint Adherence (binary: 0 or 1) ---
            constraint = 0
            m = re.search(
                r"##\s*Constraint Adherence\s*[:\-]?\s*(\d)", content, re.IGNORECASE,
            )
            if m:
                constraint = 1 if int(m.group(1)) == 1 else 0

            # --- Parse Utility & Tone (continuous 0.0–1.0) ---
            utility = 0.5
            m = re.search(
                r"##\s*Utility\s*(?:&|and)?\s*Tone\s*[:\-]?\s*(\d+\.?\d*)",
                content, re.IGNORECASE,
            )
            if m:
                val = float(m.group(1))
                if val > 1.0:
                    val = val / 100.0
                utility = max(0.0, min(1.0, val))

            # --- Composite reward ---
            reward = logic * 0.5 + constraint * 0.3 + utility * 0.2

            # --- Extract Reasoning block ---
            reasoning = content
            rm = re.search(
                r"##\s*Reasoning\s*(.*?)(\n##|$)",
                content, re.DOTALL | re.IGNORECASE,
            )
            if rm:
                reasoning = rm.group(1).strip()

            return {
                "logic": logic,
                "constraint": constraint,
                "utility": round(utility, 4),
                "reward": round(reward, 4),
                "reasoning": reasoning,
            }

        except Exception as e:
            return None

    def judge_with_panel_cot(self, prompt: str, response: str, model_id: str) -> Tuple[float, List[Dict]]:
        """Run the multi-judge panel and aggregate rubric scores.

        Each judge independently scores three factors (Logical Integrity,
        Constraint Adherence, Utility & Tone).  The final reward is the
        mean of per-judge composite scores.
        """
        judges = self.get_judges_for_model(model_id)

        system_prompt = (
            "You are a Discriminative Router Judge. Your goal is to find the "
            "failure points in LLM responses.\n\n"
            "Score the response on three factors:\n\n"
            "1. **Logical Integrity (50 %)** — Does the model show its work? "
            "If there is a single calculation or logical-step error, this "
            "factor is 0. No partial credit.\n"
            "2. **Constraint Adherence (30 %)** — Did the model follow ALL "
            "formatting and negative constraints (e.g. \"Do not use the word "
            "'AI'\")? If one constraint is missed, this factor is 0.\n"
            "3. **Utility & Tone (20 %)** — Is the answer helpful and "
            "professional? Score continuously from 0.0 (useless / rude) to "
            "1.0 (maximally helpful and professional).\n\n"
            "Format your response EXACTLY as follows:\n\n"
            "## Reasoning\n"
            "<Concise chain-of-thought analysis identifying any errors or "
            "constraint violations>\n\n"
            "## Logical Integrity\n"
            "<0 or 1>\n\n"
            "## Constraint Adherence\n"
            "<0 or 1>\n\n"
            "## Utility & Tone\n"
            "<0.0 to 1.0>"
        )
        user_content = f"PROMPT: {prompt}\n\nRESPONSE: {response}"

        results: List[Dict] = []

        with ThreadPoolExecutor(max_workers=len(judges)) as executor:
            futures = {
                executor.submit(
                    self.judge_single_cot, judge, system_prompt, user_content,
                ): judge
                for judge in judges
            }

            for future in as_completed(futures):
                parsed = future.result()
                if parsed is None:
                    continue
                judge = futures[future]
                results.append({
                    "judge": judge,
                    "logic": parsed["logic"],
                    "constraint": parsed["constraint"],
                    "utility": parsed["utility"],
                    "reward": parsed["reward"],
                    "reasoning": parsed["reasoning"],
                })

        if results:
            final_reward = float(np.mean([r["reward"] for r in results]))
        else:
            final_reward = float("nan")

        return final_reward, results

    def logit_transform(self, score: float) -> float:
        if np.isnan(score):
            return float("nan")
        score = np.clip(score, 0.01, 0.99)
        return float(np.log(score / (1 - score)))

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
        models = [m["model_id"] for m in registry["models"]]
        
        print(f"Processing {len(prompts)} prompts x {len(models)} models = {len(prompts)*len(models)} tasks")
        
        # 4. Create Tasks
        tasks = []
        for p in prompts:
            for m in models:
                tasks.append((p["prompt"], m))
                
        # 5. Load already-completed tasks from output (resume support)
        completed = set()
        if output_file.exists():
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        completed.add((entry.get("prompt", ""), entry.get("model_id", "")))
                    except json.JSONDecodeError:
                        continue
            if completed:
                print(f"Resuming: {len(completed)} tasks already completed, skipping them.")

        remaining = [t for t in tasks if t not in completed]
        print(f"Tasks to run: {len(remaining)} (skipped {len(tasks) - len(remaining)})")

        # 6. Run Parallel — flush each result to disk immediately
        print(f"Saving to {output_file} (append + flush per entry)")
        with open(output_file, 'a') as outfile:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_task, t): t for t in remaining}

                with tqdm(total=len(remaining), desc="CoT Judging") as pbar:
                    for f in as_completed(futures):
                        res = f.result()
                        with self.lock:
                            outfile.write(json.dumps(res) + "\n")
                            outfile.flush()
                        pbar.update(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate multi-judge CoT rewards for (prompt, model) pairs.",
    )
    parser.add_argument("--mode", type=str, default="pareto",
                        choices=["pareto", "distribution", "custom"],
                        help="Preset mode or 'custom' for explicit paths")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of prompts to process")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Path to prompts JSONL (required for --mode custom)")
    parser.add_argument("--models-file", type=str, default=None,
                        help="Path to models JSON (default: models.json)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to output JSONL (required for --mode custom)")
    parser.add_argument("--cache-file", type=str, default=None,
                        help="Path to response cache JSONL (optional)")
    parser.add_argument("--workers", type=int, default=64,
                        help="Max parallel workers (default: 64)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    gen = CoTRewardGenerator(max_workers=args.workers)

    models_file = Path(args.models_file) if args.models_file else root / "src/bandit_gpt/config/models.json"

    if args.mode == "custom":
        if not args.prompts_file or not args.output_file:
            parser.error("--mode custom requires --prompts-file and --output-file")
        gen.run(
            prompts_file=Path(args.prompts_file),
            models_file=models_file,
            output_file=Path(args.output_file),
            cache_file=Path(args.cache_file) if args.cache_file else Path(args.output_file).with_suffix(".cache.jsonl"),
            is_lmsys=False,
            limit=args.limit,
        )
    elif args.mode == "distribution":
        gen.run(
            prompts_file=root / "data/lmsys_needs_rewards_combined.jsonl",
            models_file=models_file,
            output_file=root / "data/lmsys_new_rewards_888.jsonl",
            cache_file=root / "data/lmsys_rewards_cache.jsonl",
            is_lmsys=True,
            limit=args.limit,
        )
    else:
        gen.run(
            prompts_file=root / "src/bandit_gpt/data/test_prompts.jsonl",
            models_file=models_file,
            output_file=root / "src/bandit_gpt/data/test_rewards_pareto.jsonl",
            cache_file=root / "src/bandit_gpt/data/test_rewards_cache.jsonl",
            limit=args.limit,
        )
