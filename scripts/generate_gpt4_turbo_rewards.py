#!/usr/bin/env python3
"""
Generate GPT-4-Turbo rewards using RouteLLM's exact judging methodology.

This script:
1. Loads prompts from evaluation splits that have mixtral + gpt-4o
2. Generates GPT-4-turbo responses via OpenRouter
3. Judges using GPT-4o pairwise comparison (RouteLLM's method)
4. Saves to data/routellm/ in BanditGPT format

RouteLLM Judging Method:
- Judge: GPT-4 (we use GPT-4o)
- Method: Pairwise comparison between response and reference
- Prompt: MT-Bench style evaluation prompt
- Output: Winner verdict [[A]], [[B]], or [[C]] (tie)
- Scoring: Convert to 0.0-1.0 scale

Cost: ~$29 for 1,871 prompts (full dataset)
Time: ~15-20 minutes
"""

import os
import sys
import json
import time
import re
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("❌ Error: OPENROUTER_API_KEY not found in .env file")
    sys.exit(1)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent  # scripts/ -> project root
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits" / "evaluation"
OUTPUT_DIR = PROJECT_ROOT / "src/bandit_gpt/data/offline_dataset"

# Models
MIXTRAL_MODEL = "mistralai/mixtral-8x7b-instruct"
GPT4_TURBO_MODEL = "openai/gpt-4-turbo"  # OpenRouter ID
GPT4O_MODEL = "openai/gpt-4-turbo"  # For existing responses and judging

# Input files
DEV_REWARDS_PATH = PROJECT_ROOT / "src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz"
HOLDOUT_REWARDS_PATH = PROJECT_ROOT / "src/bandit_gpt/data/offline_dataset/holdout_rewards_complete.jsonl.gz"

# Output files
OUTPUT_DEV = OUTPUT_DIR / "dev_rewards_gpt4turbo.jsonl"
OUTPUT_HOLDOUT = OUTPUT_DIR / "holdout_rewards_gpt4turbo.jsonl"
CACHE_FILE = OUTPUT_DIR / "gpt4turbo_generation_cache.json"

# Rate limiting
REQUESTS_PER_SECOND = 5
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND


def call_openrouter(
    model: str,
    messages: List[Dict],
    temperature: float = 1.0,
    max_tokens: int = 2048
) -> Optional[str]:
    """
    Call OpenRouter API.
    
    Args:
        model: Model ID (e.g., "openai/gpt-4-turbo")
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
    
    Returns:
        Response text or None if failed
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/banditgpt",  # Required by OpenRouter
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
        
    except requests.exceptions.Timeout:
        print(f"  ⚠️  Timeout calling {model}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  API error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"  ⚠️  Invalid response format: {e}")
        return None


def generate_response(prompt: str, model: str, cache: Dict) -> Optional[str]:
    """
    Generate a response from a model, using cache if available.
    
    Args:
        prompt: User prompt
        model: Model ID
        cache: Cache dictionary
    
    Returns:
        Response text or None
    """
    cache_key = f"{model}:{prompt[:100]}"  # Use first 100 chars as key
    
    # Check cache
    if cache_key in cache:
        return cache[cache_key]
    
    # Generate
    messages = [{"role": "user", "content": prompt}]
    response = call_openrouter(model, messages, temperature=1.0, max_tokens=2048)
    
    if response:
        cache[cache_key] = response
        save_cache(cache)
    
    time.sleep(REQUEST_DELAY)
    return response


def judge_pairwise(
    prompt: str,
    response_a: str,
    response_b: str,
    model_a_name: str = "Assistant A",
    model_b_name: str = "Assistant B"
) -> Tuple[str, str]:
    """
    Judge two responses using GPT-4o pairwise comparison (RouteLLM's method).
    
    This is the EXACT judging prompt used by RouteLLM/MT-Bench.
    
    Args:
        prompt: Original user prompt
        response_a: First response
        response_b: Second response
        model_a_name: Name for first model
        model_b_name: Name for second model
    
    Returns:
        (verdict, explanation) where verdict is "A", "B", or "tie"
    """
    judge_prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if {model_a_name} is better, "[[B]]" if {model_b_name} is better, and "[[C]]" for a tie.

[User Question]
{prompt}

[The Start of {model_a_name}'s Answer]
{response_a}
[The End of {model_a_name}'s Answer]

[The Start of {model_b_name}'s Answer]
{response_b}
[The End of {model_b_name}'s Answer]"""
    
    messages = [{"role": "user", "content": judge_prompt}]
    judgment = call_openrouter(GPT4O_MODEL, messages, temperature=0.0, max_tokens=1024)
    
    if not judgment:
        return "tie", "Failed to get judgment"
    
    # Extract verdict
    if "[[A]]" in judgment:
        verdict = "A"
    elif "[[B]]" in judgment:
        verdict = "B"
    elif "[[C]]" in judgment:
        verdict = "tie"
    else:
        # Fallback: try to parse explanation
        if "assistant a" in judgment.lower() and "better" in judgment.lower():
            verdict = "A"
        elif "assistant b" in judgment.lower() and "better" in judgment.lower():
            verdict = "B"
        else:
            verdict = "tie"
    
    time.sleep(REQUEST_DELAY)
    return verdict, judgment


def verdict_to_score(verdict: str, is_reference: bool) -> float:
    """
    Convert pairwise verdict to normalized score (0.0-1.0).
    
    RouteLLM scoring:
    - If response beats reference: 1.0 (excellent)
    - If tie: 0.85 (very good)
    - If reference wins: 0.7 (good but not as good as reference)
    
    Args:
        verdict: "A", "B", or "tie"
        is_reference: True if response A is the reference
    
    Returns:
        Score from 0.0 to 1.0
    """
    if verdict == "tie":
        return 0.85
    elif verdict == "A":
        return 0.7 if is_reference else 1.0
    else:  # verdict == "B"
        return 1.0 if is_reference else 0.7


def load_existing_responses(path: Path, model_id: str) -> Dict[str, str]:
    """
    Load existing model responses from a rewards file.
    
    Args:
        path: Path to rewards JSONL file
        model_id: Model to load responses for
    
    Returns:
        Dictionary mapping prompt -> response
    """
    responses = {}
    
    if not path.exists():
        return responses
    
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok") and entry.get("model_id") == model_id:
                prompt = entry["prompt"]
                response = entry.get("response", "")
                if response:
                    responses[prompt] = response
    
    return responses


def load_prompts_with_models(path: Path) -> Dict[str, Dict]:
    """
    Load prompts that have both mixtral and gpt-4o.
    
    Args:
        path: Path to rewards JSONL file (supports .gz)
    
    Returns:
        Dictionary mapping prompt -> {rewards, responses}
    """
    prompt_data = defaultdict(lambda: {"rewards": {}, "responses": {}})
    
    if not path.exists():
        print(f"⚠️  File not found: {path}")
        return {}
    
    # Handle gzipped files
    if str(path).endswith('.gz'):
        open_func = lambda f: gzip.open(f, 'rt')
    else:
        open_func = lambda f: open(f, 'r')
    
    with open_func(path) as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            
            prompt = entry["prompt"]
            model_id = entry["model_id"]
            
            # Only process our target models
            if model_id not in [MIXTRAL_MODEL, GPT4O_MODEL]:
                continue
            
            prompt_data[prompt]["rewards"][model_id] = entry.get("raw_score", 0.0)
            prompt_data[prompt]["responses"][model_id] = entry.get("response", "")
    
    # Filter to prompts with BOTH models
    complete_prompts = {
        prompt: data
        for prompt, data in prompt_data.items()
        if len(data["rewards"]) == 2 and len(data["responses"]) == 2
    }
    
    return complete_prompts


def load_cache() -> Dict:
    """Load generation cache from disk."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict):
    """Save generation cache to disk."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def process_split(
    input_path: Path,
    output_path: Path,
    split_name: str,
    cache: Dict,
    auto_confirm: bool = False
):
    """
    Process one data split (dev or holdout).
    
    Args:
        input_path: Input rewards file
        output_path: Output rewards file
        split_name: Name for logging
        cache: Generation cache
        auto_confirm: If True, skip confirmation prompt
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {split_name}")
    print(f"{'='*70}")
    
    # Load prompts
    print(f"📂 Loading prompts from {input_path.name}...")
    prompt_data = load_prompts_with_models(input_path)
    
    if not prompt_data:
        print(f"❌ No prompts found with both models")
        return
    
    print(f"✓ Found {len(prompt_data):,} prompts with mixtral + gpt-4o")
    
    # Check what we already have
    existing_entries = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("model_id") == GPT4_TURBO_MODEL:
                    existing_entries.add(entry["prompt"])
        print(f"✓ Found {len(existing_entries):,} existing gpt-4-turbo entries")
    
    # Process prompts
    prompts_to_process = [p for p in prompt_data.keys() if p not in existing_entries]
    print(f"📝 Need to process: {len(prompts_to_process):,} prompts")
    
    if not prompts_to_process:
        print(f"✅ All prompts already processed!")
        return
    
    # Estimate cost
    total_tokens = len(prompts_to_process) * (150 + 300 + 800)  # prompt + response + judgment
    estimated_cost = total_tokens / 1_000_000 * 15  # Rough average
    print(f"💰 Estimated cost: ${estimated_cost:.2f}")
    
    # Confirm
    if not auto_confirm:
        response = input(f"\nProcess {len(prompts_to_process):,} prompts? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    else:
        print(f"\n✅ Auto-confirmed: Processing {len(prompts_to_process):,} prompts...")
    
    # Open output file for appending
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'a') as outfile:
        for prompt in tqdm(prompts_to_process, desc=f"  {split_name}"):
            try:
                # Generate GPT-4-turbo response
                gpt4_turbo_response = generate_response(prompt, GPT4_TURBO_MODEL, cache)
                
                if not gpt4_turbo_response:
                    print(f"\n  ⚠️  Failed to generate response, skipping...")
                    continue
                
                # Get reference response (GPT-4o)
                reference_response = prompt_data[prompt]["responses"][GPT4O_MODEL]
                
                # Judge: GPT-4-turbo vs GPT-4o reference
                verdict, explanation = judge_pairwise(
                    prompt,
                    reference_response,  # A = reference (GPT-4o)
                    gpt4_turbo_response,  # B = GPT-4-turbo
                    model_a_name="GPT-4o (reference)",
                    model_b_name="GPT-4-turbo"
                )
                
                # Convert to score
                score = verdict_to_score(verdict, is_reference=True)
                
                # Create entry in BanditGPT format
                entry = {
                    "prompt": prompt,
                    "model_id": GPT4_TURBO_MODEL,
                    "response": gpt4_turbo_response,
                    "raw_score": score,
                    "ok": True,
                    "metadata": {
                        "judge_model": GPT4O_MODEL,
                        "reference_model": GPT4O_MODEL,
                        "verdict": verdict,
                        "explanation": explanation,
                        "judging_method": "routellm_pairwise"
                    }
                }
                
                # Write to file
                outfile.write(json.dumps(entry) + '\n')
                outfile.flush()
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                save_cache(cache)
                return
            except Exception as e:
                print(f"\n  ⚠️  Error processing prompt: {e}")
                continue
    
    print(f"\n✅ Completed: {split_name}")
    print(f"   Output: {output_path}")
    save_cache(cache)


def main(auto_confirm: bool = False):
    """Main execution."""
    print("="*70)
    print("GPT-4-TURBO REWARDS GENERATION")
    print("="*70)
    print(f"\nUsing RouteLLM's judging methodology:")
    print(f"  - Judge: {GPT4O_MODEL}")
    print(f"  - Method: Pairwise comparison (MT-Bench style)")
    print(f"  - Reference: {GPT4O_MODEL} responses")
    print(f"\nModels:")
    print(f"  - Target: {GPT4_TURBO_MODEL}")
    print(f"  - Reference: {GPT4O_MODEL}")
    print(f"  - Baseline: {MIXTRAL_MODEL}")
    
    # Load cache
    cache = load_cache()
    print(f"\n📦 Cache: {len(cache)} entries")
    
    # Process splits
    process_split(DEV_REWARDS_PATH, OUTPUT_DEV, "Dev Set", cache, auto_confirm)
    process_split(HOLDOUT_REWARDS_PATH, OUTPUT_HOLDOUT, "Holdout Set", cache, auto_confirm)
    
    print("\n" + "="*70)
    print("✅ COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Dev: {OUTPUT_DEV}")
    print(f"  - Holdout: {OUTPUT_HOLDOUT}")
    print(f"\nNext steps:")
    print(f"  1. Verify rewards: python -c 'import json; [print(json.loads(l)) for l in open(\"{OUTPUT_DEV}\")][:5]'")
    print(f"  2. Run comparison: cd experiments/11_routellm_comparison && python run_comparison.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate GPT-4-turbo rewards using RouteLLM's judging method")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm all prompts (skip confirmation)")
    args = parser.parse_args()
    
    main(auto_confirm=args.yes)

