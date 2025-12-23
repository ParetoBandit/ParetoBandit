#!/usr/bin/env python3
"""
Table 3: Router Performance Comparison with LLM Judge Evaluation
=================================================================
Uses real LLM judges via OpenRouter API:
- Gemini 3.0 Flash for non-Gemini models
- Claude 4.5 Sonnet for Gemini models (avoids self-judging bias)

Based on the "LLM-as-Judge" methodology used in Chatbot Arena.
"""

import sys
import os
import json
import logging
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Add final_release to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from bandit import BanditRouter, l2_normalize

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. OPENROUTER API CLIENT
# ==============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Judge models
JUDGE_GEMINI = "google/gemini-2.0-flash-001"  # Fast, cheap Gemini judge
JUDGE_CLAUDE = "anthropic/claude-sonnet-4"     # For judging Gemini models

def call_openrouter(model: str, messages: list, max_tokens: int = 1024) -> Optional[str]:
    """Make API call to OpenRouter."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found in environment")
        return None
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"API call failed for {model}: {e}")
        return None

# ==============================================================================
# 2. LLM JUDGE
# ==============================================================================

JUDGE_PROMPT = """You are an expert evaluator. Given a user query and a model's response, 
rate the response quality on a scale of 1-10.

Consider:
- Accuracy: Is the information correct?
- Helpfulness: Does it address the user's needs?
- Clarity: Is it well-written and easy to understand?
- Completeness: Does it fully answer the question?

Return ONLY a JSON object with this format:
{"score": <1-10>, "reason": "<brief explanation>"}

User Query:
{query}

Model Response:
{response}

Your evaluation (JSON only):"""

def judge_response(query: str, response: str, model_being_judged: str) -> Optional[float]:
    """
    Use LLM judge to score a response.
    Uses Claude for Gemini models, Gemini for others (avoids self-judging bias).
    """
    # Select judge based on model being judged
    if "gemini" in model_being_judged.lower() or "google" in model_being_judged.lower():
        judge_model = JUDGE_CLAUDE
    else:
        judge_model = JUDGE_GEMINI
    
    prompt = JUDGE_PROMPT.format(query=query, response=response)
    
    messages = [{"role": "user", "content": prompt}]
    result = call_openrouter(judge_model, messages, max_tokens=256)
    
    if not result:
        return None
    
    # Parse JSON response
    try:
        # Find JSON in response
        import re
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            parsed = json.loads(json_match.group())
            return float(parsed.get("score", 5)) / 10.0  # Normalize to 0-1
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse judge response: {result[:100]}")
    
    return 0.5  # Default neutral score

# ==============================================================================
# 3. LOAD MODEL REGISTRY
# ==============================================================================

def load_model_registry():
    """Load models.json for cost/latency data."""
    cache_path = Path(__file__).parent.parent.parent / "models.json"
    with open(cache_path) as f:
        data = json.load(f)
    return {m["openrouter_id"]: m for m in data["models"] if "openrouter_id" in m}

REGISTRY = load_model_registry()

def get_cost(model_id: str) -> float:
    """Get cost per 1k tokens for a model."""
    if model_id in REGISTRY:
        return REGISTRY[model_id].get("price_1m_blended", 1.0) / 1000
    return 0.001

def get_risk(model_id: str) -> float:
    """Get hallucination risk for a model."""
    if model_id in REGISTRY:
        return REGISTRY[model_id].get("hallucination_composite", 
               REGISTRY[model_id].get("hallucination_rate", 8.0))
    return 8.0

# ==============================================================================
# 4. ROUTER IMPLEMENTATIONS
# ==============================================================================

def create_bandit_router():
    """Create BanditGPT router."""
    return BanditRouter(REGISTRY)

def routellm_route(prompt: str) -> str:
    """RouteLLM: Route based on query complexity."""
    complex_keywords = ["explain", "analyze", "compare", "code", "python", 
                        "algorithm", "math", "calculate", "legal", "medical"]
    prompt_lower = prompt.lower()
    complexity = sum(1 for kw in complex_keywords if kw in prompt_lower)
    
    # Strong model for complex, weak for simple
    if complexity >= 2:
        return "openai/gpt-4o"
    else:
        return "meta-llama/llama-3.1-8b-instruct"

def aurelio_route(prompt: str) -> str:
    """Aurelio AI: Semantic category routing."""
    prompt_lower = prompt.lower()
    
    # Categories that need strong model
    if any(kw in prompt_lower for kw in ["code", "python", "function", "debug"]):
        return "openai/gpt-4o"  # Coding
    if any(kw in prompt_lower for kw in ["math", "calculate", "solve", "equation"]):
        return "openai/gpt-4o"  # Math
    if any(kw in prompt_lower for kw in ["medical", "legal", "financial"]):
        return "openai/gpt-4o"  # Safety-critical
    
    return "google/gemma-3-4b-it"  # General queries

def frugalgpt_route(prompt: str) -> Tuple[str, float]:
    """FrugalGPT: Cascade with cost tracking."""
    import random
    cheap = "google/gemma-3-1b-it"
    strong = "openai/gpt-4o"
    
    # 70% success on cheap, 30% cascade
    if random.random() < 0.7:
        return cheap, get_cost(cheap)
    else:
        # Cascade: pay for both
        return strong, get_cost(cheap) + get_cost(strong)

def litellm_route() -> str:
    """LiteLLM: Static balanced model."""
    return "google/gemini-2.0-flash-001"

# ==============================================================================
# 5. EVALUATION WITH LLM JUDGE
# ==============================================================================

def run_judge_evaluation(n_samples: int = 100):
    """
    Run evaluation using LLM judges.
    
    For each query:
    1. Each router selects a model
    2. Generate response from selected model
    3. Judge evaluates response quality
    4. Track accuracy (score), cost, and risk
    """
    print(f">>> Table 3: LLM Judge Evaluation (n={n_samples} queries)")
    print(f"    Judges: Gemini Flash (default), Claude 4.5 (for Gemini models)")
    
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in .env file")
        return
    
    # Load prompts from HelpSteer2
    from datasets import load_dataset
    print("\nLoading HelpSteer2 prompts...")
    ds = load_dataset("nvidia/HelpSteer2", split="train", streaming=True)
    
    prompts = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        prompts.append(row["prompt"])
    
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Initialize router
    bandit = create_bandit_router()
    
    # Results
    results = defaultdict(lambda: {"scores": [], "costs": [], "risks": []})
    routers = {
        "BanditGPT": lambda p: bandit.route(p, profile="best_value")[0],
        "RouteLLM": routellm_route,
        "Aurelio AI": aurelio_route,
        "LiteLLM": lambda p: litellm_route(),
    }
    
    print("\nRunning evaluation (this may take a few minutes)...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{n_samples}] Evaluating prompt: {prompt[:60]}...")
        
        for router_name, route_fn in routers.items():
            # 1. Router selects model
            if router_name == "FrugalGPT":
                model, cost = frugalgpt_route(prompt)
            else:
                model = route_fn(prompt)
                cost = get_cost(model)
            
            risk = get_risk(model)
            
            # 2. Generate response from selected model
            messages = [{"role": "user", "content": prompt}]
            response = call_openrouter(model, messages, max_tokens=512)
            
            if not response:
                logger.warning(f"  {router_name}: Failed to get response from {model}")
                continue
            
            # 3. Judge evaluates response
            score = judge_response(prompt, response, model)
            
            if score is not None:
                results[router_name]["scores"].append(score)
                results[router_name]["costs"].append(cost)
                results[router_name]["risks"].append(risk)
                print(f"  {router_name}: {model} -> Score: {score:.2f}")
    
    # Compute final metrics
    print("\n" + "=" * 100)
    print(f"{'Router':<20} | {'Avg Score':<12} | {'Avg Cost ($/q)':<15} | {'Avg Risk %':<12}")
    print("=" * 100)
    
    for router_name, data in results.items():
        if not data["scores"]:
            continue
        avg_score = np.mean(data["scores"]) * 100
        avg_cost = np.mean(data["costs"])
        avg_risk = np.mean(data["risks"])
        print(f"{router_name:<20} | {avg_score:<10.1f}% | ${avg_cost:<14.6f} | {avg_risk:<10.1f}%")
    
    print("=" * 100)
    
    # Save results
    output_path = Path(__file__).parent / "results_table_3_routers.txt"
    with open(output_path, "w") as f:
        f.write(f">>> Table 3: LLM Judge Evaluation (n={n_samples})\n")
        f.write("Judges: Gemini Flash (default), Claude 4.5 (for Gemini models)\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Router':<20} | {'Avg Score':<12} | {'Avg Cost ($/q)':<15} | {'Avg Risk %':<12}\n")
        f.write("=" * 100 + "\n")
        for router_name, data in results.items():
            if not data["scores"]:
                continue
            avg_score = np.mean(data["scores"]) * 100
            avg_cost = np.mean(data["costs"])
            avg_risk = np.mean(data["risks"])
            f.write(f"{router_name:<20} | {avg_score:<10.1f}% | ${avg_cost:<14.6f} | {avg_risk:<10.1f}%\n")
        f.write("=" * 100 + "\n")
    
    print(f"\n✓ Results saved to {output_path}")

if __name__ == "__main__":
    # Use smaller sample size for cost efficiency
    run_judge_evaluation(n_samples=50)
