#!/usr/bin/env python3
"""
Demo: Compare Quality/Verbosity predictions across different models.

Calls a small model vs a frontier model via OpenRouter and grades their responses.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import numpy as np

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import torch


def call_openrouter(model_id: str, prompt: str, max_tokens: int = 500) -> str:
    """Call a model via OpenRouter API."""
    from openai import OpenAI
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Reasoning models need max_completion_tokens
    reasoning_patterns = ['/o1', '/o3', '/gpt-5', '/deepseek-r1']
    is_reasoning = any(p in model_id.lower() for p in reasoning_patterns)
    
    try:
        if is_reasoning:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
        else:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        
        content = response.choices[0].message.content or ""
        return content.strip()
    
    except Exception as e:
        return f"[ERROR: {e}]"


def run_demo():
    """Run demo comparing models."""
    from llm_jury.neural_routing.quality_cost_predictor import (
        QualityCostPredictor, 
        get_device
    )
    
    # Model configurations (small vs frontier)
    models = [
        # NOTE: The ':free' endpoint is frequently unavailable. Use a small paid model.
        # Pulled from data/models_cache.json: openrouter_id="openai/gpt-4o-mini"
        ("openai/gpt-4o-mini", "GPT-4o mini (Small, Paid)"),
        ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash"),
        ("google/gemini-3-pro-preview", "Gemini 3 Pro Preview"),
        ("anthropic/claude-sonnet-4", "Claude Sonnet 4"),
        ("anthropic/claude-opus-4.5", "Claude Opus 4.5"),
        ("openai/gpt-5.2", "OpenAI GPT-5.2"),
        # Additional models (from data/models_cache.json)
        ("mistralai/mixtral-8x22b-instruct", "Mixtral 8x22B Instruct"),
        ("mistralai/mistral-7b-instruct", "Mistral 7B Instruct"),
        ("mistralai/ministral-3b", "Ministral 3B"),
        ("meta-llama/llama-3.2-3b-instruct", "Llama 3.2 3B Instruct"),
        ("x-ai/grok-4", "Grok 4"),
        ("x-ai/grok-3-mini", "Grok 3 Mini"),
        # Note: requested id had a typo ("eepseek..."). Cache canonical OpenRouter id:
        ("deepseek/deepseek-r1-0528-qwen3-8b", "DeepSeek R1 (Qwen3 8B)"),
        ("moonshotai/kimi-k2-0905:exacto", "Kimi K2 0905 (Exacto)"),
    ]

    # Some providers/models frequently return empty content at low token budgets.
    # Give them a larger completion budget by default.
    model_max_tokens = {
        # Gemini 3 often needs a larger budget to produce non-empty output.
        "google/gemini-3-pro-preview": 4000,
        # Reasoning-ish / long-form models can need more than 400.
        "deepseek/deepseek-r1-0528-qwen3-8b": 2000,
        "anthropic/claude-opus-4.5": 1200,
        "x-ai/grok-4": 1200,
        # New OpenRouter model id (not yet in local cache)
        "openai/gpt-5.2": 1200,
        # Safe default for everything else
        "__default__": 600,
    }
    
    # Challenging prompts to test quality differences
    test_prompts = [
        # Math reasoning
        (
            "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? "
            "Think step by step and explain your reasoning.",
            "Math riddle"
        ),
        # Coding
        (
            "Write a Python function to check if a string is a valid palindrome, "
            "considering only alphanumeric characters and ignoring case. Include docstring and examples.",
            "Coding task"
        ),
        # Complex reasoning
        (
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines "
            "to make 100 widgets? Explain your reasoning carefully.",
            "Logic puzzle"
        ),
        # Knowledge + explanation
        (
            "Explain why the sky appears blue during the day but red/orange during sunset. "
            "Be concise but technically accurate.",
            "Science explanation"
        ),
    ]
    
    # Load the trained quality predictor
    model_path = PROJECT_ROOT / "data" / "quality_predictor" / "best_quality_predictor.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Run: python -m llm_jury.neural_routing.quality_cost_predictor --epochs 3")
        return
    
    print("Loading Quality/Cost Predictor...")
    grader = QualityCostPredictor.load(model_path)
    grader.eval()

    # NOTE:
    # - For production bandit learning, you typically use a persistent online normalizer.
    # - For this demo (cross-model comparison), a *batch z-score* is more interpretable
    #   because it's not order-dependent.
    def _batch_z(x: float, mean: float, std: float, clamp: float = 3.0) -> float:
        z = (float(x) - float(mean)) / (float(std) + 1e-9)
        return float(max(min(z, clamp), -clamp))

    # Online z-score (order-dependent) used ONLY to show the signal at inference time.
    # This mirrors what you'd feed to the bandit in production (with persisted state).
    from llm_jury.neural_routing.quality_cost_predictor import RunningZScoreNormalizer
    reward_norm_online = RunningZScoreNormalizer(
        mean_init=0.65,
        std_init=0.05,
        alpha=0.01,
        clamp=3.0,
        auto_init_from_first_sample=True,  # avoid misleading all-negative early runs
    )
    
    print("\n" + "="*80)
    print("QUALITY GRADER DEMO: Comparing Models via OpenRouter")
    print("="*80)
    
    # Store all results
    all_results: List[Dict] = []
    
    for prompt, task_name in test_prompts:
        print(f"\n{'─'*80}")
        print(f"📝 Task: {task_name}")
        print(f"   Prompt: {prompt[:80]}...")
        print("─"*80)
        
        for model_id, model_name in models:
            print(f"\n🤖 {model_name} ({model_id})")
            
            # Call the model
            print("   Calling OpenRouter...", end=" ", flush=True)
            max_tokens = int(model_max_tokens.get(model_id, model_max_tokens["__default__"]))
            response = call_openrouter(model_id, prompt, max_tokens=max_tokens)

            # Retry once with a larger budget if we got an empty response.
            if response is not None and len(str(response).strip()) == 0:
                retry_tokens = max(max_tokens * 2, 1200)
                response = call_openrouter(model_id, prompt, max_tokens=retry_tokens)
            
            if response.startswith("[ERROR"):
                print(f"\n   ⚠️  {response}")
                continue
            
            print(f"✓ ({len(response)} chars)")
            
            # Grade the response
            scores = grader.predict(prompt, response)
            # Online z-score for "at this point" display (what bandit would consume).
            prod = grader.predict_production(prompt, response, reward_normalizer=reward_norm_online)
            
            # Store result
            result = {
                'task': task_name,
                'model': model_name,
                'quality': scores['quality'],
                'verbosity': scores['verbosity'],
                # Production fields for async bandit update
                'p_correct_raw': prod['p_correct_raw'],
                'p_correct_clipped': prod['p_correct_clipped'],
                'competence_risk': prod['competence_risk'],
                'route_to_strong': prod['route_to_strong'],
                'reward_raw': prod['reward_raw'],
                'reward_logit': prod.get('reward_logit'),
                # Online z (order-dependent): what bandit would consume online.
                'reward_z_online': prod.get('reward_z'),
                # Batch z (order-independent): computed after collecting all results.
                'reward_z': None,
                'response_len': len(response),
            }
            all_results.append(result)
            
            # Display
            print(f"   Response: {response[:150]}...")
            print(f"   ────────────────────────────────────")
            print(f"   📊 Quality:   {scores['quality']:.3f}")
            print(f"   📊 Verbosity: {scores['verbosity']:.3f}")
            if prod["routing_p_correct_threshold"] is None:
                route_str = "N/A (no routing threshold set)"
            else:
                route_str = "ROUTE → strong model" if prod["route_to_strong"] else "KEEP → cheap model"
            print(f"   🎯 P_correct: {prod['p_correct_raw']:.3f} | competence_risk={prod['competence_risk']:.3f}")
            print(f"   🧭 Routing:   {route_str} (T={prod['routing_p_correct_threshold']})")
            z_online = prod.get("reward_z")
            z_online_str = "N/A" if z_online is None else f"{float(z_online):.3f}"
            logit_val = prod.get("reward_logit")
            logit_str = "N/A" if logit_val is None else f"{float(logit_val):.3f}"
            print(f"   🎯 Reward:    raw={prod['reward_raw']:.3f} | logit={logit_str} | z_online={z_online_str}")
            print(f"   📊 Length:    {len(response)} chars")
    
    # Compute batch z-score across all collected rewards (shared scale).
    if all_results:
        raw_vals = np.array([r["reward_raw"] for r in all_results], dtype=np.float64)
        mu = float(np.mean(raw_vals))
        sigma = float(np.std(raw_vals) + 1e-9)
        for r in all_results:
            r["reward_z"] = _batch_z(r["reward_raw"], mu, sigma, clamp=3.0)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Quality & Verbosity Scores by Model")
    print("="*80)
    
    # Group by model
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r['model']].append(r)
    
    print(f"\n{'Model':<25} {'Avg Quality':>12} {'Avg Verbosity':>14} {'Avg Length':>12} {'Avg r(raw)':>12} {'Avg r(logit)':>13} {'Avg r(z)':>10}")
    print("-"*116)
    
    for model_name, results in by_model.items():
        avg_q = sum(r['quality'] for r in results) / len(results)
        avg_v = sum(r['verbosity'] for r in results) / len(results)
        avg_len = sum(r['response_len'] for r in results) / len(results)
        avg_r_raw = sum(r['reward_raw'] for r in results) / len(results)
        logit_vals = [r['reward_logit'] for r in results if r.get('reward_logit') is not None]
        avg_r_logit = (sum(logit_vals) / len(logit_vals)) if logit_vals else float('nan')
        z_vals = [r['reward_z'] for r in results if r.get('reward_z') is not None]
        avg_r_z = (sum(z_vals) / len(z_vals)) if z_vals else float('nan')
        avg_r_z_str = "N/A" if np.isnan(avg_r_z) else f"{avg_r_z:>10.3f}"
        avg_r_logit_str = "N/A" if np.isnan(avg_r_logit) else f"{avg_r_logit:>13.3f}"
        print(f"{model_name:<25} {avg_q:>12.3f} {avg_v:>14.3f} {avg_len:>12.0f} {avg_r_raw:>12.3f} {avg_r_logit_str} {avg_r_z_str}")
    
    # Per-task breakdown
    print("\n" + "-"*80)
    print("Per-Task Quality Scores:")
    print("-"*80)
    
    tasks = list(dict.fromkeys(r['task'] for r in all_results))
    model_names = list(by_model.keys())
    
    print(f"\n{'Task':<20}", end="")
    for m in model_names:
        print(f"{m[:15]:>16}", end="")
    print()
    print("-"*80)
    
    for task in tasks:
        print(f"{task:<20}", end="")
        for model_name in model_names:
            task_results = [r for r in all_results if r['task'] == task and r['model'] == model_name]
            if task_results:
                print(f"{task_results[0]['quality']:>16.3f}", end="")
            else:
                print(f"{'N/A':>16}", end="")
        print()

    # Per-task reward (batch z-score) breakdown
    print("\n" + "-"*80)
    print("Per-Task Reward Z-Scores (batch-normalized, clipped to [-3, 3]):")
    print("-"*80)

    print(f"\n{'Task':<20}", end="")
    for m in model_names:
        print(f"{m[:15]:>16}", end="")
    print()
    print("-"*80)

    for task in tasks:
        print(f"{task:<20}", end="")
        for model_name in model_names:
            task_results = [r for r in all_results if r['task'] == task and r['model'] == model_name]
            if task_results and task_results[0].get("reward_z") is not None:
                print(f"{float(task_results[0]['reward_z']):>16.3f}", end="")
            else:
                print(f"{'N/A':>16}", end="")
        print()
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    run_demo()
