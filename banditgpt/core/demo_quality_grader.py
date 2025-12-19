#!/usr/bin/env python3
"""
Demo: Compare Quality/Verbosity predictions across different models.

This demo calls models via OpenRouter and grades their responses.

Moved from `banditgpt.neural_routing.demo_quality_grader` to keep all async-bandit
artifacts together under `banditgpt.core`.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import numpy as np

# Load .env from current directory or user home
load_dotenv()  # Will search current dir and parents for .env

import torch


def call_openrouter(model_id: str, prompt: str, max_tokens: int = 500, *, timeout_s: float = 60.0) -> str:
    """Call a model via OpenRouter API."""
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=float(timeout_s))

    # Reasoning models need max_completion_tokens
    reasoning_patterns = ["/o1", "/o3", "/gpt-5", "/deepseek-r1"]
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
    from banditgpt.core.quality_cost_predictor import QualityCostPredictor, get_device
    from banditgpt.core.tiered_grader import TieredGrader, OpenRouterTeacherVerifier

    parser = argparse.ArgumentParser(description="Demo: grade OpenRouter model outputs")
    parser.add_argument("--use-teacher", action="store_true", help="Enable TieredGrader hard-path teacher verification (costly)")
    parser.add_argument("--teacher-model", type=str, default="openai/gpt-4o", help="OpenRouter id for teacher verifier")
    parser.add_argument("--teacher-max-tokens", type=int, default=64, help="Max tokens for teacher verifier output")
    args, _unknown = parser.parse_known_args()

    # Model configurations (small vs frontier)
    models = [
        ("openai/gpt-4o-mini", "GPT-4o mini (Small, Paid)"),
        ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash"),
        ("google/gemini-3-pro-preview", "Gemini 3 Pro Preview"),
        ("anthropic/claude-sonnet-4", "Claude Sonnet 4"),
        ("anthropic/claude-opus-4.5", "Claude Opus 4.5"),
        ("openai/gpt-5.2", "OpenAI GPT-5.2"),
        ("mistralai/mixtral-8x22b-instruct", "Mixtral 8x22B Instruct"),
        ("mistralai/mistral-7b-instruct", "Mistral 7B Instruct"),
        ("mistralai/ministral-3b", "Ministral 3B"),
        ("meta-llama/llama-3.2-3b-instruct", "Llama 3.2 3B Instruct"),
        ("x-ai/grok-4", "Grok 4"),
        ("x-ai/grok-3-mini", "Grok 3 Mini"),
        ("deepseek/deepseek-r1-0528-qwen3-8b", "DeepSeek R1 (Qwen3 8B)"),
        ("moonshotai/kimi-k2-0905:exacto", "Kimi K2 0905 (Exacto)"),
    ]

    # Max token overrides for models that need larger context windows
    # Gemini 3.0+ and reasoning models often need higher limits to generate responses
    model_max_tokens = {
        # Gemini 3.0 models - require ~4000 tokens to generate responses
        "google/gemini-3-pro-preview": 4000,
        # Gemini 2.5 models - also benefit from higher limits
        "google/gemini-2.5-pro": 4000,
        "google/gemini-2.5-pro-preview-06-05": 4000,
        "google/gemini-2.5-flash-preview-09-2025": 4000,
        "google/gemini-2.5-flash-lite": 2000,
        # Reasoning models
        "deepseek/deepseek-r1-0528-qwen3-8b": 2000,
        # Other frontier models
        "anthropic/claude-opus-4.5": 1200,
        "x-ai/grok-4": 1200,
        "openai/gpt-5.2": 1200,
        "__default__": 600,
    }

    test_prompts = [
        (
            "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? "
            "Think step by step and explain your reasoning.",
            "Math riddle",
        ),
        (
            "Write a Python function to check if a string is a valid palindrome, "
            "considering only alphanumeric characters and ignoring case. Include docstring and examples.",
            "Coding task",
        ),
        (
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines "
            "to make 100 widgets? Explain your reasoning carefully.",
            "Logic puzzle",
        ),
        (
            "Explain why the sky appears blue during the day but red/orange during sunset. "
            "Be concise but technically accurate.",
            "Science explanation",
        ),
    ]

    model_path = PROJECT_ROOT / "data" / "quality_predictor" / "best_quality_predictor.pt"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Run: python -m banditgpt.neural_routing.quality_cost_predictor --epochs 3")
        return

    print("Loading Quality/Cost Predictor...")
    soft_grader = QualityCostPredictor.load(model_path)
    soft_grader.eval()

    if bool(args.use_teacher):
        teacher = OpenRouterTeacherVerifier(model_id=str(args.teacher_model), max_tokens=int(args.teacher_max_tokens))
        grader = TieredGrader(soft_grader=soft_grader, teacher_verifier=teacher)
        print(f"TieredGrader enabled: hard prompts verified by {args.teacher_model}")
    else:
        grader = soft_grader

    def _batch_z(x: float, mean: float, std: float, clamp: float = 3.0) -> float:
        z = (float(x) - float(mean)) / (float(std) + 1e-9)
        return float(max(min(z, clamp), -clamp))

    from banditgpt.core.quality_cost_predictor import RunningZScoreNormalizer

    reward_norm_online = RunningZScoreNormalizer(
        mean_init=0.65,
        std_init=0.05,
        alpha=0.01,
        clamp=3.0,
        auto_init_from_first_sample=True,
    )

    print("\n" + "=" * 80)
    print("QUALITY GRADER DEMO: Comparing Models via OpenRouter")
    print("=" * 80)

    all_results: List[Dict] = []

    for prompt, task_name in test_prompts:
        print(f"\n{'─' * 80}")
        print(f"📝 Task: {task_name}")
        print(f"   Prompt: {prompt[:80]}...")
        print("─" * 80)

        for model_id, model_name in models:
            print(f"\n🤖 {model_name} ({model_id})")
            print("   Calling OpenRouter...", end=" ", flush=True)
            max_tokens = int(model_max_tokens.get(model_id, model_max_tokens["__default__"]))
            response = call_openrouter(model_id, prompt, max_tokens=max_tokens)

            if response is not None and len(str(response).strip()) == 0:
                retry_tokens = max(max_tokens * 2, 1200)
                response = call_openrouter(model_id, prompt, max_tokens=retry_tokens)

            if response.startswith("[ERROR"):
                print(f"\n   ⚠️  {response}")
                continue

            print(f"✓ ({len(response)} chars)")

            scores = soft_grader.predict(prompt, response)
            prod = grader.predict_production(prompt, response, reward_normalizer=reward_norm_online)

            result = {
                "task": task_name,
                "model": model_name,
                "quality": scores["quality"],
                "verbosity": scores["verbosity"],
                "p_correct_raw": prod["p_correct_raw"],
                "p_correct_clipped": prod["p_correct_clipped"],
                "competence_risk": prod["competence_risk"],
                "route_to_strong": prod["route_to_strong"],
                "reward_raw": prod["reward_raw"],
                "reward_logit": prod.get("reward_logit"),
                "reward_z_online": prod.get("reward_z"),
                "tiered_is_hard": prod.get("tiered_is_hard"),
                "tiered_used_teacher": prod.get("tiered_used_teacher"),
                "reward_z": None,
                "response_len": len(response),
            }
            all_results.append(result)

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

    if all_results:
        raw_vals = np.array([r["reward_raw"] for r in all_results], dtype=np.float64)
        mu = float(np.mean(raw_vals))
        sigma = float(np.std(raw_vals) + 1e-9)
        for r in all_results:
            r["reward_z"] = _batch_z(r["reward_raw"], mu, sigma, clamp=3.0)

    print("\n" + "=" * 80)
    print("SUMMARY: Quality & Verbosity Scores by Model")
    print("=" * 80)

    from collections import defaultdict

    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    print(
        f"\n{'Model':<25} {'Avg Quality':>12} {'Avg Verbosity':>14} {'Avg Length':>12} "
        f"{'Avg r(raw)':>12} {'Avg r(logit)':>13} {'Avg r(z)':>10}"
    )
    print("-" * 116)

    for model_name, results in by_model.items():
        avg_q = sum(r["quality"] for r in results) / len(results)
        avg_v = sum(r["verbosity"] for r in results) / len(results)
        avg_len = sum(r["response_len"] for r in results) / len(results)
        avg_r_raw = sum(r["reward_raw"] for r in results) / len(results)
        logit_vals = [r["reward_logit"] for r in results if r.get("reward_logit") is not None]
        avg_r_logit = (sum(logit_vals) / len(logit_vals)) if logit_vals else float("nan")
        z_vals = [r["reward_z"] for r in results if r.get("reward_z") is not None]
        avg_r_z = (sum(z_vals) / len(z_vals)) if z_vals else float("nan")
        avg_r_z_str = "N/A" if np.isnan(avg_r_z) else f"{avg_r_z:>10.3f}"
        avg_r_logit_str = "N/A" if np.isnan(avg_r_logit) else f"{avg_r_logit:>13.3f}"
        print(
            f"{model_name:<25} {avg_q:>12.3f} {avg_v:>14.3f} {avg_len:>12.0f} "
            f"{avg_r_raw:>12.3f} {avg_r_logit_str} {avg_r_z_str}"
        )

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    run_demo()

