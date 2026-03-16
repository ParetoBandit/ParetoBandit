#!/usr/bin/env python3
"""
End-to-end live test for pareto_bandit.

Demonstrates the full feedback loop with real LLM APIs:
    route prompt → call model → judge response → update router

Supports multiple providers via --provider flag:
    openrouter (default), openai, anthropic, gemini, ollama

Requires:
    pip install paretobandit[full]          # or a single provider extra

Usage:
    python scripts/test_e2e_live.py --provider openrouter --api-key sk-or-...
    python scripts/test_e2e_live.py --provider openai --api-key sk-...
    python scripts/test_e2e_live.py --provider ollama  # no key needed
    python scripts/test_e2e_live.py --rounds 10        # more learning signal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompts — a diverse mix so the router has real specialization to learn
# ---------------------------------------------------------------------------
PROMPTS = [
    {
        "text": "Write a Python function that finds the longest palindromic substring in a string. Include type hints and a docstring.",
        "domain": "coding",
    },
    {
        "text": "Prove that the square root of 2 is irrational. Be rigorous.",
        "domain": "math",
    },
    {
        "text": "Write a short, evocative poem (8-12 lines) about the feeling of waking up in an unfamiliar city.",
        "domain": "creative",
    },
    {
        "text": "Explain the difference between TCP and UDP to a 10-year-old. Use an analogy.",
        "domain": "explain",
    },
    {
        "text": "A farmer has 100 meters of fencing. What dimensions of a rectangular pen maximize the enclosed area? Show your work step by step.",
        "domain": "math",
    },
    {
        "text": "Implement a thread-safe LRU cache in Python with O(1) get and put. No external libraries.",
        "domain": "coding",
    },
    {
        "text": "Summarize the key ideas of Daniel Kahneman's Thinking, Fast and Slow in exactly 5 bullet points.",
        "domain": "knowledge",
    },
    {
        "text": "Write a SQL query to find the top 3 customers by total spend in the last 90 days, given tables `orders(id, customer_id, total, created_at)` and `customers(id, name)`.",
        "domain": "coding",
    },
    {
        "text": "What are three non-obvious risks of using microservices architecture? For each, suggest a mitigation strategy.",
        "domain": "reasoning",
    },
    {
        "text": "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning carefully.",
        "domain": "math",
    },
]

# ---------------------------------------------------------------------------
# Judge prompt — asks the LLM to return a structured score
# ---------------------------------------------------------------------------
JUDGE_TEMPLATE = textwrap.dedent("""\
    You are an expert evaluator. Rate the following response to the given prompt.

    ## Prompt
    {prompt}

    ## Response
    {response}

    ## Scoring criteria
    - Correctness: Is the answer factually/logically right?
    - Completeness: Does it address all parts of the prompt?
    - Clarity: Is it well-structured and easy to follow?
    - Quality: Code should run, math should be rigorous, writing should be polished.

    Return ONLY a JSON object (no markdown fences):
    {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
""")


_ENV_KEY_MAP = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": None,
}


def resolve_api_key(cli_key: str | None, provider: str) -> str | None:
    """Resolve the API key for *provider* from CLI arg, .env, or env var."""
    if cli_key:
        return cli_key
    if provider == "ollama":
        return None

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    env_var = _ENV_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")
    key = os.environ.get(env_var, "") if env_var else ""
    if not key:
        print(f"Error: No API key found for provider '{provider}'.\n")
        print("Provide it any of these ways:")
        print(f"  1. --api-key <key>")
        print(f"  2. echo '{env_var}=<key>' >> .env")
        print(f"  3. export {env_var}=<key>")
        sys.exit(1)
    return key


def build_client(provider: str, api_key: str | None):
    """Instantiate the appropriate LLMClient adapter for *provider*."""
    from pareto_bandit.providers import (
        OpenRouterClient, OpenAIClient, AnthropicClient,
        GeminiClient, OllamaClient,
    )

    builders = {
        "openrouter": lambda: OpenRouterClient(api_key=api_key),
        "openai": lambda: OpenAIClient(api_key=api_key),
        "anthropic": lambda: AnthropicClient(api_key=api_key),
        "gemini": lambda: GeminiClient(api_key=api_key),
        "ollama": lambda: OllamaClient(),
    }
    factory = builders.get(provider)
    if factory is None:
        print(f"Error: Unknown provider '{provider}'.")
        print(f"  Supported: {', '.join(builders)}")
        sys.exit(1)
    return factory()


def call_model(client, model_id: str, prompt: str, max_tokens: int = 600) -> str | None:
    """Call a model via the provider client and return the response text."""
    try:
        return client.complete(
            model_id,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
    except Exception as e:
        print(f"    API error ({model_id}): {e}")
        return None


def judge_response(client, judge_model: str, prompt: str, response: str) -> tuple[float, str]:
    """Use an LLM judge to score a response. Returns (score, reason)."""
    judge_prompt = JUDGE_TEMPLATE.format(prompt=prompt, response=response)
    try:
        raw = client.complete(
            judge_model,
            [{"role": "user", "content": judge_prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.5))
        reason = parsed.get("reason", "")
        return max(0.0, min(1.0, score)), reason
    except Exception as e:
        print(f"    Judge parse error: {e}")
        return 0.5, "judge-error"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="pareto_bandit live E2E test")
    parser.add_argument("--provider", type=str, default="openrouter",
                        choices=list(_ENV_KEY_MAP),
                        help="LLM provider (default: openrouter)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (overrides .env and env var)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of learning rounds (each round uses all prompts)")
    parser.add_argument("--judge", type=str, default="openai/gpt-4o",
                        help="Model ID for judging responses")
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key, args.provider)
    client = build_client(args.provider, api_key)

    # Load the real model registry shipped with the package
    from pareto_bandit import BanditRouter
    config_path = Path(__file__).resolve().parent.parent / "src" / "pareto_bandit" / "config" / "models.json"
    with open(config_path) as f:
        models_data = json.load(f)
    registry = {m["model_id"]: m for m in models_data["models"]}
    model_names = sorted(registry.keys())

    print("=" * 70)
    print("  pareto_bandit  end-to-end live test")
    print("=" * 70)
    print(f"\n  Models:  {', '.join(m.split('/')[-1] for m in model_names)}")
    print(f"  Judge:   {args.judge}")
    print(f"  Rounds:  {args.rounds}")
    print(f"  Prompts: {len(PROMPTS)} per round")
    print()

    router = BanditRouter.create(
        model_registry=registry,
        priors="none",
        alpha=0.3,  # moderate exploration for a short run
    )

    # Track results per model
    history: list[dict] = []
    model_stats: dict[str, dict] = {m: {"calls": 0, "total_reward": 0.0} for m in model_names}

    for round_num in range(1, args.rounds + 1):
        print(f"--- Round {round_num}/{args.rounds} {'─' * 45}")
        for prompt_info in PROMPTS:
            prompt_text = prompt_info["text"]
            domain = prompt_info["domain"]
            short = prompt_text[:60] + "..."

            # 1. Route
            model_id, log = router.route(prompt_text)
            print(f"\n  [{domain:>10}] {short}")
            print(f"  → Routed to: {model_id}")

            # 2. Call model
            t0 = time.time()
            response = call_model(client, model_id, prompt_text)
            latency = time.time() - t0
            if response is None:
                print(f"    Skipped (API failure)")
                continue
            print(f"    Response: {len(response)} chars, {latency:.1f}s")

            # 3. Judge
            score, reason = judge_response(client, args.judge, prompt_text, response)
            print(f"    Score: {score:.2f}  ({reason})")

            # 4. Update router (close the feedback loop)
            router.process_feedback(log.request_id, score)

            # Track stats
            model_stats[model_id]["calls"] += 1
            model_stats[model_id]["total_reward"] += score
            history.append({
                "round": round_num,
                "domain": domain,
                "model": model_id,
                "score": score,
                "latency": latency,
            })

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    print(f"\n  {'Model':<35} {'Calls':>6} {'Avg Score':>10} {'Avg $/call':>10}")
    print(f"  {'─' * 35} {'─' * 6} {'─' * 10} {'─' * 10}")
    for m in model_names:
        s = model_stats[m]
        if s["calls"] == 0:
            print(f"  {m:<35} {'0':>6} {'—':>10} {'—':>10}")
            continue
        avg_score = s["total_reward"] / s["calls"]
        cost_per_m = registry[m].get("price_1m_blended", 0)
        est_cost = cost_per_m * 600 / 1_000_000  # ~600 tokens/call
        print(f"  {m:<35} {s['calls']:>6} {avg_score:>10.3f} {est_cost:>9.6f}")

    # Show how selections evolved across rounds
    print(f"\n  Selection distribution by round:")
    for r in range(1, args.rounds + 1):
        round_entries = [h for h in history if h["round"] == r]
        counts = {}
        for h in round_entries:
            short_name = h["model"].split("/")[-1]
            counts[short_name] = counts.get(short_name, 0) + 1
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        avg = sum(h["score"] for h in round_entries) / max(len(round_entries), 1)
        print(f"    Round {r}: {dist}  (avg score: {avg:.3f})")

    total_cost_estimate = sum(
        registry[h["model"]].get("price_1m_blended", 0) * 600 / 1_000_000
        for h in history
    )
    judge_cost = len(history) * 2.5 * 200 / 1_000_000  # ~200 judge tokens at $2.5/M
    print(f"\n  Estimated total cost: ${total_cost_estimate + judge_cost:.4f}")
    print(f"  Total API calls: {len(history)} model + {len(history)} judge = {2 * len(history)}")
    print()


if __name__ == "__main__":
    main()
