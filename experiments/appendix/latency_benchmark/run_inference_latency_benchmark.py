#!/usr/bin/env python3
"""Appendix: LLM Inference Latency Benchmark for K=4 Portfolio.

Measures time-to-first-token (TTFT) and total response latency for
the paper's K=4 model portfolio via the OpenRouter streaming API.
Results contextualise ParetoBandit's routing overhead against the
actual inference cost of the models it routes to.

Methodology
-----------
- **Models**: the 4 arms from ``data_collection/config/models_k4.json``.
- **Prompts**: reused from the E2E latency benchmark pool, bucketed
  into short / medium / long by character length.
- **Trials**: ``--trials`` streaming calls per (model, prompt-length)
  combination (default 20).
- **Metrics**: TTFT (first chunk with content) and total latency
  (stream end), both via ``time.perf_counter()``.
- **Aggregation**: mean + 95 % bootstrap CI per configuration.

Usage::

    python experiments/appendix/latency_benchmark/run_inference_latency_benchmark.py
    python experiments/appendix/latency_benchmark/run_inference_latency_benchmark.py --trials 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci_mean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
MODELS_PATH = PROJECT_ROOT / "data_collection" / "config" / "models_k4.json"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ======================================================================
# Prompt pool (reused from run_e2e_latency_benchmark.py)
# ======================================================================

_PROMPT_POOL: List[str] = [
    "Solve the quadratic equation x^2 + 5x + 6 = 0.",
    "Translate the following English text to French: 'The quick brown fox jumps over the lazy dog.'",
    "Write a Python function that checks if a string is a palindrome.",
    "Explain the difference between supervised and unsupervised learning in two paragraphs.",
    (
        "I have a dataset of 10,000 customer records with columns: age, income, "
        "purchase_amount, region, and churn_label.  Suggest a feature engineering "
        "pipeline and a model selection strategy suitable for a churn prediction task."
    ),
    "What is the capital of Japan?",
    "Summarize the key findings of the 2023 IPCC climate report in three bullet points.",
    (
        "Design a REST API for a library management system.  Include endpoints for "
        "books, authors, borrowers, and loans.  Use standard HTTP methods and status codes."
    ),
    "Give me a regex that matches valid email addresses per RFC 5322.",
    "Compare merge sort and quicksort in terms of time complexity, space complexity, and stability.",
    (
        "You are a helpful math tutor.  A student asks: 'Why does 0.1 + 0.2 not equal "
        "0.3 in Python?'  Explain using IEEE 754 floating-point representation."
    ),
    "List five common design patterns in object-oriented programming and when to use each.",
    "Write a SQL query that returns the top 10 customers by total spend in the last 90 days.",
    (
        "Explain how transformers use self-attention, including the roles of Q, K, and V "
        "matrices, positional encoding, and multi-head attention."
    ),
    "What are the main differences between TCP and UDP?  When would you choose one over the other?",
    "Help me debug this error: 'CUDA out of memory. Tried to allocate 2.00 GiB'.",
    "Write a short story (100 words) about a robot discovering it can dream.",
    "Convert 72 degrees Fahrenheit to Celsius.",
    (
        "I'm building a recommendation engine for an e-commerce site.  Should I use "
        "collaborative filtering, content-based filtering, or a hybrid approach?  Justify."
    ),
    "Prove that the square root of 2 is irrational.",
]


def _bucket_prompts(
    prompts: List[str],
) -> Dict[str, List[str]]:
    """Split prompts into short / medium / long buckets by character length.

    Thresholds are chosen to yield roughly balanced buckets from the
    20-prompt pool above.

    Returns:
        Dictionary mapping ``"short"`` / ``"medium"`` / ``"long"`` to
        lists of prompt strings.
    """
    buckets: Dict[str, List[str]] = {"short": [], "medium": [], "long": []}
    for p in prompts:
        n = len(p)
        if n < 80:
            buckets["short"].append(p)
        elif n < 150:
            buckets["medium"].append(p)
        else:
            buckets["long"].append(p)
    return buckets


def _load_models() -> List[Dict[str, Any]]:
    """Load the K=4 model portfolio from the config file."""
    with open(MODELS_PATH) as f:
        data = json.load(f)
    return data["models"]


# ======================================================================
# Streaming latency measurement
# ======================================================================


TIMEOUT_S = 60.0
OUTLIER_CEILING_MS = 30_000.0


def _measure_streaming_latency(
    client: Any,
    model_id: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Tuple[float, float] | None:
    """Send one streaming request and return (TTFT_ms, total_ms).

    Args:
        client: An ``openai.OpenAI`` instance pointed at OpenRouter.
        model_id: OpenRouter model identifier (e.g. ``meta-llama/llama-3.1-8b-instruct``).
        prompt: The user message text.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Tuple of (time-to-first-token in ms, total latency in ms),
        or ``None`` if the request timed out or hit an outlier ceiling.
    """
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    ttft: float | None = None
    for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = (time.perf_counter() - t0) * 1000.0
    total = (time.perf_counter() - t0) * 1000.0

    if ttft is None:
        ttft = total

    if ttft > OUTLIER_CEILING_MS or total > OUTLIER_CEILING_MS:
        return None
    return ttft, total


def _summarize(
    raw_values: List[float],
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Compute mean + bootstrap CI for a list of latency measurements.

    Args:
        raw_values: Per-trial latency values in ms.
        ci_level: Confidence level for the bootstrap CI.

    Returns:
        Dictionary with ``mean``, ``ci_lo``, ``ci_hi``, and ``raw``.
    """
    arr = np.asarray(raw_values, dtype=np.float64)
    mean = float(np.mean(arr))
    ci_lo, ci_hi = bootstrap_ci_mean(arr, ci_level=ci_level)
    return {
        "mean": round(mean, 1),
        "ci_lo": round(ci_lo, 1),
        "ci_hi": round(ci_hi, 1),
        "raw": [round(v, 1) for v in raw_values],
    }


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM inference latency benchmark for K=4 portfolio models.",
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Streaming calls per (model, prompt-length) combination (default 20).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per response (default 512).",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment or .env file.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=0,
        timeout=TIMEOUT_S,
    )

    models = _load_models()
    buckets = _bucket_prompts(_PROMPT_POOL)

    logger.info("=" * 70)
    logger.info("LLM Inference Latency Benchmark (K=4 Portfolio)")
    logger.info("=" * 70)
    logger.info("  Models: %s", [m["display"] for m in models])
    logger.info("  Prompt buckets: %s",
                {k: len(v) for k, v in buckets.items()})
    logger.info("  Trials per config: %d", args.trials)
    logger.info("  Max tokens: %d", args.max_tokens)
    logger.info("")

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model in models:
        model_id = model["model_id"]
        display = model["display"]
        results[model_id] = {}

        for length_label in ("short", "medium", "long"):
            prompt_list = buckets[length_label]
            if not prompt_list:
                logger.warning("  No prompts in bucket '%s', skipping.", length_label)
                continue

            ttft_ms_raw: List[float] = []
            total_ms_raw: List[float] = []

            logger.info("  %s / %s (%d trials) ...", display, length_label, args.trials)

            trial_idx = 0
            attempts = 0
            max_attempts = args.trials * 2
            while trial_idx < args.trials and attempts < max_attempts:
                prompt = prompt_list[trial_idx % len(prompt_list)]
                attempts += 1
                try:
                    result = _measure_streaming_latency(
                        client, model_id, prompt,
                        max_tokens=args.max_tokens,
                    )
                    if result is None:
                        logger.warning(
                            "    trial %2d/%d  DROPPED (outlier/timeout), retrying",
                            trial_idx + 1, args.trials,
                        )
                        continue
                    ttft, total = result
                    ttft_ms_raw.append(ttft)
                    total_ms_raw.append(total)
                    logger.info(
                        "    trial %2d/%d  TTFT=%7.1f ms  total=%8.1f ms",
                        trial_idx + 1, args.trials, ttft, total,
                    )
                    trial_idx += 1
                except Exception:
                    logger.exception(
                        "    trial %d/%d FAILED for %s/%s, retrying",
                        trial_idx + 1, args.trials, display, length_label,
                    )

            if ttft_ms_raw:
                results[model_id][length_label] = {
                    "ttft_ms": _summarize(ttft_ms_raw),
                    "total_ms": _summarize(total_ms_raw),
                }
                logger.info(
                    "    => TTFT mean=%.1f ms [%.1f, %.1f]  "
                    "total mean=%.1f ms [%.1f, %.1f]",
                    results[model_id][length_label]["ttft_ms"]["mean"],
                    results[model_id][length_label]["ttft_ms"]["ci_lo"],
                    results[model_id][length_label]["ttft_ms"]["ci_hi"],
                    results[model_id][length_label]["total_ms"]["mean"],
                    results[model_id][length_label]["total_ms"]["ci_lo"],
                    results[model_id][length_label]["total_ms"]["ci_hi"],
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "models": [m["model_id"] for m in models],
        "model_display": {m["model_id"]: m["display"] for m in models},
        "prompt_lengths": ["short", "medium", "long"],
        "prompt_counts": {k: len(v) for k, v in buckets.items()},
        "trials_per_config": args.trials,
        "max_tokens": args.max_tokens,
        "results": results,
    }
    out_path = RESULTS_DIR / "inference_latency_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("\nResults saved to %s", out_path)


if __name__ == "__main__":
    main()
