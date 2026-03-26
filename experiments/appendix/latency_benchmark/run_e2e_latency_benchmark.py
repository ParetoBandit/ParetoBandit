#!/usr/bin/env python3
"""Appendix: End-to-End Latency Benchmark.

Measures wall-clock latency of the **full production pipeline**:

    prompt (string) → SentenceTransformer encode → PCA + whitening → route()

Each stage is timed independently so the cost breakdown is additive and
auditable.  The total E2E is also measured as a single ``router.route(prompt)``
call for cross-validation.

Stage definitions
-----------------
  1. **embed** — ``feature_service.encode_prompt(prompt)``
     SentenceTransformer (all-MiniLM-L6-v2), CPU, single-threaded.
  2. **pca** — ``pca.transform(emb)`` + whitening + bias append.
     Shipped ``pca_25.joblib`` (384 → 25 + 1 bias = 26-D).
  3. **route** — ``router.route(precomputed_vector)``
     Full production BanditRouter with pre-embedded context vector.
  4. **total** — ``router.route(prompt_string)``
     Full pipeline end-to-end (embed + PCA + route + budget pacing + logging).

A fixed pool of diverse prompts (varying length and domain) is cycled to
keep the benchmark deterministic across runs.

Outputs
-------
  ``results/e2e_latency_results.json`` with per-stage p50 / p95 / p99 in
  **milliseconds** and the full decomposition.

Usage:
    python experiments/appendix/latency_benchmark/run_e2e_latency_benchmark.py
    python experiments/appendix/latency_benchmark/run_e2e_latency_benchmark.py --rounds 500
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit import BanditRouter
from pareto_bandit.feature_service import FeatureService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in (
    "pareto_bandit.router",
    "pareto_bandit.feature_service",
    "pareto_bandit.policy",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"

# ======================================================================
# Representative prompt pool
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
    (
        "Write a short story (100 words) about a robot discovering it can dream."
    ),
    "Convert 72 degrees Fahrenheit to Celsius.",
    (
        "I'm building a recommendation engine for an e-commerce site.  Should I use "
        "collaborative filtering, content-based filtering, or a hybrid approach?  Justify."
    ),
    "Prove that the square root of 2 is irrational.",
]


# ======================================================================
# Timing data class
# ======================================================================

@dataclass
class E2ELatencyResult:
    """Per-stage latency breakdown in milliseconds."""

    embed_p50_ms: float
    embed_p95_ms: float
    embed_p99_ms: float
    pca_p50_ms: float
    pca_p95_ms: float
    pca_p99_ms: float
    route_p50_ms: float
    route_p95_ms: float
    route_p99_ms: float
    total_p50_ms: float
    total_p95_ms: float
    total_p99_ms: float
    sum_stages_p50_ms: float
    rounds_measured: int


def _percentile_ms(values_ns: List[int], q: float) -> float:
    """Convert nanosecond timings to millisecond percentile."""
    return float(np.percentile(np.asarray(values_ns, dtype=np.float64), q) / 1e6)


# ======================================================================
# Benchmark
# ======================================================================

def _build_production_router() -> tuple[BanditRouter, FeatureService]:
    """Construct a production-config BanditRouter with the real encoder + PCA.

    Returns:
        ``(router, feature_service)`` — the feature_service is the same
        object that the router uses internally, exposed for direct stage
        timing.
    """
    fs = FeatureService()
    router = BanditRouter.create(
        feature_service=fs,
        priors="none",
    )
    return router, fs


def run_e2e_benchmark(
    rounds: int,
    warmup: int,
) -> E2ELatencyResult:
    """Run the full embed → PCA → route pipeline benchmark.

    Args:
        rounds: Total measurement iterations (after warmup).
        warmup: Number of un-timed warmup iterations executed first.

    Returns:
        :class:`E2ELatencyResult` with per-stage and total timing.
    """
    router, fs = _build_production_router()

    # Force-load encoder + PCA so the first timed call isn't inflated.
    _warmup_prompt = "warmup"
    _ = fs.extract_features(_warmup_prompt)

    pca_model = fs.pca
    if pca_model is None:
        raise RuntimeError("FeatureService did not load a PCA model; cannot benchmark PCA stage.")

    total_iters = warmup + rounds
    prompts = [_PROMPT_POOL[i % len(_PROMPT_POOL)] for i in range(total_iters)]

    embed_ns: List[int] = []
    pca_ns: List[int] = []
    route_ns: List[int] = []
    total_ns: List[int] = []

    for i in range(total_iters):
        prompt = prompts[i]

        # --- Stage 1: Embedding ---
        t0 = time.perf_counter_ns()
        raw_emb = fs.encode_prompt(prompt)
        t1 = time.perf_counter_ns()

        # --- Stage 2: PCA + whitening + bias ---
        t2 = time.perf_counter_ns()
        pca_emb = pca_model.transform(raw_emb.reshape(1, -1)).flatten()
        pca_emb = fs._apply_pca_whitening(pca_emb)
        context = np.append(pca_emb, 1.0)
        t3 = time.perf_counter_ns()

        # --- Stage 3: route() with pre-embedded vector ---
        t4 = time.perf_counter_ns()
        _model_id, _log = router.route(context)
        t5 = time.perf_counter_ns()

        # --- Stage 4: Total E2E (independent measurement) ---
        t6 = time.perf_counter_ns()
        _model_id2, _log2 = router.route(prompt)
        t7 = time.perf_counter_ns()

        if i >= warmup:
            embed_ns.append(t1 - t0)
            pca_ns.append(t3 - t2)
            route_ns.append(t5 - t4)
            total_ns.append(t7 - t6)

    sum_stages_p50 = (
        _percentile_ms(embed_ns, 50)
        + _percentile_ms(pca_ns, 50)
        + _percentile_ms(route_ns, 50)
    )

    return E2ELatencyResult(
        embed_p50_ms=_percentile_ms(embed_ns, 50),
        embed_p95_ms=_percentile_ms(embed_ns, 95),
        embed_p99_ms=_percentile_ms(embed_ns, 99),
        pca_p50_ms=_percentile_ms(pca_ns, 50),
        pca_p95_ms=_percentile_ms(pca_ns, 95),
        pca_p99_ms=_percentile_ms(pca_ns, 99),
        route_p50_ms=_percentile_ms(route_ns, 50),
        route_p95_ms=_percentile_ms(route_ns, 95),
        route_p99_ms=_percentile_ms(route_ns, 99),
        total_p50_ms=_percentile_ms(total_ns, 50),
        total_p95_ms=_percentile_ms(total_ns, 95),
        total_p99_ms=_percentile_ms(total_ns, 99),
        sum_stages_p50_ms=round(sum_stages_p50, 4),
        rounds_measured=rounds,
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end latency benchmark: embed → PCA → route().",
    )
    parser.add_argument(
        "--rounds", type=int, default=200,
        help="Timed iterations (default 200; ~1.6 s of embedding time).",
    )
    parser.add_argument(
        "--warmup", type=int, default=50,
        help="Un-timed warmup iterations (default 50).",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("End-to-End Latency Benchmark: embed → PCA → route()")
    logger.info("=" * 70)
    logger.info("  Platform : %s %s", platform.machine(), platform.platform())
    logger.info("  Rounds   : %d  |  Warmup: %d", args.rounds, args.warmup)
    logger.info("  Prompts  : pool of %d (cycled)", len(_PROMPT_POOL))
    logger.info("")
    logger.info("Loading SentenceTransformer + PCA …")

    t_wall_start = time.perf_counter()
    result = run_e2e_benchmark(rounds=args.rounds, warmup=args.warmup)
    t_wall = time.perf_counter() - t_wall_start

    logger.info("")
    logger.info("-" * 70)
    logger.info("  Stage breakdown (p50 / p95 / p99):")
    logger.info(
        "    embed   : %8.3f / %8.3f / %8.3f ms",
        result.embed_p50_ms, result.embed_p95_ms, result.embed_p99_ms,
    )
    logger.info(
        "    PCA+wh  : %8.4f / %8.4f / %8.4f ms",
        result.pca_p50_ms, result.pca_p95_ms, result.pca_p99_ms,
    )
    logger.info(
        "    route() : %8.4f / %8.4f / %8.4f ms",
        result.route_p50_ms, result.route_p95_ms, result.route_p99_ms,
    )
    logger.info(
        "    TOTAL   : %8.3f / %8.3f / %8.3f ms",
        result.total_p50_ms, result.total_p95_ms, result.total_p99_ms,
    )
    logger.info("    Σ stages p50: %.3f ms  (vs. total p50: %.3f ms)",
                result.sum_stages_p50_ms, result.total_p50_ms)
    logger.info("-" * 70)

    embed_frac = result.embed_p50_ms / max(result.total_p50_ms, 1e-6) * 100
    route_frac = result.route_p50_ms / max(result.total_p50_ms, 1e-6) * 100
    logger.info("  Embedding accounts for %.1f%% of E2E p50", embed_frac)
    logger.info("  route() accounts for %.1f%% of E2E p50", route_frac)
    logger.info("  Wall time: %.1f s", t_wall)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output: Dict = {
        "platform": f"{platform.machine()} {platform.platform()}",
        "rounds": args.rounds,
        "warmup": args.warmup,
        "prompt_pool_size": len(_PROMPT_POOL),
        "stages": asdict(result),
        "fractions": {
            "embed_pct_of_total_p50": round(embed_frac, 2),
            "route_pct_of_total_p50": round(route_frac, 2),
        },
    }
    out_path = RESULTS_DIR / "e2e_latency_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("\nResults saved to %s", out_path)


if __name__ == "__main__":
    main()
