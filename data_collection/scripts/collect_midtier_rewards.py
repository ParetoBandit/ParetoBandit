#!/usr/bin/env python3
"""Collect rewards for candidate mid-tier models on the pareto prompt set.

Generates responses via OpenRouter and judges with DeepSeek-R1 only,
matching the judge configuration used for the original pareto dataset
(``build_router_pareto_dataset.py``).

Candidate models
----------------
1. ``microsoft/phi-4`` — 14B parameter, expected reward ~0.86
2. ``meta-llama/llama-3.1-70b-instruct`` — 70B, expected ~0.88
3. ``google/gemma-3-27b-it`` — 27B, expected ~0.86

The goal is to find a model with clear separation from both
Llama-3.1-8B (~0.80) and Gemini-2.5-Pro (~0.94) for K=3 experiments.

Resume support: already-completed (prompt, model_id) pairs in the
output file are skipped on re-run.

Usage
-----
    # All three candidates (default):
    python data_collection/scripts/collect_midtier_rewards.py

    # Single candidate:
    python data_collection/scripts/collect_midtier_rewards.py --model microsoft/phi-4

    # Quick test (10 prompts):
    python data_collection/scripts/collect_midtier_rewards.py --limit 10

    # More parallel workers:
    python data_collection/scripts/collect_midtier_rewards.py --workers 15

Requirements
------------
    export OPENROUTER_API_KEY=...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_collection.scripts.rejudge_cot import CoTRewardGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PARETO_CLASSIFIED = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_classified.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "data_collection" / "midtier_candidates"

CANDIDATE_MODELS = [
    "microsoft/phi-4",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemma-3-27b-it",
]


def load_pareto_prompts(path: Path = PARETO_CLASSIFIED) -> List[str]:
    """Load unique prompts from the pareto classified dataset.

    Args:
        path: Path to the pareto_classified.jsonl file.

    Returns:
        List of prompt strings in file order.
    """
    prompts: List[str] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["prompt"])
    return prompts


def collect_rewards_for_model(
    model_id: str,
    prompts: List[str],
    output_path: Path,
    *,
    workers: int = 10,
    limit: Optional[int] = None,
) -> None:
    """Collect rewards for a single model using R1-only judge.

    Args:
        model_id: OpenRouter model identifier.
        prompts: List of prompt strings.
        output_path: JSONL output file (supports resume).
        workers: Thread pool size for parallel API calls.
        limit: If set, only process the first *limit* prompts.
    """
    if limit is not None:
        prompts = prompts[:limit]

    gen = CoTRewardGenerator(max_workers=workers)
    gen.judge_panel = ["deepseek/deepseek-r1"]

    completed: Set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("ok"):
                        completed.add(entry["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if completed:
            logger.info("  Resuming: %d already completed, skipping.", len(completed))

    remaining = [p for p in prompts if p not in completed]
    logger.info(
        "  Tasks: %d remaining out of %d total",
        len(remaining), len(prompts),
    )

    if not remaining:
        logger.info("  Nothing to do for %s.", model_id)
        return

    tasks = [(p, model_id) for p in remaining]
    completed_count = 0

    with open(output_path, "a") as outfile:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(gen.process_task, t): t for t in tasks
            }
            with tqdm(
                total=len(tasks),
                desc=f"  {model_id.split('/')[-1]}",
            ) as pbar:
                for fut in as_completed(futures):
                    res = fut.result()
                    with gen.lock:
                        outfile.write(json.dumps(res) + "\n")
                        outfile.flush()
                        completed_count += 1
                    pbar.update(1)

    logger.info("  Done: %d records written to %s", completed_count, output_path.name)


def summarize_results(output_dir: Path, models: List[str]) -> None:
    """Print a comparison table of candidate model rewards.

    Args:
        output_dir: Directory containing per-model reward JSONL files.
        models: List of model identifiers to summarize.
    """
    ref_rewards: Dict[str, float] = {}
    with open(PARETO_CLASSIFIED) as f:
        llama_r, gemini_r = [], []
        for line in f:
            r = json.loads(line)
            arms = r.get("arms", {})
            llama_info = arms.get("meta-llama/llama-3.1-8b-instruct", {})
            gemini_info = arms.get("google/gemini-2.5-pro", {})
            if "reward" in llama_info:
                llama_r.append(llama_info["reward"])
            if "reward" in gemini_info:
                gemini_r.append(gemini_info["reward"])
    ref_rewards["meta-llama/llama-3.1-8b-instruct"] = (
        sum(llama_r) / len(llama_r) if llama_r else 0.0
    )
    ref_rewards["google/gemini-2.5-pro"] = (
        sum(gemini_r) / len(gemini_r) if gemini_r else 0.0
    )

    print("\n" + "=" * 75)
    print("MID-TIER CANDIDATE COMPARISON")
    print("=" * 75)
    llama_mean = ref_rewards["meta-llama/llama-3.1-8b-instruct"]
    gemini_mean = ref_rewards["google/gemini-2.5-pro"]
    print(f"  Reference: Llama-8B = {llama_mean:.4f}, Gemini-Pro = {gemini_mean:.4f}")
    print(f"  Gap to fill: {gemini_mean - llama_mean:.4f}")
    print()
    print(
        f"  {'Model':<40s} {'N':>6s} {'Mean_R':>7s} "
        f"{'Δ_Llama':>8s} {'Δ_Gemini':>9s} {'Mid%':>6s}"
    )
    print("  " + "-" * 73)

    for model_id in models:
        slug = model_id.replace("/", "_")
        path = output_dir / f"{slug}_rewards.jsonl"
        if not path.exists():
            print(f"  {model_id:<40s}  [no data]")
            continue

        rewards = []
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("ok") and "raw_score" in entry:
                        rewards.append(entry["raw_score"])
                except (json.JSONDecodeError, KeyError):
                    continue

        if not rewards:
            print(f"  {model_id:<40s}  [no valid rewards]")
            continue

        import numpy as np

        mean_r = float(np.mean(rewards))
        delta_llama = mean_r - llama_mean
        delta_gemini = gemini_mean - mean_r
        midpoint_pct = (
            (mean_r - llama_mean) / (gemini_mean - llama_mean) * 100
            if gemini_mean > llama_mean
            else 0.0
        )
        print(
            f"  {model_id:<40s} {len(rewards):6d} {mean_r:7.4f} "
            f"{delta_llama:+8.4f} {-delta_gemini:+9.4f} {midpoint_pct:5.1f}%"
        )

    print()
    print("  Mid% = position between Llama (0%) and Gemini (100%).")
    print("  Ideal mid-tier: 30-60% with clear gaps on both sides.")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run a single model (default: all three candidates).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of prompts (for testing).",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Parallel workers for API calls (default: 10).",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip collection, just print summary of existing results.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else CANDIDATE_MODELS

    if not args.summary_only:
        logger.info("Loading pareto prompts from %s", PARETO_CLASSIFIED)
        prompts = load_pareto_prompts()
        logger.info("  %d prompts loaded", len(prompts))

        for model_id in models:
            slug = model_id.replace("/", "_")
            output_path = OUTPUT_DIR / f"{slug}_rewards.jsonl"
            logger.info(
                "\nCollecting rewards for %s → %s",
                model_id, output_path.name,
            )
            collect_rewards_for_model(
                model_id, prompts, output_path,
                workers=args.workers,
                limit=args.limit,
            )

    summarize_results(OUTPUT_DIR, models)


if __name__ == "__main__":
    main()
