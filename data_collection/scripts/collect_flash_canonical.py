#!/usr/bin/env python3
"""Collect Gemini-2.5-Flash rewards using the canonical single-judge setup.

Generates flash responses (or reuses cached ones from v4/v5 data) and
judges them with **DeepSeek-R1 as the sole judge** using the v3 continuous
rubric — identical to the ``build_router_pareto_dataset.py`` pipeline that
produced the canonical K=3 train/val/test splits.  This ensures flash
rewards are on an identical scale to the existing Llama / Mistral /
Gemini-Pro arms.

After collection, run ``merge_flash_into_splits.py`` to build K=4
val/test splits for the model-onboarding experiment.

Usage
-----
    # Collect flash rewards for all val+test prompts missing flash data
    python data_collection/scripts/collect_flash_canonical.py

    # Quick test (5 prompts)
    python data_collection/scripts/collect_flash_canonical.py --limit 5

    # Resume interrupted run
    python data_collection/scripts/collect_flash_canonical.py --resume

    # Print summary of collected data
    python data_collection/scripts/collect_flash_canonical.py --summary-only

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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit.config import (
    K4_MODELS_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    HOLDOUT_DATA_PATH,
)
from data_collection.scripts.rejudge_cot import CoTRewardGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FLASH_ID = "google/gemini-2.5-flash"
OUTPUT_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "flash_canonical"
OUTPUT_FILE = OUTPUT_DIR / "gemini_flash_v3.jsonl"

V4_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "v4_metacognitive"
V5_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "v5_dual_judge"
PARETO_REWARDS = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_rewards.jsonl"
)
MIDTIER_DIR = PROJECT_ROOT / "data_collection" / "midtier_candidates"


def load_flash_response_cache() -> Dict[str, str]:
    """Load cached flash responses from all known data sources.

    Reusing cached responses avoids re-generating them (saves API cost
    and ensures deterministic responses for prompts that appeared in
    v4/v5 collections).

    Returns:
        Mapping ``{prompt_text: response_text}``.
    """
    cache: Dict[str, str] = {}
    sources = []

    for v_dir in [V4_DIR, V5_DIR]:
        for p in sorted(v_dir.glob("*flash*.jsonl")):
            sources.append(p)

    if PARETO_REWARDS.exists():
        sources.append(PARETO_REWARDS)

    if MIDTIER_DIR.is_dir():
        for p in sorted(MIDTIER_DIR.glob("*.jsonl")):
            sources.append(p)

    for source_path in sources:
        n_loaded = 0
        with open(source_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("model_id") != FLASH_ID:
                        continue
                    if not rec.get("ok") or not rec.get("response"):
                        continue
                    prompt = rec["prompt"]
                    if prompt not in cache:
                        cache[prompt] = rec["response"]
                        n_loaded += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        if n_loaded > 0:
            logger.info("  Loaded %d flash responses from %s", n_loaded, source_path.name)

    logger.info("Total cached flash responses: %d", len(cache))
    return cache


def load_canonical_prompts(
    splits: List[str] = ("val", "test"),
) -> List[str]:
    """Load unique prompts from canonical splits.

    Args:
        splits: Which splits to load (default: val + test for the
            onboarding experiment).

    Returns:
        Deduplicated prompt list preserving insertion order.
    """
    split_paths = {
        "train": TRAIN_DATA_PATH,
        "val": VAL_DATA_PATH,
        "test": HOLDOUT_DATA_PATH,
    }
    seen: Set[str] = set()
    prompts: List[str] = []
    for split_name in splits:
        path = split_paths[split_name]
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                p = rec["prompt"]
                if p not in seen:
                    prompts.append(p)
                    seen.add(p)
    return prompts


def load_completed() -> Set[str]:
    """Load prompts already completed in the output file."""
    completed: Set[str] = set()
    if not OUTPUT_FILE.exists():
        return completed
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    completed.add(rec["prompt"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def print_summary() -> None:
    """Print a summary of collected flash rewards."""
    if not OUTPUT_FILE.exists():
        print("No data collected yet.")
        return

    scores: List[float] = []
    n_cached = 0
    n_failed = 0

    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    scores.append(rec["raw_score"])
                    if rec.get("response_cached"):
                        n_cached += 1
                else:
                    n_failed += 1
            except (json.JSONDecodeError, KeyError):
                continue

    if not scores:
        print("No successful records.")
        return

    arr = np.array(scores)
    print("\n" + "=" * 70)
    print("FLASH CANONICAL (v3 PoLL panel) COLLECTION SUMMARY")
    print("=" * 70)
    print(f"  Successful records: {len(arr)}")
    print(f"  Failed records:     {n_failed}")
    print(f"  Cached responses:   {n_cached} ({n_cached / len(arr) * 100:.1f}%)")
    print(f"  Mean reward:        {arr.mean():.4f}")
    print(f"  Std reward:         {arr.std():.4f}")
    print(f"  Min / Max:          {arr.min():.4f} / {arr.max():.4f}")
    print(f"  Median:             {np.median(arr):.4f}")

    for split_name, path in [("val", VAL_DATA_PATH), ("test", HOLDOUT_DATA_PATH)]:
        split_prompts: Set[str] = set()
        with open(path) as f:
            for line in f:
                split_prompts.add(json.loads(line)["prompt"])
        collected_prompts: Set[str] = set()
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok") and rec["prompt"] in split_prompts:
                        collected_prompts.add(rec["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"  {split_name} coverage: {len(collected_prompts)} / {len(split_prompts)} "
              f"({len(collected_prompts) / len(split_prompts) * 100:.1f}%)")

    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of prompts (for testing).",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Parallel workers (default: 10).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file.",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip collection, just print summary.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["val", "test"],
        help="Which canonical splits to collect for (default: val test).",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = load_canonical_prompts(args.splits)
    if args.limit:
        prompts = prompts[: args.limit]
    logger.info("Loaded %d prompts from %s", len(prompts), args.splits)

    completed = load_completed() if args.resume else set()
    if completed:
        logger.info("Resuming: %d prompts already completed", len(completed))

    pending = [p for p in prompts if p not in completed]
    logger.info("Pending: %d prompts to collect", len(pending))

    if not pending:
        logger.info("Nothing to do — all prompts already collected.")
        print_summary()
        return

    gen = CoTRewardGenerator(max_workers=args.workers)
    gen.judge_panel = ["deepseek/deepseek-r1"]
    logger.info(
        "Judge panel: %s (v3 rubric, single R1 judge — same as canonical K=3 data)",
        gen.judge_panel,
    )

    response_cache = load_flash_response_cache()
    for prompt, response in response_cache.items():
        gen.response_cache[(FLASH_ID, prompt)] = response
    logger.info(
        "Pre-loaded %d cached flash responses into generator",
        len(response_cache),
    )

    n_cached = sum(1 for p in pending if (FLASH_ID, p) in gen.response_cache)
    logger.info(
        "  %d / %d pending prompts have cached responses (%.1f%%)",
        n_cached, len(pending), n_cached / max(len(pending), 1) * 100,
    )

    tasks: List[Tuple[str, str]] = [(p, FLASH_ID) for p in pending]

    logger.info(
        "Starting collection: %d tasks, %d workers...",
        len(tasks), args.workers,
    )

    with open(OUTPUT_FILE, "a") as out_f:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_done = 0
        n_ok = 0
        t0 = time.time()
        score_acc = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(gen.process_task, t): t for t in tasks}

            with tqdm(total=len(tasks), desc="Flash v3 PoLL") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    n_done += 1

                    if result.get("ok"):
                        n_ok += 1
                        score_acc.append(result.get("raw_score", 0))

                    pbar.update(1)

                    if n_done % 100 == 0 and score_acc:
                        elapsed = time.time() - t0
                        rate = n_done / elapsed * 60
                        logger.info(
                            "[%d/%d] %.0f tasks/min | ok=%d | "
                            "mean=%.4f | cache_hit=%.0f%%",
                            n_done, len(tasks), rate, n_ok,
                            np.mean(score_acc),
                            sum(1 for p, _ in tasks[:n_done]
                                if (FLASH_ID, p) in gen.response_cache)
                            / max(n_done, 1) * 100,
                        )

    elapsed = time.time() - t0
    logger.info(
        "Collection complete: %d/%d ok in %.1f min (%.1f tasks/min)",
        n_ok, n_done, elapsed / 60, n_done / max(elapsed, 1) * 60,
    )

    print_summary()


if __name__ == "__main__":
    main()
