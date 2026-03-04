#!/usr/bin/env python3
"""
Sample new LMSYS Arena prompts for reward collection.

Selects prompts from lmarena_battles_en.jsonl that do NOT already
appear in the dev or holdout reward files, applies basic quality
filters, and writes a deduplicated JSONL file.

Usage:
    python scripts/sample_new_prompts.py --n 1750 --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config import (
    OFFLINE_DATASET_DIR,
    LMSYS_BATTLES_PATH,
    PROMPTS_DIR,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEV_ALL_MODELS = DEV_DATA_PATH_ALL_MODELS
HOLDOUT_ALL_MODELS = HOLDOUT_DATA_PATH_ALL_MODELS

MIN_PROMPT_LEN = 20
MAX_PROMPT_LEN = 5000
MIN_ASCII_RATIO = 0.5


def _load_existing_prompts() -> Set[str]:
    """Return the set of all prompts that already have rewards or were previously sampled."""
    existing: Set[str] = set()
    for gz_path in [DEV_ALL_MODELS, HOLDOUT_ALL_MODELS]:
        if not gz_path.exists():
            logger.warning(f"  Skipping {gz_path.name} (not found)")
            continue
        with gzip.open(gz_path, "rt") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    existing.add(entry["prompt"])
    for jsonl_path in PROMPTS_DIR.glob("new_prompts_*.jsonl"):
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                if "prompt" in entry:
                    existing.add(entry["prompt"])
    return existing


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def _load_lmsys_prompts(path: Path) -> list[str]:
    """Extract unique prompts from the LMSYS battles file."""
    seen: Set[str] = set()
    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = data.get("prompt", "")
            if not text:
                try:
                    text = data["conversation"][0]["content"]
                except (KeyError, IndexError, TypeError):
                    continue
            text = text.strip()
            if text and text not in seen:
                seen.add(text)
                prompts.append(text)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample new LMSYS prompts for K=10 reward collection.")
    parser.add_argument("--n", type=int, default=1750, help="Number of prompts to sample (default: 1750)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="data/new_prompts_1750.jsonl",
                        help="Output JSONL path (default: data/new_prompts_1750.jsonl)")
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output

    logger.info("=" * 60)
    logger.info("Sample New Prompts for K=10 Reward Collection")
    logger.info("=" * 60)

    logger.info("\n1. Loading existing prompts (dev + holdout) ...")
    existing = _load_existing_prompts()
    logger.info(f"   Existing prompts with rewards: {len(existing)}")

    logger.info("\n2. Loading LMSYS Arena prompts ...")
    if not LMSYS_BATTLES_PATH.exists():
        logger.error(f"   LMSYS battles file not found: {LMSYS_BATTLES_PATH}")
        sys.exit(1)
    all_lmsys = _load_lmsys_prompts(LMSYS_BATTLES_PATH)
    logger.info(f"   Total unique LMSYS prompts: {len(all_lmsys)}")

    logger.info("\n3. Deduplicating ...")
    candidates = [p for p in all_lmsys if p not in existing]
    logger.info(f"   After removing existing: {len(candidates)}")

    logger.info("\n4. Quality filtering ...")
    filtered = []
    rejected_short = 0
    rejected_long = 0
    rejected_ascii = 0
    for p in candidates:
        if len(p) < MIN_PROMPT_LEN:
            rejected_short += 1
            continue
        if len(p) > MAX_PROMPT_LEN:
            rejected_long += 1
            continue
        if _ascii_ratio(p) < MIN_ASCII_RATIO:
            rejected_ascii += 1
            continue
        filtered.append(p)
    logger.info(f"   Passed filters: {len(filtered)}")
    logger.info(f"   Rejected (too short <{MIN_PROMPT_LEN}): {rejected_short}")
    logger.info(f"   Rejected (too long >{MAX_PROMPT_LEN}): {rejected_long}")
    logger.info(f"   Rejected (low ASCII <{MIN_ASCII_RATIO}): {rejected_ascii}")

    if len(filtered) < args.n:
        logger.error(f"   Not enough candidates ({len(filtered)}) for requested sample ({args.n})")
        sys.exit(1)

    logger.info(f"\n5. Sampling {args.n} prompts (seed={args.seed}) ...")
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(filtered), size=args.n, replace=False)
    sampled = [filtered[i] for i in sorted(indices)]

    overlap_check = set(sampled) & existing
    assert len(overlap_check) == 0, f"BUG: {len(overlap_check)} prompts overlap with existing data!"
    assert len(set(sampled)) == args.n, f"BUG: duplicates in sampled set"
    logger.info(f"   Deduplication verified: 0 overlap with existing data")

    logger.info(f"\n6. Writing to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in sampled:
            f.write(json.dumps({"prompt": p}) + "\n")
    logger.info(f"   Written {args.n} prompts")

    logger.info(f"\n   Prompt length stats:")
    lengths = [len(p) for p in sampled]
    logger.info(f"     Mean: {np.mean(lengths):.0f} chars")
    logger.info(f"     Median: {np.median(lengths):.0f} chars")
    logger.info(f"     Min: {np.min(lengths)} chars")
    logger.info(f"     Max: {np.max(lengths)} chars")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
