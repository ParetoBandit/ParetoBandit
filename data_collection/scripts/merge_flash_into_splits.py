#!/usr/bin/env python3
"""Merge canonical flash rewards into K=3 val/test splits to produce K=4 versions.

Reads the flash rewards collected by ``collect_flash_canonical.py``
(DeepSeek-R1 single judge, v3 continuous rubric — identical to the
``build_router_pareto_dataset.py`` pipeline used for canonical K=3 data)
and injects
a ``google/gemini-2.5-flash`` arm into each prompt record.  Only prompts
with complete flash data are included in the K=4 output.

The K=4 split files are used by the model-onboarding appendix experiment
(``experiments/04_model_onboarding/``).

Usage
-----
    python data_collection/scripts/merge_flash_into_splits.py

    # Verify coverage without writing
    python data_collection/scripts/merge_flash_into_splits.py --dry-run

Output
------
    data_collection/rewards/val_k4.jsonl
    data_collection/rewards/test_k4.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit.config import (
    K4_MODELS_PATH,
    OFFLINE_DATASET_DIR,
    VAL_DATA_PATH,
    HOLDOUT_DATA_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FLASH_ID = "google/gemini-2.5-flash"
FLASH_REWARDS_PATH = OFFLINE_DATASET_DIR / "flash_canonical" / "gemini_flash_v3.jsonl"
FLASH_TOKEN_COUNTS_PATH = OFFLINE_DATASET_DIR / "flash_canonical" / "flash_token_counts.jsonl"


def _load_actual_flash_costs() -> Dict[str, float]:
    """Load actual per-request Flash costs from the token-count collection.

    Returns:
        ``{prompt: cost_usd}`` for prompts with actual token counts.
        Empty dict if the token-counts file does not exist.
    """
    if not FLASH_TOKEN_COUNTS_PATH.exists():
        return {}
    costs: Dict[str, float] = {}
    with open(FLASH_TOKEN_COUNTS_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ok") and rec.get("cost_usd") is not None:
                    costs[rec["prompt"]] = rec["cost_usd"]
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info(
        "Loaded %d actual Flash costs from %s",
        len(costs), FLASH_TOKEN_COUNTS_PATH.name,
    )
    return costs


def load_flash_rewards() -> Dict[str, Dict[str, Any]]:
    """Load flash rewards keyed by prompt text.

    Costs are resolved with the following priority:
    1. Actual per-request cost from ``flash_token_counts.jsonl``
    2. Token-count-based cost from the reward record itself
    3. Blended-rate heuristic fallback (logs a warning)

    Returns:
        ``{prompt: {"reward": float, "cost": float, ...}}``.
    """
    actual_costs = _load_actual_flash_costs()

    with open(K4_MODELS_PATH) as f:
        models_cfg = json.load(f)
    flash_cfg = next(m for m in models_cfg["models"] if m["model_id"] == FLASH_ID)
    input_cost_per_m = flash_cfg["input_cost_per_m"]
    output_cost_per_m = flash_cfg["output_cost_per_m"]

    rewards: Dict[str, Dict[str, Any]] = {}
    n_actual = 0
    n_token_based = 0
    n_fallback = 0

    with open(FLASH_REWARDS_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not rec.get("ok"):
                    continue
                prompt = rec["prompt"]

                if prompt in actual_costs:
                    cost = actual_costs[prompt]
                    n_actual += 1
                elif rec.get("cost_usd") is not None:
                    cost = rec["cost_usd"]
                    n_token_based += 1
                else:
                    input_tokens = rec.get("input_tokens")
                    output_tokens = rec.get("output_tokens")
                    if input_tokens is not None and output_tokens is not None:
                        cost = (
                            input_tokens * input_cost_per_m / 1e6
                            + output_tokens * output_cost_per_m / 1e6
                        )
                        n_token_based += 1
                    else:
                        avg_cost_per_m = (input_cost_per_m + output_cost_per_m) / 2.0
                        cost = avg_cost_per_m / 1e6 * 500
                        n_fallback += 1

                rewards[prompt] = {
                    "reward": rec["raw_score"],
                    "cost": cost,
                    "near_best": False,
                }
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info(
        "Loaded %d flash rewards (%d actual cost, %d token-based, %d fallback)",
        len(rewards), n_actual, n_token_based, n_fallback,
    )
    if n_fallback > 0:
        logger.warning(
            "%d prompts used the blended-rate heuristic fallback. "
            "Run collect_flash_token_counts.py to get actual costs.",
            n_fallback,
        )
    return rewards


def merge_split(
    input_path: Path,
    output_path: Path,
    flash_rewards: Dict[str, Dict[str, Any]],
    *,
    dry_run: bool = False,
) -> int:
    """Merge flash arm into a canonical K=3 split file.

    Args:
        input_path: K=3 JSONL split (e.g. val.jsonl).
        output_path: K=4 JSONL output (e.g. val_k4.jsonl).
        flash_rewards: Flash reward data keyed by prompt.
        dry_run: If True, only report coverage without writing.

    Returns:
        Number of prompts written (or that would be written in dry-run).
    """
    n_total = 0
    n_merged = 0
    n_missing = 0
    records = []

    with open(input_path) as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            prompt = rec["prompt"]

            if prompt not in flash_rewards:
                n_missing += 1
                continue

            flash = flash_rewards[prompt]
            rec["arms"][FLASH_ID] = {
                "reward": flash["reward"],
                "cost": flash["cost"],
                "near_best": flash.get("near_best", False),
            }

            all_rewards = [info["reward"] for info in rec["arms"].values()]
            best_reward = max(all_rewards)
            near_threshold = best_reward - 0.05
            for arm_info in rec["arms"].values():
                arm_info["near_best"] = arm_info["reward"] >= near_threshold

            rec["best_reward"] = best_reward
            rec["worst_reward"] = min(all_rewards)
            rec["reward_spread"] = best_reward - min(all_rewards)
            rec["best_arm"] = max(rec["arms"], key=lambda a: rec["arms"][a]["reward"])

            records.append(rec)
            n_merged += 1

    logger.info(
        "  %s: %d total, %d merged (%.1f%%), %d missing flash",
        input_path.name, n_total, n_merged,
        n_merged / max(n_total, 1) * 100, n_missing,
    )

    if dry_run:
        return n_merged

    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("  Wrote %s (%d records)", output_path, n_merged)
    return n_merged


def print_k4_stats(path: Path) -> None:
    """Print reward statistics for the K=4 split."""
    arm_rewards: Dict[str, list] = {}
    arm_costs: Dict[str, list] = {}
    n = 0

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            for arm_id, info in rec["arms"].items():
                arm_rewards.setdefault(arm_id, []).append(info["reward"])
                arm_costs.setdefault(arm_id, []).append(info["cost"])

    print(f"\n  {path.name}: {n} prompts, {len(arm_rewards)} arms")
    for arm_id in sorted(arm_rewards):
        r = np.array(arm_rewards[arm_id])
        c = np.array(arm_costs[arm_id])
        short = arm_id.split("/")[-1]
        print(f"    {short:<30s}  reward={r.mean():.4f} ± {r.std():.4f}  "
              f"cost=${c.mean():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report coverage without writing output files.",
    )
    args = parser.parse_args()

    flash_rewards = load_flash_rewards()

    splits = [
        (VAL_DATA_PATH, OFFLINE_DATASET_DIR / "val_k4.jsonl"),
        (HOLDOUT_DATA_PATH, OFFLINE_DATASET_DIR / "test_k4.jsonl"),
    ]

    for input_path, output_path in splits:
        merge_split(input_path, output_path, flash_rewards, dry_run=args.dry_run)

    if not args.dry_run:
        print("\n" + "=" * 70)
        print("K=4 SPLIT STATISTICS")
        print("=" * 70)
        for _, output_path in splits:
            if output_path.exists():
                print_k4_stats(output_path)
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
