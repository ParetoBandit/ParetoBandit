#!/usr/bin/env python3
"""Build val/test splits of the 2K judge-robustness subset for both R1 and
GPT-4.1-mini, enabling end-to-end cross-judge regret comparison.

The wide-format ``judge_robustness_prompts.jsonl`` already carries R1 rewards
and per-model costs.  This script replaces R1 rewards with GPT-4.1-mini
rewards from the long-format ``judge_robustness_rewards.jsonl``, then
performs a stratified 1/3-val / 2/3-test split (by ``source``), writing
four JSONL files consumable by :func:`utils.simulation.load_split`.

Usage
-----
    python experiments/appendix/judge_robustness/build_cross_judge_splits.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "appendix" / "judge_robustness"))

from judge_robustness_utils import JUDGE_ID_TO_SHORT
from pareto_bandit.config import CALIBRATION_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUBSET_PROMPTS_PATH = CALIBRATION_DIR / "judge_robustness_prompts.jsonl"
SUPPLEMENTARY_REWARDS_PATH = CALIBRATION_DIR / "judge_robustness_rewards.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

RNG_SEED: int = 2026
VAL_FRACTION: float = 1 / 3


def _load_wide_records(path: Path) -> List[Dict[str, Any]]:
    """Load the wide-format subset prompts (R1 rewards + costs)."""
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d wide records from %s", len(records), path.name)
    return records


def _load_gpt_mini_rewards(
    path: Path,
) -> Dict[Tuple[str, str], float]:
    """Load GPT-4.1-mini rewards keyed by (prompt, model_id).

    Uses exact judge-ID matching via ``JUDGE_ID_TO_SHORT``.
    """
    rewards: Dict[Tuple[str, str], float] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            key = (rec["prompt"], rec["model_id"])
            for jd in rec.get("judge_details", []):
                short = JUDGE_ID_TO_SHORT.get(jd["judge"])
                if short == "GPT-4.1-mini":
                    rewards[key] = jd["reward"]
    logger.info("Loaded %d GPT-4.1-mini rewards", len(rewards))
    return rewards


def _build_gpt_mini_records(
    wide_records: List[Dict[str, Any]],
    gpt_rewards: Dict[Tuple[str, str], float],
) -> List[Dict[str, Any]]:
    """Clone wide records, replacing R1 rewards with GPT-4.1-mini rewards.

    Prompts where any model lacks a GPT-4.1-mini score are dropped.
    Costs are preserved (model property, not judge property).
    """
    out: List[Dict[str, Any]] = []
    dropped = 0
    for rec in wide_records:
        prompt = rec["prompt"]
        new_arms: Dict[str, Dict[str, Any]] = {}
        missing = False
        for model_id, arm_info in rec["arms"].items():
            key = (prompt, model_id)
            if key not in gpt_rewards:
                missing = True
                break
            new_arms[model_id] = {
                "reward": gpt_rewards[key],
                "cost": arm_info["cost"],
            }
        if missing:
            dropped += 1
            continue
        new_rec = dict(rec)
        new_rec["arms"] = new_arms
        best_model = max(new_arms, key=lambda m: new_arms[m]["reward"])
        new_rec["best_arm"] = best_model
        new_rec["best_reward"] = new_arms[best_model]["reward"]
        worst_reward = min(a["reward"] for a in new_arms.values())
        new_rec["worst_reward"] = worst_reward
        new_rec["reward_spread"] = new_rec["best_reward"] - worst_reward
        near_best_threshold = new_rec["best_reward"] - 0.01
        for arm_info in new_arms.values():
            arm_info["near_best"] = arm_info["reward"] >= near_best_threshold
        out.append(new_rec)

    if dropped:
        logger.warning("Dropped %d prompts missing GPT-4.1-mini scores", dropped)
    logger.info("Built %d GPT-4.1-mini wide records", len(out))
    return out


def _stratified_split(
    records: List[Dict[str, Any]],
    val_fraction: float = VAL_FRACTION,
    seed: int = RNG_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified split by ``source`` field, preserving proportions.

    Parameters
    ----------
    records:
        Wide-format prompt records.
    val_fraction:
        Fraction allocated to val (burn-in).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List, List]
        (val_records, test_records).
    """
    rng = np.random.default_rng(seed)

    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_source[rec.get("source", "unknown")].append(rec)

    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for source in sorted(by_source):
        pool = by_source[source]
        n_val = max(1, round(len(pool) * val_fraction))
        indices = rng.permutation(len(pool))
        for i in indices[:n_val]:
            val.append(pool[i])
        for i in indices[n_val:]:
            test.append(pool[i])

    rng.shuffle(val)
    rng.shuffle(test)

    logger.info(
        "Split: %d val (%.0f%%), %d test (%.0f%%), %d sources",
        len(val), 100 * len(val) / len(records),
        len(test), 100 * len(test) / len(records),
        len(by_source),
    )
    return val, test


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def main() -> None:
    logger.info("=" * 70)
    logger.info("Building cross-judge val/test splits")
    logger.info("=" * 70)

    r1_records = _load_wide_records(SUBSET_PROMPTS_PATH)
    gpt_rewards = _load_gpt_mini_rewards(SUPPLEMENTARY_REWARDS_PATH)
    gpt_records = _build_gpt_mini_records(r1_records, gpt_rewards)

    common_prompts = {r["prompt"] for r in gpt_records}
    r1_filtered = [r for r in r1_records if r["prompt"] in common_prompts]
    logger.info("Filtered R1 to %d prompts (matching GPT-4.1-mini coverage)", len(r1_filtered))

    r1_val, r1_test = _stratified_split(r1_filtered)

    # Build GPT splits in the same prompt order as R1 splits so that
    # the simulation can safely share embeddings across judges.
    gpt_by_prompt = {r["prompt"]: r for r in gpt_records}
    gpt_val = [gpt_by_prompt[r["prompt"]] for r in r1_val]
    gpt_test = [gpt_by_prompt[r["prompt"]] for r in r1_test]

    _write_jsonl(r1_val, RESULTS_DIR / "cross_judge_r1_val.jsonl")
    _write_jsonl(r1_test, RESULTS_DIR / "cross_judge_r1_test.jsonl")
    _write_jsonl(gpt_val, RESULTS_DIR / "cross_judge_gpt_mini_val.jsonl")
    _write_jsonl(gpt_test, RESULTS_DIR / "cross_judge_gpt_mini_test.jsonl")

    logger.info("\nSummary:")
    logger.info("  R1        val=%d  test=%d", len(r1_val), len(r1_test))
    logger.info("  GPT-mini  val=%d  test=%d", len(gpt_val), len(gpt_test))
    logger.info("Done.")


if __name__ == "__main__":
    main()
