#!/usr/bin/env python3
"""Build val/test splits of the 2K judge-robustness subset for R1,
GPT-4.1-mini, and Claude-3.7-Sonnet, enabling end-to-end cross-judge
regret comparison.

The wide-format ``judge_robustness_prompts.jsonl`` already carries R1 rewards
and per-model costs.  This script replaces R1 rewards with supplementary
judge rewards from ``judge_robustness_rewards.jsonl``, then performs a
stratified 1/3-val / 2/3-test split (by ``source``), writing six JSONL
files consumable by :func:`utils.simulation.load_split`.

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


def _load_supplementary_rewards(
    path: Path,
    judge_short_name: str,
) -> Dict[Tuple[str, str], float]:
    """Load supplementary judge rewards keyed by (prompt, model_id).

    Parameters
    ----------
    path:
        Path to ``judge_robustness_rewards.jsonl``.
    judge_short_name:
        Short judge name to filter (e.g. ``"GPT-4.1-mini"``,
        ``"Claude-3.7-Sonnet"``).  Matched via ``JUDGE_ID_TO_SHORT``.

    Returns
    -------
    Dict[Tuple[str, str], float]
        ``{(prompt, model_id): reward}``.
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
                if short == judge_short_name:
                    rewards[key] = jd["reward"]
    logger.info("Loaded %d %s rewards", len(rewards), judge_short_name)
    return rewards


def _build_supplementary_records(
    wide_records: List[Dict[str, Any]],
    supp_rewards: Dict[Tuple[str, str], float],
    judge_label: str,
) -> List[Dict[str, Any]]:
    """Clone wide records, replacing R1 rewards with supplementary scores.

    Parameters
    ----------
    wide_records:
        R1 wide-format prompt records.
    supp_rewards:
        ``{(prompt, model_id): reward}`` from the supplementary judge.
    judge_label:
        Display name for logging (e.g. ``"GPT-4.1-mini"``).

    Returns
    -------
    List[Dict[str, Any]]
        Wide-format records with the supplementary judge's rewards.
        Prompts where any model lacks a score are dropped.
    """
    out: List[Dict[str, Any]] = []
    dropped = 0
    for rec in wide_records:
        prompt = rec["prompt"]
        new_arms: Dict[str, Dict[str, Any]] = {}
        missing = False
        for model_id, arm_info in rec["arms"].items():
            key = (prompt, model_id)
            if key not in supp_rewards:
                missing = True
                break
            new_arms[model_id] = {
                "reward": supp_rewards[key],
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
        logger.warning("Dropped %d prompts missing %s scores", dropped, judge_label)
    logger.info("Built %d %s wide records", len(out), judge_label)
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


SUPPLEMENTARY_JUDGES: List[Tuple[str, str, str]] = [
    ("GPT-4.1-mini", "gpt_mini", "GPT-mini"),
    ("Claude-3.7-Sonnet", "claude", "Claude"),
]
"""(judge_short_name, file_slug, log_label) for each supplementary judge."""


def main() -> None:
    logger.info("=" * 70)
    logger.info("Building cross-judge val/test splits")
    logger.info("=" * 70)

    r1_records = _load_wide_records(SUBSET_PROMPTS_PATH)

    supp_records: Dict[str, List[Dict[str, Any]]] = {}
    for judge_short, _slug, _label in SUPPLEMENTARY_JUDGES:
        rewards = _load_supplementary_rewards(
            SUPPLEMENTARY_REWARDS_PATH, judge_short,
        )
        supp_records[judge_short] = _build_supplementary_records(
            r1_records, rewards, judge_short,
        )

    # Intersect to prompts covered by ALL judges.
    common_prompts = set.intersection(*(
        {r["prompt"] for r in recs} for recs in supp_records.values()
    ))
    r1_filtered = [r for r in r1_records if r["prompt"] in common_prompts]
    logger.info(
        "Filtered R1 to %d prompts (covered by all %d supplementary judges)",
        len(r1_filtered), len(SUPPLEMENTARY_JUDGES),
    )

    r1_val, r1_test = _stratified_split(r1_filtered)

    _write_jsonl(r1_val, RESULTS_DIR / "cross_judge_r1_val.jsonl")
    _write_jsonl(r1_test, RESULTS_DIR / "cross_judge_r1_test.jsonl")

    # Build supplementary splits in the same prompt order as R1 so that
    # the simulation can safely share embeddings across judges.
    for judge_short, slug, log_label in SUPPLEMENTARY_JUDGES:
        by_prompt = {r["prompt"]: r for r in supp_records[judge_short]}
        s_val = [by_prompt[r["prompt"]] for r in r1_val]
        s_test = [by_prompt[r["prompt"]] for r in r1_test]
        _write_jsonl(s_val, RESULTS_DIR / f"cross_judge_{slug}_val.jsonl")
        _write_jsonl(s_test, RESULTS_DIR / f"cross_judge_{slug}_test.jsonl")
        logger.info("  %-10s val=%d  test=%d", log_label, len(s_val), len(s_test))

    logger.info("\nSummary:")
    logger.info("  R1        val=%d  test=%d", len(r1_val), len(r1_test))
    logger.info("Done.")


if __name__ == "__main__":
    main()
