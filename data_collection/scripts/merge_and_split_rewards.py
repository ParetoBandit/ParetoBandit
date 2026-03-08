#!/usr/bin/env python3
"""
Merge multiple reward JSONL files into a single K=5 master and split.

Reads one or more JSONL reward files, keeps only prompts that have
complete coverage across a specified model portfolio, and produces
deterministic train / val / holdout gzipped JSONL splits.

The split is performed at the *prompt* level: every record for a given
prompt lands in the same split, preserving within-prompt correlation.

Usage::

    python data_collection/scripts/merge_and_split_rewards.py \
        --input data_collection/rewards/k4_rewards_v3.jsonl \
                data_collection/rewards/mistral_large_2512_rewards.jsonl \
        --models-file data_collection/config/models_k5.json \
        --output-dir data_collection/rewards \
        --prefix k5

Output files (in ``--output-dir``)::

    k5_rewards_v3.jsonl        — merged master (uncompressed)
    k5_train_rewards.jsonl.gz
    k5_val_rewards.jsonl.gz
    k5_holdout_rewards.jsonl.gz
    k5_dev_rewards.jsonl.gz    — train + val convenience alias
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SPLIT_SEED = 42
TRAIN_FRAC = 0.25
VAL_FRAC = 0.375
HOLDOUT_FRAC = 0.375

CANONICAL_JUDGES: Set[str] = {
    "deepseek/deepseek-r1",
    "qwen/qwen-2.5-72b-instruct",
    "anthropic/claude-3.5-haiku",
}


def _load_model_ids(models_path: Path) -> List[str]:
    """Read model IDs from a canonical JSON config file."""
    with open(models_path) as f:
        return [m["model_id"] for m in json.load(f)["models"]]


def _has_canonical_panel(rec: Dict[str, Any], required_judges: Set[str]) -> bool:
    """Return True if the record's judge panel matches *required_judges* exactly."""
    judges = {jd.get("judge", "") for jd in rec.get("judge_details", [])}
    return judges == required_judges


def _load_and_merge(
    input_files: List[Path],
    model_ids: Set[str],
    required_judges: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], int]:
    """Load JSONL files and group records by (prompt, model_id).

    Args:
        input_files: Paths to JSONL reward files.
        model_ids: Model IDs to retain.
        required_judges: If provided, only keep records whose judge
            panel matches this set exactly.

    Returns:
        Tuple of (nested dict ``{prompt: {model_id: record}}``,
        count of records skipped due to panel mismatch).
    """
    prompt_records: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    total = 0
    skipped_panel = 0
    for path in input_files:
        n = 0
        n_skip = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                mid = rec.get("model_id", "")
                if mid not in model_ids:
                    continue
                if required_judges and not _has_canonical_panel(rec, required_judges):
                    n_skip += 1
                    continue
                prompt = rec.get("prompt", "").strip()
                if not prompt:
                    continue
                prompt_records[prompt][mid] = rec
                n += 1
        logger.info(
            "Loaded %d relevant records from %s (skipped %d bad-panel)",
            n, path, n_skip,
        )
        total += n
        skipped_panel += n_skip
    logger.info("Total loaded: %d records across %d prompts",
                total, len(prompt_records))
    return prompt_records, skipped_panel


def _filter_complete(
    prompt_records: Dict[str, Dict[str, Dict[str, Any]]],
    model_ids: Set[str],
) -> List[str]:
    """Return sorted list of prompts with full model coverage."""
    K = len(model_ids)
    complete = [p for p, recs in prompt_records.items() if len(recs) == K]
    complete.sort()
    logger.info(
        "Complete prompts: %d / %d (K=%d)",
        len(complete), len(prompt_records), K,
    )
    return complete


def _split_prompts(
    prompts: List[str],
    seed: int = SPLIT_SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> Dict[str, List[str]]:
    """Deterministic prompt-level split into train / val / holdout.

    Args:
        prompts: Sorted list of unique prompts.
        seed: Random seed for reproducibility.
        train_frac: Fraction allocated to training.
        val_frac: Fraction allocated to validation.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"holdout"``, each
        mapping to a sorted list of prompts.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(prompts))
    n_train = int(len(prompts) * train_frac)
    n_val = int(len(prompts) * val_frac)

    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train:n_train + n_val])
    holdout_idx = sorted(indices[n_train + n_val:])

    return {
        "train": [prompts[i] for i in train_idx],
        "val": [prompts[i] for i in val_idx],
        "holdout": [prompts[i] for i in holdout_idx],
    }


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records as plain JSONL."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _write_jsonl_gz(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records as gzipped JSONL."""
    with gzip.open(path, "wt") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _collect_records(
    prompts: List[str],
    prompt_records: Dict[str, Dict[str, Dict[str, Any]]],
    model_ids: List[str],
) -> List[Dict[str, Any]]:
    """Flatten prompt-grouped records into a list ordered by (prompt, model)."""
    records: List[Dict[str, Any]] = []
    for p in prompts:
        for mid in model_ids:
            records.append(prompt_records[p][mid])
    return records


def _print_summary_stats(records: List[Dict[str, Any]], model_ids: List[str]) -> None:
    """Print per-model and per-judge summary statistics."""
    per_model: Dict[str, List[float]] = defaultdict(list)
    per_judge: Dict[str, List[float]] = defaultdict(list)
    per_model_judge: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        mid = rec.get("model_id", "")
        raw = rec.get("raw_score")
        if raw is not None and np.isfinite(raw):
            per_model[mid].append(raw)
        for jd in rec.get("judge_details", []):
            j = jd.get("judge", "")
            r = jd.get("reward")
            if r is not None and np.isfinite(r):
                per_judge[j].append(r)
                per_model_judge[mid][j].append(r)

    header = "\n" + "=" * 72
    logger.info(header)
    logger.info("  SUMMARY STATISTICS")
    logger.info("=" * 72)

    logger.info("\n  Per-model aggregate (raw_score = mean of 3 judges):")
    logger.info("  %-40s %6s %8s %8s %8s", "model_id", "n", "mean", "std", "median")
    logger.info("  " + "-" * 70)
    for mid in model_ids:
        scores = per_model.get(mid, [])
        if scores:
            arr = np.array(scores)
            logger.info(
                "  %-40s %6d %8.4f %8.4f %8.4f",
                mid, len(arr), arr.mean(), arr.std(), np.median(arr),
            )

    logger.info("\n  Per-judge reward (across all models):")
    logger.info("  %-40s %6s %8s %8s %8s", "judge", "n", "mean", "std", "median")
    logger.info("  " + "-" * 70)
    for j in sorted(per_judge):
        arr = np.array(per_judge[j])
        logger.info(
            "  %-40s %6d %8.4f %8.4f %8.4f",
            j, len(arr), arr.mean(), arr.std(), np.median(arr),
        )

    logger.info("\n  Per-model x judge reward:")
    logger.info(
        "  %-40s %-35s %6s %8s %8s",
        "model_id", "judge", "n", "mean", "std",
    )
    logger.info("  " + "-" * 95)
    for mid in model_ids:
        for j in sorted(per_model_judge.get(mid, {})):
            arr = np.array(per_model_judge[mid][j])
            logger.info(
                "  %-40s %-35s %6d %8.4f %8.4f",
                mid, j, len(arr), arr.mean(), arr.std(),
            )

    logger.info("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge reward files and split into train/val/holdout.",
    )
    parser.add_argument(
        "--input", nargs="+", type=str, required=True,
        help="One or more JSONL reward files to merge.",
    )
    parser.add_argument(
        "--models-file", type=str, required=True,
        help="Path to models JSON config (defines the target portfolio).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--prefix", type=str, default="k5",
        help="Filename prefix for output files (default: k5).",
    )
    parser.add_argument(
        "--seed", type=int, default=SPLIT_SEED,
        help=f"Random seed for the split (default: {SPLIT_SEED}).",
    )
    parser.add_argument(
        "--no-panel-filter", action="store_true",
        help="Skip filtering by canonical judge panel.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    models_path = Path(args.models_file)
    model_ids = _load_model_ids(models_path)
    model_id_set = set(model_ids)
    K = len(model_ids)
    logger.info("Target portfolio: K=%d models", K)
    for mid in model_ids:
        logger.info("  %s", mid)

    required_judges = None if args.no_panel_filter else CANONICAL_JUDGES
    if required_judges:
        logger.info("Filtering for canonical panel: %s", sorted(required_judges))

    input_files = [Path(p) for p in args.input]
    prompt_records, n_skipped = _load_and_merge(
        input_files, model_id_set, required_judges=required_judges,
    )
    if n_skipped:
        logger.info("Skipped %d records with non-canonical judge panel", n_skipped)
    complete_prompts = _filter_complete(prompt_records, model_id_set)

    if not complete_prompts:
        logger.error("No prompts with complete K=%d coverage — aborting.", K)
        sys.exit(1)

    splits = _split_prompts(complete_prompts, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    master_path = output_dir / f"{prefix}_rewards_v3.jsonl"
    all_records = _collect_records(complete_prompts, prompt_records, model_ids)
    _write_jsonl(all_records, master_path)

    for split_name in ("train", "val", "holdout"):
        split_prompts = splits[split_name]
        split_records = _collect_records(
            split_prompts, prompt_records, model_ids,
        )
        gz_path = output_dir / f"{prefix}_{split_name}_rewards.jsonl.gz"
        _write_jsonl_gz(split_records, gz_path)

    dev_prompts = sorted(set(splits["train"] + splits["val"]))
    dev_records = _collect_records(dev_prompts, prompt_records, model_ids)
    dev_path = output_dir / f"{prefix}_dev_rewards.jsonl.gz"
    _write_jsonl_gz(dev_records, dev_path)

    logger.info(
        "\nSplit summary (seed=%d, ratio %.0f/%.0f/%.0f):",
        args.seed,
        TRAIN_FRAC * 100, VAL_FRAC * 100, HOLDOUT_FRAC * 100,
    )
    logger.info("  Total prompts: %d", len(complete_prompts))
    logger.info("  Train:   %d prompts, %d records",
                len(splits["train"]), len(splits["train"]) * K)
    logger.info("  Val:     %d prompts, %d records",
                len(splits["val"]), len(splits["val"]) * K)
    logger.info("  Holdout: %d prompts, %d records",
                len(splits["holdout"]), len(splits["holdout"]) * K)
    logger.info("  Dev:     %d prompts, %d records",
                len(dev_prompts), len(dev_prompts) * K)

    _print_summary_stats(all_records, model_ids)
    logger.info("Done.")


if __name__ == "__main__":
    main()
