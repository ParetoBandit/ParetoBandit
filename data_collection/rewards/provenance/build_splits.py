#!/usr/bin/env python3
"""Build train / val / test splits and warmup priors for the K=3 pareto dataset.

Reads ``pareto_classified.jsonl`` and ``pareto_rewards.jsonl``, produces:

    experiments/benchmark/
        train.jsonl          – full-information reward logs (train split)
        val.jsonl            – full-information reward logs (val split)
        test.jsonl           – full-information reward logs (test split)
        train_priors.joblib  – whitened PCA-15 warmup priors from train
        split_manifest.json  – counts, seeds, and per-split statistics

Each ``.jsonl`` file has one line per prompt with the structure::

    {"prompt": str,
     "source": str,
     "difficulty": str,
     "best_arm": str,
     "reward_spread": float,
     "arms": {"model_id": {"reward": float, "cost": float, "near_best": bool}}}

Splitting strategy
------------------
Stratified by *source* so every benchmark is represented proportionally
in all three splits.  Within each source, prompts are shuffled with a
fixed seed.

    train : val : test  =  70 : 15 : 15

Warmup priors
-------------
Generated from the *train split only* using the canonical
``pareto_bandit.calibration.generate_warmup_priors`` implementation in the
production PCA-15 whitened coordinate system.

Usage
-----
    python experiments/benchmark/build_splits.py           # defaults
    python experiments/benchmark/build_splits.py --seed 0  # different seed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.calibration import generate_warmup_priors
from pareto_bandit.config import (
    DATA_COLLECTION_DIR,
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARETO_DIR = DATA_COLLECTION_DIR / "pareto_dataset"
CLASSIFIED_PATH = PARETO_DIR / "pareto_classified.jsonl"
REWARDS_PATH = PARETO_DIR / "pareto_rewards.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "benchmark"

ARM_ORDER = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-pro",
]
ARM_SHORT = {
    "meta-llama/llama-3.1-8b-instruct": "small",
    "mistralai/mistral-large-2512": "medium",
    "google/gemini-2.5-pro": "large",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_classified() -> List[Dict[str, Any]]:
    """Load pareto_classified.jsonl into a list of dicts."""
    records = []
    with open(CLASSIFIED_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_reward_lookup() -> Dict[str, Dict[str, float]]:
    """Build prompt → {model_id: reward} from pareto_rewards.jsonl.

    This is the format expected by ``generate_warmup_priors``.
    """
    lookup: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(REWARDS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok", False):
                continue
            lookup[r["prompt"]][r["model_id"]] = r.get("raw_score", 0.0)
    return dict(lookup)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    records: List[Dict[str, Any]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split records into train/val/test, stratified by source.

    Args:
        records: List of classified prompt dicts.
        train_frac: Fraction for training.
        val_frac: Fraction for validation (test gets the remainder).
        seed: Random seed for reproducibility.

    Returns:
        (train, val, test) lists.
    """
    rng = np.random.RandomState(seed)
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_source[r.get("source", "unknown")].append(r)

    train, val, test = [], [], []
    for source in sorted(by_source):
        items = by_source[source]
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac)) if n > 2 else 0
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_jsonl(records: List[Dict], path: Path) -> None:
    """Write records to a JSONL file."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("  Wrote %d records to %s", len(records), path.name)


def compute_split_stats(records: List[Dict]) -> Dict[str, Any]:
    """Summary statistics for a split."""
    rewards = {arm: [] for arm in ARM_ORDER}
    for r in records:
        for arm_id in ARM_ORDER:
            info = r.get("arms", {}).get(arm_id, {})
            if "reward" in info:
                rewards[arm_id].append(info["reward"])

    stats: Dict[str, Any] = {
        "n_prompts": len(records),
        "n_sources": len(set(r.get("source", "?") for r in records)),
    }
    for arm_id in ARM_ORDER:
        vals = np.array(rewards[arm_id])
        if len(vals) > 0:
            stats[f"mean_reward_{ARM_SHORT[arm_id]}"] = round(float(vals.mean()), 4)
    difficulty_counts = defaultdict(int)
    for r in records:
        difficulty_counts[r.get("difficulty", "?")] += 1
    stats["difficulty_distribution"] = dict(difficulty_counts)
    best_arm_counts = defaultdict(int)
    for r in records:
        best_arm_counts[ARM_SHORT.get(r.get("best_arm", ""), "?")] += 1
    stats["best_arm_distribution"] = dict(best_arm_counts)
    return stats


# ---------------------------------------------------------------------------
# Warmup prior generation
# ---------------------------------------------------------------------------

def build_warmup_priors(
    records: List[Dict[str, Any]],
    reward_lookup: Dict[str, Dict[str, float]],
    output_path: Path,
    pca_path: Path = DEFAULT_PCA_PATH,
    plasticity: float = 0.1,
    seed: int = 42,
    use_text_features: bool = False,
) -> Dict[str, Any]:
    """Generate warmup priors from classified records using the production pipeline.

    Converts the classified format to the ``{"prompt", "rewards"}`` format
    expected by ``generate_warmup_priors``, falling back to the
    ``pareto_rewards.jsonl`` lookup for per-judge scores.

    Args:
        records: Train-split records from pareto_classified.jsonl.
        reward_lookup: prompt → {model_id: raw_score} from pareto_rewards.
        output_path: Where to save the joblib priors.
        pca_path: Path to the PCA artifact.
        plasticity: Scaling factor for A, b.
        seed: Random seed.

    Returns:
        The prior state dict.
    """
    rewards_data = []
    for r in records:
        prompt = r["prompt"]
        prompt_rewards: Dict[str, float] = {}
        for arm_id in ARM_ORDER:
            arm_info = r.get("arms", {}).get(arm_id, {})
            if "reward" in arm_info:
                prompt_rewards[arm_id] = arm_info["reward"]
        if len(prompt_rewards) == len(ARM_ORDER):
            rewards_data.append({"prompt": prompt, "rewards": prompt_rewards})

    logger.info("  %d prompts with full K=%d coverage for priors",
                len(rewards_data), len(ARM_ORDER))

    try:
        from utils.embeddings import load_raw_embedding_cache
        raw_cache = load_raw_embedding_cache()
        logger.info("  Raw embedding cache: %d prompts", len(raw_cache))
    except Exception:
        raw_cache = None
        logger.info("  No raw embedding cache available; will encode on the fly")

    np.random.seed(seed)
    state = generate_warmup_priors(
        rewards_data=rewards_data,
        encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
        pca=pca_path,
        plasticity=plasticity,
        whiten_pca=True,
        output_path=output_path,
        precomputed_raw_embeddings=raw_cache,
        use_text_features=use_text_features,
    )
    state["reward_source"] = "pareto_dataset_train_split"
    state["split_mode"] = "pareto_stratified"
    joblib.dump(state, output_path)
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--plasticity", type=float, default=0.1)
    parser.add_argument("--pca", type=str, default=str(DEFAULT_PCA_PATH))
    parser.add_argument("--text-features", action="store_true",
                        help="Include text features (n_logical_ops, n_constraints, "
                             "avg_word_len, instruction_x_vague_density); outputs "
                             "train_priors_textfeat.joblib")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading classified data from %s", CLASSIFIED_PATH)
    records = load_classified()
    logger.info("  %d prompts loaded", len(records))

    logger.info("Loading reward lookup from %s", REWARDS_PATH)
    reward_lookup = load_reward_lookup()
    logger.info("  %d prompts in reward lookup", len(reward_lookup))

    # 2. Split
    logger.info("\nStratified split (seed=%d, train=%.0f%%, val=%.0f%%, test=%.0f%%)",
                args.seed, args.train_frac * 100, args.val_frac * 100,
                (1 - args.train_frac - args.val_frac) * 100)
    train, val, test = stratified_split(
        records, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed,
    )

    # Verify no leakage
    train_prompts = {r["prompt"] for r in train}
    val_prompts = {r["prompt"] for r in val}
    test_prompts = {r["prompt"] for r in test}
    assert not (train_prompts & val_prompts), "Train/val overlap!"
    assert not (train_prompts & test_prompts), "Train/test overlap!"
    assert not (val_prompts & test_prompts), "Val/test overlap!"
    assert len(train_prompts) + len(val_prompts) + len(test_prompts) == len(records), \
        "Split does not cover all prompts"
    logger.info("  No leakage: train=%d, val=%d, test=%d (total=%d)",
                len(train), len(val), len(test), len(records))

    # 3. Write splits
    logger.info("\nWriting splits to %s", OUTPUT_DIR)
    write_jsonl(train, OUTPUT_DIR / "train.jsonl")
    write_jsonl(val, OUTPUT_DIR / "val.jsonl")
    write_jsonl(test, OUTPUT_DIR / "test.jsonl")

    # 4. Generate warmup priors from train split
    priors_name = "train_priors_textfeat.joblib" if args.text_features else "train_priors.joblib"
    logger.info("\nGenerating warmup priors from train split (%s) ...", priors_name)
    priors_path = OUTPUT_DIR / priors_name
    state = build_warmup_priors(
        train, reward_lookup,
        output_path=priors_path,
        pca_path=Path(args.pca),
        plasticity=args.plasticity,
        seed=args.seed,
        use_text_features=args.text_features,
    )
    logger.info("  Priors saved to %s", priors_path.name)
    logger.info("  Context dim: %d, Models: %s",
                state["context_dim"], state["models"])

    # 5. Write manifest
    manifest = {
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": round(1 - args.train_frac - args.val_frac, 4),
        "source_file": str(CLASSIFIED_PATH),
        "pca_artifact": str(Path(args.pca).name),
        "encoder": DEFAULT_SENTENCE_TRANSFORMER,
        "plasticity": args.plasticity,
        "arm_order": ARM_ORDER,
        "train": compute_split_stats(train),
        "val": compute_split_stats(val),
        "test": compute_split_stats(test),
        "priors": {
            "path": str(priors_path.name),
            "n_prompts": state["n_prompts"],
            "context_dim": state["context_dim"],
            "models": state["models"],
            "pca_whitened": state.get("pca_whitened", False),
        },
    }
    manifest_path = OUTPUT_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("\nManifest saved to %s", manifest_path.name)

    # 6. Summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK DATASET SUMMARY")
    logger.info("=" * 60)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        stats = compute_split_stats(split_data)
        logger.info("  %-6s  %5d prompts, %2d sources, "
                    "mean r: small=%.3f med=%.3f large=%.3f",
                    split_name, stats["n_prompts"], stats["n_sources"],
                    stats.get("mean_reward_small", 0),
                    stats.get("mean_reward_medium", 0),
                    stats.get("mean_reward_large", 0))
    logger.info("  priors: dim=%d, K=%d arms, plasticity=%.2f",
                state["context_dim"], len(state["models"]), args.plasticity)
    logger.info("=" * 60)
    logger.info("\nDone. Files in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
