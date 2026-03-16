#!/usr/bin/env python3
"""Generate warmup priors for K=3 from a **single** benchmark source.

Creates priors that encode the reward landscape of one specific task
type (e.g., gsm8k math reasoning), enabling controlled experiments
where the priors are *genuinely miscalibrated* when the traffic
distribution shifts to a different task domain.

Usage:
    python scripts/generate_k3_warmup_priors_sourceonly.py --source gsm8k

Produces:
    data_collection/warmup_priors/priors_k3_25comp_<source>only.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.calibration import generate_warmup_priors
from pareto_bandit.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K3_ARM_ORDER,
    TRAIN_DATA_PATH,
    WARMUP_PRIORS_DIR,
)
from utils.embeddings import load_raw_embedding_cache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PLASTICITY = 0.1
SEED = 42


def load_train_single_source(
    path: Path,
    arm_order: List[str],
    include_source: str,
) -> List[Dict]:
    """Load arms-based JSONL, keeping only prompts from one source.

    Parameters
    ----------
    path : Path
        Canonical ``train.jsonl``.
    arm_order : list[str]
        Model IDs to include.
    include_source : str
        Value of the ``source`` field to keep (e.g. ``"gsm8k"``).

    Returns
    -------
    list[dict]
        ``{"prompt": str, "rewards": {model_id: float}}`` dicts.
    """
    arm_set = set(arm_order)
    data: List[Dict] = []
    n_excluded = 0
    n_skipped = 0

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("source", "") != include_source:
                n_excluded += 1
                continue
            arms = row.get("arms", {})
            if not arm_set <= set(arms.keys()):
                n_skipped += 1
                continue
            rewards = {m: arms[m]["reward"] for m in arm_order}
            data.append({"prompt": row["prompt"], "rewards": rewards})

    logger.info(
        "  Kept %d %s prompts (excluded %d other, skipped %d incomplete)",
        len(data), include_source, n_excluded, n_skipped,
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate K=3 warmup priors from a single benchmark source.",
    )
    parser.add_argument(
        "--source", required=True,
        help="Benchmark source to include (e.g., gsm8k, hellaswag)",
    )
    args = parser.parse_args()
    source: str = args.source
    output_path = WARMUP_PRIORS_DIR / f"priors_k3_25comp_{source}only.joblib"

    logger.info("=" * 70)
    logger.info("K=3 Warmup Prior Generation (%s only)", source)
    logger.info("  Models:  %s", K3_ARM_ORDER)
    logger.info("  Train:   %s", TRAIN_DATA_PATH)
    logger.info("  Include: source == '%s'", source)
    logger.info("  PCA:     %s", DEFAULT_PCA_PATH)
    logger.info("  Output:  %s", output_path)
    logger.info("=" * 70)

    logger.info("\n1. Loading training data (%s only) ...", source)
    rewards_data = load_train_single_source(
        TRAIN_DATA_PATH, K3_ARM_ORDER, source,
    )
    logger.info("  %d prompts with full K=3 coverage", len(rewards_data))

    logger.info("\n2. Loading raw embedding cache ...")
    raw_cache = load_raw_embedding_cache()
    logger.info("  Raw cache: %d prompts", len(raw_cache))

    logger.info("\n3. Building warmup priors (PCA-25, whitened) ...")
    np.random.seed(SEED)
    state = generate_warmup_priors(
        rewards_data=rewards_data,
        encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
        pca=DEFAULT_PCA_PATH,
        plasticity=PLASTICITY,
        whiten_pca=True,
        output_path=output_path,
        precomputed_raw_embeddings=raw_cache,
    )
    state["reward_source"] = str(TRAIN_DATA_PATH)
    state["split_mode"] = f"canonical_train_k3_{source}_only"
    state["arm_order"] = K3_ARM_ORDER
    state["included_source"] = source

    joblib.dump(state, output_path)
    logger.info("  Saved priors to %s", output_path)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  Models:       %s", state["models"])
    logger.info("  N prompts:    %d", state["n_prompts"])
    logger.info("  Context dim:  %d", state["context_dim"])
    logger.info("  Plasticity:   %s", state["plasticity"])
    logger.info("  PCA whitened: %s", state.get("pca_whitened", "?"))
    logger.info("  Source:       %s", source)

    logger.info("\n  Per-model statistics:")
    logger.info("  %-45s %10s %10s %10s", "Model", "tr(A)", "||b||", "||theta||")
    logger.info("  " + "-" * 77)
    for m in state["models"]:
        trace_a = np.trace(state["A"][m])
        norm_b = np.linalg.norm(state["b"][m])
        try:
            theta = np.linalg.solve(state["A"][m], state["b"][m])
            norm_theta = np.linalg.norm(theta)
        except np.linalg.LinAlgError:
            norm_theta = float("nan")
        logger.info(
            "  %-45s %10.1f %10.3f %10.4f", m, trace_a, norm_b, norm_theta,
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
