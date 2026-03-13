#!/usr/bin/env python3
"""Generate warmup priors for the K=3 portfolio from canonical train split.

Reads the arms-based ``train.jsonl`` format and converts it to the
``{prompt, rewards}`` format expected by ``generate_warmup_priors``.

Usage:
    python scripts/generate_k3_warmup_priors.py

Produces:
    data_collection/warmup_priors/priors_k3_25comp.joblib
"""

from __future__ import annotations

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

from bandit_gpt.calibration import generate_warmup_priors
from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K3_ARM_ORDER,
    TRAIN_DATA_PATH,
    WARMUP_PRIORS_DIR,
)
from utils.embeddings import load_raw_embedding_cache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = WARMUP_PRIORS_DIR / "priors_k3_25comp.joblib"
PLASTICITY = 0.1
SEED = 42


def load_train_as_rewards(
    path: Path,
    arm_order: List[str],
) -> List[Dict]:
    """Load arms-based JSONL and convert to the rewards-dict format.

    Args:
        path: Path to the canonical train.jsonl.
        arm_order: Model IDs to include.

    Returns:
        List of ``{"prompt": str, "rewards": {model_id: float}}`` dicts.
    """
    arm_set = set(arm_order)
    data: List[Dict] = []
    n_skipped = 0

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            arms = row.get("arms", {})
            if not arm_set <= set(arms.keys()):
                n_skipped += 1
                continue
            rewards = {m: arms[m]["reward"] for m in arm_order}
            data.append({"prompt": row["prompt"], "rewards": rewards})

    if n_skipped:
        logger.warning(
            "Skipped %d prompts missing coverage for all K=%d arms",
            n_skipped, len(arm_order),
        )
    return data


def main() -> None:
    logger.info("=" * 70)
    logger.info("K=3 Warmup Prior Generation")
    logger.info("  Models: %s", K3_ARM_ORDER)
    logger.info("  Train:  %s", TRAIN_DATA_PATH)
    logger.info("  PCA:    %s", DEFAULT_PCA_PATH)
    logger.info("  Output: %s", OUTPUT_PATH)
    logger.info("=" * 70)

    logger.info("\n1. Loading training data ...")
    rewards_data = load_train_as_rewards(TRAIN_DATA_PATH, K3_ARM_ORDER)
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
        output_path=OUTPUT_PATH,
        precomputed_raw_embeddings=raw_cache,
    )
    state["reward_source"] = str(TRAIN_DATA_PATH)
    state["split_mode"] = "canonical_train_k3"
    state["arm_order"] = K3_ARM_ORDER

    joblib.dump(state, OUTPUT_PATH)
    logger.info("  Saved priors to %s", OUTPUT_PATH)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  Models:      %s", state["models"])
    logger.info("  N prompts:   %d", state["n_prompts"])
    logger.info("  Context dim: %d", state["context_dim"])
    logger.info("  Plasticity:  %s", state["plasticity"])
    logger.info("  PCA whitened: %s", state.get("pca_whitened", "?"))

    logger.info("\n  Per-model statistics:")
    logger.info("  %-45s %10s %10s %10s", "Model", "tr(A)", "||b||", "||θ||")
    logger.info("  " + "-" * 77)
    for m in state["models"]:
        trace_a = np.trace(state["A"][m])
        norm_b = np.linalg.norm(state["b"][m])
        try:
            theta = np.linalg.solve(state["A"][m], state["b"][m])
            norm_theta = np.linalg.norm(theta)
        except np.linalg.LinAlgError:
            norm_theta = float("nan")
        logger.info("  %-45s %10.1f %10.3f %10.4f", m, trace_a, norm_b, norm_theta)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
