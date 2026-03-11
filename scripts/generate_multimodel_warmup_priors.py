#!/usr/bin/env python3
"""
Generate warmup priors for K-model portfolios.

Thin CLI wrapper around ``bandit_gpt.calibration.generate_warmup_priors``,
which is the single canonical prior-generation implementation.  This
ensures that priors are always built in the same **whitened PCA** coordinate
system used by production (``embed_prompt``) and experiments
(``project_embeddings``).

Two modes of operation:

**Legacy (three-way split)**:  Loads the combined dev set, creates a
stratified prior-training / online-learning split, and builds priors
from the prior-training portion.  Holdout is loaded for leakage checks.

    python scripts/generate_multimodel_warmup_priors.py \\
        --model-config data_collection/config/models_k3.json \\
        --pca src/bandit_gpt/data/artifacts/pca_32.joblib

**Canonical splits (``--no-split``)**:  Uses a pre-split training file
(e.g. ``k4_train_rewards.jsonl.gz``) in its entirety.  No internal
splitting is needed because the canonical data layout already guarantees
disjointness between train / cal / holdout.

    python scripts/generate_multimodel_warmup_priors.py \\
        --model-config data_collection/config/models_k3.json \\
        --data-path data_collection/rewards/k4_train_rewards.jsonl.gz \\
        --no-split \\
        --pca src/bandit_gpt/data/artifacts/pca_32.joblib \\
        --output-priors data_collection/warmup_priors/priors_warmup_k3_32comp.joblib
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import generate_warmup_priors
from bandit_gpt.rewards import extract_reward
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    K4_TRAIN_DATA_PATH,
    WARMUP_PRIORS_PATH,
    ARTIFACTS_DIR,
)
from utils.embeddings import load_raw_embedding_cache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rewards_as_dataset(
    data_path: Path,
    min_models: int = 43,
    required_models: Optional[Set[str]] = None,
    judge_id: Optional[str] = None,
) -> List[Dict]:
    """Load rewards from gzipped JSONL, returning the format expected by
    ``calibration.generate_warmup_priors``.

    Args:
        data_path: Gzipped JSONL file with per-(prompt, model) reward entries.
        min_models: Minimum number of distinct models per prompt.
        required_models: If provided, retain only prompts that cover *all*
            of these model IDs and keep only their reward entries.
        judge_id: When set, extract reward from this single judge
            rather than the full panel mean.

    Returns:
        List of ``{"prompt": str, "rewards": {model_id: float}}`` dicts.
    """
    logger.info(f"  Loading {data_path.name} ...")
    rewards: Dict[str, Dict[str, float]] = {}
    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            model_id = entry["model_id"]
            if required_models is not None and model_id not in required_models:
                continue
            prompt = entry["prompt"]
            if prompt not in rewards:
                rewards[prompt] = {}
            rewards[prompt][model_id] = extract_reward(entry, judge_id=judge_id)

    if required_models is not None:
        full = {
            p: r for p, r in rewards.items()
            if required_models <= set(r.keys())
        }
    else:
        full = {p: r for p, r in rewards.items() if len(r) >= min_models}

    logger.info(
        f"  {len(full)} prompts with full coverage "
        f"(dropped {len(rewards) - len(full)})"
    )
    return [{"prompt": p, "rewards": r} for p, r in full.items()]


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------


def verify_no_leakage(
    prior_train: List[str],
    online_learn: List[str],
    holdout_prompts: List[str],
) -> None:
    """Assert pairwise disjointness across all three splits."""
    sets = {
        "prior_train": set(prior_train),
        "online_learn": set(online_learn),
        "holdout": set(holdout_prompts),
    }
    for a_name, a_set in sets.items():
        for b_name, b_set in sets.items():
            if a_name >= b_name:
                continue
            overlap = a_set & b_set
            if overlap:
                raise ValueError(
                    f"DATA LEAKAGE: {len(overlap)} prompts in "
                    f"{a_name} ∩ {b_name}"
                )
    logger.info("  All three splits are prompt-disjoint.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(state: Dict, n_prior: int, n_online: int,
                 n_holdout: int) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-MODEL WARMUP PRIORS — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Models             : {len(state['models'])}")
    logger.info(f"  Prior training     : {state['n_prompts']} prompts")
    logger.info(f"  Online learning    : {n_online} prompts")
    logger.info(f"  Holdout (untouched): {n_holdout} prompts")
    logger.info(f"  Plasticity         : {state['plasticity']}")
    logger.info(f"  Context dim        : {state['context_dim']}")
    logger.info(f"  PCA whitened       : {state.get('pca_whitened', '?')}")

    logger.info(f"\n  Per-model prior statistics:")
    logger.info(f"  {'Model':<45} {'tr(A)':>10} {'||b||':>10} {'||θ||':>10}")
    logger.info("  " + "-" * 77)
    for m in state["models"][:10]:
        trace_a = np.trace(state["A"][m])
        norm_b = np.linalg.norm(state["b"][m])
        try:
            theta = np.linalg.solve(state["A"][m], state["b"][m])
            norm_theta = np.linalg.norm(theta)
        except np.linalg.LinAlgError:
            norm_theta = float("nan")
        logger.info(f"  {m:<45} {trace_a:>10.1f} {norm_b:>10.3f} "
                    f"{norm_theta:>10.4f}")
    if len(state["models"]) > 10:
        logger.info(f"  ... ({len(state['models']) - 10} more models)")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_model_ids(config_path: str) -> List[str]:
    """Read model IDs from a JSON config file (``{"models": [...]}``)."""
    with open(config_path) as f:
        data = json.load(f)
    entries = data.get("models", data) if isinstance(data, dict) else data
    return [
        e["model_id"] if isinstance(e, dict) else str(e) for e in entries
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-model warmup priors",
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to a JSON model config (e.g. models_k3.json). "
             "Only prompts with full coverage for these models are kept. "
             "If omitted, defaults to min_models=43 (legacy behaviour).",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to a gzipped JSONL reward file to use for prior "
             "training.  When combined with --no-split, all prompts in "
             "this file are used directly.  Defaults to the legacy "
             "DEV_DATA_PATH_ALL_MODELS.",
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Use the entire --data-path file for prior training, "
             "bypassing the three-way split.  Use when the input file "
             "is already a canonical train split (e.g. k4_train_rewards).",
    )
    parser.add_argument(
        "--pca", type=str, default=str(DEFAULT_PCA_PATH),
        help="Path to PCA model (must match live router)",
    )
    parser.add_argument(
        "--prior-ratio", type=float, default=0.40,
        help="Fraction of dev prompts for prior training (default: 0.40). "
             "Ignored when --no-split is set.",
    )
    parser.add_argument(
        "--plasticity", type=float, default=0.1,
        help="Plasticity factor applied to A and b (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split and encoder (default: 42)",
    )
    parser.add_argument(
        "--output-priors", type=str,
        default=str(WARMUP_PRIORS_PATH),
        help="Output path for warmup priors",
    )
    parser.add_argument(
        "--output-splits", type=str,
        default=str(ARTIFACTS_DIR / "splits_three_way.json"),
        help="Output path for three-way split definition "
             "(ignored when --no-split is set)",
    )
    parser.add_argument(
        "--judge", type=str, default=None,
        help="Extract reward from a single judge (e.g. 'deepseek/deepseek-r1') "
             "instead of the full panel mean.  Default: None (ensemble).",
    )
    args = parser.parse_args()

    # Determine model filter
    if args.model_config:
        model_ids = _load_model_ids(args.model_config)
        required_models: Optional[Set[str]] = set(model_ids)
        min_models = len(required_models)
        logger.info(f"Model config: {args.model_config}")
        logger.info(f"  K={len(model_ids)} models required per prompt")
    else:
        required_models = None
        min_models = 43
        logger.info("No --model-config supplied; requiring >= 43 models (legacy)")

    data_path = Path(args.data_path) if args.data_path else None

    logger.info("=" * 70)
    logger.info("Multi-Model Warmup Prior Generation")
    logger.info(f"  Mode: {'canonical split (--no-split)' if args.no_split else 'three-way split'}")
    if data_path:
        logger.info(f"  Data: {data_path}")
    logger.info(f"  Judge: {args.judge or 'ensemble (all judges)'}")
    logger.info("=" * 70)

    # ── Load raw embedding cache (shared by both modes) ────────────────
    logger.info("\nLoading raw embedding cache ...")
    raw_cache = load_raw_embedding_cache()
    logger.info(f"  Raw cache: {len(raw_cache)} prompts")

    if args.no_split:
        # ── Canonical-split mode ──────────────────────────────────────
        source_path = data_path or K4_TRAIN_DATA_PATH
        logger.info(f"\n1. Loading training data from {source_path.name} ...")
        rewards_data = load_rewards_as_dataset(
            source_path,
            min_models=min_models,
            required_models=required_models,
            judge_id=args.judge,
        )
        logger.info(f"  Using all {len(rewards_data)} prompts for prior training")

        # ── Build priors via canonical implementation ──────────────────
        logger.info("\n2. Building whitened warmup priors ...")
        np.random.seed(args.seed)
        state = generate_warmup_priors(
            rewards_data=rewards_data,
            encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
            pca=Path(args.pca),
            plasticity=args.plasticity,
            whiten_pca=True,
            output_path=Path(args.output_priors),
            precomputed_raw_embeddings=raw_cache,
        )
        state["reward_source"] = str(source_path)
        state["split_mode"] = "canonical_no_split"

        # Re-save with extra metadata (generate_warmup_priors already
        # saved once; we overwrite with the enriched state).
        output_path = Path(args.output_priors)
        joblib.dump(state, output_path)
        logger.info(f"\n3. Saved priors to {output_path}")

        assert state.get("pca_whitened", False), (
            "BUG: priors should be whitened but metadata says otherwise."
        )

        print_report(state, len(rewards_data), 0, 0)

    else:
        # ── Legacy three-way-split mode ───────────────────────────────
        from bandit_gpt.utils.experiment import ExperimentBurnIn

        source_path = data_path or DEV_DATA_PATH_ALL_MODELS

        logger.info(f"\n1. Loading evaluation data from {source_path.name} ...")
        all_data = load_rewards_as_dataset(
            source_path,
            min_models=min_models,
            required_models=required_models,
            judge_id=args.judge,
        )
        holdout_data = load_rewards_as_dataset(
            HOLDOUT_DATA_PATH_ALL_MODELS,
            min_models=min_models,
            required_models=required_models,
            judge_id=args.judge,
        )

        # ExperimentBurnIn expects {prompt: {model: reward}} format
        oracle_rewards = {d["prompt"]: d["rewards"] for d in all_data}

        # ── Three-way split ───────────────────────────────────────────
        logger.info("\n2. Creating three-way split ...")
        splits_path = Path(args.output_splits)
        prior_train, online_learn = ExperimentBurnIn.create_three_way_splits(
            oracle_rewards=oracle_rewards,
            splits_path=splits_path,
            prior_ratio=args.prior_ratio,
            random_state=args.seed,
            min_models=min_models,
        )

        # ── Leakage check ─────────────────────────────────────────────
        logger.info("\n3. Verifying data integrity ...")
        verify_no_leakage(
            prior_train, online_learn,
            [d["prompt"] for d in holdout_data],
        )

        # ── Build priors via canonical implementation ──────────────────
        prior_data = [d for d in all_data if d["prompt"] in set(prior_train)]
        logger.info(f"\n4. Building whitened warmup priors from "
                    f"{len(prior_data)} prior-train prompts ...")
        np.random.seed(args.seed)
        state = generate_warmup_priors(
            rewards_data=prior_data,
            encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
            pca=Path(args.pca),
            plasticity=args.plasticity,
            whiten_pca=True,
            output_path=Path(args.output_priors),
            precomputed_raw_embeddings=raw_cache,
        )
        state["reward_source"] = "lmsys_chatbot_arena_prior_train_pool"
        state["split_mode"] = "three_way_split"

        output_path = Path(args.output_priors)
        joblib.dump(state, output_path)
        logger.info(f"\n5. Saved priors to {output_path}")

        assert state.get("pca_whitened", False), (
            "BUG: priors should be whitened but metadata says otherwise."
        )

        print_report(state, len(prior_data), len(online_learn),
                     len(holdout_data))

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
