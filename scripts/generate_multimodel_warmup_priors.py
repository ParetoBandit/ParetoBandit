#!/usr/bin/env python3
"""
Generate warmup priors for K-model portfolios from the 43-model evaluation data.

This script implements the three-way split approach for K>>2 experiments:
  1. Loads dev prompts with full 43-model coverage
  2. Creates a stratified split: prior-training (~40%) / online-learning (~60%)
  3. Builds LinUCB warmup priors for ALL models from the prior-training set
  4. Saves priors as a joblib artifact usable by any K-model experiment

The holdout set (750 prompts) is never touched.

Usage:
    python scripts/generate_multimodel_warmup_priors.py \\
        --pca src/bandit_gpt/data/artifacts/pca_32.joblib \\
        --prior-ratio 0.40 \\
        --plasticity 0.1 \\
        --device cpu

Output:
    src/artifacts/priors_warmup_43model.joblib   — warmup priors for 43 models
    src/artifacts/splits_three_way.json          — reproducible split definition
"""

import sys
import hashlib
import json
import gzip
import argparse
import logging
import numpy as np
import joblib
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.rewards import extract_reward
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    ARTIFACTS_DIR,
    RAW_EMBEDDINGS_CACHE_PATH,
)
from bandit_gpt.utils.experiment import ExperimentBurnIn
from utils.embeddings import load_raw_embedding_cache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_models_rewards(
    data_path: Path,
    min_models: int = 43,
    required_models: set | None = None,
) -> Dict[str, Dict[str, float]]:
    """Load rewards grouped by prompt, keeping only full-coverage prompts.

    Args:
        data_path: Gzipped JSONL file with per-(prompt, model) reward entries.
        min_models: Minimum number of distinct models per prompt.
        required_models: If provided, retain only prompts that cover *all*
            of these model IDs and keep only their reward entries.  The
            ``min_models`` filter is applied *after* the required-model
            filter (set it to ``len(required_models)`` for exact match).

    Returns:
        ``{prompt: {model_id: reward}}`` for qualifying prompts.
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
            rewards[prompt][model_id] = extract_reward(entry)

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
    return full


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

def build_multimodel_priors(
    prior_prompts: List[str],
    rewards: Dict[str, Dict[str, float]],
    pca,
    encoder_model: str,
    plasticity: float = 0.1,
    raw_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """Build LinUCB warmup priors for every model observed in the prior-training set.

    For each (prompt, model) pair in the prior-training data, we:
      1. Look up the raw embedding from *raw_cache* (or encode live on miss)
      2. Project through PCA and append a bias term
      3. Perform a rank-1 LinUCB update: A[m] += x x^T, b[m] += r * x

    Args:
        prior_prompts: Prompts in the prior-training split.
        rewards: ``{prompt: {model_id: reward}}`` lookup.
        pca: Fitted PCA model.
        encoder_model: SentenceTransformer model name (lazy-loaded on
            cache miss only).
        plasticity: Scaling factor for the accumulated matrices.
        raw_cache: Optional dict mapping ``sha256(prompt) -> raw_embedding``.
            When provided, avoids loading the SentenceTransformer encoder.

    Returns:
        Dict with keys ``A``, ``b``, ``models``, etc.
    """
    context_dim = pca.n_components_ + 1  # PCA dims + bias

    all_models = sorted(
        {m for p in prior_prompts for m in rewards.get(p, {}).keys()}
    )

    A = {m: np.eye(context_dim) for m in all_models}
    b = {m: np.zeros(context_dim) for m in all_models}

    logger.info(f"  Building priors for {len(all_models)} models "
                f"from {len(prior_prompts)} prompts ...")

    skipped = 0
    processed = 0
    encoder = None  # lazy-loaded on first cache miss

    for i, prompt in enumerate(prior_prompts):
        try:
            # Look up raw embedding from cache, fall back to live encoding
            emb = None
            if raw_cache is not None:
                key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                emb = raw_cache.get(key)
            if emb is None:
                if encoder is None:
                    from sentence_transformers import SentenceTransformer
                    encoder = SentenceTransformer(encoder_model)
                    if raw_cache is not None:
                        logger.warning("Cache miss — falling back to live encoding.")
                emb = encoder.encode(prompt, convert_to_numpy=True,
                                     show_progress_bar=False)

            if np.isnan(emb).any() or np.isinf(emb).any():
                skipped += 1
                continue
            emb_pca = pca.transform(emb.reshape(1, -1)).flatten()
            if np.isnan(emb_pca).any() or np.isinf(emb_pca).any():
                skipped += 1
                continue
            x = np.append(emb_pca, 1.0).reshape(-1, 1)  # column vector
        except Exception:
            skipped += 1
            continue

        for model_id, reward in rewards[prompt].items():
            A[model_id] += x @ x.T
            b[model_id] += (reward * x).flatten()
            processed += 1

        if (i + 1) % 100 == 0:
            logger.info(f"    {i + 1}/{len(prior_prompts)} prompts encoded")

    logger.info(f"  Processed {processed:,} (prompt, model) observations "
                f"({skipped} prompts skipped)")

    # Apply plasticity
    for m in all_models:
        A[m] *= plasticity
        b[m] *= plasticity

    pca_whitens = bool(getattr(pca, "whiten", False))
    ev = getattr(pca, "explained_variance_", None)

    state = {
        "A": A,
        "b": b,
        "models": all_models,
        "n_prompts": len(prior_prompts) - skipped,
        "n_observations": processed,
        "plasticity": plasticity,
        "context_dim": context_dim,
        "pca_components": pca.n_components_,
        "pca_whitened": pca_whitens,
        "pca_explained_variance": (
            np.asarray(ev, dtype=np.float64).tolist() if ev is not None else []
        ),
        "reward_source": "43model_evaluation_data",
    }
    return state


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_no_leakage(
    prior_train: List[str],
    online_learn: List[str],
    holdout_rewards: Dict[str, Dict[str, float]],
) -> None:
    """Assert pairwise disjointness across all three splits."""
    sets = {
        "prior_train": set(prior_train),
        "online_learn": set(online_learn),
        "holdout": set(holdout_rewards.keys()),
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

def print_report(state: Dict, prior_train: List[str],
                 online_learn: List[str], holdout_size: int) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-MODEL WARMUP PRIORS — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Models             : {len(state['models'])}")
    logger.info(f"  Prior training     : {state['n_prompts']} prompts "
                f"({state['n_observations']:,} observations)")
    logger.info(f"  Online learning    : {len(online_learn)} prompts")
    logger.info(f"  Holdout (untouched): {holdout_size} prompts")
    logger.info(f"  Plasticity         : {state['plasticity']}")
    logger.info(f"  Context dim        : {state['context_dim']}")

    # Per-model diagnostics
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-model warmup priors",
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to a JSON model config (e.g. models_k10.json). "
             "Only prompts with full coverage for these models are kept. "
             "If omitted, defaults to min_models=43 (legacy behaviour).",
    )
    parser.add_argument(
        "--pca", type=str, default=str(DEFAULT_PCA_PATH),
        help="Path to PCA model (must match live router)",
    )
    parser.add_argument(
        "--prior-ratio", type=float, default=0.40,
        help="Fraction of dev prompts for prior training (default: 0.40)",
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
        "--device", type=str, default="cpu",
        help="Torch device for SentenceTransformer inference (default: cpu)",
    )
    parser.add_argument(
        "--output-priors", type=str,
        default=str(ARTIFACTS_DIR / "priors_warmup_k10_15comp.joblib"),
        help="Output path for warmup priors",
    )
    parser.add_argument(
        "--output-splits", type=str,
        default=str(ARTIFACTS_DIR / "splits_three_way.json"),
        help="Output path for three-way split definition",
    )
    args = parser.parse_args()

    # Determine model filter
    if args.model_config:
        model_ids = _load_model_ids(args.model_config)
        required_models: set | None = set(model_ids)
        min_models = len(required_models)
        logger.info(f"Model config: {args.model_config}")
        logger.info(f"  K={len(model_ids)} models required per prompt")
    else:
        required_models = None
        min_models = 43
        logger.info("No --model-config supplied; requiring >= 43 models (legacy)")

    logger.info("=" * 70)
    logger.info("Multi-Model Warmup Prior Generation")
    logger.info("=" * 70)

    # ---- Load data --------------------------------------------------------
    logger.info(f"\n1. Loading evaluation data (min_models={min_models}) ...")
    dev_rewards = load_all_models_rewards(
        DEV_DATA_PATH_ALL_MODELS,
        min_models=min_models,
        required_models=required_models,
    )
    holdout_rewards = load_all_models_rewards(
        HOLDOUT_DATA_PATH_ALL_MODELS,
        min_models=min_models,
        required_models=required_models,
    )

    # ---- Three-way split --------------------------------------------------
    logger.info("\n2. Creating three-way split ...")
    splits_path = Path(args.output_splits)
    prior_train, online_learn = ExperimentBurnIn.create_three_way_splits(
        oracle_rewards=dev_rewards,
        splits_path=splits_path,
        prior_ratio=args.prior_ratio,
        random_state=args.seed,
        min_models=min_models,
    )

    # ---- Leakage check ----------------------------------------------------
    logger.info("\n3. Verifying data integrity ...")
    verify_no_leakage(prior_train, online_learn, holdout_rewards)

    # ---- Load PCA & raw embedding cache ------------------------------------
    logger.info("\n4. Loading PCA and raw embedding cache ...")
    pca_path = Path(args.pca)
    if not pca_path.exists():
        logger.error(f"PCA model not found: {pca_path}")
        return
    pca = joblib.load(pca_path)
    logger.info(f"  PCA: {pca.n_components_} components")

    raw_cache = load_raw_embedding_cache()
    logger.info(f"  Raw cache: {len(raw_cache)} prompts")

    # ---- Build priors -----------------------------------------------------
    logger.info("\n5. Building warmup priors ...")
    np.random.seed(args.seed)
    state = build_multimodel_priors(
        prior_prompts=prior_train,
        rewards=dev_rewards,
        pca=pca,
        encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
        plasticity=args.plasticity,
        raw_cache=raw_cache,
    )
    state["reward_source"] = "lmsys_chatbot_arena_prior_train_pool"

    # ---- Save -------------------------------------------------------------
    output_path = Path(args.output_priors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, output_path)
    logger.info(f"\n6. Saved priors to {output_path}")

    # ---- Report -----------------------------------------------------------
    print_report(state, prior_train, online_learn, len(holdout_rewards))
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
