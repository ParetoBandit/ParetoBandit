#!/usr/bin/env python3
"""
PCA Dimensionality and Prior Strength (neff) Ablation
======================================================

Sweeps PCA components and neff to verify robustness of the default
configuration (d=32, neff=10).

Uses the production BanditRouter via create_experiment_router().
"""

import sys
from pathlib import Path
import json
import numpy as np
import logging
import time
from scipy import stats as sp_stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from generate_pareto_frontier import load_model_costs, load_dataset_with_split
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
)
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import joblib

from bandit_gpt.feature_service import FeatureService
from bandit_gpt.storage import EphemeralContextStore
from bandit_gpt.router import BanditRouter

sys.path.insert(0, str(project_root / "experiments"))
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_TRIALS = 20
SEED_OFFSET = 42
ALPHA_START = 2.0
LAMBDA = 0.0

PCA_DIMS = [4, 8, 16, 32, 64, 128]
NEFF_VALUES = [1, 3, 5, 10, 20, 50, 100]


def embed_raw(data, encoder):
    """Embed prompts without PCA — returns raw sentence-transformer vectors."""
    return np.array([encoder.encode(p["prompt"]) for p in data])


def evaluate_holdout(router, eval_data, eval_emb, burn_in_steps):
    total = 0.0
    for i, p in enumerate(eval_data):
        model, _log = router.route(eval_emb[i], total_steps=burn_in_steps)
        total += p["rewards"][model]
    return total / len(eval_data)


def run_with_config(train_data, eval_data, train_emb, eval_emb,
                    warmup_path, neff, n_trials):
    dim = train_emb[0].shape[0]
    burn_in_steps = len(train_data)

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if r_max > r_min else 1.0

    rewards = []
    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        fs = FeatureService.for_precomputed(dim)
        store = EphemeralContextStore()
        router = BanditRouter.create(
            model_registry=None,
            feature_service=fs,
            context_store=store,
            priors="warmup" if warmup_path else "none",
            prior_n_effective=neff,
            alpha=ALPHA_START,
            cost_penalty=LAMBDA,
            **({"warmup_path": warmup_path} if warmup_path else {}),
        )

        for i, p in enumerate(train_data):
            model, log = router.route(train_emb[i], total_steps=burn_in_steps)
            norm_r = (p["rewards"][model] - r_min) / r_range
            router.process_feedback(log.request_id, norm_r)

        r = evaluate_holdout(router, eval_data, eval_emb, burn_in_steps)
        rewards.append(r)

    return np.array(rewards)


def main():
    logger.info("=" * 70)
    logger.info("PCA DIMENSIONALITY & NEFF ABLATION")
    logger.info("=" * 70)

    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_path = str(sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH)

    logger.info("\n--- Computing raw embeddings ---")
    t0 = time.time()
    raw_train = embed_raw(train_data, encoder)
    raw_eval = embed_raw(eval_data, encoder)
    logger.info(f"  Raw embeddings: train={raw_train.shape}, eval={raw_eval.shape} in {time.time()-t0:.1f}s")

    results = {"pca_sweep": {}, "neff_sweep": {}}
    t_start = time.time()

    # --- PCA dimensionality sweep (hold neff=10) ---
    logger.info("\n--- PCA dimensionality sweep (neff=10) ---")
    for d in PCA_DIMS:
        pca = PCA(n_components=d)
        train_pca = pca.fit_transform(raw_train)
        eval_pca = pca.transform(raw_eval)

        train_emb = [np.concatenate([train_pca[i], [1.0]]) for i in range(len(train_pca))]
        eval_emb = [np.concatenate([eval_pca[i], [1.0]]) for i in range(len(eval_pca))]

        wp = warmup_path if d == 32 else None
        rewards = run_with_config(
            train_data, eval_data, train_emb, eval_emb,
            wp, neff=10.0, n_trials=N_TRIALS)

        t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1)
        ci = t_crit * rewards.std(ddof=1) / np.sqrt(N_TRIALS)
        results["pca_sweep"][str(d)] = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std(ddof=1)),
            "ci95": float(ci),
        }
        logger.info(f"  d={d:<4}: {rewards.mean():.4f} ± {ci:.4f}")

    # --- neff sweep (hold d=32, the default) ---
    logger.info("\n--- neff sweep (d=32) ---")
    default_pca = joblib.load(DEFAULT_PCA_PATH)
    train_emb_default = [embed_prompt(p["prompt"], encoder, default_pca) for p in train_data]
    eval_emb_default = [embed_prompt(p["prompt"], encoder, default_pca) for p in eval_data]

    for neff in NEFF_VALUES:
        rewards = run_with_config(
            train_data, eval_data, train_emb_default, eval_emb_default,
            warmup_path, neff=float(neff), n_trials=N_TRIALS)

        t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1)
        ci = t_crit * rewards.std(ddof=1) / np.sqrt(N_TRIALS)
        results["neff_sweep"][str(neff)] = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std(ddof=1)),
            "ci95": float(ci),
        }
        logger.info(f"  neff={neff:<4}: {rewards.mean():.4f} ± {ci:.4f}")

    elapsed = time.time() - t_start
    logger.info(f"\n--- Complete in {elapsed:.0f}s ---")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pca_neff_ablation.json"

    with open(output_file, "w") as f:
        json.dump({"metadata": {"n_trials": N_TRIALS, "lambda": LAMBDA}, **results}, f, indent=2)

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
