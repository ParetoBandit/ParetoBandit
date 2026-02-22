#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis
=====================================

Sweeps η (Corralling learning rate) and α_start (UCB exploration coefficient)
to verify that Figure 4 results are robust to hyperparameter choices.

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
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
)
from sentence_transformers import SentenceTransformer
import joblib

sys.path.insert(0, str(project_root / "experiments"))
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_TRIALS = 20
SEED_OFFSET = 42
TARGET_NEFF = 10.0
LAMBDA = 0.0

ETA_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
ALPHA_VALUES = [0.5, 1.0, 2.0, 4.0]


def precompute_embeddings(data, encoder, pca):
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


def evaluate_holdout(router, eval_data, eval_emb, burn_in_steps):
    total = 0.0
    for i, p in enumerate(eval_data):
        model, _log = router.route(eval_emb[i], total_steps=burn_in_steps)
        total += p["rewards"][model]
    return total / len(eval_data)


def run_sweep(train_data, eval_data, train_emb, eval_emb, warmup_path,
              alpha_start, eta, n_trials):
    dim = len(train_emb[0])
    burn_in_steps = len(train_data)

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if r_max > r_min else 1.0

    rewards = []
    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=None,
            feature_dim=dim,
            prior_n_effective=TARGET_NEFF,
            alpha=alpha_start,
            warmup_path=warmup_path,
            cost_penalty=LAMBDA,
            corralling_learning_rate=eta,
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
    logger.info("HYPERPARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_path = str(sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH)

    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    train_emb = precompute_embeddings(train_data, encoder, pca)
    eval_emb = precompute_embeddings(eval_data, encoder, pca)
    logger.info(f"  {len(train_emb)}+{len(eval_emb)} prompts in {time.time()-t0:.1f}s")

    results = {"eta_sweep": {}, "alpha_sweep": {}}
    t_start = time.time()

    # Sweep η (fix α=2.0)
    logger.info("\n--- η sweep (α_start=2.0) ---")
    for eta in ETA_VALUES:
        rewards = run_sweep(train_data, eval_data, train_emb, eval_emb,
                            warmup_path, alpha_start=2.0, eta=eta, n_trials=N_TRIALS)
        t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1)
        ci = t_crit * rewards.std(ddof=1) / np.sqrt(N_TRIALS)
        results["eta_sweep"][str(eta)] = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std(ddof=1)),
            "ci95": float(ci),
        }
        logger.info(f"  η={eta:<4}: {rewards.mean():.4f} ± {ci:.4f}")

    # Sweep α_start (fix η=1.0)
    logger.info("\n--- α_start sweep (η=1.0) ---")
    for alpha in ALPHA_VALUES:
        rewards = run_sweep(train_data, eval_data, train_emb, eval_emb,
                            warmup_path, alpha_start=alpha, eta=1.0, n_trials=N_TRIALS)
        t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1)
        ci = t_crit * rewards.std(ddof=1) / np.sqrt(N_TRIALS)
        results["alpha_sweep"][str(alpha)] = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std(ddof=1)),
            "ci95": float(ci),
        }
        logger.info(f"  α={alpha:<4}: {rewards.mean():.4f} ± {ci:.4f}")

    elapsed = time.time() - t_start
    logger.info(f"\n--- Complete in {elapsed:.0f}s ---")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hyperparameter_sensitivity.json"

    with open(output_file, "w") as f:
        json.dump({"metadata": {"n_trials": N_TRIALS, "lambda": LAMBDA}, **results}, f, indent=2)

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
