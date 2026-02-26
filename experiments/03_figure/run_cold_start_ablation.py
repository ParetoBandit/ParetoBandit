#!/usr/bin/env python3
"""
Cold-Start Ablation: Warm-Start vs. Cold-Start (Priors Only)
=============================================================

Compares banditGPT with full dev-set burn-in (warm-start) against
banditGPT evaluated using only warmup priors (cold-start, no online
learning).  Sweeps λ to show the gap narrows at higher cost penalties.

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
import joblib

sys.path.insert(0, str(project_root / "experiments"))
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_TRIALS = 20
SEED_OFFSET = 42
ALPHA_START = 2.0
TARGET_NEFF = 10.0
LAMBDA_VALUES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]


def precompute_embeddings(data, encoder, pca):
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


def evaluate_holdout(router, eval_data, eval_emb, model_costs, burn_in_steps):
    total_reward = 0.0
    total_cost = 0.0
    for i, p in enumerate(eval_data):
        model, _log = router.route(eval_emb[i], total_steps=burn_in_steps)
        total_reward += p["rewards"][model]
        total_cost += model_costs[model]["cost"]
    n = len(eval_data)
    return total_reward / n, total_cost / n


def run_condition(train_data, eval_data, train_emb, eval_emb,
                  warmup_path, model_costs, lam, warm_start):
    """Run one λ value for warm-start or cold-start condition."""
    dim = len(train_emb[0])
    burn_in_steps = len(train_data)

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    rewards, costs = [], []

    for trial in range(N_TRIALS):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=None,
            feature_dim=dim,
            prior_n_effective=TARGET_NEFF,
            alpha=ALPHA_START,
            warmup_path=warmup_path,
            cost_penalty=lam,
        )

        if warm_start:
            for i, p in enumerate(train_data):
                model, log = router.route(train_emb[i], total_steps=burn_in_steps)
                norm_r = (p["rewards"][model] - r_min) / r_range
                router.process_feedback(log.request_id, norm_r)

        r, c = evaluate_holdout(router, eval_data, eval_emb, model_costs, burn_in_steps)
        rewards.append(r)
        costs.append(c)

    return np.array(rewards), np.array(costs)


def main():
    logger.info("=" * 70)
    logger.info("COLD-START ABLATION")
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

    logger.info(f"\nProtocol: {N_TRIALS} seeds × {len(LAMBDA_VALUES)} λ × 2 conditions")

    results = {}
    t_start = time.time()

    for lam in LAMBDA_VALUES:
        logger.info(f"\n--- λ = {lam} ---")
        warm_r, warm_c = run_condition(
            train_data, eval_data, train_emb, eval_emb,
            warmup_path, model_costs, lam, warm_start=True,
        )
        cold_r, cold_c = run_condition(
            train_data, eval_data, train_emb, eval_emb,
            warmup_path, model_costs, lam, warm_start=False,
        )

        t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1)

        results[str(lam)] = {
            "warm_start": {
                "reward_mean": float(warm_r.mean()),
                "reward_std": float(warm_r.std(ddof=1)),
                "reward_ci95": float(t_crit * warm_r.std(ddof=1) / np.sqrt(N_TRIALS)),
                "cost_mean": float(warm_c.mean()),
            },
            "cold_start": {
                "reward_mean": float(cold_r.mean()),
                "reward_std": float(cold_r.std(ddof=1)),
                "reward_ci95": float(t_crit * cold_r.std(ddof=1) / np.sqrt(N_TRIALS)),
                "cost_mean": float(cold_c.mean()),
            },
            "delta_reward": float(warm_r.mean() - cold_r.mean()),
            "delta_pct": float((warm_r.mean() - cold_r.mean()) / cold_r.mean() * 100)
            if cold_r.mean() > 0 else 0.0,
        }

        logger.info(f"  Warm: {warm_r.mean():.4f} ± {t_crit*warm_r.std(ddof=1)/np.sqrt(N_TRIALS):.4f}")
        logger.info(f"  Cold: {cold_r.mean():.4f} ± {t_crit*cold_r.std(ddof=1)/np.sqrt(N_TRIALS):.4f}")
        logger.info(f"  Δ:    {warm_r.mean()-cold_r.mean():+.4f} ({(warm_r.mean()-cold_r.mean())/cold_r.mean()*100:+.1f}%)")

    elapsed = time.time() - t_start
    logger.info(f"\n--- Complete in {elapsed:.0f}s ---")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cold_start_ablation.json"

    with open(output_file, "w") as f:
        json.dump({"metadata": {"n_trials": N_TRIALS, "lambda_values": LAMBDA_VALUES}, **results}, f, indent=2)

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
