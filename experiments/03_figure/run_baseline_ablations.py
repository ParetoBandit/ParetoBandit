#!/usr/bin/env python3
"""
Architectural Ablation: Component Contributions
=================================================

Evaluates seven routing strategies that progressively add one capability
at a time, quantifying the marginal contribution of each component.

Methods: Random, EMA Tracker, ε-greedy (±priors), LinUCB (±priors), banditGPT-Hybrid.
Protocol: 20 seeds × 10 λ values, dev-set burn-in, holdout evaluation.

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


def run_random(eval_data, model_costs, models, n_trials):
    """Random routing baseline."""
    rewards, costs = [], []
    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        total_r, total_c = 0.0, 0.0
        for p in eval_data:
            m = np.random.choice(models)
            total_r += p["rewards"][m]
            total_c += model_costs[m]["cost"]
        rewards.append(total_r / len(eval_data))
        costs.append(total_c / len(eval_data))
    return np.array(rewards), np.array(costs)


def run_ema_tracker(train_data, eval_data, model_costs, models, lam, n_trials):
    """Non-contextual EMA tracker with ε-greedy."""
    alpha_ema = 0.15
    epsilon = 0.1
    rewards, costs = [], []

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        ema = {m: 0.5 for m in models}
        model_costs_norm = {}
        max_cost = max(model_costs[m]["cost"] for m in models)
        min_cost = min(model_costs[m]["cost"] for m in models)
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
        for m in models:
            model_costs_norm[m] = (model_costs[m]["cost"] - min_cost) / cost_range

        all_raw = [s for p in train_data for s in p["rewards"].values()]
        r_min, r_max = min(all_raw), max(all_raw)
        r_range = r_max - r_min if r_max > r_min else 1.0

        for p in train_data:
            if np.random.random() < epsilon:
                m = np.random.choice(models)
            else:
                m = max(models, key=lambda x: ema[x] - lam * model_costs_norm[x])
            norm_r = (p["rewards"][m] - r_min) / r_range
            ema[m] = alpha_ema * norm_r + (1 - alpha_ema) * ema[m]

        total_r, total_c = 0.0, 0.0
        for p in eval_data:
            m = max(models, key=lambda x: ema[x] - lam * model_costs_norm[x])
            total_r += p["rewards"][m]
            total_c += model_costs[m]["cost"]

        rewards.append(total_r / len(eval_data))
        costs.append(total_c / len(eval_data))

    return np.array(rewards), np.array(costs)


def _load_warmup_priors(warmup_path, models, dim, neff):
    """Load and scale warmup priors from joblib file."""
    priors = joblib.load(warmup_path)
    A_init, b_init = {}, {}
    for m in models:
        if m in priors:
            p = priors[m]
            A_raw = np.array(p["A"])[:dim, :dim]
            b_raw = np.array(p["b"])[:dim]
            A_init[m] = np.eye(dim) + neff * (A_raw / max(np.trace(A_raw) / dim, 1e-6) - np.eye(dim))
            b_init[m] = neff * b_raw / max(np.trace(A_raw) / dim, 1e-6)
        else:
            A_init[m] = np.eye(dim)
            b_init[m] = np.zeros(dim)
    return A_init, b_init


def run_epsilon_greedy(train_data, eval_data, train_emb, eval_emb,
                       model_costs, models, lam, warmup_path, use_priors, n_trials):
    """Contextual ε-greedy with ridge regression."""
    dim = len(train_emb[0])
    epsilon = 0.1
    burn_in_steps = len(train_data)

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if r_max > r_min else 1.0

    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
    norm_costs = {m: (model_costs[m]["cost"] - min_cost) / cost_range for m in models}

    rewards, costs = [], []

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        reg = 1.0
        A = {m: reg * np.eye(dim) for m in models}
        b = {m: np.zeros(dim) for m in models}

        if use_priors:
            A_w, b_w = _load_warmup_priors(warmup_path, models, dim, TARGET_NEFF)
            A = {m: A_w[m].copy() for m in models}
            b = {m: b_w[m].copy() for m in models}

        for i, p in enumerate(train_data):
            x = train_emb[i]
            if np.random.random() < epsilon:
                m = np.random.choice(models)
            else:
                scores = {}
                for m in models:
                    theta = np.linalg.solve(A[m], b[m])
                    scores[m] = theta @ x - lam * norm_costs[m]
                m = max(scores, key=scores.get)

            norm_r = (p["rewards"][m] - r_min) / r_range
            A[m] += np.outer(x, x)
            b[m] += norm_r * x

        total_r, total_c = 0.0, 0.0
        for i, p in enumerate(eval_data):
            x = eval_emb[i]
            scores = {}
            for m in models:
                theta = np.linalg.solve(A[m], b[m])
                scores[m] = theta @ x - lam * norm_costs[m]
            best = max(scores, key=scores.get)
            total_r += p["rewards"][best]
            total_c += model_costs[best]["cost"]

        rewards.append(total_r / len(eval_data))
        costs.append(total_c / len(eval_data))

    return np.array(rewards), np.array(costs)


def run_linucb(train_data, eval_data, train_emb, eval_emb,
               model_costs, models, lam, warmup_path, use_priors, n_trials):
    """LinUCB (no Corralling) — single expert."""
    dim = len(train_emb[0])
    burn_in_steps = len(train_data)

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if r_max > r_min else 1.0

    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
    norm_costs = {m: (model_costs[m]["cost"] - min_cost) / cost_range for m in models}

    rewards, costs = [], []

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        reg = 1.0
        A = {m: reg * np.eye(dim) for m in models}
        b = {m: np.zeros(dim) for m in models}

        if use_priors:
            A_w, b_w = _load_warmup_priors(warmup_path, models, dim, TARGET_NEFF)
            A = {m: A_w[m].copy() for m in models}
            b = {m: b_w[m].copy() for m in models}

        for step, (p, x) in enumerate(zip(train_data, train_emb)):
            alpha = ALPHA_START - (ALPHA_START - 0.1) * step / max(burn_in_steps - 1, 1)
            scores = {}
            for m in models:
                A_inv = np.linalg.inv(A[m])
                theta = A_inv @ b[m]
                ucb = alpha * np.sqrt(x @ A_inv @ x)
                scores[m] = theta @ x + ucb - lam * norm_costs[m]
            chosen = max(scores, key=scores.get)

            norm_r = (p["rewards"][chosen] - r_min) / r_range
            A[chosen] += np.outer(x, x)
            b[chosen] += norm_r * x

        total_r, total_c = 0.0, 0.0
        for i, p in enumerate(eval_data):
            x = eval_emb[i]
            scores = {}
            for m in models:
                theta = np.linalg.solve(A[m], b[m])
                scores[m] = theta @ x - lam * norm_costs[m]
            best = max(scores, key=scores.get)
            total_r += p["rewards"][best]
            total_c += model_costs[best]["cost"]

        rewards.append(total_r / len(eval_data))
        costs.append(total_c / len(eval_data))

    return np.array(rewards), np.array(costs)


def run_banditgpt_hybrid(train_data, eval_data, train_emb, eval_emb,
                         warmup_path, model_costs, lam, n_trials):
    """Full banditGPT-Hybrid via production BanditRouter."""
    dim = len(train_emb[0])
    burn_in_steps = len(train_data)

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if r_max > r_min else 1.0

    rewards, costs = [], []

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=None, feature_dim=dim,
            prior_n_effective=TARGET_NEFF, alpha=ALPHA_START,
            warmup_path=warmup_path, cost_penalty=lam,
        )

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
    logger.info("ARCHITECTURAL ABLATION STUDY")
    logger.info("=" * 70)

    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_path = str(sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH)

    models = list(eval_data[0]["rewards"].keys())

    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    train_emb = precompute_embeddings(train_data, encoder, pca)
    eval_emb = precompute_embeddings(eval_data, encoder, pca)
    logger.info(f"  {len(train_emb)}+{len(eval_emb)} prompts in {time.time()-t0:.1f}s")

    methods = [
        "Random",
        "EMA Tracker",
        "ε-greedy (no priors)",
        "ε-greedy (w/ priors)",
        "LinUCB (no priors)",
        "LinUCB (w/ priors)",
        "banditGPT-Hybrid",
    ]

    all_results = {m: {} for m in methods}
    t_start = time.time()

    for lam in LAMBDA_VALUES:
        logger.info(f"\n--- λ = {lam} ---")

        rand_r, rand_c = run_random(eval_data, model_costs, models, N_TRIALS)
        ema_r, ema_c = run_ema_tracker(train_data, eval_data, model_costs, models, lam, N_TRIALS)
        eg_np_r, eg_np_c = run_epsilon_greedy(
            train_data, eval_data, train_emb, eval_emb,
            model_costs, models, lam, warmup_path, use_priors=False, n_trials=N_TRIALS)
        eg_wp_r, eg_wp_c = run_epsilon_greedy(
            train_data, eval_data, train_emb, eval_emb,
            model_costs, models, lam, warmup_path, use_priors=True, n_trials=N_TRIALS)
        lu_np_r, lu_np_c = run_linucb(
            train_data, eval_data, train_emb, eval_emb,
            model_costs, models, lam, warmup_path, use_priors=False, n_trials=N_TRIALS)
        lu_wp_r, lu_wp_c = run_linucb(
            train_data, eval_data, train_emb, eval_emb,
            model_costs, models, lam, warmup_path, use_priors=True, n_trials=N_TRIALS)
        bg_r, bg_c = run_banditgpt_hybrid(
            train_data, eval_data, train_emb, eval_emb,
            warmup_path, model_costs, lam, N_TRIALS)

        t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1)

        for name, r_arr, c_arr in [
            ("Random", rand_r, rand_c),
            ("EMA Tracker", ema_r, ema_c),
            ("ε-greedy (no priors)", eg_np_r, eg_np_c),
            ("ε-greedy (w/ priors)", eg_wp_r, eg_wp_c),
            ("LinUCB (no priors)", lu_np_r, lu_np_c),
            ("LinUCB (w/ priors)", lu_wp_r, lu_wp_c),
            ("banditGPT-Hybrid", bg_r, bg_c),
        ]:
            ci = t_crit * r_arr.std(ddof=1) / np.sqrt(N_TRIALS) if r_arr.std(ddof=1) > 0 else 0
            all_results[name][str(lam)] = {
                "reward_mean": float(r_arr.mean()),
                "reward_std": float(r_arr.std(ddof=1)),
                "reward_ci95": float(ci),
                "cost_mean": float(c_arr.mean()),
            }
            logger.info(f"  {name:<28s}: {r_arr.mean():.4f} ± {ci:.4f}  cost=${c_arr.mean():.5f}")

    elapsed = time.time() - t_start
    logger.info(f"\n--- Complete in {elapsed:.0f}s ---")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_ablations.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "n_trials": N_TRIALS,
                "lambda_values": LAMBDA_VALUES,
                "methods": methods,
            },
            "results": all_results,
        }, f, indent=2)

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
