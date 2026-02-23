#!/usr/bin/env python3
"""
Figure 8: Cumulative Regret at K=5 and K=10
=============================================

Tracks per-step regret during online learning for five methods:
  1. banditGPT (LinUCB + Corralling + warmup priors)
  2. LinTS (Linear Thompson Sampling + warmup priors)
  3. LinTS tabula rasa (no priors)
  4. ε-greedy (online, ε=0.1)
  5. Random (uniform)

Regret_t = reward(oracle_t) - reward(selected_t)
Cumulative regret_T = Σ_{t=1}^{T} Regret_t

Sub-linear cumulative regret growth confirms the bandit's learning; the
comparison against LinTS isolates the Corralling + UCB contribution.

Output: results/cumulative_regret_results.json
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.baselines import CostAwareLinTSRouter
from utils.router_factory import create_experiment_router
from utils.multimodel import (
    MODEL_CATALOG, PORTFOLIO_K5, PORTFOLIO_K10,
    N_TRIALS, SEED_OFFSET, TARGET_NEFF, ALPHA_START,
    CORRALLING_LR, CORRALLING_GAMMA,
    build_model_registry, build_lints_costs, load_warmup_priors,
    load_multimodel_data, MULTIMODEL_WARMUP_PRIORS_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Subsample steps for JSON output to keep file reasonable
REPORT_EVERY = 5


def run_regret_banditgpt(models, train_data, train_emb, oracle_rewards,
                         r_min, r_range, n_trials):
    dim = train_emb[0].shape[0]
    burn_in = len(train_data)
    n_steps = len(train_data)
    all_cum = np.zeros((n_trials, n_steps))

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=build_model_registry(models),
            feature_dim=dim, prior_n_effective=TARGET_NEFF,
            alpha=ALPHA_START,
            warmup_path=str(MULTIMODEL_WARMUP_PRIORS_PATH),
            use_corralling=True,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
            cost_penalty=0.0,
        )
        cum = 0.0
        for t, (p, x) in enumerate(zip(train_data, train_emb)):
            m, log = router.route(x, total_steps=burn_in)
            norm_r = (p["rewards"][m] - r_min) / r_range
            router.process_feedback(log.request_id, norm_r)
            cum += oracle_rewards[t] - p["rewards"][m]
            all_cum[trial, t] = cum
        if (trial + 1) % 5 == 0:
            logger.info(f"      banditGPT trial {trial+1}/{n_trials}")
    return all_cum


def run_regret_lints(models, train_data, train_emb, oracle_rewards,
                     r_min, r_range, n_trials, use_priors=True, label="LinTS"):
    dim = train_emb[0].shape[0]
    burn_in = len(train_data)
    n_steps = len(train_data)
    lints_costs = build_lints_costs(models)
    priors = load_warmup_priors(models) if use_priors else None
    all_cum = np.zeros((n_trials, n_steps))

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        ts = CostAwareLinTSRouter(
            models=models, context_dim=dim, model_costs=lints_costs,
            cost_penalty=0.0, noise_variance=0.25,
            warmup_priors=priors, ridge_lambda=1.0,
        )
        cum = 0.0
        for t, (p, x) in enumerate(zip(train_data, train_emb)):
            m = ts.select_model(x, total_steps=burn_in)
            norm_r = (p["rewards"][m] - r_min) / r_range
            ts.update(x, m, norm_r)
            cum += oracle_rewards[t] - p["rewards"][m]
            all_cum[trial, t] = cum
        if (trial + 1) % 5 == 0:
            logger.info(f"      {label} trial {trial+1}/{n_trials}")
    return all_cum


def run_regret_epsilon_greedy(models, train_data, oracle_rewards, n_trials):
    n_steps = len(train_data)
    all_cum = np.zeros((n_trials, n_steps))

    for trial in range(n_trials):
        rng = np.random.RandomState(SEED_OFFSET + trial)
        model_sums = {m: 0.0 for m in models}
        model_counts = {m: 0 for m in models}
        cum = 0.0
        for t, p in enumerate(train_data):
            if t < len(models):
                m = models[t % len(models)]
            elif rng.random() < 0.1:
                m = models[rng.randint(len(models))]
            else:
                means = {m: model_sums[m] / max(model_counts[m], 1) for m in models}
                m = max(means, key=means.get)
            model_sums[m] += p["rewards"][m]
            model_counts[m] += 1
            cum += oracle_rewards[t] - p["rewards"][m]
            all_cum[trial, t] = cum
    return all_cum


def run_regret_random(models, train_data, oracle_rewards, n_trials):
    n_steps = len(train_data)
    all_cum = np.zeros((n_trials, n_steps))

    for trial in range(n_trials):
        rng = np.random.RandomState(SEED_OFFSET + trial)
        cum = 0.0
        for t, p in enumerate(train_data):
            m = models[rng.randint(len(models))]
            cum += oracle_rewards[t] - p["rewards"][m]
            all_cum[trial, t] = cum
    return all_cum


def subsample_regret(all_cum, every=REPORT_EVERY):
    """Subsample and compute mean/std for JSON serialisation."""
    n_steps = all_cum.shape[1]
    indices = list(range(0, n_steps, every))
    if indices[-1] != n_steps - 1:
        indices.append(n_steps - 1)
    return {
        "steps": indices,
        "mean": all_cum[:, indices].mean(axis=0).tolist(),
        "std": all_cum[:, indices].std(axis=0, ddof=1).tolist() if all_cum.shape[0] > 1 else [0.0] * len(indices),
    }


def main():
    logger.info("=" * 70)
    logger.info("Figure 8: Cumulative Regret (K=5, K=10)")
    logger.info("=" * 70)

    results_all = {}

    for portfolio_name, models in [("K5", PORTFOLIO_K5), ("K10", PORTFOLIO_K10)]:
        K = len(models)
        logger.info(f"\n{'='*70}")
        logger.info(f"PORTFOLIO: {portfolio_name} ({K} models)")
        logger.info("=" * 70)

        train_data, eval_data, train_emb, eval_emb, costs, r_min, r_max = \
            load_multimodel_data(models)
        r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
        n_steps = len(train_data)

        oracle_rewards = [max(p["rewards"][m] for m in models) for p in train_data]

        methods = {}

        # banditGPT
        t0 = time.time()
        logger.info(f"\n  banditGPT ({N_TRIALS} trials) ...")
        cum = run_regret_banditgpt(models, train_data, train_emb, oracle_rewards,
                                   r_min, r_range, N_TRIALS)
        methods["banditGPT"] = subsample_regret(cum)
        logger.info(f"  Done in {time.time()-t0:.0f}s | final regret: {cum[:,-1].mean():.1f}")

        # LinTS (warmup)
        t0 = time.time()
        logger.info(f"\n  LinTS (warmup) ({N_TRIALS} trials) ...")
        cum = run_regret_lints(models, train_data, train_emb, oracle_rewards,
                               r_min, r_range, N_TRIALS, use_priors=True, label="LinTS")
        methods["LinTS"] = subsample_regret(cum)
        logger.info(f"  Done in {time.time()-t0:.0f}s | final regret: {cum[:,-1].mean():.1f}")

        # LinTS (tabula rasa)
        t0 = time.time()
        logger.info(f"\n  LinTS (no priors) ({N_TRIALS} trials) ...")
        cum = run_regret_lints(models, train_data, train_emb, oracle_rewards,
                               r_min, r_range, N_TRIALS, use_priors=False, label="LinTS (no priors)")
        methods["LinTS (no priors)"] = subsample_regret(cum)
        logger.info(f"  Done in {time.time()-t0:.0f}s | final regret: {cum[:,-1].mean():.1f}")

        # ε-greedy
        logger.info(f"\n  ε-greedy ({N_TRIALS} trials) ...")
        cum = run_regret_epsilon_greedy(models, train_data, oracle_rewards, N_TRIALS)
        methods["ε-greedy"] = subsample_regret(cum)
        logger.info(f"  final regret: {cum[:,-1].mean():.1f}")

        # Random
        logger.info(f"\n  Random ({N_TRIALS} trials) ...")
        cum = run_regret_random(models, train_data, oracle_rewards, N_TRIALS)
        methods["Random"] = subsample_regret(cum)
        logger.info(f"  final regret: {cum[:,-1].mean():.1f}")

        results_all[portfolio_name] = {
            "K": K,
            "n_steps": n_steps,
            "n_trials": N_TRIALS,
            "methods": methods,
        }

        logger.info(f"\n  SUMMARY ({portfolio_name}):")
        for name, d in methods.items():
            logger.info(f"    {name:<20} final regret: {d['mean'][-1]:.1f} ± {d['std'][-1]:.1f}")

    out_path = Path(__file__).parent / "results" / "cumulative_regret_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
