#!/usr/bin/env python3
"""
Figure 7: LinTS Baseline Comparison at K=5 and K=10
====================================================

Compares banditGPT (LinUCB + Corralling) against Linear Thompson Sampling
(LinTS), the primary alternative contextual bandit algorithm.

Both methods use:
  - Identical warmup priors (43-model, 355 prompts)
  - Same online learning set (533 prompts)
  - Same holdout evaluation (750 prompts)
  - Same cost penalty sweep (λ = 0..5)

This experiment answers: "Does the Corralling + UCB architecture add value
over the standard posterior-sampling alternative?"

Output: results/lints_comparison_results.json
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
    load_multimodel_data, oracle_route, static_route,
    random_route, epsilon_greedy_route,
    MULTIMODEL_WARMUP_PRIORS_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LAMBDA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


# ============================================================================
# PARETO SWEEPS
# ============================================================================

def evaluate_frozen(router, eval_data, eval_emb, costs, total_steps):
    rng_state = np.random.get_state()
    r = c = 0.0
    for p, x in zip(eval_data, eval_emb):
        m, _log = router.route(x, total_steps=total_steps)
        r += p["rewards"][m]; c += costs[m]
    np.random.set_state(rng_state)
    n = len(eval_data)
    return r / n, c / n


def run_banditgpt_sweep(models, train_data, eval_data, train_emb, eval_emb,
                        costs, r_min, r_max, lambda_values, n_trials):
    dim = train_emb[0].shape[0]
    burn_in = len(train_data)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    results = []
    for lam in lambda_values:
        trial_r, trial_c = [], []
        for trial in range(n_trials):
            np.random.seed(SEED_OFFSET + trial)
            router = create_experiment_router(
                model_registry=build_model_registry(models),
                feature_dim=dim,
                prior_n_effective=TARGET_NEFF, alpha=ALPHA_START,
                warmup_path=str(MULTIMODEL_WARMUP_PRIORS_PATH),
                use_corralling=True,
                corralling_learning_rate=CORRALLING_LR,
                corralling_gamma=CORRALLING_GAMMA,
                cost_penalty=lam,
            )
            for p, x in zip(train_data, train_emb):
                m, log = router.route(x, total_steps=burn_in)
                norm_r = (p["rewards"][m] - r_min) / r_range
                router.process_feedback(log.request_id, norm_r)
            r, c = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
            trial_r.append(r); trial_c.append(c)

        results.append({
            "lambda": lam,
            "mean_reward": float(np.mean(trial_r)),
            "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
            "mean_cost": float(np.mean(trial_c)),
            "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
            "n_trials": n_trials, "label": "banditGPT",
        })
        logger.info(f"    banditGPT  λ={lam:<5} R={np.mean(trial_r):.4f}±{np.std(trial_r):.4f}")
    return results


def run_lints_sweep(models, train_data, eval_data, train_emb, eval_emb,
                    costs, r_min, r_max, lambda_values, n_trials,
                    use_priors=True, label="LinTS"):
    dim = train_emb[0].shape[0]
    burn_in = len(train_data)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    lints_costs = build_lints_costs(models)
    priors = load_warmup_priors(models) if use_priors else None

    results = []
    for lam in lambda_values:
        trial_r, trial_c = [], []
        for trial in range(n_trials):
            np.random.seed(SEED_OFFSET + trial)
            ts = CostAwareLinTSRouter(
                models=models, context_dim=dim, model_costs=lints_costs,
                cost_penalty=lam, noise_variance=0.25,
                warmup_priors=priors, ridge_lambda=1.0,
            )
            for p, x in zip(train_data, train_emb):
                m = ts.select_model(x, total_steps=burn_in)
                norm_r = (p["rewards"][m] - r_min) / r_range
                ts.update(x, m, norm_r)

            rng_state = np.random.get_state()
            r_t = c_t = 0.0
            for p, x in zip(eval_data, eval_emb):
                m = ts.select_model(x, total_steps=burn_in)
                r_t += p["rewards"][m]; c_t += costs[m]
            np.random.set_state(rng_state)
            trial_r.append(r_t / len(eval_data))
            trial_c.append(c_t / len(eval_data))

        results.append({
            "lambda": lam,
            "mean_reward": float(np.mean(trial_r)),
            "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
            "mean_cost": float(np.mean(trial_c)),
            "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
            "n_trials": n_trials, "label": label,
        })
        logger.info(f"    {label:<16} λ={lam:<5} R={np.mean(trial_r):.4f}±{np.std(trial_r):.4f}")
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("Figure 7: LinTS Baseline Comparison (K=5, K=10)")
    logger.info("=" * 70)

    results_all = {}

    for portfolio_name, models in [("K5", PORTFOLIO_K5), ("K10", PORTFOLIO_K10)]:
        K = len(models)
        logger.info(f"\n{'='*70}")
        logger.info(f"PORTFOLIO: {portfolio_name} ({K} models)")
        logger.info("=" * 70)

        train_data, eval_data, train_emb, eval_emb, costs, r_min, r_max = \
            load_multimodel_data(models)

        # Baselines
        oracle_r, oracle_c = oracle_route(eval_data, models, costs)
        static_results = {m: dict(zip(("reward", "cost"), static_route(eval_data, m, costs)))
                          for m in models}
        rand_r, rand_c = random_route(eval_data, models, costs)
        eg_r, eg_c = epsilon_greedy_route(train_data, eval_data, models, costs)

        logger.info(f"  Oracle: {oracle_r:.4f} | Best static: "
                     f"{max(static_results.values(), key=lambda x: x['reward'])['reward']:.4f} | "
                     f"Random: {rand_r:.4f}")

        # banditGPT Pareto sweep
        t0 = time.time()
        logger.info(f"\n  banditGPT sweep ({len(LAMBDA_VALUES)} λ × {N_TRIALS} trials) ...")
        pareto_bandit = run_banditgpt_sweep(
            models, train_data, eval_data, train_emb, eval_emb,
            costs, r_min, r_max, LAMBDA_VALUES, N_TRIALS,
        )
        logger.info(f"  Done in {time.time()-t0:.0f}s")

        # LinTS (warmup priors) Pareto sweep
        t0 = time.time()
        logger.info(f"\n  LinTS (warmup) sweep ({len(LAMBDA_VALUES)} λ × {N_TRIALS} trials) ...")
        pareto_lints = run_lints_sweep(
            models, train_data, eval_data, train_emb, eval_emb,
            costs, r_min, r_max, LAMBDA_VALUES, N_TRIALS,
            use_priors=True, label="LinTS",
        )
        logger.info(f"  Done in {time.time()-t0:.0f}s")

        # LinTS (tabula rasa) Pareto sweep
        t0 = time.time()
        logger.info(f"\n  LinTS (no priors) sweep ({len(LAMBDA_VALUES)} λ × {N_TRIALS} trials) ...")
        pareto_lints_tr = run_lints_sweep(
            models, train_data, eval_data, train_emb, eval_emb,
            costs, r_min, r_max, LAMBDA_VALUES, N_TRIALS,
            use_priors=False, label="LinTS (no priors)",
        )
        logger.info(f"  Done in {time.time()-t0:.0f}s")

        best_static_m = max(static_results, key=lambda m: static_results[m]["reward"])
        peak_bandit = max(pareto_bandit, key=lambda x: x["mean_reward"])
        peak_lints = max(pareto_lints, key=lambda x: x["mean_reward"])

        results_all[portfolio_name] = {
            "K": K,
            "models": [{"id": m, **MODEL_CATALOG[m]} for m in models],
            "oracle": {"reward": oracle_r, "cost": oracle_c},
            "random": {"reward": rand_r, "cost": rand_c},
            "epsilon_greedy": {"reward": eg_r, "cost": eg_c},
            "static": static_results,
            "best_static": {
                "model": best_static_m,
                "reward": static_results[best_static_m]["reward"],
                "cost": static_results[best_static_m]["cost"],
            },
            "pareto_banditgpt": pareto_bandit,
            "pareto_lints": pareto_lints,
            "pareto_lints_tabula": pareto_lints_tr,
            "n_train": len(train_data),
            "n_eval": len(eval_data),
            "n_trials": N_TRIALS,
        }

        logger.info(f"\n  SUMMARY ({portfolio_name}):")
        logger.info(f"    Oracle:         {oracle_r:.4f}")
        logger.info(f"    banditGPT peak: {peak_bandit['mean_reward']:.4f} ± {peak_bandit['std_reward']:.4f}")
        logger.info(f"    LinTS peak:     {peak_lints['mean_reward']:.4f} ± {peak_lints['std_reward']:.4f}")
        logger.info(f"    Best static:    {static_results[best_static_m]['reward']:.4f}")

    out_path = Path(__file__).parent / "results" / "lints_comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
