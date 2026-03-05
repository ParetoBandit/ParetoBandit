#!/usr/bin/env python3
"""
Figure 7: LinTS Baseline Comparison at K=3
============================================

Compares banditGPT (LinUCB + Corralling) against Linear Thompson Sampling
(LinTS), the primary alternative contextual bandit algorithm.

Both methods use:
  - Portfolio-specific warmup priors (K=3: Llama-3.1-8B, Gemini-2.5-Flash, GPT-4.1)
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
import joblib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    K3_WARMUP_PRIORS_PATH,
    K3_MODELS_PATH,
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    THREE_WAY_SPLITS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
)
from bandit_gpt.baselines import CostAwareLinTSRouter
from utils.router_factory import create_experiment_router
from utils.model_pricing import load_model_catalog
from utils.multimodel import (
    N_TRIALS, SEED_OFFSET, TARGET_NEFF, ALPHA_START,
    CORRALLING_LR, CORRALLING_GAMMA,
    compute_normalized_cost,
    load_rewards, load_holdout_rewards, load_warmup_priors,
    oracle_route, static_route, random_route, epsilon_greedy_route,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# K=3 portfolio (Llama-3.1-8B, Gemini-2.5-Flash, GPT-4.1)
K3_MODELS, K3_CATALOG = load_model_catalog(K3_MODELS_PATH)

LAMBDA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


# ============================================================================
# K3-CATALOG-AWARE HELPERS
# ============================================================================

def _build_model_registry(models: list[str]) -> dict[str, dict]:
    """Build model registry from K3_CATALOG for ``create_experiment_router``."""
    return {
        m: {
            "input_cost_per_m": K3_CATALOG[m]["input_cost_per_m"],
            "output_cost_per_m": K3_CATALOG[m]["output_cost_per_m"],
        }
        for m in models
    }


def _build_lints_costs(models: list[str]) -> dict[str, dict]:
    """Build normalized cost dict from K3_CATALOG for ``CostAwareLinTSRouter``."""
    return {
        m: {"normalized_cost": compute_normalized_cost(
            K3_CATALOG[m]["input_cost_per_m"],
            K3_CATALOG[m]["output_cost_per_m"],
        )}
        for m in models
    }


def _load_data(models: list[str]):
    """Load train/eval data and embeddings for the K=3 portfolio.

    Returns:
        (train_data, eval_data, train_emb, eval_emb, costs, r_min, r_max)
    """
    from sentence_transformers import SentenceTransformer
    from utils.embeddings import load_embedding_cache, embed_dataset_cached

    with open(THREE_WAY_SPLITS_PATH) as f:
        splits = json.load(f)
    online_prompts = splits["online_learn_pool"]

    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    _cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    train_data = load_rewards(DEV_DATA_PATH_ALL_MODELS, online_prompts, models)
    eval_data = load_holdout_rewards(models)

    logger.info(
        f"  Train: {len(train_data)} | Eval: {len(eval_data)} "
        f"| dim: {pca.n_components_}+1"
    )

    train_emb = embed_dataset_cached(train_data, _cache, encoder, pca)
    eval_emb = embed_dataset_cached(eval_data, _cache, encoder, pca)

    costs = {m: K3_CATALOG[m]["cost"] for m in models}
    all_raw = [p["rewards"][m] for p in train_data for m in models]
    r_min, r_max = min(all_raw), max(all_raw)

    return train_data, eval_data, train_emb, eval_emb, costs, r_min, r_max


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
                model_registry=_build_model_registry(models),
                feature_dim=dim,
                prior_n_effective=TARGET_NEFF, alpha=ALPHA_START,
                warmup_path=str(K3_WARMUP_PRIORS_PATH),
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
    lints_costs = _build_lints_costs(models)
    priors = load_warmup_priors(models, warmup_path=K3_WARMUP_PRIORS_PATH) if use_priors else None

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
    logger.info("Figure 7: LinTS Baseline Comparison (K=3)")
    logger.info("=" * 70)

    models = K3_MODELS
    K = len(models)
    logger.info(f"\nPORTFOLIO: K3 ({K} models)")
    logger.info("=" * 70)

    train_data, eval_data, train_emb, eval_emb, costs, r_min, r_max = \
        _load_data(models)

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

    results_all = {
        "K3": {
            "K": K,
            "models": [{"id": m, **K3_CATALOG[m]} for m in models],
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
    }

    logger.info(f"\n  SUMMARY (K3):")
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
