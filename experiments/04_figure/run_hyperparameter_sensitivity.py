#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis for the Main Routing Task
==============================================================

Sweeps key hyperparameters on the holdout set (λ=0, quality-focused) to show
that banditGPT-Hybrid's performance is robust to reasonable choices.

Hyperparameters swept:
  1. η (Corralling learning rate): {0.1, 0.3, 0.5, 1.0, 2.0, 5.0}
  2. α_start (UCB exploration start): {0.5, 1.0, 2.0, 4.0}
     (α decays linearly from α_start to 0.1)

Protocol: identical to generate_pareto_frontier.py
  - 20 seeds per configuration, λ=0 (quality-focused)
  - Metrics: holdout reward (mean ± 95% CI)
"""

import sys
from pathlib import Path
import json
import numpy as np
import logging
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from generate_pareto_frontier import (
    load_model_costs,
    load_dataset_with_split,
    normalize_prior_strength,
)
from bandit_gpt.router import (
    CorrallingRouter,
    CostAwareLinUCBRouter,
    CostAwareTabulaRasaRouter,
)
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_TRIALS = 20
SEED_OFFSET = 42
ALPHA_END = 0.1
TARGET_NEFF = 10.0
LAMBDA = 0.0  # quality-focused for sensitivity analysis

ETA_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
ALPHA_START_VALUES = [0.5, 1.0, 2.0, 4.0]


def precompute_embeddings(data, encoder, pca):
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


def run_banditgpt(
    train_data, eval_data, train_emb, eval_emb,
    warmup_priors, model_costs,
    eta=1.0, alpha_start=2.0,
):
    """Run banditGPT-Hybrid with specified hyperparameters."""
    scaled_priors = normalize_prior_strength(warmup_priors, TARGET_NEFF)
    models = list(train_data[0]["rewards"].keys())
    dim = scaled_priors["context_dim"]

    warmup_expert = CostAwareLinUCBRouter(
        models=models, warmup_priors=scaled_priors, model_costs=model_costs,
        alpha_start=alpha_start, alpha_end=ALPHA_END, cost_penalty=LAMBDA,
    )
    tabula_rasa = CostAwareTabulaRasaRouter(
        models=models, context_dim=dim, model_costs=model_costs,
        alpha_start=alpha_start, alpha_end=ALPHA_END, cost_penalty=LAMBDA,
    )
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa],
        models=models,
        learning_rate=eta,
    )

    # Reward normalization
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Phase 1: Burn-in
    for i, p in enumerate(train_data):
        x = train_emb[i]
        sel, token = router.select_model(x, total_steps=burn_in_steps)
        norm_r = (p["rewards"][sel] - r_min) / r_range
        router.update(x, sel, norm_r, selection_token=token)

    # Phase 2: Evaluation (exploitation only)
    total_reward = 0.0
    for i, p in enumerate(eval_data):
        x = eval_emb[i]
        sel, _ = router.select_model(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][sel]

    return total_reward / len(eval_data)


def sweep(name, param_values, run_fn):
    """Sweep a single hyperparameter, return dict of results."""
    results = {}
    for val in param_values:
        rewards = []
        for trial in range(N_TRIALS):
            np.random.seed(SEED_OFFSET + trial)
            r = run_fn(val)
            rewards.append(r)

        avg = np.mean(rewards)
        std = np.std(rewards, ddof=1) if N_TRIALS > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(N_TRIALS)
        results[val] = {
            "mean": avg, "std": std, "ci95": ci95,
            "n_trials": N_TRIALS,
        }
        logger.info(f"  {name}={val:<5}  Reward={avg:.4f} ± {ci95:.4f}")
    return results


def main():
    logger.info("=" * 70)
    logger.info("HYPERPARAMETER SENSITIVITY ANALYSIS (Main Routing Task)")
    logger.info("=" * 70)
    logger.info(
        f"\nProtocol: {N_TRIALS} seeds, λ={LAMBDA}, holdout evaluation\n"
        f"Sweeps: η ∈ {ETA_VALUES}, α_start ∈ {ALPHA_START_VALUES}\n"
    )

    # Load data
    logger.info("--- Loading data ---")
    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_priors = joblib.load(
        sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH
    )

    models = list(eval_data[0]["rewards"].keys())
    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost
    normalized_costs = {
        m: {
            "cost": model_costs[m]["cost"],
            "normalized_cost": (model_costs[m]["cost"] - min_cost) / cost_range
            if cost_range > 0
            else 0.0,
        }
        for m in models
    }

    # Pre-compute embeddings
    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    train_emb = precompute_embeddings(train_data, encoder, pca)
    eval_emb = precompute_embeddings(eval_data, encoder, pca)
    logger.info(f"  {len(train_emb)}+{len(eval_emb)} prompts in {time.time()-t0:.1f}s")

    t_start = time.time()

    # Sweep 1: Corralling learning rate η (α_start fixed at 2.0)
    logger.info("\n[1/2] Sweeping Corralling learning rate η (α_start=2.0 fixed)")
    eta_results = sweep(
        "η",
        ETA_VALUES,
        lambda eta: run_banditgpt(
            train_data, eval_data, train_emb, eval_emb,
            warmup_priors, normalized_costs,
            eta=eta, alpha_start=2.0,
        ),
    )

    # Sweep 2: UCB exploration α_start (η fixed at 1.0)
    logger.info("\n[2/2] Sweeping UCB α_start (η=1.0 fixed)")
    alpha_results = sweep(
        "α_start",
        ALPHA_START_VALUES,
        lambda a: run_banditgpt(
            train_data, eval_data, train_emb, eval_emb,
            warmup_priors, normalized_costs,
            eta=1.0, alpha_start=a,
        ),
    )

    elapsed = time.time() - t_start
    logger.info(f"\n--- Complete in {elapsed:.0f}s ---")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hyperparameter_sensitivity.json"

    # Convert keys to strings for JSON
    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "description": "Hyperparameter sensitivity for main routing task",
                    "lambda": LAMBDA,
                    "n_trials": N_TRIALS,
                    "n_eval": len(eval_data),
                    "n_train": len(train_data),
                },
                "eta_sweep": {str(k): v for k, v in eta_results.items()},
                "alpha_start_sweep": {str(k): v for k, v in alpha_results.items()},
            },
            f,
            indent=2,
        )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info("\nη sweep (α_start=2.0):")
    logger.info(f"  {'η':<6} {'Reward':<12} {'95% CI':<10}")
    logger.info("  " + "-" * 30)
    for eta in ETA_VALUES:
        r = eta_results[eta]
        marker = " <-- default" if eta == 1.0 else ""
        logger.info(f"  {eta:<6.1f} {r['mean']:<12.4f} ±{r['ci95']:.4f}{marker}")

    logger.info("\nα_start sweep (η=1.0):")
    logger.info(f"  {'α_start':<8} {'Reward':<12} {'95% CI':<10}")
    logger.info("  " + "-" * 30)
    for a in ALPHA_START_VALUES:
        r = alpha_results[a]
        marker = " <-- default" if a == 2.0 else ""
        logger.info(f"  {a:<8.1f} {r['mean']:<12.4f} ±{r['ci95']:.4f}{marker}")

    best_eta = max(eta_results.items(), key=lambda x: x[1]["mean"])
    best_alpha = max(alpha_results.items(), key=lambda x: x[1]["mean"])

    logger.info(f"\nBest η: {best_eta[0]} (reward={best_eta[1]['mean']:.4f})")
    logger.info(f"Best α_start: {best_alpha[0]} (reward={best_alpha[1]['mean']:.4f})")
    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
