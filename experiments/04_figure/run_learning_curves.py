#!/usr/bin/env python3
"""
Learning Curve Analysis: Convergence During Dev-Set Burn-In
============================================================

Tracks per-step reward during Phase 1 (burn-in on dev set) and Phase 2
(holdout evaluation) to show:
  1. Convergence of cumulative reward during training
  2. Expert weight evolution (warmup vs tabula rasa)
  3. Holdout reward vs. burn-in steps (early stopping analysis)

Protocol: banditGPT-Hybrid at λ=0, 20 seeds.
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
ALPHA_START = 2.0
ALPHA_END = 0.1
TARGET_NEFF = 10.0
LAMBDA = 0.0


def precompute_embeddings(data, encoder, pca):
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


def evaluate_holdout(router, eval_data, eval_emb, burn_in_steps):
    """Evaluate current router on full holdout (exploitation only)."""
    total = 0.0
    for i, p in enumerate(eval_data):
        x = eval_emb[i]
        sel, _ = router.select_model(x, total_steps=burn_in_steps)
        total += p["rewards"][sel]
    return total / len(eval_data)


def run_single_trial(
    train_data, eval_data, train_emb, eval_emb,
    warmup_priors, model_costs, seed,
    eval_checkpoints,
):
    """Run one trial, recording per-step metrics and holdout reward at checkpoints."""
    np.random.seed(seed)
    scaled_priors = normalize_prior_strength(warmup_priors, TARGET_NEFF)
    models = list(train_data[0]["rewards"].keys())
    dim = scaled_priors["context_dim"]

    warmup_expert = CostAwareLinUCBRouter(
        models=models, warmup_priors=scaled_priors, model_costs=model_costs,
        alpha_start=ALPHA_START, alpha_end=ALPHA_END, cost_penalty=LAMBDA,
    )
    tabula_rasa = CostAwareTabulaRasaRouter(
        models=models, context_dim=dim, model_costs=model_costs,
        alpha_start=ALPHA_START, alpha_end=ALPHA_END, cost_penalty=LAMBDA,
    )
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa],
        models=models, learning_rate=1.0,
    )

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Track per-step metrics
    per_step_rewards = []
    expert_weights = []
    holdout_at_checkpoints = {}

    # Phase 0: Evaluate before any training (cold-start holdout)
    if 0 in eval_checkpoints:
        holdout_at_checkpoints[0] = evaluate_holdout(
            router, eval_data, eval_emb, burn_in_steps
        )

    # Phase 1: Burn-in with tracking
    for i, p in enumerate(train_data):
        x = train_emb[i]
        sel, token = router.select_model(x, total_steps=burn_in_steps)
        raw_r = p["rewards"][sel]
        norm_r = (raw_r - r_min) / r_range
        router.update(x, sel, norm_r, selection_token=token)

        per_step_rewards.append(raw_r)
        expert_weights.append(router.weights.copy())

        step = i + 1
        if step in eval_checkpoints:
            holdout_at_checkpoints[step] = evaluate_holdout(
                router, eval_data, eval_emb, burn_in_steps
            )

    # Final holdout (full burn-in)
    if burn_in_steps not in holdout_at_checkpoints:
        holdout_at_checkpoints[burn_in_steps] = evaluate_holdout(
            router, eval_data, eval_emb, burn_in_steps
        )

    return {
        "per_step_rewards": per_step_rewards,
        "expert_weights": [w.tolist() for w in expert_weights],
        "holdout_at_checkpoints": holdout_at_checkpoints,
    }


def main():
    logger.info("=" * 70)
    logger.info("LEARNING CURVE ANALYSIS")
    logger.info("=" * 70)

    # Load data
    logger.info("\n--- Loading data ---")
    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    sanitized_path = (
        Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    )
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
            if cost_range > 0 else 0.0,
        }
        for m in models
    }

    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    train_emb = precompute_embeddings(train_data, encoder, pca)
    eval_emb = precompute_embeddings(eval_data, encoder, pca)
    logger.info(f"  {len(train_emb)}+{len(eval_emb)} prompts in {time.time()-t0:.1f}s")

    n_train = len(train_data)
    # Evaluate holdout at these checkpoints during burn-in
    eval_checkpoints = set([0, 10, 25, 50, 100, 200, 300, 500, 750, 1000, n_train])
    eval_checkpoints = {c for c in eval_checkpoints if c <= n_train}

    logger.info(
        f"\nProtocol: {N_TRIALS} seeds, λ={LAMBDA}, "
        f"burn-in={n_train} steps, holdout={len(eval_data)}"
    )
    logger.info(f"Checkpoints: {sorted(eval_checkpoints)}")

    t_start = time.time()
    all_trials = []

    for trial in range(N_TRIALS):
        seed = SEED_OFFSET + trial
        result = run_single_trial(
            train_data, eval_data, train_emb, eval_emb,
            warmup_priors, normalized_costs, seed,
            eval_checkpoints,
        )
        all_trials.append(result)
        if trial % 5 == 0:
            final_holdout = result["holdout_at_checkpoints"][n_train]
            logger.info(f"  Trial {trial:2d}: holdout reward = {final_holdout:.4f}")

    elapsed = time.time() - t_start
    logger.info(f"\n--- {N_TRIALS} trials in {elapsed:.0f}s ---")

    # Aggregate: cumulative training reward (smoothed)
    window = 50
    cum_rewards_all = []
    for trial in all_trials:
        r = np.array(trial["per_step_rewards"])
        # Moving average
        cum = np.convolve(r, np.ones(window) / window, mode="valid")
        cum_rewards_all.append(cum)

    cum_rewards_all = np.array(cum_rewards_all)
    cum_mean = cum_rewards_all.mean(axis=0).tolist()
    cum_std = cum_rewards_all.std(axis=0, ddof=1).tolist()

    # Aggregate: expert weights
    weights_all = np.array([t["expert_weights"] for t in all_trials])  # (trials, steps, 2)
    weights_mean = weights_all.mean(axis=0).tolist()
    weights_std = weights_all.std(axis=0, ddof=1).tolist()

    # Aggregate: holdout at checkpoints
    checkpoints_sorted = sorted(eval_checkpoints)
    holdout_agg = {}
    for cp in checkpoints_sorted:
        vals = [t["holdout_at_checkpoints"][cp] for t in all_trials]
        avg = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(vals))
        holdout_agg[cp] = {
            "mean": avg, "std": std, "ci95": ci95, "n": len(vals),
        }

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "learning_curves.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "description": "Learning curve analysis: convergence during burn-in",
                    "n_trials": N_TRIALS,
                    "n_train": n_train,
                    "n_eval": len(eval_data),
                    "smoothing_window": window,
                    "lambda": LAMBDA,
                },
                "training_reward_smoothed": {
                    "mean": cum_mean,
                    "std": cum_std,
                    "window": window,
                    "n_points": len(cum_mean),
                },
                "expert_weights": {
                    "mean": weights_mean,
                    "std": weights_std,
                    "labels": ["warmup", "tabula_rasa"],
                },
                "holdout_at_checkpoints": {
                    str(k): v for k, v in holdout_agg.items()
                },
            },
            f,
            indent=2,
        )

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("HOLDOUT REWARD vs. BURN-IN STEPS")
    logger.info("=" * 60)
    logger.info(f"{'Steps':<8} {'Holdout Reward':<16} {'95% CI':<10}")
    logger.info("-" * 40)
    for cp in checkpoints_sorted:
        h = holdout_agg[cp]
        logger.info(f"{cp:<8d} {h['mean']:<16.4f} ±{h['ci95']:.4f}")

    # Expert weight summary
    final_weights = weights_all[:, -1, :]  # (trials, 2)
    logger.info(f"\nFinal expert weights (step {n_train}):")
    logger.info(f"  Warmup:      {final_weights[:,0].mean():.3f} ± {final_weights[:,0].std():.3f}")
    logger.info(f"  Tabula Rasa: {final_weights[:,1].mean():.3f} ± {final_weights[:,1].std():.3f}")

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
