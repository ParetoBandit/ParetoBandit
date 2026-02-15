#!/usr/bin/env python3
"""
Cold-Start Ablation: Quantifying the Value of Dev-Set Training
===============================================================

Addresses the fairness concern that banditGPT trains on 1,121 dev prompts
while RouteLLM uses only pre-trained weights (no access to dev set).

Compares two conditions:
  - Warm-start: banditGPT trained on dev set, then evaluated on holdout
  - Cold-start: banditGPT evaluated directly on holdout (priors only, no dev training)

Protocol: identical to generate_pareto_frontier.py
  - Same 10 lambda values, 20 seeds, hyperparameters
  - Cold-start skips Phase 1 (burn-in) entirely
"""

import sys
from pathlib import Path
import json
import numpy as np
import logging
import time
from typing import Dict, List, Tuple

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
TARGET_SAMPLE_SIZE = 10.0
COST_PENALTIES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def precompute_embeddings(data, encoder, pca):
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


def banditgpt_run(
    train_data, eval_data, train_emb, eval_emb,
    warmup_priors, model_costs, lambda_penalty, cold_start=False,
):
    """Run banditGPT-Hybrid with or without dev-set training."""
    scaled_priors = normalize_prior_strength(warmup_priors, TARGET_SAMPLE_SIZE)
    models = list(train_data[0]["rewards"].keys())
    dim = scaled_priors["context_dim"]

    warmup_expert = CostAwareLinUCBRouter(
        models=models, warmup_priors=scaled_priors, model_costs=model_costs,
        alpha_start=ALPHA_START, alpha_end=ALPHA_END, cost_penalty=lambda_penalty,
    )
    tabula_rasa = CostAwareTabulaRasaRouter(
        models=models, context_dim=dim, model_costs=model_costs,
        alpha_start=ALPHA_START, alpha_end=ALPHA_END, cost_penalty=lambda_penalty,
    )
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa], models=models, learning_rate=1.0,
    )

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Phase 1: Burn-in (skipped for cold-start)
    if not cold_start:
        for i, p in enumerate(train_data):
            x = train_emb[i]
            sel, token = router.select_model(x, total_steps=burn_in_steps)
            norm_r = (p["rewards"][sel] - r_min) / r_range
            router.update(x, sel, norm_r, selection_token=token)

    # Phase 2: Evaluation
    total_reward, total_cost = 0.0, 0.0
    for i, p in enumerate(eval_data):
        x = eval_emb[i]
        sel, _ = router.select_model(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][sel]
        total_cost += model_costs[sel]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


def run_sweep(func, label):
    """Run a method across all lambdas and seeds, return results list."""
    results = []
    for lam in COST_PENALTIES:
        rewards, costs = [], []
        for trial in range(N_TRIALS):
            np.random.seed(SEED_OFFSET + trial)
            r, c = func(lam)
            rewards.append(r)
            costs.append(c)

        avg_r, avg_c = np.mean(rewards), np.mean(costs)
        std_r = np.std(rewards, ddof=1) if N_TRIALS > 1 else 0.0
        std_c = np.std(costs, ddof=1) if N_TRIALS > 1 else 0.0
        ci95_r = 1.96 * std_r / np.sqrt(N_TRIALS)
        ci95_c = 1.96 * std_c / np.sqrt(N_TRIALS)

        results.append({
            "lambda": lam, "reward": avg_r, "cost": avg_c,
            "reward_std": std_r, "cost_std": std_c,
            "reward_ci95": ci95_r, "cost_ci95": ci95_c,
            "n_trials": N_TRIALS,
        })
        logger.info(
            f"  λ={lam:<5.2f}  Reward={avg_r:.4f}±{ci95_r:.4f}  "
            f"Cost=${avg_c:.6f}±${ci95_c:.6f}"
        )
    return results


def main():
    logger.info("=" * 70)
    logger.info("COLD-START ABLATION: Value of Dev-Set Training")
    logger.info("=" * 70)
    logger.info(
        "\nFairness context:\n"
        "  RouteLLM: Pre-trained weights only (no dev set access)\n"
        "  banditGPT warm-start: Trains on 1,121 dev prompts, then evaluates\n"
        "  banditGPT cold-start: Evaluates directly (priors only, no dev training)\n"
        f"\nProtocol: {N_TRIALS} seeds, {len(COST_PENALTIES)} λ values, "
        f"α {ALPHA_START}→{ALPHA_END}, neff={TARGET_SAMPLE_SIZE}"
    )

    # Load everything
    logger.info("\n--- Loading data ---")
    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_priors = joblib.load(sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH)

    models = list(eval_data[0]["rewards"].keys())
    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost
    normalized_costs = {
        m: {"cost": model_costs[m]["cost"],
            "normalized_cost": (model_costs[m]["cost"] - min_cost) / cost_range if cost_range > 0 else 0.0}
        for m in models
    }

    # Pre-compute embeddings
    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    train_emb = precompute_embeddings(train_data, encoder, pca)
    eval_emb = precompute_embeddings(eval_data, encoder, pca)
    logger.info(f"  {len(train_emb)}+{len(eval_emb)} prompts in {time.time()-t0:.1f}s")

    t_start = time.time()

    # Warm-start
    logger.info("\n[1/2] Warm-start (with dev training)")
    warm_results = run_sweep(
        lambda lam: banditgpt_run(
            train_data, eval_data, train_emb, eval_emb,
            warmup_priors, normalized_costs, lam, cold_start=False,
        ),
        "warm-start",
    )

    # Cold-start
    logger.info("\n[2/2] Cold-start (NO dev training)")
    cold_results = run_sweep(
        lambda lam: banditgpt_run(
            train_data, eval_data, train_emb, eval_emb,
            warmup_priors, normalized_costs, lam, cold_start=True,
        ),
        "cold-start",
    )

    elapsed = time.time() - t_start
    logger.info(f"\n--- Complete in {elapsed:.0f}s ---")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cold_start_ablation.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "description": "Cold-start ablation: warm-start vs cold-start (priors only)",
                "n_eval_prompts": len(eval_data),
                "n_train_prompts": len(train_data),
                "n_trials": N_TRIALS,
                "cost_penalties": COST_PENALTIES,
            },
            "warm_start": warm_results,
            "cold_start": cold_results,
        }, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 78)
    logger.info("SUMMARY: Warm-Start vs Cold-Start")
    logger.info("=" * 78)
    logger.info(f"{'λ':<6} | {'Warm-Start':<18} | {'Cold-Start':<18} | {'Δ Reward':<12} | {'Δ%'}")
    logger.info("-" * 78)
    for w, c in zip(warm_results, cold_results):
        delta = w["reward"] - c["reward"]
        pct = 100 * delta / w["reward"] if w["reward"] > 0 else 0
        logger.info(
            f"{w['lambda']:<6.2f} | {w['reward']:.4f}±{w['reward_ci95']:.4f}   | "
            f"{c['reward']:.4f}±{c['reward_ci95']:.4f}   | "
            f"{delta:+.4f}      | {pct:+.1f}%"
        )

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
