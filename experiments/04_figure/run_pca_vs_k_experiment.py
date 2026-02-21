#!/usr/bin/env python3
"""
PCA Dimensionality vs Number of Models (K) — Hypothesis Test
=============================================================

Tests the hypothesis: optimal PCA dimensionality scales with the number of
models (K) in the routing portfolio.

Design:
  - Real LMSYS prompt embeddings (all-MiniLM-L6-v2) as feature source
  - Synthetic context-dependent rewards:
      reward(prompt, model_k) = base + scale * tanh(z_k_norm) + noise
    where z_k_norm is the k-th PCA score, normalized to unit variance.
    Each model's "expertise" is aligned with a different principal component.
  - K in {2, 5, 8}
  - PCA dim in {2, 4, 8, 16, 32, 64}
  - 20 seeds per (K, dim) combination
  - Router: LinUCB (no priors) — isolates the PCA dimensionality effect

Expected result:
  - K=2:  d >= 2 captures both PCs -> flat performance
  - K=5:  d < 5 loses signal -> lower reward; d >= 8 captures all
  - K=8:  d < 8 loses signal -> lower reward; d >= 8 captures all

This validates the paper's choice of d=32 "for generality to larger portfolios":
  d=4 is optimal for K=2 but suboptimal for K >= 5.

Protocol matches Figure 4: train on dev set (N=1,121) with bandit feedback,
evaluate on holdout (N=750) with pure exploitation, 20 independent seeds.
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

from generate_pareto_frontier import load_dataset_with_split
from bandit_gpt.router import CostAwareTabulaRasaRouter
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---- Experiment Parameters ----
N_TRIALS = 20
SEED_OFFSET = 42
ALPHA_START = 2.0
ALPHA_END = 0.1

K_VALUES = [2, 5, 8]
PCA_DIMS = [2, 4, 8, 16, 32, 64]

# Reward parameters
REWARD_BASE = 0.50
REWARD_SCALE = 0.15  # max context bonus: ±0.15 via tanh
REWARD_NOISE_STD = 0.05
REWARD_NOISE_SEED = 999


def generate_synthetic_rewards(raw_embeddings, pca_full, z_std, K):
    """Generate fixed synthetic rewards for K models.

    Model k's expertise is aligned with the k-th principal component.
    PCA scores are normalized to unit variance so each model has equally
    strong context-dependent preferences regardless of PC index.

    Returns:
        rewards: list of dicts {model_name: reward} per prompt
        model_names: list of model name strings
    """
    rng = np.random.RandomState(REWARD_NOISE_SEED)
    z_full = pca_full.transform(raw_embeddings)

    model_names = [f"model_{k}" for k in range(K)]
    rewards = []

    for i in range(len(raw_embeddings)):
        prompt_rewards = {}
        for k, model in enumerate(model_names):
            if k < z_full.shape[1]:
                z_norm = z_full[i, k] / (z_std[k] + 1e-8)
                context_signal = REWARD_SCALE * np.tanh(z_norm)
            else:
                context_signal = 0.0
            r = REWARD_BASE + context_signal + rng.normal(0, REWARD_NOISE_STD)
            prompt_rewards[model] = float(np.clip(r, 0.0, 1.0))
        rewards.append(prompt_rewards)

    return rewards, model_names


def compute_oracle_reward(eval_rewards):
    """Compute oracle (best-model-per-prompt) average reward."""
    return np.mean([max(p.values()) for p in eval_rewards])


def compute_random_reward(eval_rewards):
    """Compute random-routing average reward."""
    K = len(eval_rewards[0])
    return np.mean([sum(p.values()) / K for p in eval_rewards])


def run_linucb(train_rewards, eval_rewards, train_emb, eval_emb, models, seed):
    """Run LinUCB (no priors) with train/eval protocol."""
    np.random.seed(seed)
    dim = len(train_emb[0])
    model_costs = {m: {"cost": 0.0, "normalized_cost": 0.0} for m in models}

    router = CostAwareTabulaRasaRouter(
        models=models, context_dim=dim, model_costs=model_costs,
        alpha_start=ALPHA_START, alpha_end=ALPHA_END, cost_penalty=0.0,
    )

    # Normalize rewards for bandit update
    all_raw = [r for p in train_rewards for r in p.values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    # Burn-in on training data
    for i in range(len(train_rewards)):
        x = train_emb[i]
        sel = router.select_model(x, total_steps=len(train_rewards))
        norm_r = (train_rewards[i][sel] - r_min) / r_range
        router.update(x, sel, norm_r)

    # Pure exploitation on eval data
    total_reward = 0.0
    for i in range(len(eval_rewards)):
        x = eval_emb[i]
        sel = router.select_model(x, total_steps=len(train_rewards))
        total_reward += eval_rewards[i][sel]

    return total_reward / len(eval_rewards)


def main():
    logger.info("=" * 70)
    logger.info("PCA DIMENSIONALITY vs NUMBER OF MODELS (K)")
    logger.info("Hypothesis: optimal PCA dim scales with K")
    logger.info("=" * 70)

    # Load LMSYS prompts (only need prompts for embeddings; rewards are synthetic)
    train_data, eval_data, stats = load_dataset_with_split()
    logger.info(f"Loaded {len(train_data)} train, {len(eval_data)} eval prompts")

    # Encode raw 384-dim embeddings
    logger.info("\n--- Encoding raw embeddings ---")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    t0 = time.time()
    raw_train = encoder.encode(
        [p["prompt"] for p in train_data],
        normalize_embeddings=True, show_progress_bar=False,
    )
    raw_eval = encoder.encode(
        [p["prompt"] for p in eval_data],
        normalize_embeddings=True, show_progress_bar=False,
    )
    logger.info(f"  Encoded {len(raw_train)}+{len(raw_eval)} in {time.time()-t0:.1f}s")

    # Fit ground-truth PCA at max dimensionality
    max_dim = max(PCA_DIMS)
    pca_full = PCA(n_components=max_dim)
    pca_full.fit(raw_train)
    explained = pca_full.explained_variance_ratio_
    logger.info(f"  PCA({max_dim}) explained variance: {explained.sum():.3f}")
    logger.info(f"  Top 8 PCs: {', '.join(f'{v:.3f}' for v in explained[:8])}")

    # Compute PCA score std on training data (for reward normalization)
    z_train_full = pca_full.transform(raw_train)
    z_std = z_train_full.std(axis=0)
    logger.info(f"  PCA score stds (first 8): {', '.join(f'{s:.3f}' for s in z_std[:8])}")

    t_start = time.time()
    all_results = {}

    for K in K_VALUES:
        logger.info(f"\n{'='*60}")
        logger.info(f"K = {K} models")
        logger.info(f"{'='*60}")

        # Generate fixed synthetic rewards
        train_rewards, models = generate_synthetic_rewards(
            raw_train, pca_full, z_std, K,
        )
        eval_rewards, _ = generate_synthetic_rewards(
            raw_eval, pca_full, z_std, K,
        )

        oracle_r = compute_oracle_reward(eval_rewards)
        random_r = compute_random_reward(eval_rewards)
        logger.info(f"  Oracle reward: {oracle_r:.4f}")
        logger.info(f"  Random reward: {random_r:.4f}")

        k_results = {"oracle": oracle_r, "random": random_r, "pca_sweep": {}}

        for d in PCA_DIMS:
            # Fit PCA at this dimensionality
            pca_d = PCA(n_components=d)
            pca_d.fit(raw_train)

            # Compute d-dim embeddings + bias
            train_emb = [
                np.append(pca_d.transform(raw_train[i:i+1])[0], 1.0)
                for i in range(len(raw_train))
            ]
            eval_emb = [
                np.append(pca_d.transform(raw_eval[i:i+1])[0], 1.0)
                for i in range(len(raw_eval))
            ]

            # Run N_TRIALS
            rewards = []
            for trial in range(N_TRIALS):
                seed = SEED_OFFSET + trial
                r = run_linucb(
                    train_rewards, eval_rewards,
                    train_emb, eval_emb, models, seed,
                )
                rewards.append(r)

            avg = np.mean(rewards)
            std = np.std(rewards, ddof=1)
            t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1) if N_TRIALS > 1 else 1.96
            ci95 = t_crit * std / np.sqrt(N_TRIALS)
            k_results["pca_sweep"][d] = {
                "mean": float(avg),
                "std": float(std),
                "ci95": float(ci95),
            }

            # Indicate whether d captures all K expertise dimensions
            captures_all = "yes" if d >= K else f"no (captures {d}/{K})"
            logger.info(
                f"  d={d:<4}  Reward={avg:.4f} ± {ci95:.4f}  "
                f"[captures all: {captures_all}]"
            )

        all_results[K] = k_results

    elapsed = time.time() - t_start
    logger.info(f"\n--- Experiment complete in {elapsed:.0f}s ---")

    # ---- Save results ----
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pca_vs_k_experiment.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "description": "PCA dimensionality vs K (number of models)",
                    "hypothesis": "Optimal PCA dim scales with K",
                    "n_trials": N_TRIALS,
                    "reward_scale": REWARD_SCALE,
                    "reward_noise_std": REWARD_NOISE_STD,
                    "reward_base": REWARD_BASE,
                    "alpha_start": ALPHA_START,
                    "alpha_end": ALPHA_END,
                    "n_train": len(train_data),
                    "n_eval": len(eval_data),
                    "K_values": K_VALUES,
                    "PCA_dims": PCA_DIMS,
                },
                "results": {str(k): v for k, v in all_results.items()},
            },
            f, indent=2,
        )

    # ---- Summary table ----
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Mean Holdout Reward by (K, PCA dim)")
    logger.info("=" * 70)

    header = f"{'K':>4} | {'Oracle':>7} | {'Random':>7} | " + " | ".join(
        f"d={d:<3}" for d in PCA_DIMS
    )
    logger.info(header)
    logger.info("-" * len(header))

    for K in K_VALUES:
        res = all_results[K]
        row = f"{K:>4} | {res['oracle']:>7.4f} | {res['random']:>7.4f} | "
        row += " | ".join(
            f"{res['pca_sweep'][d]['mean']:.4f}" for d in PCA_DIMS
        )
        logger.info(row)

    # ---- Key finding ----
    logger.info("\n--- Hypothesis Check ---")
    for K in K_VALUES:
        res = all_results[K]
        best_d = max(PCA_DIMS, key=lambda d: res["pca_sweep"][d]["mean"])
        worst_d = min(PCA_DIMS, key=lambda d: res["pca_sweep"][d]["mean"])
        gap = res["pca_sweep"][best_d]["mean"] - res["pca_sweep"][worst_d]["mean"]
        logger.info(
            f"  K={K}: best d={best_d} ({res['pca_sweep'][best_d]['mean']:.4f}), "
            f"worst d={worst_d} ({res['pca_sweep'][worst_d]['mean']:.4f}), "
            f"gap={gap:.4f}"
        )

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
