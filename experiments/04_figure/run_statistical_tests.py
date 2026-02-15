#!/usr/bin/env python3
"""
Statistical Significance Tests: banditGPT vs Baselines
=======================================================

Tests whether banditGPT-Hybrid's advantage over RouteLLM-MF and static
baselines is statistically significant.

Approach:
  - banditGPT produces 20 independent trial means (one per seed)
  - RouteLLM-MF is deterministic at each threshold → a single point
  - We use a one-sample t-test (and Wilcoxon) to test whether banditGPT's
    mean is significantly different from RouteLLM-MF's best aggregate reward.
  - We also compute per-prompt comparisons against static baselines
    (which have known per-prompt selections).
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
N_BOOTSTRAP = 10000

# RouteLLM-MF's best aggregate reward from pareto_results.json
ROUTELLM_BEST_REWARD = 0.8827


def precompute_embeddings(data, encoder, pca):
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


def run_banditgpt_trials(
    train_data, eval_data, train_emb, eval_emb,
    warmup_priors, model_costs,
):
    """Run 20 seeds, return per-trial mean rewards and per-prompt selections."""
    models = list(train_data[0]["rewards"].keys())
    n_eval = len(eval_data)

    trial_means = []
    all_per_prompt = []  # (n_trials, n_eval)

    for trial in range(N_TRIALS):
        np.random.seed(SEED_OFFSET + trial)
        scaled_priors = normalize_prior_strength(warmup_priors, TARGET_NEFF)
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

        for i, p in enumerate(train_data):
            x = train_emb[i]
            sel, token = router.select_model(x, total_steps=len(train_data))
            norm_r = (p["rewards"][sel] - r_min) / r_range
            router.update(x, sel, norm_r, selection_token=token)

        per_prompt = []
        for i, p in enumerate(eval_data):
            x = eval_emb[i]
            sel, _ = router.select_model(x, total_steps=len(train_data))
            per_prompt.append(p["rewards"][sel])

        trial_means.append(np.mean(per_prompt))
        all_per_prompt.append(per_prompt)

    return np.array(trial_means), np.array(all_per_prompt)


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap CI for mean."""
    rng = np.random.RandomState(42)
    boot_means = [data[rng.choice(len(data), size=len(data), replace=True)].mean()
                  for _ in range(n_boot)]
    boot_means = np.array(boot_means)
    return np.percentile(boot_means, 100 * alpha / 2), np.percentile(boot_means, 100 * (1 - alpha / 2))


def main():
    logger.info("=" * 70)
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info("=" * 70)

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
            if cost_range > 0 else 0.0,
        }
        for m in models
    }

    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    train_emb = precompute_embeddings(train_data, encoder, pca)
    eval_emb = precompute_embeddings(eval_data, encoder, pca)
    logger.info(f"  {len(train_emb)}+{len(eval_emb)} prompts in {time.time()-t0:.1f}s")

    t_start = time.time()

    # Run banditGPT
    logger.info("\n--- Running banditGPT (20 seeds) ---")
    trial_means, per_prompt = run_banditgpt_trials(
        train_data, eval_data, train_emb, eval_emb,
        warmup_priors, normalized_costs,
    )
    logger.info(f"  Trial means: {trial_means.mean():.4f} ± {trial_means.std(ddof=1):.4f}")

    # Static baselines (deterministic, per-prompt)
    mixtral_rewards = np.array([p["rewards"]["mistralai/mixtral-8x7b-instruct"] for p in eval_data])
    gpt4_rewards = np.array([p["rewards"]["openai/gpt-4-turbo"] for p in eval_data])
    oracle_rewards = np.maximum(mixtral_rewards, gpt4_rewards)

    logger.info(f"\n  Static Mixtral mean:  {mixtral_rewards.mean():.4f}")
    logger.info(f"  Static GPT-4T mean:   {gpt4_rewards.mean():.4f}")
    logger.info(f"  Oracle mean:          {oracle_rewards.mean():.4f}")
    logger.info(f"  RouteLLM-MF best:     {ROUTELLM_BEST_REWARD:.4f}")

    results = {}

    # ================================================================
    # TEST 1: banditGPT vs RouteLLM-MF (one-sample t-test)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: banditGPT vs RouteLLM-MF (best threshold)")
    logger.info("=" * 70)
    logger.info(f"  H0: banditGPT mean = {ROUTELLM_BEST_REWARD}")
    logger.info(f"  H1: banditGPT mean ≠ {ROUTELLM_BEST_REWARD}")

    t_stat, p_val = sp_stats.ttest_1samp(trial_means, ROUTELLM_BEST_REWARD)
    boot_lo, boot_hi = bootstrap_ci(trial_means)
    cohens_d = (trial_means.mean() - ROUTELLM_BEST_REWARD) / trial_means.std(ddof=1)

    logger.info(f"\n  banditGPT mean:     {trial_means.mean():.4f} ± {1.96*trial_means.std(ddof=1)/np.sqrt(N_TRIALS):.4f}")
    logger.info(f"  RouteLLM-MF best:   {ROUTELLM_BEST_REWARD:.4f}")
    logger.info(f"  Difference:         {trial_means.mean() - ROUTELLM_BEST_REWARD:+.4f}")
    logger.info(f"  t-statistic:        {t_stat:.4f}")
    logger.info(f"  p-value:            {p_val:.2e}")
    logger.info(f"  Bootstrap 95% CI:   [{boot_lo:.4f}, {boot_hi:.4f}]")
    logger.info(f"  Cohen's d:          {cohens_d:.2f}")
    logger.info(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'} (α=0.05)")

    # Wilcoxon signed-rank (against constant)
    w_stat, p_wilcoxon = sp_stats.wilcoxon(trial_means - ROUTELLM_BEST_REWARD)

    logger.info(f"  Wilcoxon p-value:   {p_wilcoxon:.2e}")

    results["vs_routellm"] = {
        "bandit_mean": float(trial_means.mean()),
        "bandit_std": float(trial_means.std(ddof=1)),
        "bandit_ci95": float(1.96 * trial_means.std(ddof=1) / np.sqrt(N_TRIALS)),
        "routellm_best": ROUTELLM_BEST_REWARD,
        "difference": float(trial_means.mean() - ROUTELLM_BEST_REWARD),
        "ttest": {"t": float(t_stat), "p": float(p_val)},
        "wilcoxon": {"w": float(w_stat), "p": float(p_wilcoxon)},
        "bootstrap_ci": [float(boot_lo), float(boot_hi)],
        "cohens_d": float(cohens_d),
    }

    # ================================================================
    # TEST 2: banditGPT vs Static Mixtral (per-prompt paired)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: banditGPT (majority vote) vs Static Mixtral")
    logger.info("=" * 70)

    # Majority vote per-prompt
    majority_rewards = per_prompt.mean(axis=0)  # average reward per prompt across seeds
    diff_mixtral = majority_rewards - mixtral_rewards

    t_stat_m, p_val_m = sp_stats.ttest_rel(majority_rewards, mixtral_rewards)
    nonzero = diff_mixtral[diff_mixtral != 0]
    if len(nonzero) > 0:
        w_m, p_w_m = sp_stats.wilcoxon(nonzero)
    else:
        w_m, p_w_m = 0, 1.0

    pooled_std = np.sqrt((majority_rewards.std(ddof=1)**2 + mixtral_rewards.std(ddof=1)**2) / 2)
    d_m = diff_mixtral.mean() / pooled_std if pooled_std > 0 else 0

    logger.info(f"  banditGPT (avg):  {majority_rewards.mean():.4f}")
    logger.info(f"  Static Mixtral:   {mixtral_rewards.mean():.4f}")
    logger.info(f"  Mean difference:  {diff_mixtral.mean():+.4f}")
    logger.info(f"  Paired t:         {t_stat_m:.4f}, p = {p_val_m:.2e}")
    logger.info(f"  Wilcoxon:         p = {p_w_m:.2e}")
    logger.info(f"  Cohen's d:        {d_m:.2f}")

    results["vs_mixtral"] = {
        "difference": float(diff_mixtral.mean()),
        "ttest_p": float(p_val_m),
        "wilcoxon_p": float(p_w_m),
        "cohens_d": float(d_m),
    }

    # ================================================================
    # TEST 3: banditGPT vs Oracle (upper bound gap)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: banditGPT vs Oracle (gap characterization)")
    logger.info("=" * 70)

    diff_oracle = majority_rewards - oracle_rewards
    logger.info(f"  Oracle:           {oracle_rewards.mean():.4f}")
    logger.info(f"  banditGPT:        {majority_rewards.mean():.4f}")
    logger.info(f"  Gap:              {diff_oracle.mean():+.4f}")
    logger.info(f"  Oracle gap closure: {(majority_rewards.mean() - mixtral_rewards.mean()) / (oracle_rewards.mean() - mixtral_rewards.mean()) * 100:.1f}%")

    results["vs_oracle"] = {
        "oracle_mean": float(oracle_rewards.mean()),
        "gap": float(diff_oracle.mean()),
        "gap_closure_pct": float(
            (majority_rewards.mean() - mixtral_rewards.mean())
            / (oracle_rewards.mean() - mixtral_rewards.mean()) * 100
        ),
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t_start

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  banditGPT > RouteLLM-MF: {trial_means.mean() - ROUTELLM_BEST_REWARD:+.4f} "
                f"(p = {results['vs_routellm']['ttest']['p']:.2e}, d = {results['vs_routellm']['cohens_d']:.2f})")
    logger.info(f"  banditGPT > Mixtral:     {diff_mixtral.mean():+.4f} "
                f"(p = {results['vs_mixtral']['ttest_p']:.2e}, d = {results['vs_mixtral']['cohens_d']:.2f})")
    logger.info(f"  Oracle gap closure:      {results['vs_oracle']['gap_closure_pct']:.1f}%")
    logger.info(f"  Runtime: {elapsed:.0f}s")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "statistical_tests.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "n_trials": N_TRIALS,
                    "n_eval": len(eval_data),
                    "lambda": LAMBDA,
                    "n_bootstrap": N_BOOTSTRAP,
                },
                "trial_means": trial_means.tolist(),
                **results,
            },
            f, indent=2,
        )

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
