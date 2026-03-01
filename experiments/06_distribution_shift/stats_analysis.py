#!/usr/bin/env python3
"""
Statistical analysis of the Corralling benefit.

Runs the 4 adaptive conditions at λ=0 with paired seeds,
saves per-trial rewards, and computes:
  - Paired t-tests (Corralling vs no-Corralling)
  - Wilcoxon signed-rank tests
  - Cohen's d (paired)
  - Bootstrap CIs on the mean difference
  - Per-trial differences for visual inspection
"""

import sys
import json
import gzip
import logging
import tempfile
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER, DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH, CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
)
from utils.router_factory import create_experiment_router
from utils.rewards import extract_reward
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_A = "mistralai/mixtral-8x7b-instruct"
MODEL_B = "openai/gpt-4-turbo"
MODEL_REGISTRY = {
    MODEL_A: {"display_name": "Mixtral-8x7B", "input_cost_per_m": 0.54,
              "output_cost_per_m": 0.60, "provider": "mistral"},
    MODEL_B: {"display_name": "GPT-4-Turbo", "input_cost_per_m": 10.00,
              "output_cost_per_m": 30.00, "provider": "openai"},
}
MODELS = [MODEL_A, MODEL_B]
SEED = 42
PLASTICITY = 0.1
PRIOR_POOL_SIZE = 355
N_TRIALS = 50  # more trials for statistical power


def load_k2_deployment_data(path):
    prompt_rewards = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt_rewards[entry["prompt"]][entry["model_id"]] = extract_reward(entry)
    return [{"prompt": p, "rewards": r} for p, r in prompt_rewards.items() if len(r) == 2]


def build_same_dist_priors(data, embeddings, models, plasticity):
    dim = embeddings[0].shape[0]
    A = {m: np.eye(dim) for m in models}
    b = {m: np.zeros(dim) for m in models}
    for d, x in zip(data, embeddings):
        x_col = x.reshape(-1, 1)
        for m in models:
            r = d["rewards"].get(m)
            if r is not None:
                A[m] += x_col @ x_col.T
                b[m] += r * x
    for m in models:
        A[m] *= plasticity
        b[m] *= plasticity
    state = {"A": A, "b": b, "models": models, "n_prompts": len(data),
             "n_total": len(data), "n_skipped": 0, "plasticity": plasticity,
             "context_dim": dim, "pca_applied": True, "pca_components": dim - 1,
             "reward_source": "lmsys_same_distribution", "seed": SEED}
    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    joblib.dump(state, tmp.name)
    return tmp.name


def run_trial(train_data, train_emb, holdout_data, holdout_emb,
              models, warmup_path, cost_penalty, use_corralling, seed):
    """Run a single trial, return holdout reward."""
    dim = train_emb[0].shape[0]
    n_train = len(train_data)
    all_r = [d["rewards"][m] for d in train_data for m in models]
    r_min = min(all_r)
    r_range = max(max(all_r) - r_min, 1e-6)
    registry = {m: MODEL_REGISTRY[m] for m in models}

    np.random.seed(seed)
    router = create_experiment_router(
        model_registry=registry, feature_dim=dim,
        prior_n_effective=10.0, alpha=2.0,
        warmup_path=str(warmup_path),
        use_corralling=use_corralling,
        corralling_learning_rate=0.1, corralling_gamma=0.05,
        cost_penalty=cost_penalty, policy="hybrid",
    )

    indices = list(range(n_train))
    np.random.RandomState(seed).shuffle(indices)
    for idx in indices:
        d, x = train_data[idx], train_emb[idx]
        m, log = router.route(x, total_steps=n_train)
        norm_r = (d["rewards"][m] - r_min) / r_range
        router.process_feedback(log.request_id, norm_r)

    rewards = []
    for d, x in zip(holdout_data, holdout_emb):
        m, _ = router.route(x, total_steps=n_train)
        rewards.append(d["rewards"][m])

    return float(np.mean(rewards))


def paired_cohens_d(x, y):
    diff = x - y
    return float(diff.mean() / diff.std(ddof=1))


def bootstrap_ci(x, y, n_boot=10000, alpha=0.05):
    rng = np.random.RandomState(42)
    diffs = x - y
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(len(diffs), size=len(diffs), replace=True)
        boot_means.append(diffs[idx].mean())
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    logger.info("=" * 70)
    logger.info("Statistical Analysis: Is the Corralling benefit significant?")
    logger.info(f"N_TRIALS = {N_TRIALS} (paired seeds)")
    logger.info("=" * 70)

    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    dev_data = load_k2_deployment_data(CANONICAL_DEV_DATA_PATH)
    holdout_data = load_k2_deployment_data(CANONICAL_HOLDOUT_DATA_PATH)

    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(dev_data))
    prior_pool = [dev_data[i] for i in indices[:PRIOR_POOL_SIZE]]
    online_pool = [dev_data[i] for i in indices[PRIOR_POOL_SIZE:]]

    logger.info(f"Online pool: {len(online_pool)}, Holdout: {len(holdout_data)}")

    prior_emb = [embed_prompt(d["prompt"], encoder, pca) for d in prior_pool]
    online_emb = [embed_prompt(d["prompt"], encoder, pca) for d in online_pool]
    holdout_emb = [embed_prompt(d["prompt"], encoder, pca) for d in holdout_data]

    same_dist_path = build_same_dist_priors(prior_pool, prior_emb, MODELS, PLASTICITY)

    # Run 4 conditions × N_TRIALS, paired by seed
    conditions = {
        "cross_nocorral": (DEFAULT_WARMUP_PRIORS_PATH, False),
        "cross_corral":   (DEFAULT_WARMUP_PRIORS_PATH, True),
        "same_nocorral":  (same_dist_path, False),
        "same_corral":    (same_dist_path, True),
    }

    trial_data = {}
    for name, (wp, corral) in conditions.items():
        logger.info(f"\nRunning {name} ({N_TRIALS} trials) ...")
        rewards = []
        for trial in range(N_TRIALS):
            r = run_trial(online_pool, online_emb, holdout_data, holdout_emb,
                          MODELS, wp, cost_penalty=0.0,
                          use_corralling=corral, seed=SEED + trial)
            rewards.append(r)
            if (trial + 1) % 10 == 0:
                logger.info(f"  trial {trial+1}/{N_TRIALS}: R={r:.4f}")
        trial_data[name] = np.array(rewards)
        logger.info(f"  {name}: mean={np.mean(rewards):.4f} std={np.std(rewards, ddof=1):.4f}")

    # Statistical tests
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL TESTS (paired, λ=0)")
    logger.info("=" * 70)

    comparisons = [
        ("Corralling benefit (cross-dist)",
         "cross_corral", "cross_nocorral"),
        ("Corralling benefit (same-dist)",
         "same_corral", "same_nocorral"),
        ("Corralling benefit (pooled)",
         None, None),  # handled specially
        ("Dist shift effect (no Corral)",
         "same_nocorral", "cross_nocorral"),
        ("Dist shift effect (Corralling)",
         "same_corral", "cross_corral"),
    ]

    results = {}

    for label, key_a, key_b in comparisons:
        if key_a is None:
            # Pooled: stack both cross and same diffs
            d1 = trial_data["cross_corral"] - trial_data["cross_nocorral"]
            d2 = trial_data["same_corral"] - trial_data["same_nocorral"]
            diffs = np.concatenate([d1, d2])
            a = np.concatenate([trial_data["cross_corral"], trial_data["same_corral"]])
            b = np.concatenate([trial_data["cross_nocorral"], trial_data["same_nocorral"]])
        else:
            a = trial_data[key_a]
            b = trial_data[key_b]
            diffs = a - b

        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std(ddof=1))
        se_diff = std_diff / np.sqrt(len(diffs))

        t_stat, t_p = stats.ttest_rel(a, b) if key_a is not None else stats.ttest_1samp(diffs, 0)
        w_stat, w_p = stats.wilcoxon(diffs, alternative='two-sided')
        d = paired_cohens_d(a, b) if key_a is not None else float(diffs.mean() / diffs.std(ddof=1))
        ci_lo, ci_hi = bootstrap_ci(a, b) if key_a is not None else (
            float(np.percentile([diffs[np.random.RandomState(42).choice(len(diffs), len(diffs), True)].mean()
                                 for _ in range(10000)], 2.5)),
            float(np.percentile([diffs[np.random.RandomState(42).choice(len(diffs), len(diffs), True)].mean()
                                 for _ in range(10000)], 97.5)),
        )

        # Parametric CI
        t_crit = stats.t.ppf(0.975, df=len(diffs) - 1)
        param_ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

        n_positive = int((diffs > 0).sum())
        n_negative = int((diffs < 0).sum())
        n_zero = int((diffs == 0).sum())

        logger.info(f"\n  {label}:")
        logger.info(f"    Mean diff:      {mean_diff:+.4f} ± {std_diff:.4f}")
        logger.info(f"    95% CI (param): [{param_ci[0]:+.4f}, {param_ci[1]:+.4f}]")
        logger.info(f"    95% CI (boot):  [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        logger.info(f"    Paired t-test:  t={t_stat:.3f}, p={t_p:.4f}")
        logger.info(f"    Wilcoxon:       W={w_stat:.1f}, p={w_p:.4f}")
        logger.info(f"    Cohen's d:      {d:.3f}")
        logger.info(f"    Sign count:     {n_positive}+ / {n_negative}- / {n_zero}=")

        sig_05 = "YES" if t_p < 0.05 else "NO"
        sig_01 = "YES" if t_p < 0.01 else "NO"
        logger.info(f"    Significant at α=0.05? {sig_05}")
        logger.info(f"    Significant at α=0.01? {sig_01}")

        effect_size = "negligible" if abs(d) < 0.2 else (
            "small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large"))
        logger.info(f"    Effect size:    {effect_size}")

        results[label] = {
            "mean_diff": mean_diff, "std_diff": std_diff,
            "ci_95_parametric": list(param_ci),
            "ci_95_bootstrap": [ci_lo, ci_hi],
            "t_stat": float(t_stat), "t_p": float(t_p),
            "wilcoxon_W": float(w_stat), "wilcoxon_p": float(w_p),
            "cohens_d": d, "effect_size": effect_size,
            "n_positive": n_positive, "n_negative": n_negative,
        }

    # Plot: per-trial paired differences
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Corralling vs no-Corralling (cross-dist)
    ax = axes[0]
    diffs_cross = trial_data["cross_corral"] - trial_data["cross_nocorral"]
    ax.hist(diffs_cross, bins=15, alpha=0.7, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(diffs_cross.mean(), color="green", linestyle="-", linewidth=2)
    ax.set_xlabel("Reward difference (Corralling − no Corralling)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cross-dist: Corralling benefit\n"
                 f"mean={diffs_cross.mean():+.4f}, p={results['Corralling benefit (cross-dist)']['t_p']:.4f}")

    # Panel 2: Corralling vs no-Corralling (same-dist)
    ax = axes[1]
    diffs_same = trial_data["same_corral"] - trial_data["same_nocorral"]
    ax.hist(diffs_same, bins=15, alpha=0.7, color="#DD8452", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(diffs_same.mean(), color="green", linestyle="-", linewidth=2)
    ax.set_xlabel("Reward difference (Corralling − no Corralling)")
    ax.set_ylabel("Count")
    ax.set_title(f"Same-dist: Corralling benefit\n"
                 f"mean={diffs_same.mean():+.4f}, p={results['Corralling benefit (same-dist)']['t_p']:.4f}")

    # Panel 3: Scatter of paired trials
    ax = axes[2]
    ax.scatter(trial_data["cross_nocorral"], trial_data["cross_corral"],
               alpha=0.6, color="#4C72B0", label="Cross-dist", s=30)
    ax.scatter(trial_data["same_nocorral"], trial_data["same_corral"],
               alpha=0.6, color="#DD8452", label="Same-dist", s=30)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Reward (no Corralling)")
    ax.set_ylabel("Reward (Corralling)")
    ax.set_title("Paired trial comparison\n(above diagonal = Corralling wins)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()
    out_fig = Path(__file__).parent / "results" / "figure_stats_analysis.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\n  Stats figure saved: {out_fig}")

    # Save
    out_json = Path(__file__).parent / "results" / "stats_analysis.json"
    serializable = {k: {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                        for kk, vv in v.items()} for k, v in results.items()}
    serializable["trial_data"] = {k: v.tolist() for k, v in trial_data.items()}
    serializable["n_trials"] = N_TRIALS
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"  Stats results saved: {out_json}")

    # Final verdict
    logger.info("\n" + "=" * 70)
    logger.info("VERDICT")
    logger.info("=" * 70)
    corral_cross_p = results["Corralling benefit (cross-dist)"]["t_p"]
    corral_same_p = results["Corralling benefit (same-dist)"]["t_p"]
    corral_pooled_p = results["Corralling benefit (pooled)"]["t_p"]
    logger.info(f"  Corralling benefit (cross): p={corral_cross_p:.4f}")
    logger.info(f"  Corralling benefit (same):  p={corral_same_p:.4f}")
    logger.info(f"  Corralling benefit (pooled): p={corral_pooled_p:.4f}")
    if corral_pooled_p < 0.01:
        logger.info("  → Corralling benefit is STATISTICALLY SIGNIFICANT (p < 0.01)")
    elif corral_pooled_p < 0.05:
        logger.info("  → Corralling benefit is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        logger.info("  → Corralling benefit is NOT statistically significant")


if __name__ == "__main__":
    main()
