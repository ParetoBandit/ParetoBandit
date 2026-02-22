#!/usr/bin/env python3
"""
Model Onboarding Experiment: Isolating the Hybrid Family Sharing Effect
========================================================================

Controlled comparison: the SAME newcomer model (openai/gpt-4.1) is
introduced under two family assignment conditions:

  Condition A (shared):   gpt-4.1 joins family openai/gpt-4 (auto-inferred)
                          -> shares hybrid beta with gpt-4-turbo
  Condition B (isolated): gpt-4.1 is forced into a singleton family
                          -> no shared knowledge (control)

By using identical model, data, embeddings, seeds, and semantic
bootstrapping, the ONLY treatment variable is family sharing.

Protocol:
  Phase 1 (steps 0-500):   Burn-in on 2 base models (gpt-4-turbo + mixtral)
  Phase 2 (steps 501-1121): register_model() adds gpt-4.1, continue with 3
  Phase 3:                  Evaluate on holdout (750 prompts, 3 models)

Baselines: Oracle (3-model), Always-newcomer, 2-model (no newcomer)
Statistics: Paired t-test across 20 seeds

Output: JSON results + 3-panel PNG figure
"""

import sys
import json
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.router_factory import create_experiment_router
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_TRIALS = 20
SEED_OFFSET = 42
ALPHA_START = 2.0
TARGET_NEFF = 10.0
LAMBDA = 0.0
INTRODUCTION_STEP = 500
SMOOTHING_WINDOW = 50

BASE_MODELS = [
    "openai/gpt-4-turbo",
    "mistralai/mixtral-8x7b-instruct",
]
NEWCOMER = "openai/gpt-4.1"
ISOLATED_FAMILY = "openai/gpt-4.1-isolated"


def load_data():
    """Load dev and holdout data with raw_score rewards for 3 models."""
    import gzip
    from collections import defaultdict

    needed = set(BASE_MODELS + [NEWCOMER])

    def _load_split(path):
        prompt_rewards = defaultdict(lambda: {"prompt": None, "rewards": {}})
        with gzip.open(path, "rt") as f:
            for line in f:
                entry = json.loads(line)
                mid = entry["model_id"]
                if mid not in needed:
                    continue
                sid = entry.get("sample_id", hash(entry["prompt"]))
                prompt_rewards[sid]["prompt"] = entry["prompt"]
                prompt_rewards[sid]["rewards"][mid] = entry["raw_score"]

        data = []
        for sid in sorted(prompt_rewards.keys()):
            rec = prompt_rewards[sid]
            if len(rec["rewards"]) == len(needed):
                data.append(rec)
        return data

    dev = _load_split(DEV_DATA_PATH_ALL_MODELS)
    holdout = _load_split(HOLDOUT_DATA_PATH_ALL_MODELS)
    logger.info(f"Loaded {len(dev)} dev + {len(holdout)} holdout prompts "
                f"({len(needed)} models, raw_score 0-1)")
    return dev, holdout


def precompute_embeddings(data, encoder, pca):
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


def compute_baselines(holdout_data):
    """Compute static reference points on the holdout set."""
    oracle_rewards = []
    always_newcomer = []
    for p in holdout_data:
        r = p["rewards"]
        oracle_rewards.append(max(r[m] for m in BASE_MODELS + [NEWCOMER]))
        always_newcomer.append(r[NEWCOMER])

    return {
        "oracle_3model": float(np.mean(oracle_rewards)),
        "always_newcomer": float(np.mean(always_newcomer)),
    }


def run_single_trial(dev_data, holdout_data, dev_emb, holdout_emb,
                     warmup_path, encoder, isolated, seed):
    """Run one trial. If isolated=True, force newcomer into singleton family."""
    np.random.seed(seed)
    dim = len(dev_emb[0])
    burn_in_total = len(dev_data)

    all_models = BASE_MODELS + [NEWCOMER]
    all_raw = [dev_data[i]["rewards"][m]
               for i in range(len(dev_data)) for m in all_models]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    router = create_experiment_router(
        model_registry=None,
        feature_dim=dim,
        prior_n_effective=TARGET_NEFF,
        alpha=ALPHA_START,
        warmup_path=warmup_path,
        cost_penalty=LAMBDA,
    )

    per_step_reward = []
    per_step_model = []
    newcomer_selections = []
    newcomer_pred_errors = []

    for i, p in enumerate(dev_data):
        step = i + 1

        if step == INTRODUCTION_STEP + 1:
            if isolated:
                router._family_map_override = {NEWCOMER: ISOLATED_FAMILY}
            router.encoder = encoder
            router.register_model(NEWCOMER, speed="balanced")
            router.encoder = None
            if isolated:
                router._family_map_override = None

        model, log = router.route(dev_emb[i], total_steps=burn_in_total)
        raw_r = p["rewards"][model]
        norm_r = (raw_r - r_min) / r_range
        router.process_feedback(log.request_id, norm_r)

        per_step_reward.append(raw_r)
        per_step_model.append(model)

        if step > INTRODUCTION_STEP and model == NEWCOMER:
            actual_norm = (p["rewards"][NEWCOMER] - r_min) / r_range
            predicted = _get_predicted_reward(router, dev_emb[i], NEWCOMER)
            newcomer_selections.append(step)
            newcomer_pred_errors.append(abs(predicted - actual_norm))

    holdout_rewards = []
    holdout_models = []
    for i, p in enumerate(holdout_data):
        model, _log = router.route(holdout_emb[i], total_steps=burn_in_total)
        holdout_rewards.append(p["rewards"][model])
        holdout_models.append(model)

    return {
        "per_step_reward": per_step_reward,
        "per_step_model": per_step_model,
        "newcomer_selections": newcomer_selections,
        "newcomer_pred_errors": newcomer_pred_errors,
        "holdout_mean_reward": float(np.mean(holdout_rewards)),
        "holdout_model_counts": {m: holdout_models.count(m) for m in set(holdout_models)},
    }


def _get_predicted_reward(router, embedding, model_id):
    """Corralling-weighted predicted reward from both experts."""
    if router.corralling_router is None:
        return 0.0

    cr = router.corralling_router
    weights = cr.weights / cr.weights.sum()
    pred = 0.0

    for i, expert in enumerate(cr.experts):
        if model_id not in getattr(expert, "A_inv", {}):
            continue
        try:
            theta = expert.A_inv[model_id] @ expert.b[model_id]
            fmap = getattr(expert, "family_map", None)
            if fmap is not None:
                family = fmap.get(model_id, model_id)
                if family in getattr(expert, "A0_inv", {}):
                    beta = expert.A0_inv[family] @ expert.b0[family]
                    pred += weights[i] * float(embedding @ (beta + theta))
                    continue
            pred += weights[i] * float(theta @ embedding)
        except Exception:
            pass
    return pred


def aggregate_trials(all_trials, n_steps):
    """Aggregate per-trial results into means and CIs."""
    n_trials = len(all_trials)
    t_crit = sp_stats.t.ppf(0.975, n_trials - 1)

    reward_matrix = np.array([t["per_step_reward"] for t in all_trials])
    smoothed = np.array([
        np.convolve(row, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid")
        for row in reward_matrix
    ])
    reward_mean = smoothed.mean(axis=0)
    reward_ci = t_crit * smoothed.std(axis=0, ddof=1) / np.sqrt(n_trials)

    newcomer_share = np.zeros((n_trials, n_steps))
    window = 50
    for t_idx, trial in enumerate(all_trials):
        models = trial["per_step_model"]
        for s in range(n_steps):
            start = max(0, s - window + 1)
            chunk = models[start:s + 1]
            newcomer_share[t_idx, s] = sum(1 for m in chunk if m == NEWCOMER) / len(chunk)

    share_mean = newcomer_share[:, INTRODUCTION_STEP:].mean(axis=0)
    share_ci = t_crit * newcomer_share[:, INTRODUCTION_STEP:].std(axis=0, ddof=1) / np.sqrt(n_trials)

    max_sel = max(len(t["newcomer_pred_errors"]) for t in all_trials) if all_trials else 0
    pred_err_padded = []
    for trial in all_trials:
        errs = trial["newcomer_pred_errors"]
        pred_err_padded.append(errs + [np.nan] * (max_sel - len(errs)))
    pred_err_matrix = np.array(pred_err_padded)

    bin_size = 5
    n_bins = max_sel // bin_size
    pred_err_binned_mean, pred_err_binned_ci = [], []
    for b in range(n_bins):
        chunk = pred_err_matrix[:, b * bin_size:(b + 1) * bin_size]
        bin_means = np.nanmean(chunk, axis=1)
        valid = ~np.isnan(bin_means)
        m = float(np.nanmean(bin_means))
        c = float(t_crit * np.nanstd(bin_means[valid], ddof=1) / np.sqrt(valid.sum())) if valid.sum() > 1 else 0.0
        pred_err_binned_mean.append(m)
        pred_err_binned_ci.append(c)

    holdout_rewards = [t["holdout_mean_reward"] for t in all_trials]

    return {
        "reward_smoothed_mean": reward_mean.tolist(),
        "reward_smoothed_ci": reward_ci.tolist(),
        "newcomer_share_mean": share_mean.tolist(),
        "newcomer_share_ci": share_ci.tolist(),
        "pred_err_binned_mean": pred_err_binned_mean,
        "pred_err_binned_ci": pred_err_binned_ci,
        "pred_err_bin_size": bin_size,
        "holdout_mean": float(np.mean(holdout_rewards)),
        "holdout_std": float(np.std(holdout_rewards, ddof=1)),
        "holdout_ci95": float(t_crit * np.std(holdout_rewards, ddof=1) / np.sqrt(n_trials)),
    }


def paired_test(shared_trials, isolated_trials):
    """Paired t-test on per-trial holdout rewards."""
    s = np.array([t["holdout_mean_reward"] for t in shared_trials])
    iso = np.array([t["holdout_mean_reward"] for t in isolated_trials])
    diff = s - iso
    t_stat, p_val = sp_stats.ttest_rel(s, iso)
    return {
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "n_positive": int((diff > 0).sum()),
        "n_negative": int((diff < 0).sum()),
        "n_zero": int((diff == 0).sum()),
    }


def plot_figure(agg_shared, agg_isolated, baselines, n_steps, output_path):
    """Generate 3-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel A: Reward trajectory ---
    ax = axes[0]
    offset = SMOOTHING_WINDOW - 1
    x = np.arange(offset, n_steps)

    for agg, label, color in [
        (agg_shared, "Shared family", "#2196F3"),
        (agg_isolated, "Isolated family", "#FF5722"),
    ]:
        m = np.array(agg["reward_smoothed_mean"])
        c = np.array(agg["reward_smoothed_ci"])
        ax.plot(x, m, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, m - c, m + c, alpha=0.15, color=color)

    ax.axvline(INTRODUCTION_STEP, color="gray", ls="--", lw=1, label="gpt-4.1 introduced")
    ax.axhline(baselines["oracle_3model"], color="#4CAF50", ls=":", lw=1, label=f"Oracle ({baselines['oracle_3model']:.3f})")
    ax.axhline(baselines["always_newcomer"], color="#9E9E9E", ls=":", lw=1, label=f"Always gpt-4.1 ({baselines['always_newcomer']:.3f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("(a) Reward Trajectory")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- Panel B: Newcomer traffic share ---
    ax = axes[1]
    x_share = np.arange(len(agg_shared["newcomer_share_mean"]))
    for agg, label, color in [
        (agg_shared, "Shared family", "#2196F3"),
        (agg_isolated, "Isolated family", "#FF5722"),
    ]:
        m = np.array(agg["newcomer_share_mean"])
        c = np.array(agg["newcomer_share_ci"])
        ax.plot(x_share, m, label=label, color=color, linewidth=1.5)
        ax.fill_between(x_share, m - c, m + c, alpha=0.15, color=color)
    ax.set_xlabel("Steps After Introduction")
    ax.set_ylabel("Newcomer Traffic Share")
    ax.set_title("(b) Newcomer Adoption Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Prediction error ---
    ax = axes[2]
    for agg, label, color in [
        (agg_shared, "Shared family", "#2196F3"),
        (agg_isolated, "Isolated family", "#FF5722"),
    ]:
        bs = agg["pred_err_bin_size"]
        nb = len(agg["pred_err_binned_mean"])
        xp = np.arange(nb) * bs + bs / 2
        m = np.array(agg["pred_err_binned_mean"])
        c = np.array(agg["pred_err_binned_ci"])
        ax.plot(xp, m, label=label, color=color, linewidth=1.5)
        ax.fill_between(xp, m - c, m + c, alpha=0.15, color=color)
    ax.set_xlabel("Prompts Routed to Newcomer")
    ax.set_ylabel("Prediction Error (MAE)")
    ax.set_title("(c) Newcomer Calibration Speed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved: {output_path}")


def main():
    logger.info("=" * 70)
    logger.info("MODEL ONBOARDING — CONTROLLED FAMILY SHARING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Newcomer: {NEWCOMER} (same model in both conditions)")
    logger.info(f"Treatment: family assignment (shared vs isolated)")

    dev_data, holdout_data = load_data()
    n_steps = len(dev_data)

    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    sanitized = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_path = str(sanitized if sanitized.exists() else DEFAULT_WARMUP_PRIORS_PATH)

    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    dev_emb = precompute_embeddings(dev_data, encoder, pca)
    holdout_emb = precompute_embeddings(holdout_data, encoder, pca)
    logger.info(f"  {len(dev_emb)}+{len(holdout_emb)} prompts in {time.time()-t0:.1f}s")

    baselines = compute_baselines(holdout_data)
    logger.info(f"\nBaselines: Oracle={baselines['oracle_3model']:.4f}, "
                f"Always-newcomer={baselines['always_newcomer']:.4f}")

    all_results = {}
    all_trials = {}

    for label, isolated in [("shared", False), ("isolated", True)]:
        logger.info(f"\n{'='*50}")
        logger.info(f"CONDITION: {label} (isolated={isolated})")
        logger.info(f"{'='*50}")

        trials = []
        t_start = time.time()
        for trial in range(N_TRIALS):
            seed = SEED_OFFSET + trial
            result = run_single_trial(
                dev_data, holdout_data, dev_emb, holdout_emb,
                warmup_path, encoder, isolated, seed,
            )
            trials.append(result)
            if trial % 5 == 0:
                logger.info(
                    f"  Trial {trial:2d}: holdout={result['holdout_mean_reward']:.4f}, "
                    f"newcomer routed={len(result['newcomer_selections'])}"
                )

        elapsed = time.time() - t_start
        logger.info(f"  {N_TRIALS} trials in {elapsed:.0f}s")

        agg = aggregate_trials(trials, n_steps)
        all_trials[label] = trials
        all_results[label] = {
            "aggregated": agg,
            "per_trial_holdout": [t["holdout_mean_reward"] for t in trials],
            "per_trial_newcomer_count": [len(t["newcomer_selections"]) for t in trials],
        }
        logger.info(f"  Holdout: {agg['holdout_mean']:.4f} +/- {agg['holdout_ci95']:.4f}")

    # Paired statistical test
    pt = paired_test(all_trials["shared"], all_trials["isolated"])
    logger.info(f"\n--- Paired t-test (shared - isolated) ---")
    logger.info(f"  Mean diff: {pt['mean_diff']:.4f} +/- {pt['std_diff']:.4f}")
    logger.info(f"  t={pt['t_statistic']:.3f}, p={pt['p_value']:.4f}")
    logger.info(f"  Direction: {pt['n_positive']}+ / {pt['n_negative']}- / {pt['n_zero']}=")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "model_onboarding.json"
    with open(results_file, "w") as f:
        json.dump({
            "metadata": {
                "n_trials": N_TRIALS,
                "introduction_step": INTRODUCTION_STEP,
                "newcomer": NEWCOMER,
                "isolated_family_override": ISOLATED_FAMILY,
                "base_models": BASE_MODELS,
                "n_dev": len(dev_data),
                "n_holdout": len(holdout_data),
                "smoothing_window": SMOOTHING_WINDOW,
                "design": "Same model in both conditions; only family assignment varies",
            },
            "baselines": baselines,
            "paired_test": pt,
            "results": all_results,
        }, f, indent=2)
    logger.info(f"\nResults saved: {results_file}")

    fig_path = output_dir / "model_onboarding.png"
    plot_figure(
        all_results["shared"]["aggregated"],
        all_results["isolated"]["aggregated"],
        baselines, n_steps, fig_path,
    )

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for label in ["shared", "isolated"]:
        agg = all_results[label]["aggregated"]
        nc = all_results[label]["per_trial_newcomer_count"]
        logger.info(
            f"  {label:10s}: holdout={agg['holdout_mean']:.4f}+/-{agg['holdout_ci95']:.4f}, "
            f"newcomer routed={np.mean(nc):.0f}+/-{np.std(nc):.0f}"
        )
    sig = "SIGNIFICANT" if pt["p_value"] < 0.05 else "NOT significant"
    logger.info(f"  Paired test: diff={pt['mean_diff']:.4f}, p={pt['p_value']:.4f} ({sig})")
    logger.info(f"  Oracle: {baselines['oracle_3model']:.4f}")
    logger.info(f"  Always-newcomer: {baselines['always_newcomer']:.4f}")


if __name__ == "__main__":
    main()
