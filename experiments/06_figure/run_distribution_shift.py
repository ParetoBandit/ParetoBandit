#!/usr/bin/env python3
"""
Distribution Shift Analysis (Experiment 10) — Full Factorial
=============================================================

Separates three effects: DISTRIBUTION SHIFT, ONLINE LEARNING, and CORRALLING.

                         Frozen         Online (no Corral)   Online (Corralling)
  Cross-dist priors      (A)            (E) NEW              (B)
  Same-dist priors       (C)            (F) NEW              (D)

Plus learning curves at λ=0 for all 4 adaptive conditions (B, D, E, F)
evaluated at checkpoints [50, 100, 200, 400, 766] to show convergence speed.

K=2 topology (Mixtral-8x7B vs GPT-4-Turbo).

Data splits:
  - Prior pool:    355 dev prompts  → builds same-dist priors
  - Online pool:   766 dev prompts  → online learning
  - Holdout:       750 prompts      → evaluation (all conditions)

Key comparisons:
  (A vs C)  → distribution shift effect (frozen)
  (E vs B)  → Corralling benefit under cross-dist priors
  (F vs D)  → Corralling benefit under same-dist priors
  (E vs F)  → distribution shift effect (online, no Corralling)
  (B vs D)  → distribution shift effect (online, with Corralling)
  Learning curves → convergence speed: does Corralling recover faster?

Usage:
    python3 experiments/06_distribution_shift/run_distribution_shift.py
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
from typing import Dict, List
from scipy.stats import ks_2samp, gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    ROUTELLM_BATTLES_REWARDS_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
)
from utils.router_factory import create_experiment_router
from utils.rewards import extract_reward
from sentence_transformers import SentenceTransformer
from utils.model_pricing import get_prices_for_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_A = "mistralai/mixtral-8x7b-instruct"
MODEL_B = "openai/gpt-4-turbo"

_PRICES = get_prices_for_models([MODEL_A, MODEL_B])

MODEL_REGISTRY = {
    MODEL_A: {
        "display_name": "Mixtral-8x7B",
        **_PRICES[MODEL_A],
        "provider": "mistral",
    },
    MODEL_B: {
        "display_name": "GPT-4-Turbo",
        **_PRICES[MODEL_B],
        "provider": "openai",
    },
}

def _req_cost(inp, out):
    return (100 * inp + 400 * out) / 1_000_000

COSTS = {m: _req_cost(v["input_cost_per_m"], v["output_cost_per_m"])
         for m, v in MODEL_REGISTRY.items()}

MODELS = [MODEL_A, MODEL_B]

PRIOR_POOL_SIZE = 355
MAX_SOURCE_PROMPTS = 8000
N_BOOTSTRAP = 1000
N_PSI_BINS = 10
N_TRIALS = 20
SEED = 42
PLASTICITY = 0.1

LAMBDA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
LEARNING_CURVE_CHECKPOINTS = [50, 100, 200, 400, 766]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_routellm_prompts(path: Path, max_samples: int) -> List[str]:
    prompts, seen = [], set()
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            p = entry.get("prompt", "")
            if p and p not in seen:
                seen.add(p)
                prompts.append(p)
                if len(prompts) >= max_samples:
                    break
    return prompts


def load_k2_deployment_data(path: Path) -> List[Dict]:
    prompt_rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt_rewards[entry["prompt"]][entry["model_id"]] = extract_reward(entry)
    return [
        {"prompt": p, "rewards": r}
        for p, r in prompt_rewards.items()
        if len(r) == 2
    ]

# ---------------------------------------------------------------------------
# Same-distribution prior builder
# ---------------------------------------------------------------------------

def build_same_dist_priors(data: List[Dict], embeddings: List[np.ndarray],
                           models: List[str], plasticity: float) -> str:
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

    state = {
        "A": A, "b": b,
        "models": models,
        "n_prompts": len(data),
        "n_total": len(data),
        "n_skipped": 0,
        "plasticity": plasticity,
        "context_dim": dim,
        "pca_applied": True,
        "pca_components": dim - 1,
        "reward_source": "lmsys_same_distribution",
        "seed": SEED,
    }

    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    joblib.dump(state, tmp.name)
    logger.info(f"   Same-dist priors: {len(data)} prompts, dim={dim}, ρ={plasticity}")
    return tmp.name

# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def compute_psi(source: np.ndarray, deploy: np.ndarray,
                n_bins: int = N_PSI_BINS) -> float:
    eps = 1e-8
    combined = np.concatenate([source, deploy])
    bin_edges = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6
    src_pct = np.histogram(source, bins=bin_edges)[0].astype(float)
    dep_pct = np.histogram(deploy, bins=bin_edges)[0].astype(float)
    src_pct = src_pct / src_pct.sum() + eps
    dep_pct = dep_pct / dep_pct.sum() + eps
    return float(np.sum((dep_pct - src_pct) * np.log(dep_pct / src_pct)))


def bootstrap_psi_ci(source, deploy, n_bootstrap=N_BOOTSTRAP, rng=None):
    if rng is None:
        rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(n_bootstrap):
        s = rng.choice(source, size=len(source), replace=True)
        d = rng.choice(deploy, size=len(deploy), replace=True)
        vals.append(compute_psi(s, d))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

# ---------------------------------------------------------------------------
# Prior miscalibration
# ---------------------------------------------------------------------------

def compute_prior_miscalibration(warmup_path, deploy_data, models):
    priors = joblib.load(warmup_path)
    A_dict, b_dict = priors.get("A", {}), priors.get("b", {})
    observed = {m: float(np.mean([d["rewards"][m] for d in deploy_data])) for m in models}
    results = {}
    for m in models:
        A, b = A_dict.get(m), b_dict.get(m)
        prior_mean = None
        if A is not None and b is not None:
            try:
                theta = np.linalg.solve(A, b)
                prior_mean = float(theta[-1])
            except np.linalg.LinAlgError:
                pass
        obs = observed[m]
        results[m] = {
            "display_name": MODEL_REGISTRY[m]["display_name"],
            "prior_mean_estimate": prior_mean,
            "observed_deployment_mean": obs,
            "absolute_error": float(prior_mean - obs) if prior_mean is not None else None,
            "relative_error_pct": float((prior_mean - obs) / obs * 100) if prior_mean is not None else None,
        }
    return results

# ---------------------------------------------------------------------------
# Pareto sweep
# ---------------------------------------------------------------------------

def _reward_normalisers(data, models):
    all_r = [d["rewards"][m] for d in data for m in models]
    return min(all_r), max(max(all_r) - min(all_r), 1e-6)


def run_pareto_point(train_data, train_emb, holdout_data, holdout_emb,
                     models, warmup_path, cost_penalty,
                     *, use_corralling, enable_feedback=True):
    dim = train_emb[0].shape[0]
    n_train = len(train_data)
    r_min, r_range = _reward_normalisers(train_data, models)
    registry = {m: MODEL_REGISTRY[m] for m in models}

    trial_rewards, trial_costs = [], []
    for trial in range(N_TRIALS):
        np.random.seed(SEED + trial)
        router = create_experiment_router(
            model_registry=registry, feature_dim=dim,
            prior_n_effective=10.0, alpha=2.0,
            warmup_path=str(warmup_path),
            use_corralling=use_corralling,
            corralling_learning_rate=0.1, corralling_gamma=0.05,
            cost_penalty=cost_penalty,
        )

        if enable_feedback:
            indices = list(range(n_train))
            np.random.RandomState(SEED + trial).shuffle(indices)
            for idx in indices:
                d, x = train_data[idx], train_emb[idx]
                m, log = router.route(x, total_steps=n_train)
                norm_r = (d["rewards"][m] - r_min) / r_range
                router.process_feedback(log.request_id, norm_r)

        rewards, costs_list = [], []
        for d, x in zip(holdout_data, holdout_emb):
            m, _ = router.route(x, total_steps=n_train)
            rewards.append(d["rewards"][m])
            costs_list.append(COSTS[m])

        trial_rewards.append(float(np.mean(rewards)))
        trial_costs.append(float(np.mean(costs_list)))

    return {
        "cost_penalty": cost_penalty,
        "mean_reward": float(np.mean(trial_rewards)),
        "std_reward": float(np.std(trial_rewards, ddof=1)),
        "mean_cost": float(np.mean(trial_costs)),
        "std_cost": float(np.std(trial_costs, ddof=1)),
        "n_trials": N_TRIALS,
    }


def run_pareto_sweep(train_data, train_emb, holdout_data, holdout_emb,
                     models, warmup_path, lambda_values,
                     use_corralling, enable_feedback, label):
    points = []
    for lam in lambda_values:
        logger.info(f"     λ={lam:.3f} ...")
        pt = run_pareto_point(
            train_data, train_emb, holdout_data, holdout_emb,
            models, warmup_path, lam,
            use_corralling=use_corralling, enable_feedback=enable_feedback,
        )
        pt["label"] = label
        points.append(pt)
    return points

# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def run_learning_curve(train_data, train_emb, holdout_data, holdout_emb,
                       models, warmup_path, cost_penalty,
                       *, use_corralling, checkpoints):
    """Train incrementally and evaluate at each checkpoint."""
    dim = train_emb[0].shape[0]
    n_train = len(train_data)
    r_min, r_range = _reward_normalisers(train_data, models)
    registry = {m: MODEL_REGISTRY[m] for m in models}

    curve = {cp: {"rewards": [], "costs": []} for cp in checkpoints}

    for trial in range(N_TRIALS):
        np.random.seed(SEED + trial)
        router = create_experiment_router(
            model_registry=registry, feature_dim=dim,
            prior_n_effective=10.0, alpha=2.0,
            warmup_path=str(warmup_path),
            use_corralling=use_corralling,
            corralling_learning_rate=0.1, corralling_gamma=0.05,
            cost_penalty=cost_penalty,
        )

        indices = list(range(n_train))
        np.random.RandomState(SEED + trial).shuffle(indices)

        steps_done = 0
        cp_idx = 0
        for idx in indices:
            d, x = train_data[idx], train_emb[idx]
            m, log = router.route(x, total_steps=n_train)
            norm_r = (d["rewards"][m] - r_min) / r_range
            router.process_feedback(log.request_id, norm_r)
            steps_done += 1

            if cp_idx < len(checkpoints) and steps_done == checkpoints[cp_idx]:
                rewards, costs_list = [], []
                for hd, hx in zip(holdout_data, holdout_emb):
                    hm, _ = router.route(hx, total_steps=n_train)
                    rewards.append(hd["rewards"][hm])
                    costs_list.append(COSTS[hm])
                curve[checkpoints[cp_idx]]["rewards"].append(float(np.mean(rewards)))
                curve[checkpoints[cp_idx]]["costs"].append(float(np.mean(costs_list)))
                cp_idx += 1

    results = []
    for cp in checkpoints:
        r_arr = np.array(curve[cp]["rewards"])
        results.append({
            "steps": cp,
            "mean_reward": float(r_arr.mean()),
            "std_reward": float(r_arr.std(ddof=1)),
            "n_trials": N_TRIALS,
        })
    return results

# ---------------------------------------------------------------------------
# Plotting — 3-panel figure
# ---------------------------------------------------------------------------

def plot_results(source_pc1, deploy_pc1, psi, psi_ci, ks_stat, ks_p,
                 all_pareto, learning_curves,
                 oracle_reward, oracle_cost, out_path):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # --- Panel A: Feature distribution shift ---
    ax = axes[0]
    x_grid = np.linspace(
        min(source_pc1.min(), deploy_pc1.min()) - 0.5,
        max(source_pc1.max(), deploy_pc1.max()) + 0.5, 300)
    kde_src = gaussian_kde(source_pc1)
    kde_dep = gaussian_kde(deploy_pc1)
    ax.fill_between(x_grid, kde_src(x_grid), alpha=0.4, color="#4C72B0",
                    label=f"RouteLLM battles (N={len(source_pc1):,})")
    ax.fill_between(x_grid, kde_dep(x_grid), alpha=0.4, color="#DD8452",
                    label=f"LMSYS Arena (N={len(deploy_pc1):,})")
    ax.set_xlabel("PC1 projection")
    ax.set_ylabel("Density")
    ax.set_title(f"(a) Feature Distribution Shift\n"
                 f"PSI={psi:.3f} [{psi_ci[0]:.3f}, {psi_ci[1]:.3f}]  "
                 f"KS D={ks_stat:.3f}")
    ax.legend(fontsize=8.5)

    # --- Panel B: 6-condition Pareto ---
    ax = axes[1]

    def _extract(pts):
        c = np.array([p["mean_cost"] for p in pts])
        r = np.array([p["mean_reward"] for p in pts])
        s = np.array([p["std_reward"] for p in pts])
        return c, r, s

    n_t = N_TRIALS
    ci_z = 1.96 / np.sqrt(n_t)

    COL_CROSS = "#C44E52"
    COL_SAME  = "#4C72B0"
    COL_CROSS_NOCORRAL = "#E8A0A0"
    COL_SAME_NOCORRAL  = "#8FB8DE"

    styles = [
        ("cross_frozen",        COL_CROSS,          "s--", "Cross-dist (frozen)"),
        ("cross_nocorral",      COL_CROSS_NOCORRAL, "s:",  "Cross-dist (online, no Corral)"),
        ("cross_adaptive",      COL_CROSS,          "s-",  "Cross-dist (online, Corralling)"),
        ("same_frozen",         COL_SAME,           "o--", "Same-dist (frozen)"),
        ("same_nocorral",       COL_SAME_NOCORRAL,  "o:",  "Same-dist (online, no Corral)"),
        ("same_adaptive",       COL_SAME,           "o-",  "Same-dist (online, Corralling)"),
    ]

    for key, col, fmt, label in styles:
        c, r, s = _extract(all_pareto[key])
        ax.plot(c, r, fmt, color=col, zorder=3, label=label, linewidth=1.5)
        if s.max() > 0:
            ax.fill_between(c, r - ci_z * s, r + ci_z * s, alpha=0.10, color=col)

    ax.scatter([oracle_cost], [oracle_reward], marker="*", s=200, color="green",
               zorder=5, label=f"Oracle ({oracle_reward:.3f})")

    ax.set_xlabel("Mean cost per request ($)")
    ax.set_ylabel("Holdout reward (binary)")
    ax.set_title(f"(b) Pareto Frontiers: Prior × Adaptation × Architecture\n"
                 f"(K=2, {n_t} trials, 95% CI)")
    ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Figure saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("Experiment 10: Distribution Shift — Full Factorial + Learning Curves")
    logger.info("=" * 70)

    # 1. Load encoder / PCA
    logger.info("\n1. Loading encoder and PCA ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    # 2. Load RouteLLM battle prompts
    logger.info("\n2. Loading offline RouteLLM battle prompts ...")
    source_prompts = load_routellm_prompts(ROUTELLM_BATTLES_REWARDS_PATH, MAX_SOURCE_PROMPTS)
    logger.info(f"   Loaded {len(source_prompts)} offline prompts")

    # 3. Load K=2 deployment data
    logger.info("\n3. Loading K=2 deployment data ...")
    dev_data = load_k2_deployment_data(CANONICAL_DEV_DATA_PATH)
    holdout_data = load_k2_deployment_data(CANONICAL_HOLDOUT_DATA_PATH)
    logger.info(f"   Dev: {len(dev_data)}  Holdout: {len(holdout_data)}")

    # 4. Split dev into prior_pool and online_pool
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(dev_data))
    prior_idx = indices[:PRIOR_POOL_SIZE]
    online_idx = indices[PRIOR_POOL_SIZE:]

    prior_pool = [dev_data[i] for i in prior_idx]
    online_pool = [dev_data[i] for i in online_idx]
    logger.info(f"\n4. Data split: prior_pool={len(prior_pool)}, "
                f"online_pool={len(online_pool)}, holdout={len(holdout_data)}")

    # 5. Embed all prompts
    logger.info("\n5. Embedding prompts ...")
    source_emb = np.array([embed_prompt(p, encoder, pca) for p in source_prompts])
    deploy_prompts = [d["prompt"] for d in dev_data + holdout_data]
    deploy_emb_all = np.array([embed_prompt(p, encoder, pca) for p in deploy_prompts])

    prior_emb = [embed_prompt(d["prompt"], encoder, pca) for d in prior_pool]
    online_emb = [embed_prompt(d["prompt"], encoder, pca) for d in online_pool]
    holdout_emb = [embed_prompt(d["prompt"], encoder, pca) for d in holdout_data]
    logger.info(f"   Embedding dim: {source_emb.shape[1]}")

    source_pc1 = source_emb[:, 0]
    deploy_pc1 = deploy_emb_all[:, 0]

    # 6. PSI and KS test
    logger.info("\n6. Distribution shift metrics ...")
    psi = compute_psi(source_pc1, deploy_pc1)
    psi_lo, psi_hi = bootstrap_psi_ci(source_pc1, deploy_pc1,
                                       rng=np.random.RandomState(SEED))
    ks_stat, ks_p = ks_2samp(source_pc1, deploy_pc1)
    psi_severity = "Negligible" if psi < 0.1 else ("Moderate" if psi < 0.25 else "Substantial")
    logger.info(f"   PSI = {psi:.4f}  95% CI [{psi_lo:.4f}, {psi_hi:.4f}]  ({psi_severity})")
    logger.info(f"   KS  D = {ks_stat:.4f}, p = {ks_p:.3e}")

    # 7. Prior miscalibration
    logger.info("\n7. Prior miscalibration ...")
    cross_miscal = compute_prior_miscalibration(
        DEFAULT_WARMUP_PRIORS_PATH, dev_data + holdout_data, MODELS)
    for m, v in cross_miscal.items():
        pr = f"{v['prior_mean_estimate']:.3f}" if v["prior_mean_estimate"] is not None else "N/A"
        err = (f"{v['relative_error_pct']:+.1f}%" if v["relative_error_pct"] is not None else "N/A")
        logger.info(f"   Cross-dist {v['display_name']}: prior≈{pr}  obs={v['observed_deployment_mean']:.3f}  err={err}")

    # 8. Build same-distribution priors
    logger.info("\n8. Building same-distribution priors ...")
    same_dist_path = build_same_dist_priors(prior_pool, prior_emb, MODELS, PLASTICITY)

    same_miscal = compute_prior_miscalibration(same_dist_path, holdout_data, MODELS)
    for m, v in same_miscal.items():
        pr = f"{v['prior_mean_estimate']:.3f}" if v["prior_mean_estimate"] is not None else "N/A"
        err = (f"{v['relative_error_pct']:+.1f}%" if v["relative_error_pct"] is not None else "N/A")
        logger.info(f"   Same-dist {v['display_name']}: prior≈{pr}  obs={v['observed_deployment_mean']:.3f}  err={err}")

    # 9. Oracle
    oracle_reward = float(np.mean([max(d["rewards"][m] for m in MODELS) for d in holdout_data]))
    oracle_cost = float(np.mean([
        COSTS[max(MODELS, key=lambda m: d["rewards"][m])] for d in holdout_data]))
    logger.info(f"\n9. Oracle: R={oracle_reward:.4f}  C=${oracle_cost:.6f}")

    # 10-15. Six Pareto sweeps
    # (key, label, warmup_path, use_corralling, enable_feedback)
    sweep_cfg = [
        ("cross_frozen",    "Cross-dist frozen (Corralling, no feedback)",
         DEFAULT_WARMUP_PRIORS_PATH, True,  False),
        ("cross_nocorral",  "Cross-dist online (no Corralling)",
         DEFAULT_WARMUP_PRIORS_PATH, False, True),
        ("cross_adaptive",  "Cross-dist online (Corralling)",
         DEFAULT_WARMUP_PRIORS_PATH, True,  True),
        ("same_frozen",     "Same-dist frozen (Corralling, no feedback)",
         same_dist_path,             True,  False),
        ("same_nocorral",   "Same-dist online (no Corralling)",
         same_dist_path,             False, True),
        ("same_adaptive",   "Same-dist online (Corralling)",
         same_dist_path,             True,  True),
    ]

    all_pareto = {}
    step_num = 10
    for key, name, wp, corral, feedback in sweep_cfg:
        logger.info(f"\n{step_num}. {name} ({len(LAMBDA_VALUES)} λ × {N_TRIALS} trials) ...")
        pts = run_pareto_sweep(
            online_pool, online_emb, holdout_data, holdout_emb,
            MODELS, wp, LAMBDA_VALUES,
            use_corralling=corral, enable_feedback=feedback, label=key)
        all_pareto[key] = pts
        for pt in pts:
            logger.info(f"   λ={pt['cost_penalty']:.3f}  R={pt['mean_reward']:.4f}±{pt['std_reward']:.4f}  C=${pt['mean_cost']:.6f}")
        step_num += 1

    # 16. Learning curves at λ=0
    logger.info(f"\n{step_num}. Learning curves at λ=0 ...")
    lc_cfg = [
        ("cross_nocorral", DEFAULT_WARMUP_PRIORS_PATH, False),
        ("cross_corral",   DEFAULT_WARMUP_PRIORS_PATH, True),
        ("same_nocorral",  same_dist_path,             False),
        ("same_corral",    same_dist_path,             True),
    ]

    learning_curves = {}
    for key, wp, corral in lc_cfg:
        logger.info(f"   {key} ...")
        lc = run_learning_curve(
            online_pool, online_emb, holdout_data, holdout_emb,
            MODELS, wp, cost_penalty=0.0,
            use_corralling=corral, checkpoints=LEARNING_CURVE_CHECKPOINTS)
        learning_curves[key] = lc
        for pt in lc:
            logger.info(f"     step={pt['steps']:4d}  R={pt['mean_reward']:.4f}±{pt['std_reward']:.4f}")

    # 17. Plot
    out_fig = Path(__file__).parent / "results" / "figure_distribution_shift.png"
    plot_results(
        source_pc1, deploy_pc1, psi, (psi_lo, psi_hi), ks_stat, ks_p,
        all_pareto, learning_curves,
        oracle_reward, oracle_cost, out_fig)

    # 18. Save JSON
    bests = {k: max(v, key=lambda x: x["mean_reward"]) for k, v in all_pareto.items()}
    results = {
        "distribution_shift": {
            "psi": psi, "psi_ci_95": [psi_lo, psi_hi], "psi_severity": psi_severity,
            "ks_stat": ks_stat, "ks_p_value": ks_p,
            "n_source": len(source_pc1), "n_deploy": len(deploy_pc1),
        },
        "prior_miscalibration_cross": cross_miscal,
        "prior_miscalibration_same": same_miscal,
        "oracle_reward": oracle_reward, "oracle_cost": oracle_cost,
        "data_split": {
            "prior_pool": len(prior_pool),
            "online_pool": len(online_pool),
            "holdout": len(holdout_data),
        },
        **{f"pareto_{k}": v for k, v in all_pareto.items()},
        **{f"best_{k}": v for k, v in bests.items()},
        "learning_curves": learning_curves,
        "models": {m: MODEL_REGISTRY[m]["display_name"] for m in MODELS},
        "lambda_values": LAMBDA_VALUES,
        "learning_curve_checkpoints": LEARNING_CURVE_CHECKPOINTS,
    }
    out_json = Path(__file__).parent / "results" / "distribution_shift_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — Full Factorial")
    logger.info("=" * 70)
    logger.info(f"  PSI = {psi:.3f} [{psi_lo:.3f}, {psi_hi:.3f}]  ({psi_severity})")
    logger.info(f"  Oracle:              R={oracle_reward:.4f}  C=${oracle_cost:.6f}")
    for k, b in bests.items():
        logger.info(f"  {k:22s} R={b['mean_reward']:.4f}±{b['std_reward']:.4f}  C=${b['mean_cost']:.6f}")

    # Effect decomposition
    cf = bests["cross_frozen"]["mean_reward"]
    ca = bests["cross_adaptive"]["mean_reward"]
    cn = bests["cross_nocorral"]["mean_reward"]
    sf = bests["same_frozen"]["mean_reward"]
    sa = bests["same_adaptive"]["mean_reward"]
    sn = bests["same_nocorral"]["mean_reward"]

    logger.info(f"\n  === Effect Decomposition ===")
    logger.info(f"  Distribution shift (frozen):             {sf - cf:+.4f}  (same - cross)")
    logger.info(f"  Distribution shift (online, no Corral):  {sn - cn:+.4f}  (same - cross)")
    logger.info(f"  Distribution shift (online, Corralling): {sa - ca:+.4f}  (same - cross)")
    logger.info(f"  Corralling benefit (cross-dist):         {ca - cn:+.4f}  (Corral - no Corral)")
    logger.info(f"  Corralling benefit (same-dist):          {sa - sn:+.4f}  (Corral - no Corral)")
    logger.info(f"  Online learning (cross, no Corral):      {cn - cf:+.4f}  (online - frozen)")
    logger.info(f"  Online learning (cross, Corralling):     {ca - cf:+.4f}  (online - frozen)")
    logger.info(f"  Online learning (same, no Corral):       {sn - sf:+.4f}  (online - frozen)")
    logger.info(f"  Online learning (same, Corralling):      {sa - sf:+.4f}  (online - frozen)")

    # Learning curve summary
    logger.info(f"\n  === Learning Curves (λ=0) ===")
    for key, lc in learning_curves.items():
        first = lc[0]["mean_reward"]
        last = lc[-1]["mean_reward"]
        logger.info(f"  {key:22s}  step 50: {first:.4f}  step 766: {last:.4f}  gain: {last-first:+.4f}")


if __name__ == "__main__":
    main()
