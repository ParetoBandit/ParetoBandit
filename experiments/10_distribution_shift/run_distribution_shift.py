#!/usr/bin/env python3
"""
Distribution Shift Analysis (Experiment 10)
============================================

Measures covariate shift between the offline training distribution (RouteLLM
battles, used to build warmup priors) and the deployment distribution (LMSYS
Arena dev/holdout prompts used in all routing experiments).

This experiment supports the claim in the paper that "deployment distributions
drift from offline training data, which silently degrades static routing
policies."  It does NOT simulate temporal drift; it measures the static gap
between the two datasets that already exist.

Uses the K=5 portfolio (Llama-3.1-8B, Mixtral-8x7B, Gemini-2.5-Flash,
Claude-Sonnet-4, GPT-4.1) to match the primary multi-model experiments in
the paper (Section 4, Figure 5/6), making the drift results directly
comparable.

Analyses performed
------------------
1. Feature distribution shift (PC1)
   - KDE density plots for both distributions
   - Population Stability Index (PSI) with bootstrap 95% CI
   - Kolmogorov–Smirnov test (D-statistic, p-value)

2. Prior miscalibration due to shift (K=5)
   - Compare offline prior reward estimates vs. observed deployment rewards
   - Quantify systematic over/under-estimation per model across the portfolio

3. Adaptive router recovery
   - Static router (frozen prior, no online updates): evaluated on deployment
     holdout. Measures quality gap vs. oracle.
   - Hybrid banditGPT router (K=5, policy="hybrid"): trains online on
     deployment dev-set, evaluated on holdout at checkpoints.
   - Demonstrates that online adaptation recovers the gap caused by
     prior miscalibration at deployment time.

Output
------
  results/distribution_shift_results.json
  results/figure_distribution_shift.png

Usage
-----
    python3 experiments/10_distribution_shift/run_distribution_shift.py
"""

import sys
import json
import gzip
import logging
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy.stats import ks_2samp, gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    ROUTELLM_BATTLES_REWARDS_PATH,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)
from utils.router_factory import create_experiment_router
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — K=5 portfolio (mirrors experiment 06_figure)
# ---------------------------------------------------------------------------

def _req_cost(inp, out):
    return (100 * inp + 400 * out) / 1_000_000

PORTFOLIO_K5 = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "google/gemini-2.5-flash-preview-09-2025",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
]

MODEL_REGISTRY = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display_name": "Llama-3.1-8B",
        "input_cost_per_million_tokens": 0.05,
        "output_cost_per_million_tokens": 0.05,
        "provider": "meta",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "display_name": "Mixtral-8x7B",
        "input_cost_per_million_tokens": 0.54,
        "output_cost_per_million_tokens": 0.60,
        "provider": "mistral",
    },
    "google/gemini-2.5-flash-preview-09-2025": {
        "display_name": "Gemini-2.5-Flash",
        "input_cost_per_million_tokens": 0.15,
        "output_cost_per_million_tokens": 0.60,
        "provider": "google",
    },
    "anthropic/claude-sonnet-4": {
        "display_name": "Claude-Sonnet-4",
        "input_cost_per_million_tokens": 3.00,
        "output_cost_per_million_tokens": 15.00,
        "provider": "anthropic",
    },
    "openai/gpt-4.1": {
        "display_name": "GPT-4.1",
        "input_cost_per_million_tokens": 2.00,
        "output_cost_per_million_tokens": 8.00,
        "provider": "openai",
    },
}

COSTS = {m: _req_cost(
    MODEL_REGISTRY[m]["input_cost_per_million_tokens"],
    MODEL_REGISTRY[m]["output_cost_per_million_tokens"],
) for m in PORTFOLIO_K5}

MAX_SOURCE_PROMPTS = 8000   # RouteLLM battles (offline/prior distribution)
N_BOOTSTRAP = 1000          # for PSI confidence interval
N_PSI_BINS = 10
N_TRIALS = 20               # multi-seed router evaluation
SEED = 42

# Lambda sweep mirrors experiment 06_figure for direct comparability
LAMBDA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_routellm_prompts(path: Path, max_samples: int) -> List[str]:
    """Load unique prompts from RouteLLM battles JSONL (offline / prior dist)."""
    prompts = []
    seen = set()
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


def load_deployment_data(path: Path, models: List[str],
                         prompt_allowlist: Optional[List[str]] = None) -> List[Dict]:
    """Load prompt+rewards from gzipped JSONL deployment data.

    Parameters
    ----------
    path:
        Path to a ``.jsonl.gz`` reward file.
    models:
        Only keep entries whose model_id is in this list.
    prompt_allowlist:
        If provided, only keep prompts in this set (used to honour the
        three-way split so dev and holdout sets stay disjoint).
    """
    model_set = set(models)
    allow_set = set(prompt_allowlist) if prompt_allowlist is not None else None
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            p, m = entry["prompt"], entry["model_id"]
            if m not in model_set:
                continue
            if allow_set is not None and p not in allow_set:
                continue
            judges = entry.get("judge_details")
            r = float(np.mean([j["vote"] for j in judges])) if judges else float(entry["raw_score"])
            rewards[p][m] = r
    return [
        {"prompt": p, "rewards": r}
        for p, r in rewards.items()
        if len(r) == len(models)
    ]

# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def compute_psi(source: np.ndarray, deploy: np.ndarray,
                n_bins: int = N_PSI_BINS) -> float:
    """Population Stability Index between source and deploy distributions."""
    eps = 1e-8
    combined = np.concatenate([source, deploy])
    bin_edges = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    src_counts = np.histogram(source, bins=bin_edges)[0].astype(float)
    dep_counts = np.histogram(deploy, bins=bin_edges)[0].astype(float)

    src_pct = src_counts / src_counts.sum() + eps
    dep_pct = dep_counts / dep_counts.sum() + eps

    return float(np.sum((dep_pct - src_pct) * np.log(dep_pct / src_pct)))


def bootstrap_psi_ci(source: np.ndarray, deploy: np.ndarray,
                     n_bootstrap: int = N_BOOTSTRAP,
                     rng: Optional[np.random.RandomState] = None,
                     ) -> Tuple[float, float]:
    """95% bootstrap confidence interval for PSI."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    psi_values = []
    for _ in range(n_bootstrap):
        s = rng.choice(source, size=len(source), replace=True)
        d = rng.choice(deploy, size=len(deploy), replace=True)
        psi_values.append(compute_psi(s, d))
    return float(np.percentile(psi_values, 2.5)), float(np.percentile(psi_values, 97.5))

# ---------------------------------------------------------------------------
# Prior miscalibration
# ---------------------------------------------------------------------------

def compute_prior_miscalibration(
    warmup_path: Path, deploy_data: List[Dict], models: List[str]
) -> Dict[str, Dict]:
    """Compare offline prior estimates vs. observed deployment rewards.

    Loads the full 43-model warmup priors file but extracts only the K=5
    portfolio models (``models``).  For each model, the prior estimate is
    theta[-1] — the bias/intercept term from A^{-1}b — which represents the
    router's expected reward for a neutral prompt before seeing any deployment
    data.  This is compared against the actual mean reward observed in the
    deployment dataset to quantify miscalibration caused by distribution shift.

    Warmup priors file structure:
        {"A": {model_id: ndarray(33,33)},
         "b": {model_id: ndarray(33,)},
         "models": [...43 ids...]}
    """
    priors = joblib.load(warmup_path)
    # Only pull the K=5 entries — ignore the other 38 models entirely
    A_dict = {m: priors["A"][m] for m in models if m in priors.get("A", {})}
    b_dict = {m: priors["b"][m] for m in models if m in priors.get("b", {})}

    observed = {m: float(np.mean([d["rewards"][m] for d in deploy_data])) for m in models}

    results = {}
    for m in models:
        A = A_dict.get(m)
        b = b_dict.get(m)
        if A is not None and b is not None:
            try:
                theta = np.linalg.solve(A, b)
                # theta[-1] is the bias/intercept (constant feature = 1).
                # It represents the prior's expected reward for an average prompt.
                prior_mean = float(theta[-1])
            except np.linalg.LinAlgError:
                prior_mean = None
        else:
            prior_mean = None

        obs = observed[m]
        abs_err = float(prior_mean - obs) if prior_mean is not None else None
        rel_err = float((prior_mean - obs) / obs * 100) if prior_mean is not None else None
        results[m] = {
            "display_name": MODEL_REGISTRY[m]["display_name"],
            "prior_mean_estimate": prior_mean,
            "observed_deployment_mean": obs,
            "absolute_error": abs_err,
            "relative_error_pct": rel_err,
            "in_priors": A is not None,
        }
    return results

# ---------------------------------------------------------------------------
# Router evaluation — Pareto sweep (mirrors experiment 06_figure)
# ---------------------------------------------------------------------------

def build_model_registry_for_router(models: List[str]) -> Dict:
    """Convert display-name registry to the format expected by BanditRouter.

    BanditRouter._get_normalized_cost() reads ``input_cost_per_m`` and
    ``output_cost_per_m`` (per-million-token pricing), not the verbose
    ``input_cost_per_million_tokens`` key used in our config dict.
    """
    return {
        m: {
            "display_name": MODEL_REGISTRY[m]["display_name"],
            "input_cost_per_m": MODEL_REGISTRY[m]["input_cost_per_million_tokens"],
            "output_cost_per_m": MODEL_REGISTRY[m]["output_cost_per_million_tokens"],
            "provider": MODEL_REGISTRY[m]["provider"],
        }
        for m in models
    }


def _reward_normalisers(data: List[Dict], models: List[str]) -> Tuple[float, float]:
    """Compute global min/max reward for normalisation — same as experiment 06."""
    all_r = [d["rewards"][m] for d in data for m in models]
    r_min, r_max = min(all_r), max(all_r)
    return r_min, max(r_max - r_min, 1e-6)


def run_pareto_point(
    train_data: List[Dict], train_emb: List[np.ndarray],
    holdout_data: List[Dict], holdout_emb: List[np.ndarray],
    models: List[str], warmup_path: Path,
    cost_penalty: float,
    *,
    use_corralling: bool,
    freeze_after_training: bool = False,
    n_trials: int = N_TRIALS,
) -> Dict:
    """Run one (cost_penalty, policy) combination and return mean reward & cost.

    Parameters
    ----------
    freeze_after_training:
        If True, the router is trained on train_data but the HOLDOUT evaluation
        uses a fresh router with the SAME priors and NO feedback — representing
        the "static frozen prior at this lambda" baseline.  This ensures the
        static baseline and the adaptive baseline use identical cost penalties
        and reward normalisation; the only difference is whether online learning
        happened before holdout evaluation.
    """
    dim = train_emb[0].shape[0]
    n_train = len(train_data)
    r_min, r_range = _reward_normalisers(train_data, models)
    registry = build_model_registry_for_router(models)

    trial_rewards, trial_costs = [], []

    for trial in range(n_trials):
        np.random.seed(SEED + trial)

        # --- Training phase -------------------------------------------------
        router = create_experiment_router(
            model_registry=registry,
            feature_dim=dim,
            prior_n_effective=10.0,
            alpha=2.0,
            warmup_path=str(warmup_path),
            use_corralling=use_corralling,
            corralling_learning_rate=0.1,
            corralling_gamma=0.05,
            cost_penalty=cost_penalty,
            policy="hybrid",
        )

        if not freeze_after_training:
            # Adaptive: train on dev set with online feedback
            indices = list(range(n_train))
            np.random.RandomState(SEED + trial).shuffle(indices)
            for idx in indices:
                d, x = train_data[idx], train_emb[idx]
                m, log = router.route(x, total_steps=n_train)
                norm_r = (d["rewards"][m] - r_min) / r_range
                router.process_feedback(log.request_id, norm_r)
            eval_router = router
        else:
            # Static frozen prior: create a fresh router (same priors, same lambda)
            # with NO training, so holdout sees the raw prior at this cost penalty.
            eval_router = create_experiment_router(
                model_registry=registry,
                feature_dim=dim,
                prior_n_effective=10.0,
                alpha=2.0,
                warmup_path=str(warmup_path),
                use_corralling=False,   # no Corralling exploration overhead
                cost_penalty=cost_penalty,
                policy="hybrid",
            )

        # --- Holdout evaluation (no feedback in either case) ----------------
        rewards, costs = [], []
        for d, x in zip(holdout_data, holdout_emb):
            m, _ = eval_router.route(x, total_steps=n_train)
            rewards.append(d["rewards"][m])
            costs.append(COSTS[m])

        trial_rewards.append(float(np.mean(rewards)))
        trial_costs.append(float(np.mean(costs)))

    return {
        "cost_penalty": cost_penalty,
        "mean_reward": float(np.mean(trial_rewards)),
        "std_reward": float(np.std(trial_rewards, ddof=1)),
        "mean_cost": float(np.mean(trial_costs)),
        "std_cost": float(np.std(trial_costs, ddof=1)),
        "n_trials": n_trials,
    }


def run_pareto_sweep(
    train_data, train_emb, holdout_data, holdout_emb,
    models, warmup_path, lambda_values,
    use_corralling: bool, freeze_after_training: bool,
    label: str,
) -> List[Dict]:
    """Sweep lambda values and return the Pareto-optimal frontier."""
    points = []
    for lam in lambda_values:
        logger.info(f"     λ={lam:.3f} ...")
        pt = run_pareto_point(
            train_data, train_emb, holdout_data, holdout_emb,
            models, warmup_path, lam,
            use_corralling=use_corralling,
            freeze_after_training=freeze_after_training,
        )
        pt["label"] = label
        points.append(pt)
    return points

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    source_pc1: np.ndarray,
    deploy_pc1: np.ndarray,
    psi: float,
    psi_ci: Tuple[float, float],
    ks_stat: float, ks_p: float,
    pareto_static: List[Dict],
    pareto_adaptive: List[Dict],
    oracle_reward: float,
    oracle_cost: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Panel A: Feature distribution shift --------------------------------
    ax = axes[0]
    x_grid = np.linspace(
        min(source_pc1.min(), deploy_pc1.min()) - 0.5,
        max(source_pc1.max(), deploy_pc1.max()) + 0.5,
        300,
    )
    kde_src = gaussian_kde(source_pc1)
    kde_dep = gaussian_kde(deploy_pc1)
    ax.fill_between(x_grid, kde_src(x_grid), alpha=0.4, color="#4C72B0",
                    label=f"Offline / Prior (N={len(source_pc1):,})")
    ax.fill_between(x_grid, kde_dep(x_grid), alpha=0.4, color="#DD8452",
                    label=f"Deployment / LMSYS (N={len(deploy_pc1):,})")
    ax.set_xlabel("PC1 projection")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Feature Distribution Shift\n"
        f"PSI={psi:.3f} [{psi_ci[0]:.3f}, {psi_ci[1]:.3f}]  "
        f"KS D={ks_stat:.3f} (p={ks_p:.3e})"
    )
    ax.legend(fontsize=8)

    # --- Panel B: Pareto frontiers (static prior vs adaptive Hybrid) --------
    ax = axes[1]
    s_costs = [pt["mean_cost"] for pt in pareto_static]
    s_rwds  = [pt["mean_reward"] for pt in pareto_static]
    a_costs = [pt["mean_cost"] for pt in pareto_adaptive]
    a_rwds  = [pt["mean_reward"] for pt in pareto_adaptive]

    ax.plot(s_costs, s_rwds, "s--", color="#4C72B0", label="Static frozen prior")
    ax.plot(a_costs, a_rwds, "o-",  color="#DD8452", label="Hybrid banditGPT (online)")
    ax.scatter([oracle_cost], [oracle_reward], marker="*", s=200, color="green",
               zorder=5, label=f"Oracle ({oracle_reward:.3f})")

    ax.set_xlabel("Mean cost per request ($)")
    ax.set_ylabel("Holdout reward (mean judge agreement)")
    ax.set_title("Cost–Quality Pareto: Static Prior vs. Adaptive Hybrid\n"
                 "(K=5, deployment distribution, same λ sweep)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Figure saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("Experiment 10: Distribution Shift Analysis (K=5)")
    logger.info("=" * 70)

    models = PORTFOLIO_K5
    logger.info(f"   Portfolio: {[MODEL_REGISTRY[m]['display_name'] for m in models]}")

    # 1. Load encoder / PCA
    logger.info("\n1. Loading encoder and PCA ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    logger.info(f"   PCA: {pca.n_components_} components")

    # 2. Load offline (source/prior) prompts — RouteLLM battles
    logger.info("\n2. Loading offline RouteLLM battle prompts ...")
    source_prompts = load_routellm_prompts(ROUTELLM_BATTLES_REWARDS_PATH, MAX_SOURCE_PROMPTS)
    logger.info(f"   Loaded {len(source_prompts)} offline prompts")

    # 3. Load deployment data — use three-way split to honour dev/holdout separation
    logger.info("\n3. Loading deployment data (K=5 portfolio, three-way split) ...")
    with open(THREE_WAY_SPLITS_PATH) as f:
        splits = json.load(f)
    online_prompts = splits["online_learn_pool"]
    # Load all-models reward file, filtering to K=5 portfolio and split prompts
    dev_data = load_deployment_data(DEV_DATA_PATH_ALL_MODELS, models,
                                    prompt_allowlist=online_prompts)
    # Holdout: all prompts NOT in the online split
    online_set = set(online_prompts)
    holdout_data_raw = load_deployment_data(HOLDOUT_DATA_PATH_ALL_MODELS, models)
    holdout_data = [d for d in holdout_data_raw if d["prompt"] not in online_set]

    deploy_prompts = [d["prompt"] for d in dev_data + holdout_data]
    logger.info(f"   Dev (online train): {len(dev_data)} prompts")
    logger.info(f"   Holdout (eval):     {len(holdout_data)} prompts")

    # 4. Embed all prompts
    logger.info("\n4. Embedding prompts ...")
    logger.info(f"   Embedding {len(source_prompts)} offline (source) prompts ...")
    source_emb = np.array([embed_prompt(p, encoder, pca) for p in source_prompts])
    logger.info(f"   Embedding {len(deploy_prompts)} deployment prompts ...")
    deploy_emb_all = np.array([embed_prompt(p, encoder, pca) for p in deploy_prompts])
    dev_emb = [embed_prompt(d["prompt"], encoder, pca) for d in dev_data]
    holdout_emb = [embed_prompt(d["prompt"], encoder, pca) for d in holdout_data]
    logger.info(f"   Embedding dim: {source_emb.shape[1]}")

    # PC1 projections
    source_pc1 = source_emb[:, 0]
    deploy_pc1 = deploy_emb_all[:, 0]

    # 5. PSI and KS test
    logger.info("\n5. Computing distribution shift metrics ...")
    psi = compute_psi(source_pc1, deploy_pc1)
    rng = np.random.RandomState(SEED)
    psi_lo, psi_hi = bootstrap_psi_ci(source_pc1, deploy_pc1, rng=rng)
    ks_stat, ks_p = ks_2samp(source_pc1, deploy_pc1)
    logger.info(f"   PSI = {psi:.4f}  95% CI [{psi_lo:.4f}, {psi_hi:.4f}]")
    logger.info(f"   KS  D = {ks_stat:.4f}, p = {ks_p:.3e}")

    psi_severity = "Negligible" if psi < 0.1 else ("Moderate" if psi < 0.25 else "Substantial")
    logger.info(f"   Severity: {psi_severity} (threshold: 0.25)")

    # 6. Prior miscalibration
    logger.info("\n6. Computing prior miscalibration ...")
    prior_miscal = compute_prior_miscalibration(
        MULTIMODEL_WARMUP_PRIORS_PATH, dev_data + holdout_data, models
    )
    for m, v in prior_miscal.items():
        prior_str = f"{v['prior_mean_estimate']:.3f}" if v["prior_mean_estimate"] is not None else "N/A"
        err_str = (f"{v['absolute_error']:+.3f} ({v['relative_error_pct']:+.1f}%)"
                   if v["absolute_error"] is not None else "N/A")
        logger.info(f"   {v['display_name']}: prior≈{prior_str}  "
                    f"observed={v['observed_deployment_mean']:.3f}  error={err_str}")

    # 7. Oracle reward & cost on K=5 holdout
    oracle_reward = float(np.mean([
        max(d["rewards"][m] for m in models) for d in holdout_data
    ]))
    oracle_cost = float(np.mean([
        COSTS[max(models, key=lambda m: d["rewards"][m])] for d in holdout_data
    ]))
    logger.info(f"\n7. Oracle: reward={oracle_reward:.4f}  cost=${oracle_cost:.6f}")

    # 8. Pareto sweep — static frozen prior (same λ, no online learning)
    logger.info(f"\n8. Static frozen-prior Pareto sweep "
                f"({len(LAMBDA_VALUES)} λ × {N_TRIALS} trials) ...")
    pareto_static = run_pareto_sweep(
        dev_data, dev_emb, holdout_data, holdout_emb,
        models, MULTIMODEL_WARMUP_PRIORS_PATH, LAMBDA_VALUES,
        use_corralling=False, freeze_after_training=True,
        label="static_frozen_prior",
    )
    for pt in pareto_static:
        logger.info(f"   λ={pt['cost_penalty']:.3f}  "
                    f"R={pt['mean_reward']:.4f}±{pt['std_reward']:.4f}  "
                    f"C=${pt['mean_cost']:.6f}")

    # 9. Pareto sweep — Hybrid banditGPT with online learning (Corralling + hybrid)
    logger.info(f"\n9. Adaptive Hybrid banditGPT Pareto sweep "
                f"({len(LAMBDA_VALUES)} λ × {N_TRIALS} trials) ...")
    pareto_adaptive = run_pareto_sweep(
        dev_data, dev_emb, holdout_data, holdout_emb,
        models, MULTIMODEL_WARMUP_PRIORS_PATH, LAMBDA_VALUES,
        use_corralling=True, freeze_after_training=False,
        label="hybrid_banditgpt",
    )
    for pt in pareto_adaptive:
        logger.info(f"   λ={pt['cost_penalty']:.3f}  "
                    f"R={pt['mean_reward']:.4f}±{pt['std_reward']:.4f}  "
                    f"C=${pt['mean_cost']:.6f}")

    # 10. Summary metrics: best quality and max cost saving
    best_static  = max(pareto_static,  key=lambda x: x["mean_reward"])
    best_adaptive = max(pareto_adaptive, key=lambda x: x["mean_reward"])
    # Find adaptive point with similar quality to best static — measure cost difference
    quality_target = best_static["mean_reward"]
    comparable = [p for p in pareto_adaptive if p["mean_reward"] >= quality_target - 0.005]
    cheapest_comparable = min(comparable, key=lambda x: x["mean_cost"]) if comparable else None

    # 11. Plot
    out_fig = Path(__file__).parent / "results" / "figure_distribution_shift.png"
    plot_results(
        source_pc1, deploy_pc1, psi, (psi_lo, psi_hi), ks_stat, ks_p,
        pareto_static, pareto_adaptive, oracle_reward, oracle_cost, out_fig,
    )

    # 12. Save JSON results
    results = {
        "distribution_shift": {
            "psi": psi,
            "psi_ci_95": [psi_lo, psi_hi],
            "psi_severity": psi_severity,
            "ks_stat": ks_stat,
            "ks_p_value": ks_p,
            "n_source": len(source_pc1),
            "n_deploy": len(deploy_pc1),
        },
        "prior_miscalibration": prior_miscal,
        "oracle_reward": oracle_reward,
        "oracle_cost": oracle_cost,
        "pareto_static": pareto_static,
        "pareto_adaptive": pareto_adaptive,
        "best_static": best_static,
        "best_adaptive": best_adaptive,
        "cheapest_comparable_adaptive": cheapest_comparable,
        "models": {m: MODEL_REGISTRY[m]["display_name"] for m in models},
        "lambda_values": LAMBDA_VALUES,
    }
    out_json = Path(__file__).parent / "results" / "distribution_shift_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Results saved: {out_json}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  PSI = {psi:.3f} [{psi_lo:.3f}, {psi_hi:.3f}]  ({psi_severity})")
    logger.info(f"  KS D = {ks_stat:.3f}, p = {ks_p:.3e}")
    logger.info(f"  Prior miscalibration range: "
                f"{min(v['relative_error_pct'] for v in prior_miscal.values() if v['relative_error_pct']):+.1f}% "
                f"to "
                f"{max(v['relative_error_pct'] for v in prior_miscal.values() if v['relative_error_pct']):+.1f}%")
    logger.info(f"  Oracle:               R={oracle_reward:.4f}  C=${oracle_cost:.6f}")
    logger.info(f"  Best static:          R={best_static['mean_reward']:.4f}  C=${best_static['mean_cost']:.6f}")
    logger.info(f"  Best adaptive:        R={best_adaptive['mean_reward']:.4f}  C=${best_adaptive['mean_cost']:.6f}")
    if cheapest_comparable:
        cost_saving = (best_static["mean_cost"] - cheapest_comparable["mean_cost"]) / best_static["mean_cost"] * 100
        logger.info(f"  Cost saving at comparable quality: {cost_saving:.1f}%")


if __name__ == "__main__":
    main()
