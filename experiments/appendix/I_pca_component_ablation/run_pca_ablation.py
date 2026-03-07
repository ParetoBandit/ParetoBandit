#!/usr/bin/env python3
"""
Appendix I: PCA Component-Count Ablation
=========================================

Sweeps the number of PCA components used for contextual features in
the K=3 routing setting.  For each component count, this script runs the same hyperparameter grid as
Appendix H (alpha x n_eff x gamma, each with a lambda sweep) to
ensure every dimensionality gets its own optimal hyperparameters.

This avoids the common pitfall of holding hparams fixed while varying
feature dimensionality — optimal exploration (alpha) and prior strength
(n_eff) depend on the samples-per-feature ratio, which changes with d.

Protocol
--------
The canonical validation split (K4_CAL) is sub-split into two disjoint
halves — **val_tune** (hyperparameter selection) and **val_report**
(unbiased metric for the ablation curve).  This avoids the maximization
bias that arises when the grid-search winner is selected on and reported
from the same data.  The holdout set is never touched — it is reserved
for final paper claims.

For each component count k in {4, 6, 8, 10, 12, 15, 20, 24, 32}:

  1. Truncate the production 32-component PCA to k dimensions.
  2. Truncate the portfolio-filtered 32-comp warmup priors to k dims
     (keep first k PCA dims + bias row/col of A and b).
  3. Encode all prompts once (raw sentence embeddings), then project
     through the truncated PCA + whitening + bias.
  4. Train on the canonical train split; sweep (alpha x n_eff x gamma)
     with a lambda sweep at each, selecting the configuration with
     the highest Pareto AUC on **val_tune**.
  5. Re-evaluate only the selected configuration on **val_report** to
     obtain the plotted ablation metric.
  6. Record samples-per-feature ratio and explained variance for each
     component count.

Selection criterion: **Pareto AUC** (area under the cost-reward frontier),
matching the Appendix H grid ablation.

Portfolio: K=3 (using K=4 canonical data splits).

Outputs (``results/``)
    pca_component_ablation.json   — full results for K=3
    pca_component_ablation.png    — summary plot

Usage::

    cd experiments/appendix/I_pca_component_ablation
    python run_pca_ablation.py
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    FULL_PCA_PATH,
    K3_MODELS_PATH,
    K3_WARMUP_PRIORS_32_PATH,
    K4_TRAIN_DATA_PATH,
    K4_CAL_DATA_PATH,
)
from utils.embeddings import load_raw_embedding_cache, get_raw_embeddings_for_prompts
from utils.rewards import extract_reward
from utils.router_factory import create_experiment_router
from utils.model_pricing import get_prices_for_models, req_cost
from utils.pareto import pareto_auc

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Configuration
# ============================================================================

COMPONENT_COUNTS: List[int] = [4, 6, 8, 10, 12, 15, 20, 24, 32]

ALPHA_VALUES: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0]
NEFF_VALUES: List[float] = [10.0, 100.0, 1000.0, 5000.0]
GAMMA_VALUES: List[float] = [0.995, 0.999, 0.9999, 1.0]
LAMBDA_SWEEP: List[float] = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

N_SEEDS: int = 20
SEED_OFFSET: int = 42
CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05

REWARD_THEORETICAL_MIN: float = 0.0
REWARD_THEORETICAL_MAX: float = 1.0

VAL_REPORT_FRACTION: float = 0.5
VAL_SPLIT_SEED: int = 2026



# ============================================================================
# Portfolios
# ============================================================================

def _load_k3_portfolio() -> Tuple[List[str], Dict[str, Dict]]:
    """Load K=3 model list and catalog from ``models_k3.json``."""
    with open(K3_MODELS_PATH) as f:
        k3_cfg = json.load(f)
    models = [m["model_id"] for m in k3_cfg["models"]]
    prices = get_prices_for_models(models)
    catalog: Dict[str, Dict] = {}
    for m_entry in k3_cfg["models"]:
        mid = m_entry["model_id"]
        catalog[mid] = {
            "display": m_entry.get("display", mid.split("/")[-1]),
            **prices[mid],
        }
    return models, catalog


K3_MODELS, K3_CATALOG = _load_k3_portfolio()



def build_model_registry(
    models: List[str],
    catalog: Dict[str, Dict],
) -> Dict[str, Dict[str, float]]:
    """Build the registry dict that ``create_experiment_router`` expects."""
    return {
        m: {
            "input_cost_per_m": catalog[m]["input_cost_per_m"],
            "output_cost_per_m": catalog[m]["output_cost_per_m"],
        }
        for m in models
    }


# ============================================================================
# Data loading
# ============================================================================


def load_rewards_from_file(
    data_path: Path,
    models: List[str],
    *,
    prompt_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    """Load rewards for specific models from gzipped JSONL.

    Only prompts with rewards for *all* requested models are included.
    """
    model_set = set(models)
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)

    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            prompt = entry["prompt"]
            if prompt_filter is not None and prompt not in prompt_filter:
                continue
            model_id = entry["model_id"]
            if model_id not in model_set:
                continue
            rewards[prompt][model_id] = extract_reward(entry)

    return [
        {"prompt": p, "rewards": rmap}
        for p, rmap in rewards.items()
        if len(rmap) == len(models)
    ]


# ============================================================================
# PCA truncation and warmup prior handling
# ============================================================================


def truncate_pca(pca_full: PCA, n_components: int) -> PCA:
    """Create a reduced-dimension PCA by slicing the first *n_components*."""
    if n_components >= pca_full.n_components_:
        return pca_full
    pca = PCA(n_components=n_components, whiten=pca_full.whiten)
    pca.components_ = pca_full.components_[:n_components]
    pca.explained_variance_ = pca_full.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca_full.singular_values_[:n_components]
    pca.mean_ = pca_full.mean_
    pca.n_components_ = n_components
    pca.n_features_in_ = pca_full.n_features_in_
    pca.n_samples_ = pca_full.n_samples_
    pca.noise_variance_ = pca_full.noise_variance_
    return pca


def truncate_warmup_priors(
    priors: Dict[str, Any],
    n_components: int,
) -> Dict[str, Any]:
    """Truncate sufficient statistics to fewer PCA components.

    The priors contain A (d x d) and b (d,) per model where
    d = pca_components + 1 (bias).  Truncating keeps the first
    n_components PCA dimensions and the last (bias) row/col.
    """
    old_dim = priors["context_dim"]
    new_dim = n_components + 1

    if new_dim >= old_dim:
        return priors

    keep_idx = list(range(n_components)) + [old_dim - 1]

    new_A = {}
    new_b = {}
    for m in priors["models"]:
        A_full = priors["A"][m]
        b_full = priors["b"][m]
        new_A[m] = A_full[np.ix_(keep_idx, keep_idx)]
        new_b[m] = b_full[keep_idx]

    truncated = dict(priors)
    truncated["A"] = new_A
    truncated["b"] = new_b
    truncated["context_dim"] = new_dim
    truncated["pca_components"] = n_components
    return truncated


# ============================================================================
# Embedding: encode once, project per component count
# ============================================================================


def get_raw_embeddings(
    data: List[Dict],
    raw_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Look up raw sentence embeddings from the pre-computed cache.

    Args:
        data: List of dicts, each with a ``"prompt"`` key.
        raw_cache: Pre-loaded cache from ``load_raw_embedding_cache()``.

    Returns:
        Array of shape ``(n_prompts, embedding_dim)``.
    """
    prompts = [d["prompt"] for d in data]
    return get_raw_embeddings_for_prompts(prompts, raw_cache)


def project_embeddings(
    raw_emb: np.ndarray,
    pca: PCA,
) -> List[np.ndarray]:
    """Project raw embeddings through PCA + whitening + bias.

    Mirrors ``bandit_gpt.calibration.embed_prompt`` but operates on
    pre-encoded raw embeddings in batch.

    Args:
        raw_emb: (n_prompts, raw_dim) array from SentenceTransformer.
        pca: Fitted (possibly truncated) PCA model.

    Returns:
        List of (pca_components + 1,) context vectors.
    """
    projected = pca.transform(raw_emb)

    if not bool(getattr(pca, "whiten", False)):
        ev = getattr(pca, "explained_variance_", None)
        if ev is not None:
            scale = 1.0 / np.sqrt(
                np.maximum(np.asarray(ev, dtype=np.float64), 1e-12)
            )
            projected = projected * scale

    bias = np.ones((projected.shape[0], 1))
    with_bias = np.hstack([projected, bias])
    return [with_bias[i] for i in range(with_bias.shape[0])]


# ============================================================================
# Train-then-freeze evaluation (mirrors Appendix H)
# ============================================================================


def _train_and_eval_single_lambda(
    models: List[str],
    registry: Dict[str, Dict[str, float]],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_path: Optional[str],
    costs: Dict[str, float],
    dim: int,
    *,
    alpha: float,
    n_eff: float,
    gamma: float,
    cost_penalty: float,
    n_seeds: int,
) -> Dict[str, float]:
    """Train-then-freeze for one (alpha, n_eff, gamma, lambda) point.

    Returns:
        Dict with ``mean_reward``, ``mean_cost``, and per-seed arrays
        ``seed_rewards`` and ``seed_costs`` (each length *n_seeds*).
    """
    r_min = REWARD_THEORETICAL_MIN
    r_range = REWARD_THEORETICAL_MAX - REWARD_THEORETICAL_MIN
    burn_in = len(train_data)

    trial_r: List[float] = []
    trial_c: List[float] = []

    for trial in range(n_seeds):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=registry,
            feature_dim=dim,
            prior_n_effective=n_eff,
            alpha=alpha,
            warmup_path=warmup_path,
            use_corralling=True,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
            cost_penalty=cost_penalty,
            forgetting_factor=gamma,
        )

        order = np.random.permutation(len(train_data))
        for idx in order:
            p, x = train_data[idx], train_emb[idx]
            model, log = router.route(x, total_steps=burn_in)
            raw_reward = p["rewards"][model]
            norm_reward = (
                (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
            )
            router.process_feedback(log.request_id, norm_reward)

        rng_state = np.random.get_state()
        r_total = c_total = 0.0
        with router.exploit():
            for p, x in zip(eval_data, eval_emb):
                model, _log = router.route(x, total_steps=burn_in)
                r_total += p["rewards"][model]
                c_total += costs[model]
        np.random.set_state(rng_state)

        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)

    return {
        "mean_reward": float(np.mean(trial_r)),
        "mean_cost": float(np.mean(trial_c)),
        "seed_rewards": trial_r,
        "seed_costs": trial_c,
    }


def train_and_evaluate_pareto(
    models: List[str],
    catalog: Dict[str, Dict],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_path: Optional[str],
    costs: Dict[str, float],
    *,
    alpha: float,
    n_eff: float,
    gamma: float,
    lambda_values: List[float],
    cost_lo: float,
    cost_hi: float,
    n_seeds: int,
) -> Dict[str, Any]:
    """Evaluate one (alpha, n_eff, gamma) across multiple lambdas.

    Sweeps ``lambda_values`` to trace the Pareto frontier, then
    computes Pareto AUC over ``[cost_lo, cost_hi]``.  Also computes
    per-seed Pareto AUC to enable confidence intervals.
    """
    dim = train_emb[0].shape[0]
    registry = build_model_registry(models, catalog)

    sweep_points: List[Dict[str, Any]] = []
    seed_sweep_rewards: List[List[float]] = []
    seed_sweep_costs: List[List[float]] = []
    lam0_reward: float = 0.0

    for lam in lambda_values:
        pt = _train_and_eval_single_lambda(
            models, registry,
            train_data, eval_data, train_emb, eval_emb,
            warmup_path, costs, dim,
            alpha=alpha, n_eff=n_eff, gamma=gamma,
            cost_penalty=lam, n_seeds=n_seeds,
        )
        sweep_points.append({
            "lambda": lam,
            "mean_reward": pt["mean_reward"],
            "mean_cost": pt["mean_cost"],
        })
        seed_sweep_rewards.append(pt["seed_rewards"])
        seed_sweep_costs.append(pt["seed_costs"])
        if lam == 0.0:
            lam0_reward = pt["mean_reward"]

    sweep_costs = [sp["mean_cost"] for sp in sweep_points]
    sweep_rewards = [sp["mean_reward"] for sp in sweep_points]
    p_auc = pareto_auc(sweep_costs, sweep_rewards, cost_lo, cost_hi)

    n_lam = len(lambda_values)
    seed_pareto_aucs: List[float] = []
    for s in range(n_seeds):
        s_costs = [seed_sweep_costs[l][s] for l in range(n_lam)]
        s_rewards = [seed_sweep_rewards[l][s] for l in range(n_lam)]
        seed_pareto_aucs.append(
            pareto_auc(s_costs, s_rewards, cost_lo, cost_hi)
        )

    auc_arr = np.asarray(seed_pareto_aucs)

    return {
        "alpha": alpha,
        "n_eff": n_eff,
        "gamma": gamma,
        "pareto_auc": p_auc,
        "pareto_auc_se": float(np.std(auc_arr, ddof=1) / np.sqrt(n_seeds)),
        "pareto_auc_seeds": seed_pareto_aucs,
        "lam0_reward": lam0_reward,
        "sweep_points": sweep_points,
        "n_seeds": n_seeds,
    }


# ============================================================================
# Best single model baseline (non-contextual)
# ============================================================================


def best_single_model(
    train_data: List[Dict],
    eval_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
) -> Dict[str, Any]:
    """Best single model baseline via uniform empirical mean (non-contextual).

    Computes the mean reward for each arm uniformly over *train_data*,
    selects the argmax, and evaluates that fixed arm on *eval_data*.
    Deterministic — no seed dependence or exploration noise.

    Args:
        train_data: Training prompts, each with ``rewards[model]``.
        eval_data: Evaluation prompts (typically val split), each with
            ``rewards[model]``.
        models: Portfolio model IDs.
        costs: Per-model representative cost.

    Returns:
        Dict with ``reward``, ``cost``, ``greedy_arm``, and
        ``train_means`` (per-arm empirical means on train set).
    """
    train_means: Dict[str, float] = {
        m: float(np.mean([d["rewards"][m] for d in train_data]))
        for m in models
    }
    best_arm: str = max(train_means, key=train_means.get)  # type: ignore[arg-type]
    eval_reward = float(
        np.mean([d["rewards"][best_arm] for d in eval_data])
    )
    return {
        "reward": eval_reward,
        "cost": costs.get(best_arm, 0.0),
        "greedy_arm": best_arm,
        "train_means": train_means,
    }


# ============================================================================
# Validation sub-split (tune / report)
# ============================================================================


def _split_val_tune_report(
    data: List[Dict],
    raw_emb: np.ndarray,
    *,
    report_fraction: float = VAL_REPORT_FRACTION,
    seed: int = VAL_SPLIT_SEED,
) -> Tuple[List[Dict], np.ndarray, List[Dict], np.ndarray]:
    """Split validation data into *tune* and *report* halves.

    Hyperparameters are selected on *tune*; the plotted ablation metric
    is evaluated on *report*.  This eliminates the maximization bias
    that arises from selecting and reporting on the same split.

    Args:
        data: Full validation prompts (each with ``"prompt"`` and
            ``"rewards"`` keys).
        raw_emb: Raw sentence embeddings, shape ``(len(data), raw_dim)``.
        report_fraction: Fraction reserved for unbiased reporting.
        seed: Random seed for reproducible splitting.

    Returns:
        ``(tune_data, tune_emb, report_data, report_emb)``
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    n_report = int(n * report_fraction)
    perm = rng.permutation(n)
    report_idx = perm[:n_report]
    tune_idx = perm[n_report:]

    tune_data = [data[i] for i in tune_idx]
    report_data = [data[i] for i in report_idx]
    tune_emb = raw_emb[tune_idx]
    report_emb = raw_emb[report_idx]

    return tune_data, tune_emb, report_data, report_emb


# ============================================================================
# Per-component-count ablation
# ============================================================================


def run_ablation_for_n_components(
    n_comp: int,
    *,
    pca_truncated: PCA,
    priors_truncated: Dict[str, Any],
    models: List[str],
    catalog: Dict[str, Dict],
    train_data: List[Dict],
    val_tune_data: List[Dict],
    val_report_data: List[Dict],
    raw_train_emb: np.ndarray,
    raw_val_tune_emb: np.ndarray,
    raw_val_report_emb: np.ndarray,
    costs: Dict[str, float],
    cost_lo: float,
    cost_hi: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the full hparam grid for one component count.

    Trains the bandit on the canonical train split.  The hyperparameter
    grid is evaluated on **val_tune**; the selected configuration is
    then re-evaluated on **val_report** to produce an unbiased metric
    free of maximization bias.  The holdout set is never touched.

    Args:
        n_comp: Number of PCA components.
        pca_truncated: PCA model truncated to *n_comp*.
        priors_truncated: Warmup priors truncated to *n_comp* dims.
        models: Portfolio model IDs.
        catalog: Model metadata catalog.
        train_data: Canonical train-split prompts with rewards.
        val_tune_data: Validation sub-split for hyperparameter selection.
        val_report_data: Validation sub-split for unbiased reporting.
        raw_train_emb: Raw sentence embeddings for train set.
        raw_val_tune_emb: Raw sentence embeddings for val_tune set.
        raw_val_report_emb: Raw sentence embeddings for val_report set.
        costs: Per-model cost dict.
        cost_lo: Min cost for Pareto AUC.
        cost_hi: Max cost for Pareto AUC.
        output_dir: Directory for temp prior files.

    Returns:
        Dict with best config, Pareto AUC on both val_tune and
        val_report, variance explained, etc.
    """
    feat_dim = n_comp + 1
    var_explained = float(np.sum(pca_truncated.explained_variance_ratio_))

    train_emb = project_embeddings(raw_train_emb, pca_truncated)
    tune_emb = project_embeddings(raw_val_tune_emb, pca_truncated)
    report_emb = project_embeddings(raw_val_report_emb, pca_truncated)

    samples_per_arm = len(train_data) / len(models)
    samples_per_feature = len(train_data) / (len(models) * feat_dim)

    logger.info(
        f"    feat_dim={feat_dim}  var={var_explained:.1%}  "
        f"s/arm={samples_per_arm:.0f}  s/feat={samples_per_feature:.1f}  "
        f"train={len(train_data)}  "
        f"val_tune={len(val_tune_data)}  val_report={len(val_report_data)}"
    )

    tmp_priors = output_dir / f"_tmp_priors_{n_comp}comp.joblib"
    joblib.dump(priors_truncated, tmp_priors)

    # ── Grid sweep on val_tune ────────────────────────────────────────
    total = len(ALPHA_VALUES) * len(NEFF_VALUES) * len(GAMMA_VALUES)
    results: List[Dict[str, Any]] = []
    idx = 0

    for gamma in GAMMA_VALUES:
        for n_eff in NEFF_VALUES:
            for alpha in ALPHA_VALUES:
                idx += 1
                res = train_and_evaluate_pareto(
                    models, catalog,
                    train_data, val_tune_data, train_emb, tune_emb,
                    str(tmp_priors), costs,
                    alpha=alpha, n_eff=n_eff, gamma=gamma,
                    lambda_values=LAMBDA_SWEEP,
                    cost_lo=cost_lo, cost_hi=cost_hi,
                    n_seeds=N_SEEDS,
                )
                results.append(res)
                if idx % 20 == 0 or idx == total:
                    logger.info(
                        f"      [{idx:3d}/{total}] "
                        f"a={alpha:<4} neff={n_eff:<6} g={gamma:<6} "
                        f"AUC={res['pareto_auc']:.4f} "
                        f"R@0={res['lam0_reward']:.4f}"
                    )

    ranked = sorted(results, key=lambda r: r["pareto_auc"], reverse=True)
    best_tune = ranked[0]

    logger.info(
        f"    BEST (tune): a={best_tune['alpha']} "
        f"neff={int(best_tune['n_eff'])} g={best_tune['gamma']} "
        f"-> AUC={best_tune['pareto_auc']:.4f} "
        f"R@0={best_tune['lam0_reward']:.4f}"
    )

    # ── Re-evaluate selected config on val_report ─────────────────────
    logger.info("    Re-evaluating selected config on val_report ...")
    best_report = train_and_evaluate_pareto(
        models, catalog,
        train_data, val_report_data, train_emb, report_emb,
        str(tmp_priors), costs,
        alpha=best_tune["alpha"],
        n_eff=best_tune["n_eff"],
        gamma=best_tune["gamma"],
        lambda_values=LAMBDA_SWEEP,
        cost_lo=cost_lo, cost_hi=cost_hi,
        n_seeds=N_SEEDS,
    )

    tmp_priors.unlink(missing_ok=True)

    logger.info(
        f"    REPORT: AUC={best_report['pareto_auc']:.4f} "
        f"± {best_report['pareto_auc_se']:.4f} (SE)  "
        f"R@0={best_report['lam0_reward']:.4f}"
    )

    return {
        "n_components": n_comp,
        "feature_dim": feat_dim,
        "variance_explained": var_explained,
        "samples_per_arm": samples_per_arm,
        "samples_per_feature_ratio": samples_per_feature,
        "best_alpha": best_tune["alpha"],
        "best_n_eff": best_tune["n_eff"],
        "best_gamma": best_tune["gamma"],
        "tune_pareto_auc": best_tune["pareto_auc"],
        "tune_lam0_reward": best_tune["lam0_reward"],
        "report_pareto_auc": best_report["pareto_auc"],
        "report_pareto_auc_se": best_report["pareto_auc_se"],
        "report_pareto_auc_seeds": best_report["pareto_auc_seeds"],
        "report_lam0_reward": best_report["lam0_reward"],
        "top5_tune": [
            {
                "alpha": r["alpha"],
                "n_eff": r["n_eff"],
                "gamma": r["gamma"],
                "pareto_auc": r["pareto_auc"],
                "lam0_reward": r["lam0_reward"],
            }
            for r in ranked[:5]
        ],
    }


# ============================================================================
# Plotting
# ============================================================================


def plot_ablation(
    k3_results: List[Dict],
    bsm_k3: Dict,
    out_path: Path,
) -> None:
    """Generate a summary plot of Pareto AUC vs component count.

    The primary curve uses the **val_report** metric (unbiased); the
    val_tune metric is shown as a faded reference to make the
    maximization gap visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    comps = [r["n_components"] for r in k3_results]
    report_auc = [r["report_pareto_auc"] for r in k3_results]
    report_se = [r["report_pareto_auc_se"] for r in k3_results]
    tune_auc = [r["tune_pareto_auc"] for r in k3_results]

    report_lo = [a - 1.96 * s for a, s in zip(report_auc, report_se)]
    report_hi = [a + 1.96 * s for a, s in zip(report_auc, report_se)]

    ax.fill_between(comps, report_lo, report_hi, alpha=0.15, color="C0")
    ax.plot(
        comps, report_auc, "o-", color="C0",
        label="Val-report Pareto AUC ± 95% CI",
    )
    ax.plot(
        comps, tune_auc, "s--", color="C0", alpha=0.3, markersize=4,
        label="Val-tune Pareto AUC (selection)",
    )
    ax.axhline(
        bsm_k3["reward"], ls=":", color="gray", alpha=0.7,
        label=f"Best single model ({bsm_k3['reward']:.4f})",
    )
    ax.axvline(15, ls="--", color="C3", alpha=0.5, label="d=15 (production)")

    ax2 = ax.twinx()
    var_exp = [r["variance_explained"] for r in k3_results]
    ax2.fill_between(
        comps, 0, [v * 100 for v in var_exp],
        alpha=0.08, color="green",
    )
    ax2.plot(
        comps, [v * 100 for v in var_exp],
        "^-", color="green", alpha=0.4, markersize=4,
        label="Variance explained (%)",
    )
    ax2.set_ylabel("Variance explained (%)", fontsize=10, color="green")
    ax2.set_ylim(0, 40)

    ax.set_xlabel("PCA components", fontsize=11)
    ax.set_ylabel("Pareto AUC (reward units)", fontsize=11)
    ax.set_title("K=3 Portfolio", fontsize=12, fontweight="bold")
    ax.set_xticks(comps)
    ax.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "PCA Component-Count Ablation\n"
        "Each point uses its own optimal (alpha, n_eff, gamma);\n"
        "selected on val-tune, evaluated on val-report",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Shared resources ──────────────────────────────────────────────
    logger.info("Loading full 32-component PCA and raw embedding cache ...")
    pca32 = joblib.load(FULL_PCA_PATH)
    raw_cache = load_raw_embedding_cache()
    logger.info(
        f"  PCA: {pca32.n_components_} components "
        f"(whiten={pca32.whiten})"
    )

    # ── Load K=3 warmup priors (32 components) ──────────────────────
    logger.info("Loading K=3 32-component warmup priors ...")
    k3_priors_32 = joblib.load(K3_WARMUP_PRIORS_32_PATH)
    logger.info(
        f"  Source: {K3_WARMUP_PRIORS_32_PATH.name} "
        f"({len(k3_priors_32['models'])} models, "
        f"context_dim={k3_priors_32['context_dim']})"
    )

    # ── K=3 data (from K=4 canonical splits) ─────────────────────────
    logger.info("\nLoading K=3 data (K=4 canonical train/cal splits) ...")
    train_k3 = load_rewards_from_file(K4_TRAIN_DATA_PATH, K3_MODELS)
    val_k3 = load_rewards_from_file(K4_CAL_DATA_PATH, K3_MODELS)
    logger.info(f"  Train: {len(train_k3)}  Val (full): {len(val_k3)}")

    costs_k3 = {
        m: req_cost(K3_CATALOG[m]["input_cost_per_m"],
                    K3_CATALOG[m]["output_cost_per_m"])
        for m in K3_MODELS
    }
    cost_lo_k3, cost_hi_k3 = min(costs_k3.values()), max(costs_k3.values())

    # ── Look up raw embeddings from cache ────────────────────────────
    logger.info("\nLoading raw embeddings from cache ...")
    raw_train_k3 = get_raw_embeddings(train_k3, raw_cache)
    raw_val_k3 = get_raw_embeddings(val_k3, raw_cache)
    logger.info(
        f"  Raw embedding dim: {raw_train_k3.shape[1]}"
    )

    # ── Split val into tune (hparam selection) / report (unbiased) ───
    logger.info(
        f"\nSplitting val into tune/report "
        f"({1 - VAL_REPORT_FRACTION:.0%}/{VAL_REPORT_FRACTION:.0%}, "
        f"seed={VAL_SPLIT_SEED}) ..."
    )
    val_tune_k3, raw_val_tune_k3, val_report_k3, raw_val_report_k3 = (
        _split_val_tune_report(val_k3, raw_val_k3)
    )
    logger.info(
        f"  Val-tune: {len(val_tune_k3)}  Val-report: {len(val_report_k3)}"
    )

    # ── Best single model baseline (evaluated on val_report) ──────────
    logger.info("\nComputing best-single-model baseline (on val_report) ...")
    bsm_k3 = best_single_model(
        train_k3, val_report_k3, K3_MODELS, costs_k3,
    )
    logger.info(
        f"  Best single model: R={bsm_k3['reward']:.4f} "
        f"(arm: {bsm_k3['greedy_arm']})"
    )

    # ── Grid info ─────────────────────────────────────────────────────
    n_hparam = len(ALPHA_VALUES) * len(NEFF_VALUES) * len(GAMMA_VALUES)
    n_lam = len(LAMBDA_SWEEP)
    logger.info(f"\n{'='*70}")
    logger.info("PCA Component-Count Ablation")
    logger.info(f"  Components: {COMPONENT_COUNTS}")
    logger.info(
        f"  Grid: {n_hparam} hparam configs x {n_lam} lambda "
        f"x {N_SEEDS} seeds"
    )
    logger.info(
        f"  Per component count: "
        f"{n_hparam * n_lam * N_SEEDS:,} trials"
    )
    logger.info(
        f"  Total: {len(COMPONENT_COUNTS) * n_hparam * n_lam * N_SEEDS:,} "
        f"trials"
    )
    logger.info(f"{'='*70}")

    # ── Sweep component counts ────────────────────────────────────────
    k3_results: List[Dict] = []

    for n_comp in COMPONENT_COUNTS:
        pca_trunc = truncate_pca(pca32, n_comp)
        k3_priors_trunc = truncate_warmup_priors(k3_priors_32, n_comp)

        logger.info(f"\n{'='*60}")
        logger.info(f"  K=3, n_components={n_comp}")
        logger.info(f"{'='*60}")
        res_k3 = run_ablation_for_n_components(
            n_comp,
            pca_truncated=pca_trunc,
            priors_truncated=k3_priors_trunc,
            models=K3_MODELS,
            catalog=K3_CATALOG,
            train_data=train_k3,
            val_tune_data=val_tune_k3,
            val_report_data=val_report_k3,
            raw_train_emb=raw_train_k3,
            raw_val_tune_emb=raw_val_tune_k3,
            raw_val_report_emb=raw_val_report_k3,
            costs=costs_k3,
            cost_lo=cost_lo_k3,
            cost_hi=cost_hi_k3,
            output_dir=output_dir,
        )
        k3_results.append(res_k3)

    # ── Save results ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": (
                "PCA component-count ablation with per-component "
                "hyperparameter re-tuning.  K=3 portfolio on K=4 "
                "canonical train split; cal split sub-divided into "
                "val_tune (hparam selection) and val_report (unbiased "
                "metric).  Holdout never touched.  "
                "Selection criterion: Pareto AUC."
            ),
            "component_counts": COMPONENT_COUNTS,
            "alpha_values": ALPHA_VALUES,
            "neff_values": NEFF_VALUES,
            "gamma_values": GAMMA_VALUES,
            "lambda_sweep": LAMBDA_SWEEP,
            "n_seeds": N_SEEDS,
            "n_hparam_configs": n_hparam,
            "selection_criterion": "pareto_auc",
            "val_report_fraction": VAL_REPORT_FRACTION,
            "val_split_seed": VAL_SPLIT_SEED,
            "elapsed_seconds": elapsed,
        },
        "K3": {
            "models": K3_MODELS,
            "n_train": len(train_k3),
            "n_val_total": len(val_k3),
            "n_val_tune": len(val_tune_k3),
            "n_val_report": len(val_report_k3),
            "cost_range": [cost_lo_k3, cost_hi_k3],
            "best_single_model_baseline": bsm_k3,
            "ablation_results": k3_results,
        },
    }

    out_path = output_dir / "pca_component_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_ablation(
        k3_results, bsm_k3,
        output_dir / "pca_component_ablation.png",
    )

    # ── Summary table ─────────────────────────────────────────────────
    logger.info(f"\n{'='*105}")
    logger.info("SUMMARY  (K=3)")
    logger.info(f"{'='*105}")
    logger.info(
        f"  {'Comp':>5s}  {'Dim':>4s}  {'Var%':>6s}  "
        f"{'s/feat':>6s}  {'AUC_tune':>8s}  {'AUC_rpt':>8s}  "
        f"{'±SE':>6s}  {'R@0_rpt':>8s}  "
        f"{'best_a':>6s}  {'best_n':>6s}  {'best_g':>7s}"
    )
    logger.info("  " + "-" * 95)
    for r in k3_results:
        logger.info(
            f"  {r['n_components']:5d}  {r['feature_dim']:4d}  "
            f"{r['variance_explained']:5.1%}  "
            f"{r['samples_per_feature_ratio']:6.1f}  "
            f"{r['tune_pareto_auc']:8.4f}  "
            f"{r['report_pareto_auc']:8.4f}  "
            f"{r['report_pareto_auc_se']:6.4f}  "
            f"{r['report_lam0_reward']:8.4f}  "
            f"{r['best_alpha']:6.2f}  "
            f"{int(r['best_n_eff']):6d}  "
            f"{r['best_gamma']:7.4f}"
        )
    logger.info(f"  Best single model (val_report): {bsm_k3['reward']:.4f}")

    logger.info(f"\nElapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
