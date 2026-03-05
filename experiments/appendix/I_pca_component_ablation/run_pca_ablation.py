#!/usr/bin/env python3
"""
Appendix I: PCA Component-Count Ablation
=========================================

Sweeps the number of PCA components used for contextual features in
the K=2 and K=3 routing settings.  For each component count **and**
each portfolio, this script runs the same hyperparameter grid as
Appendix H (alpha x n_eff x gamma, each with a lambda sweep) to
ensure every dimensionality gets its own optimal hyperparameters.

This avoids the common pitfall of holding hparams fixed while varying
feature dimensionality — optimal exploration (alpha) and prior strength
(n_eff) depend on the samples-per-feature ratio, which changes with d.

Protocol
--------
For each component count k in {4, 6, 8, 10, 12, 15, 20, 24, 32}:

  1. Truncate the production 32-component PCA to k dimensions.
  2. Truncate the portfolio-filtered 32-comp warmup priors to k dims
     (keep first k PCA dims + bias row/col of A and b).
  3. Encode all prompts once (raw sentence embeddings), then project
     through the truncated PCA + whitening + bias.
  4. Sweep (alpha x n_eff x gamma) with a lambda sweep at each,
     selecting the configuration with the highest Pareto AUC on dev-val.
  5. Evaluate the selected config on holdout (Pareto AUC + lambda=0 reward).
  6. Record the dev-holdout gap, samples-per-feature ratio, and
     explained variance for each component count.

Selection criterion: **Pareto AUC** (area under the cost-reward frontier),
matching the Appendix H grid ablation.

Portfolios: K=2 and K=3 (no K=10).

Outputs (``results/``)
    pca_component_ablation.json   — full results for K=2 and K=3
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
    ARTIFACTS_DIR,
    FULL_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    K3_MODELS_PATH,
    THREE_WAY_SPLITS_PATH,
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

DEV_VAL_FRACTION: float = 0.2
DEV_VAL_SEED: int = 7

REWARD_THEORETICAL_MIN: float = 0.0
REWARD_THEORETICAL_MAX: float = 1.0

MULTIMODEL_PRIORS_32_PATH = ARTIFACTS_DIR / "priors_warmup_43model.joblib"


# ============================================================================
# Portfolios
# ============================================================================

K2_MODELS: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-4.1",
]

_PRICES_K2 = get_prices_for_models(K2_MODELS)

K2_CATALOG: Dict[str, Dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        **_PRICES_K2["meta-llama/llama-3.1-8b-instruct"],
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        **_PRICES_K2["openai/gpt-4.1"],
    },
}


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


def extract_portfolio_priors(
    priors_full: Dict[str, Any],
    target_models: List[str],
) -> Dict[str, Any]:
    """Filter multi-model priors to a portfolio subset.

    Args:
        priors_full: Full 43-model priors dict with A, b, models, etc.
        target_models: Model IDs to retain.

    Returns:
        New priors dict with only the target models.
    """
    missing = [m for m in target_models if m not in priors_full["models"]]
    if missing:
        raise ValueError(
            f"Models not found in priors: {missing}. "
            f"Available: {priors_full['models'][:5]}..."
        )
    filtered = dict(priors_full)
    filtered["A"] = {m: priors_full["A"][m] for m in target_models}
    filtered["b"] = {m: priors_full["b"][m] for m in target_models}
    filtered["models"] = list(target_models)
    return filtered


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


def _set_exploit_mode(router: Any, *, enable: bool) -> Dict[str, Any]:
    """Switch to greedy exploitation on a frozen router."""
    if not enable:
        return {}
    saved: Dict[str, Any] = {"expert_alphas": [], "meta_exploit": False}
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts"):
        for expert in cr.experts:
            saved["expert_alphas"].append((expert.alpha_start, expert.alpha_end))
            expert.alpha_start = 0.0
            expert.alpha_end = 0.0
        saved["meta_exploit"] = cr.exploit_mode
        cr.exploit_mode = True
    return saved


def _restore_exploit_mode(
    router: Any, saved: Dict[str, Any],
) -> None:
    """Restore expert alpha values and meta exploit mode."""
    if not saved:
        return
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts") and saved.get("expert_alphas"):
        for expert, (a_s, a_e) in zip(cr.experts, saved["expert_alphas"]):
            expert.alpha_start = a_s
            expert.alpha_end = a_e
        cr.exploit_mode = saved.get("meta_exploit", False)


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
    """Train-then-freeze for one (alpha, n_eff, gamma, lambda) point."""
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

        saved = _set_exploit_mode(router, enable=True)
        rng_state = np.random.get_state()
        r_total = c_total = 0.0
        for p, x in zip(eval_data, eval_emb):
            model, _log = router.route(x, total_steps=burn_in)
            r_total += p["rewards"][model]
            c_total += costs[model]
        np.random.set_state(rng_state)
        _restore_exploit_mode(router, saved)

        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)

    return {
        "mean_reward": float(np.mean(trial_r)),
        "mean_cost": float(np.mean(trial_c)),
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
    computes Pareto AUC over ``[cost_lo, cost_hi]``.
    """
    dim = train_emb[0].shape[0]
    registry = build_model_registry(models, catalog)

    sweep_points: List[Dict[str, Any]] = []
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
        if lam == 0.0:
            lam0_reward = pt["mean_reward"]

    sweep_costs = [sp["mean_cost"] for sp in sweep_points]
    sweep_rewards = [sp["mean_reward"] for sp in sweep_points]
    p_auc = pareto_auc(sweep_costs, sweep_rewards, cost_lo, cost_hi)

    return {
        "alpha": alpha,
        "n_eff": n_eff,
        "gamma": gamma,
        "pareto_auc": p_auc,
        "lam0_reward": lam0_reward,
        "sweep_points": sweep_points,
        "n_seeds": n_seeds,
    }


# ============================================================================
# Dev train/val split
# ============================================================================


def _split_dev_train_val(
    data: List[Dict],
    emb: List[np.ndarray],
    *,
    val_fraction: float = DEV_VAL_FRACTION,
    seed: int = DEV_VAL_SEED,
) -> Tuple[List[Dict], List[np.ndarray], List[Dict], List[np.ndarray]]:
    """Deterministically split (data, emb) into train and val portions."""
    n = len(data)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    train_d = [data[i] for i in range(n) if i not in val_idx]
    train_e = [emb[i] for i in range(n) if i not in val_idx]
    val_d = [data[i] for i in range(n) if i in val_idx]
    val_e = [emb[i] for i in range(n) if i in val_idx]
    return train_d, train_e, val_d, val_e


# ============================================================================
# UCB1 baseline (non-contextual)
# ============================================================================


def ucb1_online_route(
    train_data: List[Dict],
    holdout_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
    *,
    n_trials: int,
) -> Dict[str, Any]:
    """Non-contextual UCB1 baseline (independent of feature dimension)."""
    from collections import Counter

    all_rewards: List[float] = []
    all_costs: List[float] = []
    greedy_arms: List[str] = []

    for trial in range(n_trials):
        rng = np.random.default_rng(SEED_OFFSET + trial)
        counts = {m: 0 for m in models}
        sums = {m: 0.0 for m in models}
        order = rng.permutation(len(train_data))

        for idx in order:
            total = sum(counts.values())
            if total < len(models):
                arm = models[total]
            else:
                ucb = {
                    m: (sums[m] / counts[m])
                    + np.sqrt(2 * np.log(total) / counts[m])
                    for m in models
                }
                arm = max(ucb, key=lambda m: ucb[m])  # type: ignore[arg-type]
            r = train_data[idx]["rewards"][arm]
            counts[arm] += 1
            sums[arm] += r

        greedy = max(models, key=lambda m: sums[m] / max(counts[m], 1))
        greedy_arms.append(greedy)
        all_rewards.append(float(np.mean([d["rewards"][greedy] for d in holdout_data])))
        all_costs.append(costs.get(greedy, 0.0))

    return {
        "reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards, ddof=1)) if n_trials > 1 else 0.0,
        "cost": float(np.mean(all_costs)),
        "greedy_arm": Counter(greedy_arms).most_common(1)[0][0],
    }


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
    holdout_data: List[Dict],
    raw_train_emb: np.ndarray,
    raw_holdout_emb: np.ndarray,
    costs: Dict[str, float],
    cost_lo: float,
    cost_hi: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the full hparam grid for one component count.

    Args:
        n_comp: Number of PCA components.
        pca_truncated: PCA model truncated to n_comp.
        priors_truncated: Warmup priors truncated to n_comp dims.
        models: Portfolio model IDs.
        catalog: Model metadata catalog.
        train_data: Dev prompts with rewards.
        holdout_data: Holdout prompts with rewards.
        raw_train_emb: Raw sentence embeddings for train set.
        raw_holdout_emb: Raw sentence embeddings for holdout set.
        costs: Per-model cost dict.
        cost_lo: Min cost for Pareto AUC.
        cost_hi: Max cost for Pareto AUC.
        output_dir: Directory for temp prior files.

    Returns:
        Dict with best config, Pareto AUC, variance explained, etc.
    """
    feat_dim = n_comp + 1
    var_explained = float(np.sum(pca_truncated.explained_variance_ratio_))

    train_emb = project_embeddings(raw_train_emb, pca_truncated)
    holdout_emb = project_embeddings(raw_holdout_emb, pca_truncated)

    dev_train, dev_train_emb, dev_val, dev_val_emb = _split_dev_train_val(
        train_data, train_emb,
    )

    samples_per_arm = len(dev_train) / len(models)
    samples_per_feature = len(dev_train) / (len(models) * feat_dim)

    logger.info(
        f"    feat_dim={feat_dim}  var={var_explained:.1%}  "
        f"s/arm={samples_per_arm:.0f}  s/feat={samples_per_feature:.1f}  "
        f"train={len(dev_train)}  val={len(dev_val)}"
    )

    tmp_priors = output_dir / f"_tmp_priors_{n_comp}comp.joblib"
    joblib.dump(priors_truncated, tmp_priors)

    total = len(ALPHA_VALUES) * len(NEFF_VALUES) * len(GAMMA_VALUES)
    results: List[Dict[str, Any]] = []
    idx = 0

    for gamma in GAMMA_VALUES:
        for n_eff in NEFF_VALUES:
            for alpha in ALPHA_VALUES:
                idx += 1
                res = train_and_evaluate_pareto(
                    models, catalog,
                    dev_train, dev_val, dev_train_emb, dev_val_emb,
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

    tmp_priors.unlink(missing_ok=True)

    ranked = sorted(results, key=lambda r: r["pareto_auc"], reverse=True)
    best = ranked[0]

    logger.info(
        f"    BEST: a={best['alpha']} neff={int(best['n_eff'])} "
        f"g={best['gamma']} -> AUC={best['pareto_auc']:.4f} "
        f"R@0={best['lam0_reward']:.4f}"
    )

    # Evaluate selected config on holdout.
    holdout_res = train_and_evaluate_pareto(
        models, catalog,
        dev_train, holdout_data, dev_train_emb, holdout_emb,
        str(joblib.dump(priors_truncated,
                        output_dir / f"_tmp_priors_{n_comp}comp_hld.joblib")[0]),
        costs,
        alpha=float(best["alpha"]),
        n_eff=float(best["n_eff"]),
        gamma=float(best["gamma"]),
        lambda_values=LAMBDA_SWEEP,
        cost_lo=cost_lo, cost_hi=cost_hi,
        n_seeds=N_SEEDS,
    )
    (output_dir / f"_tmp_priors_{n_comp}comp_hld.joblib").unlink(missing_ok=True)

    logger.info(
        f"    Holdout: AUC={holdout_res['pareto_auc']:.4f} "
        f"R@0={holdout_res['lam0_reward']:.4f}"
    )

    return {
        "n_components": n_comp,
        "feature_dim": feat_dim,
        "variance_explained": var_explained,
        "samples_per_arm": samples_per_arm,
        "samples_per_feature_ratio": samples_per_feature,
        "best_alpha": best["alpha"],
        "best_n_eff": best["n_eff"],
        "best_gamma": best["gamma"],
        "dev_val_pareto_auc": best["pareto_auc"],
        "dev_val_lam0_reward": best["lam0_reward"],
        "holdout_pareto_auc": holdout_res["pareto_auc"],
        "holdout_lam0_reward": holdout_res["lam0_reward"],
        "dev_holdout_auc_gap": best["pareto_auc"] - holdout_res["pareto_auc"],
        "dev_holdout_r0_gap": best["lam0_reward"] - holdout_res["lam0_reward"],
        "top5": [
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
    k2_results: List[Dict],
    k3_results: List[Dict],
    ucb1_k2: Dict,
    ucb1_k3: Dict,
    out_path: Path,
) -> None:
    """Generate a summary plot of Pareto AUC vs component count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for ax, results, ucb1, k_label in [
        (axes[0], k2_results, ucb1_k2, "K=2"),
        (axes[1], k3_results, ucb1_k3, "K=3"),
    ]:
        comps = [r["n_components"] for r in results]
        dev_auc = [r["dev_val_pareto_auc"] for r in results]
        hld_auc = [r["holdout_pareto_auc"] for r in results]

        ax.plot(comps, dev_auc, "o-", color="C0", label="Dev-val Pareto AUC")
        ax.plot(comps, hld_auc, "s--", color="C1", label="Holdout Pareto AUC")
        ax.axhline(
            ucb1["reward"], ls=":", color="gray", alpha=0.7,
            label=f"UCB1 baseline ({ucb1['reward']:.4f})",
        )
        ax.axvline(15, ls="--", color="C3", alpha=0.5, label="d=15 (production)")

        ax2 = ax.twinx()
        var_exp = [r["variance_explained"] for r in results]
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
        ax.set_title(f"{k_label} Portfolio", fontsize=12, fontweight="bold")
        ax.set_xticks(comps)
        ax.legend(loc="lower right", fontsize=8)
        ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "PCA Component-Count Ablation\n"
        "Each point uses its own optimal (alpha, n_eff, gamma) from a full grid sweep",
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

    # ── Extract 32-comp portfolio priors ──────────────────────────────
    logger.info("Loading 43-model 32-component warmup priors ...")
    priors_43 = joblib.load(MULTIMODEL_PRIORS_32_PATH)
    logger.info(
        f"  Source: {MULTIMODEL_PRIORS_32_PATH.name} "
        f"({len(priors_43['models'])} models, "
        f"context_dim={priors_43['context_dim']})"
    )

    k2_priors_32 = extract_portfolio_priors(priors_43, K2_MODELS)
    k3_priors_32 = extract_portfolio_priors(priors_43, K3_MODELS)
    logger.info(
        f"  Extracted K=2 priors ({len(k2_priors_32['models'])} models) "
        f"and K=3 priors ({len(k3_priors_32['models'])} models)"
    )

    # ── K=2 data ──────────────────────────────────────────────────────
    logger.info("\nLoading K=2 data ...")
    dev_k2 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K2_MODELS)
    holdout_k2 = load_rewards_from_file(HOLDOUT_DATA_PATH_ALL_MODELS, K2_MODELS)
    logger.info(f"  Dev: {len(dev_k2)}  Holdout: {len(holdout_k2)}")

    costs_k2 = {
        m: req_cost(K2_CATALOG[m]["input_cost_per_m"],
                    K2_CATALOG[m]["output_cost_per_m"])
        for m in K2_MODELS
    }
    cost_lo_k2, cost_hi_k2 = min(costs_k2.values()), max(costs_k2.values())

    # ── K=3 data ──────────────────────────────────────────────────────
    logger.info("\nLoading K=3 data ...")
    prior_train_prompts: Set[str] = set()
    if THREE_WAY_SPLITS_PATH.exists():
        with open(THREE_WAY_SPLITS_PATH) as f:
            splits_3way = json.load(f)
        prior_train_prompts = set(splits_3way.get("prior_train_pool", []))
        logger.info(
            f"  Excluding {len(prior_train_prompts)} prior-train prompts"
        )

    all_dev_k3 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K3_MODELS)
    dev_k3 = [d for d in all_dev_k3 if d["prompt"] not in prior_train_prompts]
    holdout_k3 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K3_MODELS,
    )
    logger.info(f"  Dev (excl. prior-train): {len(dev_k3)}  Holdout: {len(holdout_k3)}")

    costs_k3 = {
        m: req_cost(K3_CATALOG[m]["input_cost_per_m"],
                    K3_CATALOG[m]["output_cost_per_m"])
        for m in K3_MODELS
    }
    cost_lo_k3, cost_hi_k3 = min(costs_k3.values()), max(costs_k3.values())

    # ── Look up raw embeddings from cache ────────────────────────────
    logger.info("\nLoading raw embeddings from cache ...")
    raw_dev_k2 = get_raw_embeddings(dev_k2, raw_cache)
    raw_holdout_k2 = get_raw_embeddings(holdout_k2, raw_cache)
    raw_dev_k3 = get_raw_embeddings(dev_k3, raw_cache)
    raw_holdout_k3 = get_raw_embeddings(holdout_k3, raw_cache)
    logger.info(
        f"  Raw embedding dim: {raw_dev_k2.shape[1]}"
    )

    # ── UCB1 baselines ────────────────────────────────────────────────
    logger.info("\nComputing UCB1 baselines ...")
    dev_train_k2_data, _, _, _ = _split_dev_train_val(
        dev_k2, [np.zeros(1)] * len(dev_k2),
    )
    ucb1_k2 = ucb1_online_route(
        dev_train_k2_data, holdout_k2, K2_MODELS, costs_k2,
        n_trials=N_SEEDS,
    )
    logger.info(
        f"  UCB1 K=2: R={ucb1_k2['reward']:.4f} "
        f"(greedy: {ucb1_k2['greedy_arm']})"
    )

    dev_train_k3_data, _, _, _ = _split_dev_train_val(
        dev_k3, [np.zeros(1)] * len(dev_k3),
    )
    ucb1_k3 = ucb1_online_route(
        dev_train_k3_data, holdout_k3, K3_MODELS, costs_k3,
        n_trials=N_SEEDS,
    )
    logger.info(
        f"  UCB1 K=3: R={ucb1_k3['reward']:.4f} "
        f"(greedy: {ucb1_k3['greedy_arm']})"
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
        f"  Per component count per portfolio: "
        f"{n_hparam * n_lam * N_SEEDS:,} trials"
    )
    logger.info(
        f"  Total: {len(COMPONENT_COUNTS) * 2 * n_hparam * n_lam * N_SEEDS:,} "
        f"trials"
    )
    logger.info(f"{'='*70}")

    # ── Sweep component counts ────────────────────────────────────────
    k2_results: List[Dict] = []
    k3_results: List[Dict] = []

    for n_comp in COMPONENT_COUNTS:
        pca_trunc = truncate_pca(pca32, n_comp)
        k2_priors_trunc = truncate_warmup_priors(k2_priors_32, n_comp)
        k3_priors_trunc = truncate_warmup_priors(k3_priors_32, n_comp)

        # ── K=2 ──
        logger.info(f"\n{'='*60}")
        logger.info(f"  K=2, n_components={n_comp}")
        logger.info(f"{'='*60}")
        res_k2 = run_ablation_for_n_components(
            n_comp,
            pca_truncated=pca_trunc,
            priors_truncated=k2_priors_trunc,
            models=K2_MODELS,
            catalog=K2_CATALOG,
            train_data=dev_k2,
            holdout_data=holdout_k2,
            raw_train_emb=raw_dev_k2,
            raw_holdout_emb=raw_holdout_k2,
            costs=costs_k2,
            cost_lo=cost_lo_k2,
            cost_hi=cost_hi_k2,
            output_dir=output_dir,
        )
        k2_results.append(res_k2)

        # ── K=3 ──
        logger.info(f"\n{'='*60}")
        logger.info(f"  K=3, n_components={n_comp}")
        logger.info(f"{'='*60}")
        res_k3 = run_ablation_for_n_components(
            n_comp,
            pca_truncated=pca_trunc,
            priors_truncated=k3_priors_trunc,
            models=K3_MODELS,
            catalog=K3_CATALOG,
            train_data=dev_k3,
            holdout_data=holdout_k3,
            raw_train_emb=raw_dev_k3,
            raw_holdout_emb=raw_holdout_k3,
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
                "hyperparameter re-tuning.  Selection criterion: "
                "Pareto AUC.  Portfolios: K=2 and K=3."
            ),
            "component_counts": COMPONENT_COUNTS,
            "alpha_values": ALPHA_VALUES,
            "neff_values": NEFF_VALUES,
            "gamma_values": GAMMA_VALUES,
            "lambda_sweep": LAMBDA_SWEEP,
            "n_seeds": N_SEEDS,
            "n_hparam_configs": n_hparam,
            "selection_criterion": "pareto_auc",
            "elapsed_seconds": elapsed,
        },
        "K2": {
            "models": K2_MODELS,
            "n_dev": len(dev_k2),
            "n_holdout": len(holdout_k2),
            "cost_range": [cost_lo_k2, cost_hi_k2],
            "ucb1_baseline": ucb1_k2,
            "ablation_results": k2_results,
        },
        "K3": {
            "models": K3_MODELS,
            "n_dev": len(dev_k3),
            "n_holdout": len(holdout_k3),
            "cost_range": [cost_lo_k3, cost_hi_k3],
            "ucb1_baseline": ucb1_k3,
            "ablation_results": k3_results,
        },
    }

    out_path = output_dir / "pca_component_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_ablation(
        k2_results, k3_results, ucb1_k2, ucb1_k3,
        output_dir / "pca_component_ablation.png",
    )

    # ── Summary table ─────────────────────────────────────────────────
    logger.info(f"\n{'='*90}")
    logger.info("SUMMARY")
    logger.info(f"{'='*90}")

    for k_label, results, ucb1 in [
        ("K=2", k2_results, ucb1_k2),
        ("K=3", k3_results, ucb1_k3),
    ]:
        logger.info(f"\n  {k_label}:")
        logger.info(
            f"  {'Comp':>5s}  {'Dim':>4s}  {'Var%':>6s}  "
            f"{'s/feat':>6s}  {'AUC_val':>8s}  {'AUC_hld':>8s}  "
            f"{'R@0_hld':>8s}  {'Gap':>7s}  "
            f"{'best_a':>6s}  {'best_n':>6s}  {'best_g':>7s}"
        )
        logger.info("  " + "-" * 88)
        for r in results:
            logger.info(
                f"  {r['n_components']:5d}  {r['feature_dim']:4d}  "
                f"{r['variance_explained']:5.1%}  "
                f"{r['samples_per_feature_ratio']:6.1f}  "
                f"{r['dev_val_pareto_auc']:8.4f}  "
                f"{r['holdout_pareto_auc']:8.4f}  "
                f"{r['holdout_lam0_reward']:8.4f}  "
                f"{r['dev_holdout_auc_gap']:+7.4f}  "
                f"{r['best_alpha']:6.2f}  "
                f"{int(r['best_n_eff']):6d}  "
                f"{r['best_gamma']:7.4f}"
            )
        logger.info(f"  UCB1 baseline: {ucb1['reward']:.4f}")

    logger.info(f"\nElapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
