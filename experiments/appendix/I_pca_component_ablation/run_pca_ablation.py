#!/usr/bin/env python3
"""
Appendix I: Joint PCA-Dimension × Hyperparameter Ablation
==========================================================

Single source of truth for (PCA dimension × hyperparameter) joint
optimization across all four routing variants:

  * **K=3 BanditGPT** — full portfolio with corralling + warmup priors
  * **K=3 Tabula Rasa** — full portfolio, no priors, no corralling
  * **K=2 BanditGPT** — cheapest + most expensive model, corralling +
    warmup priors (derived from K=3)
  * **K=2 Tabula Rasa** — cheapest + most expensive, no priors, no
    corralling

BanditGPT bundles warmup priors *and* the Corralling meta-learner.
The marginal contribution of each component is disentangled in the
Architectural Ablation (Appendix C2, ``C2_architectural_ablation.tex``),
which evaluates a 7-method progressive factorial including LinUCB with
and without priors.  Key finding: after full burn-in, priors contribute
zero delta; Corralling's value appears at moderate-to-high λ.

Note on forgetting factor (γ < 1) in offline single-pass training:
in the train-then-freeze protocol each sample is seen once in random
order, so there is no genuine non-stationarity.  When the grid search
selects γ < 1, it is acting as implicit regularization (reducing
effective sample size ≈ 1 / (1 − γ)) rather than adapting to temporal
drift.  The sweep is retained because it is a valid regularisation
knob, but results should be interpreted accordingly.

For each variant and component count k ∈ {4, 6, 8, 10, 12, 15, 20, 24, 32}:

  1. Truncate the production 32-component PCA to k dimensions.
  2. (BanditGPT only) Truncate the warmup priors to k dims (keep first
     k PCA dims + bias row/col of A and b).
  3. Encode all prompts once (raw sentence embeddings), then project
     through the truncated PCA + whitening + bias.
  4. Train on the canonical train split; sweep (α × n_eff × γ) with a
     λ sweep at each, selecting the configuration with the highest
     Pareto AUC on **val_tune**.
  5. Re-evaluate only the selected configuration on **val_report** to
     obtain the plotted ablation metric.
  6. At the optimal d* per variant, generate heatmap visualizations and
     save ``best_hparams_<variant>.json`` for downstream experiments.

Protocol
--------
The canonical validation split is sub-split into two disjoint halves —
**val_tune** (hyperparameter selection) and **val_report** (unbiased
metric for the ablation curve).  This eliminates the maximization bias
that arises when the grid-search winner is selected on and reported
from the same data.  The holdout set is never touched — it is reserved
for final paper claims.

Outputs (``results/``)
    pca_component_ablation.json         — full results for all variants
    pca_component_ablation.png          — summary ablation curves
    heatmap_<variant>_d<N>.png          — heatmaps at optimal d* per variant
    best_hparams_<variant>.json         — optimal hyperparameters per variant

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
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    WARMUP_PRIORS_PATH,
)
from utils.embeddings import (
    get_raw_embeddings_for_data,
    load_raw_embedding_cache,
    project_embeddings,
)
from utils.rewards import extract_reward
from utils.router_factory import create_experiment_router
from utils.model_pricing import load_model_catalog, req_cost
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

PRIMARY_JUDGE: str = "deepseek/deepseek-r1"

VAL_REPORT_FRACTION: float = 0.5
VAL_SPLIT_SEED: int = 2026



# ============================================================================
# Portfolio
# ============================================================================

K3_MODELS, K3_CATALOG = load_model_catalog(K3_MODELS_PATH)


def _derive_k2_portfolio(
    models: List[str],
    catalog: Dict[str, Dict],
) -> Tuple[List[str], Dict[str, Dict]]:
    """Derive K=2 portfolio (cheapest + most expensive) from K=3.

    Selects the two extreme-cost models from the K=3 portfolio,
    preserving their catalog metadata.  Cost is computed via
    ``req_cost`` (representative per-request cost).
    """
    model_costs = {
        m: req_cost(catalog[m]["input_cost_per_m"],
                    catalog[m]["output_cost_per_m"])
        for m in models
    }
    cheapest = min(model_costs, key=model_costs.get)  # type: ignore[arg-type]
    priciest = max(model_costs, key=model_costs.get)  # type: ignore[arg-type]
    k2_models = [cheapest, priciest]
    k2_catalog = {m: catalog[m] for m in k2_models}
    return k2_models, k2_catalog


K2_MODELS, K2_CATALOG = _derive_k2_portfolio(K3_MODELS, K3_CATALOG)


def _subset_warmup_priors(
    priors: Dict[str, Any],
    model_subset: List[str],
) -> Dict[str, Any]:
    """Subset warmup priors to a smaller set of models.

    Keeps the same context dimensionality and PCA metadata; only
    filters the per-model sufficient statistics (A and b matrices)
    to the requested subset.

    Args:
        priors: Full K=3 warmup priors dict with ``"A"``, ``"b"``,
            ``"models"`` keys.
        model_subset: Model IDs to retain (must be a subset of
            ``priors["models"]``).

    Returns:
        New priors dict with only *model_subset* entries.

    Raises:
        KeyError: If any model in *model_subset* is missing from priors.
    """
    missing = set(model_subset) - set(priors["models"])
    if missing:
        raise KeyError(
            f"Models not found in warmup priors: {missing}. "
            f"Available: {priors['models']}"
        )
    subset = dict(priors)
    subset["models"] = list(model_subset)
    subset["A"] = {m: priors["A"][m] for m in model_subset}
    subset["b"] = {m: priors["b"][m] for m in model_subset}
    return subset


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
    judge_id: Optional[str] = PRIMARY_JUDGE,
) -> List[Dict]:
    """Load rewards for specific models from gzipped JSONL.

    Only prompts with rewards for *all* requested models are included.

    Args:
        data_path: Gzipped JSONL reward file.
        models: Model IDs to load.
        prompt_filter: If provided, only include these prompts.
        judge_id: When set, extract reward from this single judge
            rather than the full panel mean.  Defaults to
            :data:`PRIMARY_JUDGE`.
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
            rewards[prompt][model_id] = extract_reward(entry, judge_id=judge_id)

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


def resize_warmup_priors(
    priors: Dict[str, Any],
    n_components: int,
) -> Dict[str, Any]:
    """Resize sufficient statistics to match the target PCA dimension.

    The priors contain A (d x d) and b (d,) per model where
    d = pca_components + 1 (bias).

    When ``n_components`` is smaller than the priors' dimension, the
    matrices are truncated (keeping the first *n_components* PCA dims
    plus the bias row/col).  When larger, the matrices are padded:
    extra PCA dimensions get identity rows/cols in A (uninformed prior)
    and zeros in b, preserving the learned structure for the original
    dimensions.
    """
    old_pca = priors["context_dim"] - 1
    if n_components == old_pca:
        return priors

    old_dim = priors["context_dim"]
    new_dim = n_components + 1

    new_A: Dict[str, np.ndarray] = {}
    new_b: Dict[str, np.ndarray] = {}

    if n_components < old_pca:
        keep_idx = list(range(n_components)) + [old_dim - 1]
        for m in priors["models"]:
            A_full = priors["A"][m]
            b_full = priors["b"][m]
            new_A[m] = A_full[np.ix_(keep_idx, keep_idx)]
            new_b[m] = b_full[keep_idx]
    else:
        for m in priors["models"]:
            A_old = priors["A"][m]
            b_old = priors["b"][m]
            A_new = np.eye(new_dim, dtype=np.float64)
            b_new = np.zeros(new_dim, dtype=np.float64)
            A_new[:old_pca, :old_pca] = A_old[:old_pca, :old_pca]
            b_new[:old_pca] = b_old[:old_pca]
            A_new[-1, :old_pca] = A_old[-1, :old_pca]
            A_new[:old_pca, -1] = A_old[:old_pca, -1]
            A_new[-1, -1] = A_old[-1, -1]
            b_new[-1] = b_old[-1]
            new_A[m] = A_new
            new_b[m] = b_new

    resized = dict(priors)
    resized["A"] = new_A
    resized["b"] = new_b
    resized["context_dim"] = new_dim
    resized["pca_components"] = n_components
    return resized


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
    use_corralling: bool = True,
    return_per_prompt: bool = False,
) -> Dict[str, Any]:
    """Train-then-freeze for one (alpha, n_eff, gamma, lambda) point.

    Args:
        return_per_prompt: When ``True``, also return per-seed,
            per-prompt rewards and costs as 2-D arrays
            (shape ``[n_seeds, n_eval]``).  Used for eval-data
            bootstrap CIs.

    Returns:
        Dict with ``mean_reward``, ``mean_cost``, and per-seed arrays
        ``seed_rewards`` and ``seed_costs`` (each length *n_seeds*).
        When *return_per_prompt* is ``True``, also includes
        ``prompt_rewards`` and ``prompt_costs`` arrays.
    """
    r_min = REWARD_THEORETICAL_MIN
    r_range = REWARD_THEORETICAL_MAX - REWARD_THEORETICAL_MIN
    burn_in = len(train_data)

    trial_r: List[float] = []
    trial_c: List[float] = []
    prompt_rewards: List[List[float]] = [] if return_per_prompt else None  # type: ignore[assignment]
    prompt_costs: List[List[float]] = [] if return_per_prompt else None  # type: ignore[assignment]

    for trial in range(n_seeds):
        # Global seed is intentional: BanditRouter internals (tiebreaking,
        # corralling weight sampling) draw from the global numpy RNG, so
        # a local RandomState would only control the permutation while
        # leaving router behaviour non-deterministic.  Re-seeding at the
        # top of every trial ensures full reproducibility.
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=registry,
            feature_dim=dim,
            prior_n_effective=n_eff,
            alpha=alpha,
            warmup_path=warmup_path,
            use_corralling=use_corralling,
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

        r_total = c_total = 0.0
        seed_pr: List[float] = []
        seed_pc: List[float] = []
        with router.exploit():
            for p, x in zip(eval_data, eval_emb):
                model, _log = router.route(x, total_steps=burn_in)
                r = p["rewards"][model]
                c = costs[model]
                r_total += r
                c_total += c
                if return_per_prompt:
                    seed_pr.append(r)
                    seed_pc.append(c)

        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)
        if return_per_prompt:
            prompt_rewards.append(seed_pr)
            prompt_costs.append(seed_pc)

    result: Dict[str, Any] = {
        "mean_reward": float(np.mean(trial_r)),
        "mean_cost": float(np.mean(trial_c)),
        "seed_rewards": trial_r,
        "seed_costs": trial_c,
    }
    if return_per_prompt:
        result["prompt_rewards"] = np.asarray(prompt_rewards)
        result["prompt_costs"] = np.asarray(prompt_costs)
    return result


N_BOOTSTRAP: int = 1000
BOOTSTRAP_SEED: int = 9999


def _bootstrap_pareto_auc(
    prompt_rewards_by_lam: List[np.ndarray],
    prompt_costs_by_lam: List[np.ndarray],
    cost_lo: float,
    cost_hi: float,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Compute bootstrap CI for Pareto AUC over eval-data resamples.

    For each bootstrap iteration, resample eval-prompt indices with
    replacement and recompute the seed-averaged Pareto AUC.  This
    captures **data-sampling uncertainty** (sensitivity to which prompts
    appear in the eval set) as opposed to the per-seed SE which captures
    training stochasticity only.

    Args:
        prompt_rewards_by_lam: List (one per lambda) of arrays with
            shape ``[n_seeds, n_eval]``.
        prompt_costs_by_lam: Same layout for costs.
        cost_lo: Min cost for Pareto AUC integration.
        cost_hi: Max cost for Pareto AUC integration.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with ``boot_mean``, ``boot_se``, ``boot_ci_lo``,
        ``boot_ci_hi`` (95% percentile interval).
    """
    rng = np.random.RandomState(seed)
    n_eval = prompt_rewards_by_lam[0].shape[1]
    n_lam = len(prompt_rewards_by_lam)

    boot_aucs: List[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n_eval, size=n_eval, replace=True)
        # Seed-averaged (cost, reward) for each lambda on this bootstrap sample
        lam_costs: List[float] = []
        lam_rewards: List[float] = []
        for l in range(n_lam):
            lam_rewards.append(float(prompt_rewards_by_lam[l][:, idx].mean()))
            lam_costs.append(float(prompt_costs_by_lam[l][:, idx].mean()))
        boot_aucs.append(pareto_auc(lam_costs, lam_rewards, cost_lo, cost_hi))

    arr = np.asarray(boot_aucs)
    return {
        "boot_mean": float(arr.mean()),
        "boot_se": float(arr.std(ddof=1)),
        "boot_ci_lo": float(np.percentile(arr, 2.5)),
        "boot_ci_hi": float(np.percentile(arr, 97.5)),
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
    use_corralling: bool = True,
    bootstrap: bool = False,
) -> Dict[str, Any]:
    """Evaluate one (alpha, n_eff, gamma) across multiple lambdas.

    Sweeps ``lambda_values`` to trace the Pareto frontier, then
    computes Pareto AUC over ``[cost_lo, cost_hi]``.  Also computes
    per-seed Pareto AUC to enable confidence intervals.

    When *bootstrap* is ``True``, additionally computes a bootstrap CI
    over eval-data resamples (captures data-sampling uncertainty).
    """
    dim = train_emb[0].shape[0]
    registry = build_model_registry(models, catalog)

    sweep_points: List[Dict[str, Any]] = []
    seed_sweep_rewards: List[List[float]] = []
    seed_sweep_costs: List[List[float]] = []
    prompt_rewards_by_lam: List[np.ndarray] = []
    prompt_costs_by_lam: List[np.ndarray] = []
    lam0_reward: float = 0.0

    for lam in lambda_values:
        pt = _train_and_eval_single_lambda(
            models, registry,
            train_data, eval_data, train_emb, eval_emb,
            warmup_path, costs, dim,
            alpha=alpha, n_eff=n_eff, gamma=gamma,
            cost_penalty=lam, n_seeds=n_seeds,
            use_corralling=use_corralling,
            return_per_prompt=bootstrap,
        )
        sweep_points.append({
            "lambda": lam,
            "mean_reward": pt["mean_reward"],
            "mean_cost": pt["mean_cost"],
        })
        seed_sweep_rewards.append(pt["seed_rewards"])
        seed_sweep_costs.append(pt["seed_costs"])
        if bootstrap:
            prompt_rewards_by_lam.append(pt["prompt_rewards"])
            prompt_costs_by_lam.append(pt["prompt_costs"])
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

    result: Dict[str, Any] = {
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

    if bootstrap and prompt_rewards_by_lam:
        result["bootstrap"] = _bootstrap_pareto_auc(
            prompt_rewards_by_lam, prompt_costs_by_lam,
            cost_lo, cost_hi,
        )

    return result


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


def portfolio_pareto_auc(
    eval_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
    cost_lo: float,
    cost_hi: float,
) -> Dict[str, Any]:
    """Non-contextual portfolio Pareto AUC baseline.

    Each model is a single (cost, mean_reward) point.  The Pareto hull
    of these K points represents the best any *static* routing policy
    can achieve — it defines an upper envelope over fixed model
    assignments with no per-prompt context.

    This is the correct Pareto AUC reference for the bandit: the bandit
    must exceed this frontier to demonstrate that contextual routing
    adds value beyond simply picking a model at each cost level.

    **Conservative comparison:** per-model means are computed on
    *eval_data* (oracle access to the true eval-set rewards).  The
    bandit, trained on *train_data*, must beat this oracle frontier.
    Reported deltas therefore **underestimate** the practical routing
    gain a deployed bandit would achieve relative to a static policy
    whose quality estimates come from a separate (potentially noisier)
    data source.

    Args:
        eval_data: Evaluation prompts with per-model rewards.
        models: Portfolio model IDs.
        costs: Per-model representative cost.
        cost_lo: Lower bound of integration range.
        cost_hi: Upper bound of integration range.

    Returns:
        Dict with ``pareto_auc``, per-model ``(cost, reward)`` points,
        and hull points.
    """
    model_points: Dict[str, Dict[str, float]] = {}
    point_costs: List[float] = []
    point_rewards: List[float] = []

    for m in models:
        r = float(np.mean([d["rewards"][m] for d in eval_data]))
        c = costs[m]
        model_points[m] = {"cost": c, "reward": r}
        point_costs.append(c)
        point_rewards.append(r)

    from utils.pareto import pareto_hull
    hull_c, hull_r = pareto_hull(point_costs, point_rewards)
    auc = pareto_auc(point_costs, point_rewards, cost_lo, cost_hi)

    return {
        "pareto_auc": auc,
        "model_points": model_points,
        "hull_costs": hull_c,
        "hull_rewards": hull_r,
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
    warmup_priors_truncated: Optional[Dict[str, Any]],
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
    use_corralling: bool = True,
    variant_label: str = "BanditGPT",
) -> Dict[str, Any]:
    """Run the full hparam grid for one component count.

    Trains the bandit on the canonical train split.  The hyperparameter
    grid is evaluated on **val_tune**; the selected configuration is
    then re-evaluated on **val_report** to produce an unbiased metric
    free of maximization bias.  The holdout set is never touched.

    **Selection noise:** With 80 grid configurations and a halved val
    set (~350 prompts), the top configs may be within noise of each
    other.  The tune/report split ensures the *reported* metric is
    unbiased regardless of selection noise.  The ``top5_tune`` field
    in the output records the five best configs so downstream analysis
    can assess how tightly clustered they are.  If adjacent component
    counts produce overlapping bootstrap CIs, the system is robust to
    PCA dimension choice in that range.

    Args:
        n_comp: Number of PCA components.
        pca_truncated: PCA model truncated to *n_comp*.
        warmup_priors_truncated: Warmup priors truncated to *n_comp*
            dims, or ``None`` for Tabula Rasa (no priors).
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
        use_corralling: Whether to enable Corralling meta-learner.
        variant_label: Human-readable label for logging.

    Returns:
        Dict with best config, Pareto AUC on both val_tune and
        val_report, variance explained, the full grid results, etc.
    """
    feat_dim = n_comp + 1
    var_explained = float(np.sum(pca_truncated.explained_variance_ratio_))

    train_emb = project_embeddings(raw_train_emb, pca_truncated)
    tune_emb = project_embeddings(raw_val_tune_emb, pca_truncated)
    report_emb = project_embeddings(raw_val_report_emb, pca_truncated)

    samples_per_arm = len(train_data) / len(models)
    samples_per_feature = len(train_data) / (len(models) * feat_dim)

    logger.info(
        f"    [{variant_label}] feat_dim={feat_dim}  "
        f"var={var_explained:.1%}  "
        f"s/arm={samples_per_arm:.0f}  s/feat={samples_per_feature:.1f}  "
        f"train={len(train_data)}  "
        f"val_tune={len(val_tune_data)}  val_report={len(val_report_data)}"
    )

    warmup_path: Optional[str] = None
    if warmup_priors_truncated is not None:
        safe_label = variant_label.replace(" ", "_").replace("=", "")
        tmp_priors = output_dir / f"_tmp_priors_{safe_label}_{n_comp}comp.joblib"
        joblib.dump(warmup_priors_truncated, tmp_priors)
        warmup_path = str(tmp_priors)

    # ── Grid sweep on val_tune ────────────────────────────────────────
    total = len(ALPHA_VALUES) * len(NEFF_VALUES) * len(GAMMA_VALUES)
    grid_results: List[Dict[str, Any]] = []
    idx = 0

    for gamma in GAMMA_VALUES:
        for n_eff in NEFF_VALUES:
            for alpha in ALPHA_VALUES:
                idx += 1
                res = train_and_evaluate_pareto(
                    models, catalog,
                    train_data, val_tune_data, train_emb, tune_emb,
                    warmup_path, costs,
                    alpha=alpha, n_eff=n_eff, gamma=gamma,
                    lambda_values=LAMBDA_SWEEP,
                    cost_lo=cost_lo, cost_hi=cost_hi,
                    n_seeds=N_SEEDS,
                    use_corralling=use_corralling,
                )
                grid_results.append(res)
                if idx % 20 == 0 or idx == total:
                    logger.info(
                        f"      [{idx:3d}/{total}] "
                        f"a={alpha:<4} neff={n_eff:<6} g={gamma:<6} "
                        f"AUC={res['pareto_auc']:.4f} "
                        f"R@0={res['lam0_reward']:.4f}"
                    )

    ranked = sorted(grid_results, key=lambda r: r["pareto_auc"], reverse=True)
    best_tune = ranked[0]

    logger.info(
        f"    BEST (tune): a={best_tune['alpha']} "
        f"neff={int(best_tune['n_eff'])} g={best_tune['gamma']} "
        f"-> AUC={best_tune['pareto_auc']:.4f} "
        f"R@0={best_tune['lam0_reward']:.4f}"
    )

    # ── Re-evaluate selected config on val_report ─────────────────────
    logger.info("    Re-evaluating selected config on val_report (+ bootstrap) ...")
    best_report = train_and_evaluate_pareto(
        models, catalog,
        train_data, val_report_data, train_emb, report_emb,
        warmup_path, costs,
        alpha=best_tune["alpha"],
        n_eff=best_tune["n_eff"],
        gamma=best_tune["gamma"],
        lambda_values=LAMBDA_SWEEP,
        cost_lo=cost_lo, cost_hi=cost_hi,
        n_seeds=N_SEEDS,
        use_corralling=use_corralling,
        bootstrap=True,
    )

    if warmup_path is not None:
        Path(warmup_path).unlink(missing_ok=True)

    boot = best_report.get("bootstrap", {})
    boot_ci_str = (
        f"  boot 95% CI=[{boot['boot_ci_lo']:.4f}, {boot['boot_ci_hi']:.4f}]"
        if boot else ""
    )
    logger.info(
        f"    REPORT: AUC={best_report['pareto_auc']:.4f} "
        f"± {best_report['pareto_auc_se']:.4f} (seed SE)"
        f"{boot_ci_str}  "
        f"R@0={best_report['lam0_reward']:.4f}"
    )

    out: Dict[str, Any] = {
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
        "grid_results": grid_results,
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
        "variant_label": variant_label,
        "use_corralling": use_corralling,
    }
    if "bootstrap" in best_report:
        out["report_bootstrap"] = best_report["bootstrap"]
    return out


# ============================================================================
# Plotting
# ============================================================================


VARIANT_STYLES: Dict[str, Dict[str, Any]] = {
    "k3_banditgpt": {"color": "C0", "marker": "o", "ls": "-"},
    "k3_tabula_rasa": {"color": "C0", "marker": "s", "ls": "--"},
    "k2_banditgpt": {"color": "C1", "marker": "D", "ls": "-"},
    "k2_tabula_rasa": {"color": "C1", "marker": "^", "ls": "--"},
}

VARIANT_DISPLAY: Dict[str, str] = {
    "k3_banditgpt": "K=3 BanditGPT",
    "k3_tabula_rasa": "K=3 Tabula Rasa",
    "k2_banditgpt": "K=2 BanditGPT",
    "k2_tabula_rasa": "K=2 Tabula Rasa",
}


def plot_ablation(
    all_results: Dict[str, List[Dict]],
    bsm: Dict[str, Dict],
    ppa: Dict[str, Dict],
    out_path: Path,
) -> None:
    """Plot contextual routing gain (delta AUC) vs component count.

    The y-axis is the **delta** between the bandit's Pareto AUC and
    the non-contextual portfolio Pareto AUC baseline.  A value of 0
    means the bandit adds no value over static model selection; positive
    values indicate genuine contextual routing benefit.

    The primary curve uses the **val_report** delta (unbiased); the
    val_tune delta is shown as a faded reference.  ``delta_tune`` and
    ``delta_report`` must already be present in each result dict (see
    post-processing in ``main()``).

    Args:
        all_results: ``{variant_key: [per-component result dicts]}``.
            Each dict must contain ``delta_report``, ``delta_tune``,
            and ``report_pareto_auc_se``.
        bsm: ``{variant_key: best-single-model dict}``.
        ppa: ``{variant_key: {"tune": {...}, "report": {...}}}``
            with ``"pareto_auc"`` in each.
        out_path: Output PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for vkey, results in all_results.items():
        style = VARIANT_STYLES.get(vkey, {"color": "gray", "marker": ".", "ls": "-"})
        label = VARIANT_DISPLAY.get(vkey, vkey)

        comps = [r["n_components"] for r in results]
        delta_report = [r["delta_report"] for r in results]
        delta_tune = [r["delta_tune"] for r in results]
        report_se = [r["report_pareto_auc_se"] for r in results]

        # Bootstrap CI over eval data (wider band, captures data-sampling uncertainty)
        has_boot = all("report_bootstrap" in r for r in results)
        if has_boot:
            ppa_report_auc = ppa[vkey]["report"]["pareto_auc"] if vkey in ppa else 0.0
            boot_lo = [r["report_bootstrap"]["boot_ci_lo"] - ppa_report_auc for r in results]
            boot_hi = [r["report_bootstrap"]["boot_ci_hi"] - ppa_report_auc for r in results]
            ax.fill_between(comps, boot_lo, boot_hi, alpha=0.06, color=style["color"])

        # Seed-level CI (narrower band, captures training stochasticity)
        lo = [d - 1.96 * s for d, s in zip(delta_report, report_se)]
        hi = [d + 1.96 * s for d, s in zip(delta_report, report_se)]
        ax.fill_between(comps, lo, hi, alpha=0.12, color=style["color"])

        ci_label = "seed ± boot 95% CI" if has_boot else "seed 95% CI"
        ax.plot(
            comps, delta_report,
            marker=style["marker"], ls=style["ls"], color=style["color"],
            label=f"{label} (report, {ci_label})",
        )
        ax.plot(
            comps, delta_tune,
            marker=style["marker"], ls=":", color=style["color"],
            alpha=0.25, markersize=3,
        )

        k_prefix = vkey.split("_")[0]
        if vkey in bsm and vkey in ppa:
            bsm_delta = bsm[vkey]["reward"] - ppa[vkey]["report"]["pareto_auc"]
            ax.axhline(
                bsm_delta, ls=":", color=style["color"],
                alpha=0.4, lw=1.0,
                label=f"{k_prefix.upper()} best-single-model delta",
            )

    ax.axhline(0, ls="-", color="black", alpha=0.3, lw=0.8)
    ax.axvline(15, ls="--", color="C3", alpha=0.5, label="d=15 (production)")

    ref_key = "k3_banditgpt" if "k3_banditgpt" in all_results else None
    if ref_key:
        ax2 = ax.twinx()
        ref_results = all_results[ref_key]
        comps = [r["n_components"] for r in ref_results]
        var_exp = [r["variance_explained"] for r in ref_results]
        ax2.fill_between(comps, 0, [v * 100 for v in var_exp],
                         alpha=0.08, color="green")
        ax2.plot(comps, [v * 100 for v in var_exp],
                 "^-", color="green", alpha=0.4, markersize=4,
                 label="Variance explained (%)")
        ax2.set_ylabel("Variance explained (%)", fontsize=10, color="green")
        ax2.set_ylim(0, 40)
        ax2.legend(loc="upper left", fontsize=8)

    ax.set_xlabel("PCA components", fontsize=11)
    ax.set_ylabel("Contextual Routing Gain (Pareto AUC delta)", fontsize=11)
    ax.set_xticks(COMPONENT_COUNTS)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Contextual Routing Gain vs PCA Dimension (All Variants)\n"
        r"$\Delta$ = bandit Pareto AUC $-$ portfolio Pareto AUC; "
        "selected on val-tune, evaluated on val-report",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_heatmap(
    grid_results: List[Dict[str, Any]],
    alpha_values: List[float],
    neff_values: List[float],
    gamma_values: List[float],
    best_config: Dict[str, Any],
    out_path: Path,
    *,
    variant_label: str,
    n_comp: int,
    portfolio_auc_tune: float = 0.0,
) -> None:
    """Generate a two-row heatmap figure (ported from Appendix H).

    Row 1: Pareto AUC **delta** (bandit - portfolio baseline).
    Row 2: Mean reward at lambda=0 (for reference).
    The global best cell (by Pareto AUC) is starred in both rows.

    Args:
        grid_results: Full grid output for one (variant, n_comp).
        alpha_values: Exploration coefficients (x-axis).
        neff_values: Prior effective sample sizes (y-axis).
        gamma_values: Forgetting factors — one panel per value.
        best_config: The globally selected configuration dict.
        out_path: Output PNG path.
        variant_label: Human-readable variant name for the title.
        n_comp: Number of PCA components (shown in title).
        portfolio_auc_tune: Portfolio Pareto AUC on val_tune, subtracted
            from each grid cell's AUC to show delta.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    lookup_delta: Dict[Tuple[float, float, float], float] = {}
    lookup_reward: Dict[Tuple[float, float, float], float] = {}
    for r in grid_results:
        key = (r["alpha"], r["n_eff"], r["gamma"])
        lookup_delta[key] = r["pareto_auc"] - portfolio_auc_tune
        lookup_reward[key] = r["lam0_reward"]

    n_panels = len(gamma_values)
    fig, axes = plt.subplots(
        2, n_panels, figsize=(4.5 * n_panels, 9),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    all_delta = list(lookup_delta.values())
    all_reward = [r["lam0_reward"] for r in grid_results]
    delta_abs_max = max(abs(min(all_delta)), abs(max(all_delta)), 0.001)
    norm_d = TwoSlopeNorm(vmin=-delta_abs_max, vcenter=0, vmax=delta_abs_max)
    norm_r = Normalize(
        vmin=min(all_reward) - 0.002, vmax=max(all_reward) + 0.002,
    )

    for row_idx, (lookup, norm, metric_label, fmt, cmap) in enumerate([
        (lookup_delta, norm_d, r"$\Delta$ Pareto AUC", "+.4f", "RdYlGn"),
        (lookup_reward, norm_r, r"Reward ($\lambda{=}0$)", ".3f", "YlOrRd"),
    ]):
        for col_idx, gamma in enumerate(gamma_values):
            ax = axes[row_idx, col_idx]
            grid = np.zeros((len(neff_values), len(alpha_values)))
            for i, n_eff in enumerate(neff_values):
                for j, alpha in enumerate(alpha_values):
                    grid[i, j] = lookup.get((alpha, n_eff, gamma), np.nan)

            ax.imshow(
                grid, aspect="auto", origin="lower", norm=norm,
                cmap=cmap,
            )
            ax.set_xticks(range(len(alpha_values)))
            ax.set_xticklabels([str(a) for a in alpha_values], fontsize=8)
            ax.set_yticks(range(len(neff_values)))
            ax.set_yticklabels(
                [str(int(n)) for n in neff_values], fontsize=8,
            )
            if row_idx == 1:
                ax.set_xlabel(r"$\alpha$ (exploration)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(
                    rf"$n_{{\mathrm{{eff}}}}$ — {metric_label}",
                    fontsize=9,
                )
            if row_idx == 0:
                gamma_label = (
                    f"{gamma}" if gamma < 1.0 else "1.0 (stationary)"
                )
                ax.set_title(
                    rf"$\gamma = {gamma_label}$",
                    fontsize=10, fontweight="bold",
                )

            for i in range(len(neff_values)):
                for j in range(len(alpha_values)):
                    val = grid[i, j]
                    is_best = (
                        alpha_values[j] == best_config["alpha"]
                        and neff_values[i] == best_config["n_eff"]
                        and gamma == best_config["gamma"]
                    )
                    text = f"{val:{fmt}}"
                    if is_best:
                        text += "\n\u2605"
                    midpoint = (norm.vmin + norm.vmax) / 2
                    color = (
                        "white"
                        if abs(val - midpoint) > (norm.vmax - norm.vmin) * 0.3
                        else "black"
                    )
                    ax.text(
                        j, i, text, ha="center", va="center",
                        fontsize=7,
                        fontweight="bold" if is_best else "normal",
                        color=color,
                    )

    best_delta = best_config["pareto_auc"] - portfolio_auc_tune
    fig.suptitle(
        rf"Hparam Sensitivity: $\alpha \times n_{{\mathrm{{eff}}}} "
        rf"\times \gamma$ — {variant_label}, d={n_comp}"
        f"\nBest: α={best_config['alpha']}, "
        f"n_eff={int(best_config['n_eff'])}, "
        f"γ={best_config['gamma']} → "
        rf"$\Delta$={best_delta:+.4f}",
        fontsize=11, fontweight="bold",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ============================================================================
# Main
# ============================================================================


def _find_best_d_star(results: List[Dict]) -> Dict:
    """Return the result dict at the d* maximizing tune_pareto_auc.

    The PCA component count is a hyperparameter, so it must be selected
    on the tuning split to avoid maximization bias on the report split.
    The report-split metrics for the selected d* are used only for
    unbiased evaluation.
    """
    return max(results, key=lambda r: r["tune_pareto_auc"])


def _save_best_hparams(
    result_at_dstar: Dict,
    variant_key: str,
    output_dir: Path,
    portfolio_auc_report: float,
) -> Path:
    """Write ``best_hparams_<variant>.json`` at the optimal d*."""
    payload = {
        "variant": variant_key,
        "n_components": result_at_dstar["n_components"],
        "alpha": result_at_dstar["best_alpha"],
        "n_eff": result_at_dstar["best_n_eff"],
        "gamma": result_at_dstar["best_gamma"],
        "report_pareto_auc": result_at_dstar["report_pareto_auc"],
        "report_pareto_auc_se": result_at_dstar["report_pareto_auc_se"],
        "portfolio_pareto_auc_report": portfolio_auc_report,
        "delta_report": result_at_dstar["delta_report"],
    }
    path = output_dir / f"best_hparams_{variant_key}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"  Saved {path}")
    return path


def _print_summary_table(
    variant_key: str,
    results: List[Dict],
    bsm: Dict,
    ppa: Dict,
) -> None:
    """Log a formatted summary table for one variant."""
    label = VARIANT_DISPLAY.get(variant_key, variant_key)
    ppa_report_auc = ppa["report"]["pareto_auc"]
    logger.info(f"\n{'='*120}")
    logger.info(f"SUMMARY  ({label})")
    logger.info(f"{'='*120}")
    logger.info(
        f"  {'Comp':>5s}  {'Dim':>4s}  {'Var%':>6s}  "
        f"{'s/feat':>6s}  {'AUC_rpt':>8s}  {'±SE':>6s}  "
        f"{'Δ_rpt':>8s}  {'Δ_tune':>8s}  "
        f"{'R@0_rpt':>8s}  "
        f"{'best_a':>6s}  {'best_n':>6s}  {'best_g':>7s}"
    )
    logger.info("  " + "-" * 110)
    for r in results:
        logger.info(
            f"  {r['n_components']:5d}  {r['feature_dim']:4d}  "
            f"{r['variance_explained']:5.1%}  "
            f"{r['samples_per_feature_ratio']:6.1f}  "
            f"{r['report_pareto_auc']:8.4f}  "
            f"{r['report_pareto_auc_se']:6.4f}  "
            f"{r['delta_report']:+8.4f}  "
            f"{r['delta_tune']:+8.4f}  "
            f"{r['report_lam0_reward']:8.4f}  "
            f"{r['best_alpha']:6.2f}  "
            f"{int(r['best_n_eff']):6d}  "
            f"{r['best_gamma']:7.4f}"
        )
    logger.info(f"  Portfolio Pareto AUC (report): {ppa_report_auc:.4f}")
    logger.info(f"  Best single model (val_report): {bsm['reward']:.4f}")
    logger.info(f"  Best-single-model delta: {bsm['reward'] - ppa_report_auc:+.4f}")


def _run_variant(
    variant_key: str,
    *,
    models: List[str],
    catalog: Dict[str, Dict],
    warmup_priors_32: Optional[Dict[str, Any]],
    use_corralling: bool,
    train_data: List[Dict],
    val_tune_data: List[Dict],
    val_report_data: List[Dict],
    raw_train_emb: np.ndarray,
    raw_val_tune_emb: np.ndarray,
    raw_val_report_emb: np.ndarray,
    costs: Dict[str, float],
    cost_lo: float,
    cost_hi: float,
    pca32: PCA,
    output_dir: Path,
) -> List[Dict]:
    """Sweep all component counts for one variant.

    Returns the list of per-component-count result dicts.
    """
    label = VARIANT_DISPLAY.get(variant_key, variant_key)
    results: List[Dict] = []

    for n_comp in COMPONENT_COUNTS:
        pca_trunc = truncate_pca(pca32, n_comp)
        priors_trunc: Optional[Dict[str, Any]] = None
        if warmup_priors_32 is not None:
            priors_trunc = resize_warmup_priors(warmup_priors_32, n_comp)

        logger.info(f"\n{'='*60}")
        logger.info(f"  {label}, n_components={n_comp}")
        logger.info(f"{'='*60}")

        res = run_ablation_for_n_components(
            n_comp,
            pca_truncated=pca_trunc,
            warmup_priors_truncated=priors_trunc,
            models=models,
            catalog=catalog,
            train_data=train_data,
            val_tune_data=val_tune_data,
            val_report_data=val_report_data,
            raw_train_emb=raw_train_emb,
            raw_val_tune_emb=raw_val_tune_emb,
            raw_val_report_emb=raw_val_report_emb,
            costs=costs,
            cost_lo=cost_lo,
            cost_hi=cost_hi,
            output_dir=output_dir,
            use_corralling=use_corralling,
            variant_label=label,
        )
        results.append(res)

    return results


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

    # ── Load warmup priors (canonical K=3, 32 components) ─────────────
    logger.info("Loading canonical K=3 warmup priors ...")
    k3_priors = joblib.load(WARMUP_PRIORS_PATH)
    assert k3_priors.get("pca_whitened", False), (
        "Priors must be in whitened PCA space to match project_embeddings(). "
        "Regenerate with: python scripts/generate_multimodel_warmup_priors.py"
    )
    logger.info(
        f"  Source: {WARMUP_PRIORS_PATH.name} "
        f"({len(k3_priors['models'])} models, "
        f"context_dim={k3_priors['context_dim']}, "
        f"pca_whitened={k3_priors.get('pca_whitened')})"
    )

    # ── Derive K=2 warmup priors from K=3 ───────────────────────────
    logger.info("Deriving K=2 warmup priors (cheapest + most expensive) ...")
    k2_priors = _subset_warmup_priors(k3_priors, K2_MODELS)
    logger.info(
        f"  K=2 models: {K2_MODELS} "
        f"(subset of {len(k3_priors['models'])} K=3 models)"
    )

    # ── Load K=3 data from canonical splits ───────────────────────────
    logger.info("\nLoading K=3 data (canonical train/val splits) ...")
    train_k3 = load_rewards_from_file(TRAIN_DATA_PATH, K3_MODELS)
    val_k3 = load_rewards_from_file(VAL_DATA_PATH, K3_MODELS)
    logger.info(f"  K=3 Train: {len(train_k3)}  Val: {len(val_k3)}")

    costs_k3 = {
        m: req_cost(K3_CATALOG[m]["input_cost_per_m"],
                    K3_CATALOG[m]["output_cost_per_m"])
        for m in K3_MODELS
    }
    cost_lo_k3, cost_hi_k3 = min(costs_k3.values()), max(costs_k3.values())

    # ── Load K=2 data (same canonical splits, filtered to 2 models) ──
    logger.info("Loading K=2 data (canonical train/val splits) ...")
    train_k2 = load_rewards_from_file(TRAIN_DATA_PATH, K2_MODELS)
    val_k2 = load_rewards_from_file(VAL_DATA_PATH, K2_MODELS)
    logger.info(f"  K=2 Train: {len(train_k2)}  Val: {len(val_k2)}")

    costs_k2 = {
        m: req_cost(K2_CATALOG[m]["input_cost_per_m"],
                    K2_CATALOG[m]["output_cost_per_m"])
        for m in K2_MODELS
    }
    cost_lo_k2, cost_hi_k2 = min(costs_k2.values()), max(costs_k2.values())

    # ── Look up raw embeddings from cache ────────────────────────────
    logger.info("\nLoading raw embeddings from cache ...")
    raw_train_k3 = get_raw_embeddings_for_data(train_k3, raw_cache)
    raw_val_k3 = get_raw_embeddings_for_data(val_k3, raw_cache)
    raw_train_k2 = get_raw_embeddings_for_data(train_k2, raw_cache)
    raw_val_k2 = get_raw_embeddings_for_data(val_k2, raw_cache)
    logger.info(f"  Raw embedding dim: {raw_train_k3.shape[1]}")

    # ── Split val into tune / report ─────────────────────────────────
    logger.info(
        f"\nSplitting val into tune/report "
        f"({1 - VAL_REPORT_FRACTION:.0%}/{VAL_REPORT_FRACTION:.0%}, "
        f"seed={VAL_SPLIT_SEED}) ..."
    )
    vt_k3, rvt_k3, vr_k3, rvr_k3 = _split_val_tune_report(val_k3, raw_val_k3)
    vt_k2, rvt_k2, vr_k2, rvr_k2 = _split_val_tune_report(val_k2, raw_val_k2)
    logger.info(f"  K=3 tune={len(vt_k3)} report={len(vr_k3)}")
    logger.info(f"  K=2 tune={len(vt_k2)} report={len(vr_k2)}")

    # ── Best single model baselines ───────────────────────────────────
    logger.info("\nComputing best-single-model baselines (on val_report) ...")
    bsm_k3 = best_single_model(train_k3, vr_k3, K3_MODELS, costs_k3)
    bsm_k2 = best_single_model(train_k2, vr_k2, K2_MODELS, costs_k2)
    logger.info(f"  K=3: R={bsm_k3['reward']:.4f} (arm: {bsm_k3['greedy_arm']})")
    logger.info(f"  K=2: R={bsm_k2['reward']:.4f} (arm: {bsm_k2['greedy_arm']})")

    # ── Portfolio Pareto AUC baselines (non-contextual frontier) ─────
    logger.info("\nComputing portfolio Pareto AUC baselines ...")
    ppa_k3_report = portfolio_pareto_auc(
        vr_k3, K3_MODELS, costs_k3, cost_lo_k3, cost_hi_k3,
    )
    ppa_k3_tune = portfolio_pareto_auc(
        vt_k3, K3_MODELS, costs_k3, cost_lo_k3, cost_hi_k3,
    )
    ppa_k2_report = portfolio_pareto_auc(
        vr_k2, K2_MODELS, costs_k2, cost_lo_k2, cost_hi_k2,
    )
    ppa_k2_tune = portfolio_pareto_auc(
        vt_k2, K2_MODELS, costs_k2, cost_lo_k2, cost_hi_k2,
    )
    logger.info(f"  K=3 portfolio AUC: tune={ppa_k3_tune['pareto_auc']:.4f}  "
                f"report={ppa_k3_report['pareto_auc']:.4f}")
    for m, pt in ppa_k3_report["model_points"].items():
        logger.info(f"    {m:<45s}  cost={pt['cost']:.6f}  R={pt['reward']:.4f}")
    logger.info(f"  K=2 portfolio AUC: tune={ppa_k2_tune['pareto_auc']:.4f}  "
                f"report={ppa_k2_report['pareto_auc']:.4f}")
    for m, pt in ppa_k2_report["model_points"].items():
        logger.info(f"    {m:<45s}  cost={pt['cost']:.6f}  R={pt['reward']:.4f}")

    # ── Grid info ─────────────────────────────────────────────────────
    n_hparam = len(ALPHA_VALUES) * len(NEFF_VALUES) * len(GAMMA_VALUES)
    n_lam = len(LAMBDA_SWEEP)
    n_variants = 4
    total_trials = (
        n_variants * len(COMPONENT_COUNTS) * n_hparam * n_lam * N_SEEDS
    )
    logger.info(f"\n{'='*70}")
    logger.info("Joint PCA-Dimension × Hyperparameter Ablation")
    logger.info(f"  Variants:   {n_variants} (K=3/K=2 × BanditGPT/Tabula Rasa)")
    logger.info(f"  Components: {COMPONENT_COUNTS}")
    logger.info(f"  Grid:       {n_hparam} hparam × {n_lam} λ × {N_SEEDS} seeds")
    logger.info(f"  Total:      {total_trials:,} trials")
    logger.info(f"{'='*70}")

    # ── Define variant configurations ─────────────────────────────────
    variant_configs = {
        "k3_banditgpt": dict(
            models=K3_MODELS, catalog=K3_CATALOG,
            warmup_priors_32=k3_priors, use_corralling=True,
            train_data=train_k3, val_tune_data=vt_k3, val_report_data=vr_k3,
            raw_train_emb=raw_train_k3, raw_val_tune_emb=rvt_k3,
            raw_val_report_emb=rvr_k3,
            costs=costs_k3, cost_lo=cost_lo_k3, cost_hi=cost_hi_k3,
        ),
        "k3_tabula_rasa": dict(
            models=K3_MODELS, catalog=K3_CATALOG,
            warmup_priors_32=None, use_corralling=False,
            train_data=train_k3, val_tune_data=vt_k3, val_report_data=vr_k3,
            raw_train_emb=raw_train_k3, raw_val_tune_emb=rvt_k3,
            raw_val_report_emb=rvr_k3,
            costs=costs_k3, cost_lo=cost_lo_k3, cost_hi=cost_hi_k3,
        ),
        "k2_banditgpt": dict(
            models=K2_MODELS, catalog=K2_CATALOG,
            warmup_priors_32=k2_priors, use_corralling=True,
            train_data=train_k2, val_tune_data=vt_k2, val_report_data=vr_k2,
            raw_train_emb=raw_train_k2, raw_val_tune_emb=rvt_k2,
            raw_val_report_emb=rvr_k2,
            costs=costs_k2, cost_lo=cost_lo_k2, cost_hi=cost_hi_k2,
        ),
        "k2_tabula_rasa": dict(
            models=K2_MODELS, catalog=K2_CATALOG,
            warmup_priors_32=None, use_corralling=False,
            train_data=train_k2, val_tune_data=vt_k2, val_report_data=vr_k2,
            raw_train_emb=raw_train_k2, raw_val_tune_emb=rvt_k2,
            raw_val_report_emb=rvr_k2,
            costs=costs_k2, cost_lo=cost_lo_k2, cost_hi=cost_hi_k2,
        ),
    }

    bsm_all = {
        "k3_banditgpt": bsm_k3, "k3_tabula_rasa": bsm_k3,
        "k2_banditgpt": bsm_k2, "k2_tabula_rasa": bsm_k2,
    }
    ppa_all = {
        "k3_banditgpt": {"tune": ppa_k3_tune, "report": ppa_k3_report},
        "k3_tabula_rasa": {"tune": ppa_k3_tune, "report": ppa_k3_report},
        "k2_banditgpt": {"tune": ppa_k2_tune, "report": ppa_k2_report},
        "k2_tabula_rasa": {"tune": ppa_k2_tune, "report": ppa_k2_report},
    }

    # ── Run all variants ──────────────────────────────────────────────
    all_results: Dict[str, List[Dict]] = {}

    for vkey, vcfg in variant_configs.items():
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"  VARIANT: {VARIANT_DISPLAY[vkey]}")
        logger.info(f"{'#'*70}")
        all_results[vkey] = _run_variant(
            vkey, pca32=pca32, output_dir=output_dir, **vcfg,
        )

    # ── Post-process: compute delta (contextual routing gain) ────────
    for vkey, results in all_results.items():
        ppa_tune_auc = ppa_all[vkey]["tune"]["pareto_auc"]
        ppa_report_auc = ppa_all[vkey]["report"]["pareto_auc"]
        for r in results:
            r["delta_tune"] = r["tune_pareto_auc"] - ppa_tune_auc
            r["delta_report"] = r["report_pareto_auc"] - ppa_report_auc

    # ── Save full results JSON ────────────────────────────────────────
    elapsed = time.time() - t0

    def _strip_grid(results: List[Dict]) -> List[Dict]:
        return [{k: v for k, v in r.items() if k != "grid_results"}
                for r in results]

    output_json: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": (
                "Joint PCA dimension × hyperparameter ablation.  "
                "4 variants (K=3/K=2 × BanditGPT/Tabula Rasa) on "
                "canonical splits.  K=2 derived from K=3 (cheapest + "
                "most expensive).  Val split sub-divided into val_tune "
                "(hparam selection) and val_report (unbiased metric).  "
                "Holdout never touched.  Selection criterion: Pareto AUC."
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
    }

    for vkey in variant_configs:
        k_size = 3 if vkey.startswith("k3") else 2
        models = K3_MODELS if k_size == 3 else K2_MODELS
        train_data = train_k3 if k_size == 3 else train_k2
        val_full = val_k3 if k_size == 3 else val_k2
        vt = vt_k3 if k_size == 3 else vt_k2
        vr = vr_k3 if k_size == 3 else vr_k2
        cost_range = (
            [cost_lo_k3, cost_hi_k3] if k_size == 3
            else [cost_lo_k2, cost_hi_k2]
        )
        output_json[vkey] = {
            "models": models,
            "n_train": len(train_data),
            "n_val_total": len(val_full),
            "n_val_tune": len(vt),
            "n_val_report": len(vr),
            "cost_range": cost_range,
            "best_single_model_baseline": bsm_all[vkey],
            "portfolio_pareto_auc_tune": ppa_all[vkey]["tune"]["pareto_auc"],
            "portfolio_pareto_auc_report": ppa_all[vkey]["report"]["pareto_auc"],
            "ablation_results": _strip_grid(all_results[vkey]),
        }

    out_path = output_dir / "pca_component_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output_json, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # ── Plot ablation curves ──────────────────────────────────────────
    plot_ablation(
        all_results, bsm_all, ppa_all,
        output_dir / "pca_component_ablation.png",
    )

    # ── Heatmaps + best_hparams at d* per variant ────────────────────
    logger.info("\nGenerating heatmaps and best_hparams at d* per variant ...")
    for vkey, results in all_results.items():
        dstar = _find_best_d_star(results)
        label = VARIANT_DISPLAY.get(vkey, vkey)
        ppa_tune_auc = ppa_all[vkey]["tune"]["pareto_auc"]
        ppa_report_auc = ppa_all[vkey]["report"]["pareto_auc"]
        logger.info(
            f"  {label}: d*={dstar['n_components']} "
            f"(tune AUC={dstar['tune_pareto_auc']:.4f}, "
            f"Δ_tune={dstar['delta_tune']:+.4f} | "
            f"report AUC={dstar['report_pareto_auc']:.4f}, "
            f"Δ_report={dstar['delta_report']:+.4f})"
        )

        if "grid_results" in dstar and dstar["grid_results"]:
            plot_heatmap(
                dstar["grid_results"],
                ALPHA_VALUES, NEFF_VALUES, GAMMA_VALUES,
                best_config={
                    "alpha": dstar["best_alpha"],
                    "n_eff": dstar["best_n_eff"],
                    "gamma": dstar["best_gamma"],
                    "pareto_auc": dstar["tune_pareto_auc"],
                },
                out_path=output_dir / f"heatmap_{vkey}_d{dstar['n_components']}.png",
                variant_label=label,
                n_comp=dstar["n_components"],
                portfolio_auc_tune=ppa_tune_auc,
            )

        _save_best_hparams(dstar, vkey, output_dir, ppa_report_auc)

    # ── Summary tables ────────────────────────────────────────────────
    for vkey in variant_configs:
        _print_summary_table(vkey, all_results[vkey], bsm_all[vkey], ppa_all[vkey])

    logger.info(f"\nElapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
