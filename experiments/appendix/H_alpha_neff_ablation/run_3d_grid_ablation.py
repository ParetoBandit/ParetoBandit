#!/usr/bin/env python3
"""
Appendix H: Hyperparameter Sensitivity Analysis
================================================

Sweeps three key hyperparameters (alpha, n_eff, forgetting_factor)
on a 5x4x4 grid to identify the best configuration for both BanditGPT
(Corralling + warmup priors) and Tabula Rasa (single LinUCB, no priors)
on the K=2 and K=3 portfolios.

Purpose
-------
This is a **hyperparameter tuning** experiment, not a warmup-vs-tabula-rasa
significance test.  Each variant is tuned independently so that downstream
experiments (Figure 3, Figure 4, Appendix J) use dev-val-selected
hyperparameters rather than defaults.

Selection criterion
-------------------
Configurations are ranked by **normalized AUCPC** — the area under the
cost-reward Pareto frontier traced by sweeping the cost penalty
parameter lambda.  For each (alpha, n_eff, gamma) triple, we train
separate routers at several lambda values, freeze each, evaluate on
dev-val, and compute the Pareto hull of the resulting (cost, reward)
points.  The AUCPC integrates the Pareto hull after **normalizing both
axes** so that the cheap-model baseline is at (0, 0) and the frontier-model
baseline is at (1, 1).  This makes the scalar directly interpretable
(random-diagonal ≈ 0.5, near-oracle ≈ 1.0).

*Why not mean reward at lambda=0?*  When one model dominates on average
(e.g. GPT-4.1 in K=2), quality-only tuning converges to a degenerate
"always pick the strongest model" policy.  The router never learns
cost-quality trade-offs because the cost signal is absent.  Pareto AUC
captures the full value proposition of a contextual bandit router:
finding the best quality model at *every* cost level.  This directly
rewards hyperparameters that produce diverse, intelligent routing across
the cost spectrum.

Protocol
--------
For each (alpha, n_eff, gamma) triple:

1. For each lambda in the cost-penalty sweep:
   a. Instantiate BanditRouter with that lambda.
   b. Train on the dev-train split (N seeds, shuffled order).
   c. Freeze the router (alpha=0 for greedy exploitation).
   d. Evaluate on the dev-val split — record (mean_cost, mean_reward).
2. Compute the Pareto hull of the (cost, reward) points.
3. Compute normalized AUCPC over the portfolio endpoints.
4. Select the config with the highest normalized AUCPC on dev-val.
5. Report its holdout normalized AUCPC (sweep repeated on holdout).

Grid
----
- alpha:             [0.1, 0.25, 0.5, 1.0, 2.0]
- n_eff:             [10, 100, 1000, 5000]
- forgetting_factor: [0.995, 0.999, 0.9999, 1.0]
- lambda (K=2):      [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
- lambda (K=3):      [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
- Total per portfolio: 80 configs x 7 lambda x 20 seeds = 11,200 trials

Outputs (``results/``)
    alpha_neff_gamma_grid_results.json      (K=2 BanditGPT)
    alpha_neff_gamma_grid_figure.png        (K=2 BanditGPT)
    alpha_neff_gamma_grid_k3_results.json   (K=3 BanditGPT)
    alpha_neff_gamma_grid_k3_figure.png     (K=3 BanditGPT)
    alpha_neff_gamma_grid_tabula_rasa_*.json/png  (Tabula Rasa variants)
    best_hparams_{k2,k3}[_tabula_rasa].json  (selected configs)
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
from scipy import stats as sp_stats
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K2_WARMUP_PRIORS_PATH,
    K3_WARMUP_PRIORS_PATH,
    K3_MODELS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    THREE_WAY_SPLITS_PATH,
)
from utils.rewards import extract_reward
from utils.router_factory import create_experiment_router
from utils.model_pricing import get_prices_for_models, req_cost
from utils.embeddings import load_embedding_cache, embed_dataset_cached
from utils.pareto import pareto_aucpc_normalized, pareto_frontier_nondominated

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# Grid parameters
# ============================================================================

ALPHA_VALUES: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0]
NEFF_VALUES: List[float] = [10.0, 100.0, 1000.0, 5000.0]
GAMMA_VALUES: List[float] = [0.995, 0.999, 0.9999, 1.0]

# Lambda sweep for Pareto frontier tracing.
LAMBDA_SWEEP_K2: List[float] = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
LAMBDA_SWEEP_K3: List[float] = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

N_SEEDS: int = 20
SEED_OFFSET: int = 42
CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05

DEV_VAL_FRACTION: float = 0.2
DEV_VAL_SEED: int = 7

# ── K=2 portfolio ─────────────────────────────────────────────────────

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

# ── K=3 portfolio (loaded from canonical config) ────────────────────


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

REWARD_THEORETICAL_MIN: float = 0.0
REWARD_THEORETICAL_MAX: float = 1.0


# ============================================================================
# Data loading (mirrors run_prequential.py)
# ============================================================================



def load_rewards_from_file(
    data_path: Path,
    models: List[str],
    *,
    prompt_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    """Load rewards for specific models from gzipped JSONL.

    Only prompts with rewards for *all* requested models are included.
    If ``prompt_filter`` is given, only those prompts are retained.
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

    data = []
    n_models = len(models)
    for prompt, rmap in rewards.items():
        if len(rmap) == n_models:
            data.append({"prompt": prompt, "rewards": rmap})
    return data


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
# Core evaluation
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
    use_corralling: bool,
) -> Dict[str, float]:
    """Train-then-freeze for one (alpha, n_eff, gamma, lambda) point.

    Returns the seed-averaged (mean_reward, mean_cost) plus routing
    fractions for diagnostics.
    """
    r_min = REWARD_THEORETICAL_MIN
    r_range = REWARD_THEORETICAL_MAX - REWARD_THEORETICAL_MIN
    burn_in = len(train_data)

    trial_r: List[float] = []
    trial_c: List[float] = []
    agg_mc: Dict[str, int] = {m: 0 for m in models}

    for trial in range(n_seeds):
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

        rng_state = np.random.get_state()
        r_total = c_total = 0.0
        with router.exploit():
            for p, x in zip(eval_data, eval_emb):
                model, _log = router.route(x, total_steps=burn_in)
                r_total += p["rewards"][model]
                c_total += costs[model]
                agg_mc[model] = agg_mc.get(model, 0) + 1
        np.random.set_state(rng_state)

        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)

    total_routed = sum(agg_mc.values()) or 1
    return {
        "mean_reward": float(np.mean(trial_r)),
        "mean_cost": float(np.mean(trial_c)),
        "routing_fractions": {m: agg_mc[m] / total_routed for m in models},
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
    cheap_baseline_reward: float,
    frontier_baseline_reward: float,
    n_seeds: int,
    use_corralling: bool = True,
) -> Dict[str, Any]:
    """Evaluate one (alpha, n_eff, gamma) config across multiple lambdas.

    Sweeps ``lambda_values`` to trace the Pareto frontier of cost vs.
    reward, then computes the **normalized AUCPC** by first normalizing:

    - cost: ``cost_lo -> 0`` and ``cost_hi -> 1`` (cheapest vs frontier model),
    - reward: ``cheap_baseline_reward -> 0`` and
      ``frontier_baseline_reward -> 1`` (static baselines on the eval split).

    Args:
        models: Candidate model IDs.
        catalog: Model metadata catalog.
        train_data: Training prompts with rewards.
        eval_data: Evaluation prompts with rewards.
        train_emb: Pre-computed feature vectors for training set.
        eval_emb: Pre-computed feature vectors for eval set.
        warmup_path: Path to warmup priors (None for tabula rasa).
        costs: Per-model cost dict.
        alpha: Exploration coefficient.
        n_eff: Prior effective sample size.
        gamma: Forgetting factor (1.0 = stationary).
        lambda_values: Cost penalty values to sweep.
        cost_lo: Cheapest model cost (integration lower bound).
        cost_hi: Frontier model cost (integration upper bound).
        cheap_baseline_reward: Mean reward of always using the cheapest model
            on the evaluation split used for AUC.
        frontier_baseline_reward: Mean reward of always using the frontier
            model on the same evaluation split.
        n_seeds: Number of random seeds per lambda.
        use_corralling: Enable Corralling meta-learner.

    Returns:
        Dict with ``pareto_auc`` (normalized AUCPC), per-lambda sweep
        points, lambda=0
        reward, and routing fractions at the median-cost lambda.
    """
    dim = train_emb[0].shape[0]
    registry = build_model_registry(models, catalog)

    sweep_points: List[Dict[str, Any]] = []
    lam0_reward: float = 0.0
    lam0_routing: Dict[str, float] = {}

    for lam in lambda_values:
        pt = _train_and_eval_single_lambda(
            models, registry,
            train_data, eval_data, train_emb, eval_emb,
            warmup_path, costs, dim,
            alpha=alpha, n_eff=n_eff, gamma=gamma,
            cost_penalty=lam, n_seeds=n_seeds,
            use_corralling=use_corralling,
        )
        sweep_points.append({
            "lambda": lam,
            "mean_reward": pt["mean_reward"],
            "mean_cost": pt["mean_cost"],
            "routing_fractions": pt["routing_fractions"],
        })
        if lam == 0.0:
            lam0_reward = pt["mean_reward"]
            lam0_routing = pt["routing_fractions"]

    sweep_costs = [sp["mean_cost"] for sp in sweep_points]
    sweep_rewards = [sp["mean_reward"] for sp in sweep_points]
    p_auc = pareto_aucpc_normalized(
        sweep_costs,
        sweep_rewards,
        cheap_cost=cost_lo,
        frontier_cost=cost_hi,
        cheap_reward=cheap_baseline_reward,
        frontier_reward=frontier_baseline_reward,
        clip_quality_to_unit=True,
    )

    # Routing fractions at the lambda closest to the portfolio median cost.
    median_cost = (cost_lo + cost_hi) / 2
    mid_idx = int(np.argmin([abs(sp["mean_cost"] - median_cost) for sp in sweep_points]))
    mid_routing = sweep_points[mid_idx]["routing_fractions"]

    return {
        "alpha": alpha,
        "n_eff": n_eff,
        "gamma": gamma,
        "pareto_auc": p_auc,
        "lam0_reward": lam0_reward,
        "lam0_routing_fractions": lam0_routing,
        "mid_routing_fractions": mid_routing,
        "sweep_points": sweep_points,
        "n_seeds": n_seeds,
    }


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
# Grid sweep
# ============================================================================


def run_grid(
    models: List[str],
    catalog: Dict[str, Dict],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_path: Optional[str],
    costs: Dict[str, float],
    *,
    alpha_values: List[float],
    neff_values: List[float],
    gamma_values: List[float],
    lambda_values: List[float],
    cost_lo: float,
    cost_hi: float,
    cheap_baseline_reward: float,
    frontier_baseline_reward: float,
    n_seeds: int,
    use_corralling: bool = True,
) -> List[Dict[str, Any]]:
    """Sweep the 3D grid, evaluating normalized AUCPC for each configuration."""
    total = len(alpha_values) * len(neff_values) * len(gamma_values)
    results: List[Dict[str, Any]] = []
    idx = 0

    for gamma in gamma_values:
        for n_eff in neff_values:
            for alpha in alpha_values:
                idx += 1
                res = train_and_evaluate_pareto(
                    models, catalog,
                    train_data, eval_data, train_emb, eval_emb,
                    warmup_path, costs,
                    alpha=alpha, n_eff=n_eff, gamma=gamma,
                    lambda_values=lambda_values,
                    cost_lo=cost_lo, cost_hi=cost_hi,
                    cheap_baseline_reward=cheap_baseline_reward,
                    frontier_baseline_reward=frontier_baseline_reward,
                    n_seeds=n_seeds,
                    use_corralling=use_corralling,
                )
                results.append(res)
                logger.info(
                    f"  [{idx:3d}/{total}] "
                    f"alpha={alpha:<5} n_eff={n_eff:<7} gamma={gamma:<7} "
                    f"| AUCPC={res['pareto_auc']:.4f} "
                    f"R@lam0={res['lam0_reward']:.4f}"
                )

    return results


# ============================================================================
# Plotting
# ============================================================================


def plot_grid(
    results: List[Dict[str, Any]],
    alpha_values: List[float],
    neff_values: List[float],
    gamma_values: List[float],
    out: Path,
    *,
    best_config: Dict[str, Any],
    k_label: int = 2,
    filename: str = "alpha_neff_gamma_grid_figure.png",
) -> None:
    """Generate a two-row heatmap figure (one panel per forgetting factor).

    Row 1: Normalized AUCPC (selection criterion).
    Row 2: Mean reward at lambda=0 (for reference).
    The global best cell (by AUCPC) is starred in both rows.

    Args:
        results: Full grid output from :func:`run_grid`.  Each entry
            must have ``alpha``, ``n_eff``, ``gamma``, ``pareto_auc``,
            ``lam0_reward``, and ``n_seeds`` keys.
        alpha_values: Exploration coefficients (x-axis tick values).
        neff_values: Prior effective sample sizes (y-axis tick values).
        gamma_values: Forgetting factors — one heatmap panel per value.
        out: Output directory for the saved figure.
        best_config: The globally selected configuration dict (same
            schema as *results* entries); its cell is starred.
        k_label: Portfolio size (2 or 3) shown in the title.
        filename: Output PNG filename.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    lookup_auc: Dict[Tuple[float, float, float], float] = {}
    lookup_reward: Dict[Tuple[float, float, float], float] = {}
    for r in results:
        key = (r["alpha"], r["n_eff"], r["gamma"])
        lookup_auc[key] = r["pareto_auc"]
        lookup_reward[key] = r["lam0_reward"]

    n_panels = len(gamma_values)
    ncols = n_panels
    fig, axes = plt.subplots(
        2, ncols, figsize=(4.5 * ncols, 9),
        constrained_layout=True,
    )
    if ncols == 1:
        axes = axes.reshape(2, 1)

    all_auc = [r["pareto_auc"] for r in results]
    all_reward = [r["lam0_reward"] for r in results]
    norm_a = Normalize(vmin=min(all_auc) - 0.005, vmax=max(all_auc) + 0.005)
    norm_r = Normalize(vmin=min(all_reward) - 0.002, vmax=max(all_reward) + 0.002)

    for row_idx, (lookup, norm, metric_label, fmt) in enumerate([
        (lookup_auc, norm_a, "AUCPC (normalized)", ".4f"),
        (lookup_reward, norm_r, r"Reward ($\lambda{=}0$)", ".3f"),
    ]):
        for col_idx, gamma in enumerate(gamma_values):
            ax = axes[row_idx, col_idx]
            grid = np.zeros((len(neff_values), len(alpha_values)))
            for i, n_eff in enumerate(neff_values):
                for j, alpha in enumerate(alpha_values):
                    grid[i, j] = lookup.get((alpha, n_eff, gamma), np.nan)

            ax.imshow(
                grid, aspect="auto", origin="lower", norm=norm,
                cmap="YlOrRd",
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

            gamma_label = (
                f"{gamma}" if gamma < 1.0 else "1.0 (stationary)"
            )
            if row_idx == 0:
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
                    color = (
                        "white" if val > (norm.vmin + norm.vmax) / 2
                        else "black"
                    )
                    ax.text(
                        j, i, text, ha="center", va="center",
                        fontsize=7,
                        fontweight="bold" if is_best else "normal",
                        color=color,
                    )

    fig.suptitle(
        r"3D Ablation: $\alpha \times n_{\mathrm{eff}} \times \gamma$ "
        f"(K={k_label}, {results[0]['n_seeds']} seeds)\n"
        f"Best (by AUCPC): alpha={best_config['alpha']}, "
        f"n_eff={int(best_config['n_eff'])}, "
        f"gamma={best_config['gamma']} "
        f"-> AUC={best_config['pareto_auc']:.4f}, "
        f"R@lam0={best_config['lam0_reward']:.4f}",
        fontsize=11, fontweight="bold",
    )

    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_lambda_sweep_curve(
    sweep_points: List[Dict[str, Any]],
    *,
    cheap_cost: float,
    frontier_cost: float,
    cheap_baseline_reward: float,
    frontier_baseline_reward: float,
    out: Path,
    filename: str,
    title: str,
) -> None:
    """Plot the empirical (cost, quality) pairs induced by a λ sweep.

    The points in *sweep_points* are produced by training (or selecting) a router
    under different values of the cost-penalty preference λ. We plot them in the
    **normalized** cost–quality space used for AUCPC:

    - x: normalized cost (cheap → 0, frontier → 1)
    - y: normalized quality (cheap baseline → 0, frontier baseline → 1)

    Args:
        sweep_points: List of dicts with keys ``lambda``, ``mean_cost``,
            ``mean_reward``.
        cheap_cost: Cheapest model cost in the portfolio.
        frontier_cost: Frontier model cost in the portfolio.
        cheap_baseline_reward: Mean reward of always choosing the cheapest model
            on the same split as *sweep_points*.
        frontier_baseline_reward: Mean reward of always choosing the frontier
            model on the same split.
        out: Output directory.
        filename: Filename for the saved figure.
        title: Figure title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    costs = [float(sp["mean_cost"]) for sp in sweep_points]
    rewards = [float(sp["mean_reward"]) for sp in sweep_points]
    lambdas = [float(sp["lambda"]) for sp in sweep_points]

    aucpc = pareto_aucpc_normalized(
        costs,
        rewards,
        cheap_cost=cheap_cost,
        frontier_cost=frontier_cost,
        cheap_reward=cheap_baseline_reward,
        frontier_reward=frontier_baseline_reward,
        clip_quality_to_unit=True,
    )

    cost_range = float(frontier_cost - cheap_cost)
    qual_range = float(frontier_baseline_reward - cheap_baseline_reward)
    if cost_range <= 0:
        raise ValueError("Expected frontier_cost > cheap_cost for plotting.")

    x = (np.array(costs, dtype=float) - cheap_cost) / cost_range
    if abs(qual_range) <= 1e-12:
        y = np.full_like(x, 0.5, dtype=float)
    else:
        y = (np.array(rewards, dtype=float) - cheap_baseline_reward) / qual_range
        y = np.clip(y, 0.0, 1.0)

    hull_c, hull_r = pareto_frontier_nondominated(costs, rewards)
    hull_x = (np.array(hull_c, dtype=float) - cheap_cost) / cost_range
    if abs(qual_range) <= 1e-12:
        hull_y = np.full_like(hull_x, 0.5, dtype=float)
    else:
        hull_y = (np.array(hull_r, dtype=float) - cheap_baseline_reward) / qual_range
        hull_y = np.clip(hull_y, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    sc = ax.scatter(
        x,
        y,
        c=np.array(lambdas, dtype=float),
        cmap="viridis",
        s=60,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.plot(hull_x, hull_y, color="black", linewidth=2.0, zorder=4, label="Pareto hull")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, zorder=1, label="Diagonal (AUC=0.5)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Normalized cost (cheap → 0, frontier → 1)")
    ax.set_ylabel("Normalized quality (cheap baseline → 0, frontier baseline → 1)")
    ax.set_title(f"{title}\nAUCPC={aucpc:.3f}", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.03)
    cbar.set_label("Cost penalty λ", rotation=90)

    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ============================================================================
# Single-portfolio ablation runner
# ============================================================================


def run_portfolio_ablation(
    models: List[str],
    catalog: Dict[str, Dict],
    train_data: List[Dict],
    holdout_data: List[Dict],
    encoder: SentenceTransformer,
    pca: PCA,
    warmup_path: Optional[str],
    output_dir: Path,
    *,
    embedding_cache: Dict[str, np.ndarray],
    k_label: int,
    lambda_values: List[float],
    json_filename: str,
    figure_filename: str,
    main_results_key: str,
    use_corralling: bool = True,
    variant_label: str = "BanditGPT",
) -> Dict[str, Any]:
    """Run the full 3D grid ablation for a single portfolio.

    For each (alpha, n_eff, gamma) triple, sweeps ``lambda_values`` to
    trace the Pareto frontier of cost vs. quality, then selects the
    configuration with the highest normalized AUCPC on the dev-val split.
    The selected configuration is re-evaluated on holdout.

    Args:
        models: Ordered list of candidate model IDs in the portfolio.
        catalog: Per-model metadata keyed by model ID.  Each entry must
            contain ``"input_cost_per_m"`` and ``"output_cost_per_m"``.
        train_data: Dev-split prompts (used for online learning).  Each
            dict has ``"prompt"`` and ``"rewards"`` keys.
        holdout_data: Holdout prompts for final evaluation (same schema).
        encoder: SentenceTransformer used for live embedding fallback
            on cache misses.
        pca: Fitted PCA model for dimensionality reduction.
        warmup_path: Path to the ``.joblib`` warmup priors file, or
            ``None`` for tabula-rasa (no priors).
        output_dir: Directory for JSON results, heatmap figures, and
            lambda-sweep curve plots.
        embedding_cache: Pre-computed ``{sha256(prompt) -> vector}``
            cache loaded from disk.  Passed to
            :func:`embed_dataset_cached` to avoid live encoding.
        k_label: Portfolio size label (2 or 3) used in logging, plot
            titles, and the JSON output filename.
        lambda_values: Cost-penalty values to sweep when tracing the
            Pareto frontier (e.g. ``[0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]``).
        json_filename: Filename for the JSON results file written to
            *output_dir*.
        figure_filename: Filename for the heatmap PNG written to
            *output_dir*.
        main_results_key: Key into the main ``prequential_results.json``
            (e.g. ``"K2"`` or ``"K3"``) used to look up the supervised
            peak baseline for annotation.
        use_corralling: If ``True``, use the Corralling meta-learner
            with two experts.  If ``False``, use a single LinUCB
            (tabula rasa).
        variant_label: Human-readable label for this variant (e.g.
            ``"BanditGPT"`` or ``"Tabula Rasa"``), used in log messages,
            plot titles, and output metadata.

    Returns:
        Dict with the selected hyperparameters and their dev-val /
        holdout normalized AUCPC and mean reward at lambda=0.  Keys:
        ``alpha``, ``n_eff``, ``gamma``, ``dev_val_pareto_auc``,
        ``dev_val_mean_reward``, ``holdout_pareto_auc``,
        ``holdout_mean_reward``.
    """
    def _baseline_reward(data: List[Dict], model_id: str) -> float:
        if not data:
            return 0.0
        return float(np.mean([p["rewards"][model_id] for p in data]))

    costs = {
        m: req_cost(
            catalog[m]["input_cost_per_m"],
            catalog[m]["output_cost_per_m"],
        )
        for m in models
    }
    cost_lo = min(costs.values())
    cost_hi = max(costs.values())
    cheap_model = min(models, key=lambda m: (costs[m], m))
    frontier_model = max(models, key=lambda m: (costs[m], m))

    logger.info(
        f"  K={k_label}: {len(train_data)} dev (online-learn pool), "
        f"{len(holdout_data)} holdout prompts"
    )
    logger.info(
        f"  Cost range: [{cost_lo:.6f}, {cost_hi:.6f}]"
    )

    logger.info(f"  Embedding K={k_label} prompts ...")
    dev_emb = embed_dataset_cached(train_data, embedding_cache, encoder, pca)
    holdout_emb = embed_dataset_cached(holdout_data, embedding_cache, encoder, pca)
    dim = dev_emb[0].shape[0]
    logger.info(f"    Feature dim: {dim}")

    logger.info(
        f"  Splitting dev into train/val ({1 - DEV_VAL_FRACTION:.0%}/"
        f"{DEV_VAL_FRACTION:.0%}) for hyperparameter selection ..."
    )
    dev_train, dev_train_emb, dev_val, dev_val_emb = _split_dev_train_val(
        train_data, dev_emb
    )
    logger.info(f"    Dev-train: {len(dev_train)}  Dev-val: {len(dev_val)}")

    dev_val_cheap_r = _baseline_reward(dev_val, cheap_model)
    dev_val_frontier_r = _baseline_reward(dev_val, frontier_model)
    holdout_cheap_r = _baseline_reward(holdout_data, cheap_model)
    holdout_frontier_r = _baseline_reward(holdout_data, frontier_model)
    logger.info(
        "  AUC normalization endpoints:\n"
        f"    cheap:    {catalog.get(cheap_model, {}).get('display', cheap_model.split('/')[-1])} "
        f"(cost={costs[cheap_model]:.6f}, baseline_R={dev_val_cheap_r:.4f} dev-val)\n"
        f"    frontier: {catalog.get(frontier_model, {}).get('display', frontier_model.split('/')[-1])} "
        f"(cost={costs[frontier_model]:.6f}, baseline_R={dev_val_frontier_r:.4f} dev-val)"
    )

    oracle_reward = float(np.mean([
        max(p["rewards"][m] for m in models) for p in holdout_data
    ]))

    main_results_path = (
        Path(__file__).parent.parent.parent
        / "03_figure" / "results" / "prequential_results.json"
    )
    supervised_peak: Optional[float] = None
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_res = json.load(f)
        supervised = (
            main_res
            .get(main_results_key, {})
            .get("supervised", {})
        )
        if supervised:
            supervised_peak = max(s["reward"] for s in supervised.values())
        logger.info(f"    Supervised peak ({main_results_key}): {supervised_peak}")

    total_configs = (
        len(ALPHA_VALUES) * len(NEFF_VALUES) * len(GAMMA_VALUES)
    )
    n_lambda = len(lambda_values)
    logger.info(
        f"\n  3D grid: {total_configs} configs x {n_lambda} lambda "
        f"x {N_SEEDS} seeds = {total_configs * n_lambda * N_SEEDS} trials"
    )

    results = run_grid(
        models, catalog,
        dev_train, dev_val, dev_train_emb, dev_val_emb,
        warmup_path, costs,
        alpha_values=ALPHA_VALUES,
        neff_values=NEFF_VALUES,
        gamma_values=GAMMA_VALUES,
        lambda_values=lambda_values,
        cost_lo=cost_lo,
        cost_hi=cost_hi,
        cheap_baseline_reward=dev_val_cheap_r,
        frontier_baseline_reward=dev_val_frontier_r,
        n_seeds=N_SEEDS,
        use_corralling=use_corralling,
    )

    ranked = sorted(results, key=lambda r: r["pareto_auc"], reverse=True)
    best = ranked[0]

    best_by_reward = sorted(
        results, key=lambda r: r["lam0_reward"], reverse=True,
    )[0]

    logger.info(f"\n{'=' * 70}")
    logger.info(
        f"K={k_label} {variant_label} TOP-10 CONFIGURATIONS "
        f"(by AUCPC)"
    )
    logger.info(f"{'=' * 70}")
    for i, r in enumerate(ranked[:10]):
        rf = r.get("mid_routing_fractions", {})
        top_model = max(rf, key=rf.get) if rf else "?"
        top_frac = rf.get(top_model, 0.0)
        top_short = catalog.get(top_model, {}).get(
            "display", top_model.split("/")[-1],
        )
        logger.info(
            f"  #{i + 1:2d}  alpha={r['alpha']:<5} n_eff={r['n_eff']:<7} "
            f"gamma={r['gamma']:<7} "
            f"| AUC={r['pareto_auc']:.4f} "
            f"R@lam0={r['lam0_reward']:.4f} "
            f"MidTop={top_short}({top_frac:.0%})"
        )
    logger.info(f"\n  Oracle (per-prompt best): {oracle_reward:.4f}")
    if supervised_peak is not None:
        logger.info(f"  Supervised peak:         {supervised_peak:.4f}")
    logger.info(
        f"\n  BEST (by AUCPC): alpha={best['alpha']}, "
        f"n_eff={int(best['n_eff'])}, gamma={best['gamma']} "
        f"-> AUC={best['pareto_auc']:.4f}, "
        f"R@lam0={best['lam0_reward']:.4f}"
    )
    if (best_by_reward["alpha"] != best["alpha"]
            or best_by_reward["n_eff"] != best["n_eff"]
            or best_by_reward["gamma"] != best["gamma"]):
        logger.info(
            f"  (cf. best by lam0 reward: alpha={best_by_reward['alpha']}, "
            f"n_eff={int(best_by_reward['n_eff'])}, "
            f"gamma={best_by_reward['gamma']} "
            f"-> AUC={best_by_reward['pareto_auc']:.4f}, "
            f"R@lam0={best_by_reward['lam0_reward']:.4f})"
        )

    # Holdout normalized AUCPC for the selected config.
    logger.info("\n  Evaluating selected config on holdout ...")
    best_holdout = train_and_evaluate_pareto(
        models=models,
        catalog=catalog,
        train_data=dev_train,
        eval_data=holdout_data,
        train_emb=dev_train_emb,
        eval_emb=holdout_emb,
        warmup_path=warmup_path,
        costs=costs,
        alpha=float(best["alpha"]),
        n_eff=float(best["n_eff"]),
        gamma=float(best["gamma"]),
        lambda_values=lambda_values,
        cost_lo=cost_lo,
        cost_hi=cost_hi,
        cheap_baseline_reward=holdout_cheap_r,
        frontier_baseline_reward=holdout_frontier_r,
        n_seeds=N_SEEDS,
        use_corralling=use_corralling,
    )
    logger.info(
        f"  Holdout: AUC={best_holdout['pareto_auc']:.4f} "
        f"R@lam0={best_holdout['lam0_reward']:.4f}"
    )

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": (
                "3D hyperparameter ablation (alpha x n_eff x forgetting_factor) "
                f"for {variant_label} "
                f"({'Corralling + warmup priors' if use_corralling else 'single LinUCB, no priors'}). "
                f"Pareto frontier traced over lambda sweep on K={k_label}. "
                "Selection criterion: normalized AUCPC (area under the normalized cost–quality curve)."
            ),
        },
        "config": {
            "alpha_values": ALPHA_VALUES,
            "neff_values": NEFF_VALUES,
            "gamma_values": GAMMA_VALUES,
            "lambda_values": lambda_values,
            "cost_lo": cost_lo,
            "cost_hi": cost_hi,
            "cheap_model": cheap_model,
            "frontier_model": frontier_model,
            "dev_val_cheap_baseline_reward": dev_val_cheap_r,
            "dev_val_frontier_baseline_reward": dev_val_frontier_r,
            "n_seeds": N_SEEDS,
            "corralling_lr": CORRALLING_LR,
            "corralling_gamma": CORRALLING_GAMMA,
            "dev_val_fraction": DEV_VAL_FRACTION,
            "dev_val_seed": DEV_VAL_SEED,
            "selection_criterion": "pareto_aucpc_normalized",
        },
        "n_dev": len(train_data),
        "n_dev_train": len(dev_train),
        "n_dev_val": len(dev_val),
        "n_holdout": len(holdout_data),
        "oracle_reward": oracle_reward,
        "supervised_peak": supervised_peak,
        "best": {
            "alpha": best["alpha"],
            "n_eff": best["n_eff"],
            "gamma": best["gamma"],
            "dev_val_pareto_auc": best["pareto_auc"],
            "dev_val_lam0_reward": best["lam0_reward"],
            "dev_val_sweep_points": best["sweep_points"],
            "dev_val_mid_routing_fractions": best["mid_routing_fractions"],
            "holdout_pareto_auc": best_holdout["pareto_auc"],
            "holdout_lam0_reward": best_holdout["lam0_reward"],
            "holdout_sweep_points": best_holdout["sweep_points"],
            "holdout_mid_routing_fractions": best_holdout["mid_routing_fractions"],
        },
        "ranked_top10": [
            {k: v for k, v in r.items() if k != "sweep_points"}
            for r in ranked[:10]
        ],
        "full_grid": [
            {k: v for k, v in r.items() if k != "sweep_points"}
            for r in results
        ],
    }

    out_path = output_dir / json_filename
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults -> {out_path}")

    plot_grid(
        results, ALPHA_VALUES, NEFF_VALUES, GAMMA_VALUES,
        output_dir,
        best_config=best,
        k_label=k_label,
        filename=figure_filename,
    )

    variant_slug = (
        variant_label.lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )
    plot_lambda_sweep_curve(
        best["sweep_points"],
        cheap_cost=cost_lo,
        frontier_cost=cost_hi,
        cheap_baseline_reward=dev_val_cheap_r,
        frontier_baseline_reward=dev_val_frontier_r,
        out=output_dir,
        filename=f"aucpc_curve_k{k_label}_{variant_slug}_dev_val.png",
        title=f"K={k_label} {variant_label} (dev-val λ sweep)",
    )
    plot_lambda_sweep_curve(
        best_holdout["sweep_points"],
        cheap_cost=cost_lo,
        frontier_cost=cost_hi,
        cheap_baseline_reward=holdout_cheap_r,
        frontier_baseline_reward=holdout_frontier_r,
        out=output_dir,
        filename=f"aucpc_curve_k{k_label}_{variant_slug}_holdout.png",
        title=f"K={k_label} {variant_label} (holdout λ sweep)",
    )

    return {
        "alpha": float(best["alpha"]),
        "n_eff": float(best["n_eff"]),
        "gamma": float(best["gamma"]),
        "dev_val_pareto_auc": float(best["pareto_auc"]),
        "dev_val_mean_reward": float(best["lam0_reward"]),
        "holdout_pareto_auc": float(best_holdout["pareto_auc"]),
        "holdout_mean_reward": float(best_holdout["lam0_reward"]),
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("Loading encoder, PCA, and embedding cache ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    embedding_cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    total_configs = (
        len(ALPHA_VALUES) * len(NEFF_VALUES) * len(GAMMA_VALUES)
    )
    logger.info(
        f"Grid: {total_configs} configs x {N_SEEDS} seeds "
        f"x {len(LAMBDA_SWEEP_K2)} lambda (K=2) / "
        f"{len(LAMBDA_SWEEP_K3)} lambda (K=3)"
    )
    logger.info(
        f"  alpha:  {ALPHA_VALUES}\n"
        f"  n_eff:  {NEFF_VALUES}\n"
        f"  gamma:  {GAMMA_VALUES}"
    )

    # ── K=2 ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("K=2 Portfolio Ablation")
    logger.info("=" * 70)

    dev_k2 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K2_MODELS)
    holdout_k2 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K2_MODELS,
    )

    best_k2 = run_portfolio_ablation(
        K2_MODELS, K2_CATALOG,
        dev_k2, holdout_k2, encoder, pca,
        str(K2_WARMUP_PRIORS_PATH),
        output_dir,
        embedding_cache=embedding_cache,
        k_label=2,
        lambda_values=LAMBDA_SWEEP_K2,
        json_filename="alpha_neff_gamma_grid_results.json",
        figure_filename="alpha_neff_gamma_grid_figure.png",
        main_results_key="K2",
    )

    # ── K=3 ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("K=3 Portfolio Ablation")
    logger.info("=" * 70)

    prior_train_prompts: Set[str] = set()
    if THREE_WAY_SPLITS_PATH.exists():
        with open(THREE_WAY_SPLITS_PATH) as f:
            splits_3way = json.load(f)
        prior_train_prompts = set(splits_3way.get("prior_train_pool", []))

    all_dev_k3 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K3_MODELS)
    train_k3 = [
        d for d in all_dev_k3 if d["prompt"] not in prior_train_prompts
    ]
    holdout_k3 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K3_MODELS,
    )

    best_k3 = run_portfolio_ablation(
        K3_MODELS, K3_CATALOG,
        train_k3, holdout_k3, encoder, pca,
        str(K3_WARMUP_PRIORS_PATH),
        output_dir,
        embedding_cache=embedding_cache,
        k_label=3,
        lambda_values=LAMBDA_SWEEP_K3,
        json_filename="alpha_neff_gamma_grid_k3_results.json",
        figure_filename="alpha_neff_gamma_grid_k3_figure.png",
        main_results_key="K3",
    )

    # ── K=2 Tabula Rasa ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("K=2 Tabula Rasa Portfolio Ablation")
    logger.info("=" * 70)

    best_k2_tr = run_portfolio_ablation(
        K2_MODELS, K2_CATALOG,
        dev_k2, holdout_k2, encoder, pca,
        None,
        output_dir,
        embedding_cache=embedding_cache,
        k_label=2,
        lambda_values=LAMBDA_SWEEP_K2,
        json_filename="alpha_neff_gamma_grid_tabula_rasa_results.json",
        figure_filename="alpha_neff_gamma_grid_tabula_rasa_figure.png",
        main_results_key="K2",
        use_corralling=False,
        variant_label="Tabula Rasa",
    )

    # ── K=3 Tabula Rasa ────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("K=3 Tabula Rasa Portfolio Ablation")
    logger.info("=" * 70)

    best_k3_tr = run_portfolio_ablation(
        K3_MODELS, K3_CATALOG,
        train_k3, holdout_k3, encoder, pca,
        None,
        output_dir,
        embedding_cache=embedding_cache,
        k_label=3,
        lambda_values=LAMBDA_SWEEP_K3,
        json_filename="alpha_neff_gamma_grid_tabula_rasa_k3_results.json",
        figure_filename="alpha_neff_gamma_grid_tabula_rasa_k3_figure.png",
        main_results_key="K3",
        use_corralling=False,
        variant_label="Tabula Rasa",
    )

    # Persist the selected hyperparameters for downstream experiments.
    (output_dir / "best_hparams_k2.json").write_text(
        json.dumps({"K2": best_k2}, indent=2)
    )
    (output_dir / "best_hparams_k3.json").write_text(
        json.dumps({"K3": best_k3}, indent=2)
    )
    (output_dir / "best_hparams_k2_tabula_rasa.json").write_text(
        json.dumps({"K2": best_k2_tr}, indent=2)
    )
    (output_dir / "best_hparams_k3_tabula_rasa.json").write_text(
        json.dumps({"K3": best_k3_tr}, indent=2)
    )
    logger.info(f"\nSaved best hyperparams to {output_dir / 'best_hparams_k2.json'}")
    logger.info(f"Saved best hyperparams to {output_dir / 'best_hparams_k3.json'}")
    logger.info(f"Saved best hyperparams to {output_dir / 'best_hparams_k2_tabula_rasa.json'}")
    logger.info(f"Saved best hyperparams to {output_dir / 'best_hparams_k3_tabula_rasa.json'}")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY — Hyperparameter Sensitivity Analysis")
    logger.info(f"{'=' * 70}")
    logger.info(
        "  Selection criterion: normalized AUCPC (area under normalized cost–quality curve)."
    )
    logger.info(
        "  Rewards contextual routing that finds the best quality at"
    )
    logger.info(
        "  every cost level.  See Appendix J for learning curves."
    )
    logger.info("")

    def _fmt(b: Dict) -> str:
        return (
            f"alpha={b['alpha']}, n_eff={int(b['n_eff'])}, "
            f"gamma={b['gamma']} "
            f"-> AUC={b['dev_val_pareto_auc']:.4f} "
            f"R@lam0={b['dev_val_mean_reward']:.4f} "
            f"(holdout: AUC={b['holdout_pareto_auc']:.4f} "
            f"R@lam0={b['holdout_mean_reward']:.4f})"
        )

    logger.info(f"  K=2 BanditGPT:   {_fmt(best_k2)}")
    logger.info(f"  K=2 Tabula Rasa: {_fmt(best_k2_tr)}")
    logger.info(f"  K=3 BanditGPT:   {_fmt(best_k3)}")
    logger.info(f"  K=3 Tabula Rasa: {_fmt(best_k3_tr)}")
    logger.info(f"\nElapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
