#!/usr/bin/env python3
"""
Appendix H: Hyperparameter Sensitivity Analysis
================================================

Sweeps three key hyperparameters (alpha, n_eff, forgetting_factor)
on a 4x4x4 grid to identify the best configuration for both BanditGPT
(Corralling + warmup priors) and Tabula Rasa (single LinUCB, no priors)
on the K=2 and K=10 portfolios.

Purpose
-------
This is a **hyperparameter tuning** experiment, not a warmup-vs-tabula-rasa
significance test.  Each variant is tuned independently so that downstream
experiments (Figure 3, Figure 4, Appendix J) use dev-val-selected
hyperparameters rather than defaults.

The final train-then-freeze reward is expected to be **similar** across
BanditGPT and Tabula Rasa given sufficient data — the advantage of warmup
priors is in **sample efficiency** (faster learning), not asymptotic
performance.  See Appendix J for the learning-curve comparison.

Protocol
--------
Train-then-freeze with lambda=0 (quality-only, no cost penalty).
For each (alpha, n_eff, gamma) triple:

1. Instantiate BanditRouter (Corralling + warmup, or single cold LinUCB).
2. Train on the dev-train split (N seeds, shuffled order).
3. Freeze the router (alpha=0 for greedy exploitation).
4. Evaluate on the dev-val split (selection metric).
5. After selecting the best config on dev-val, report its holdout score.
6. Record mean reward and 95% CI across seeds.

Grid
----
- alpha:             [0.1, 0.25, 0.5, 1.0]
- n_eff:             [10, 100, 1000, 5000]
- forgetting_factor: [0.995, 0.999, 0.9999, 1.0]
- Total: 64 configurations x 20 seeds = 1,280 trials per portfolio

Outputs (``results/``)
    alpha_neff_gamma_grid_results.json      (K=2 BanditGPT)
    alpha_neff_gamma_grid_figure.png        (K=2 BanditGPT)
    alpha_neff_gamma_grid_k10_results.json  (K=10 BanditGPT)
    alpha_neff_gamma_grid_k10_figure.png    (K=10 BanditGPT)
    alpha_neff_gamma_grid_tabula_rasa_*.json/png  (Tabula Rasa variants)
    best_hparams_{k2,k10}[_tabula_rasa].json  (selected configs)
"""

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
from scipy import stats as sp_stats
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K2_WARMUP_FROM_MULTIMODEL_PATH,
    K10_MODELS_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
)
from utils.rewards import extract_reward
from utils.router_factory import create_experiment_router
from utils.model_pricing import get_prices_for_models
from utils.embeddings import load_embedding_cache, embed_dataset_cached

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# Grid parameters
# ============================================================================

ALPHA_VALUES: List[float] = [0.1, 0.25, 0.5, 1.0]
NEFF_VALUES: List[float] = [10.0, 100.0, 1000.0, 5000.0]
GAMMA_VALUES: List[float] = [0.995, 0.999, 0.9999, 1.0]

N_SEEDS: int = 20
SEED_OFFSET: int = 42
CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05

# Dev train/val split for hyperparameter selection (holdout remains untouched).
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

# ── K=10 portfolio (loaded from canonical config) ────────────────────

def _load_k10_portfolio() -> Tuple[List[str], Dict[str, Dict]]:
    """Load K=10 model list and catalog from ``models_k10.json``."""
    with open(K10_MODELS_PATH) as f:
        k10_cfg = json.load(f)
    models = [m["model_id"] for m in k10_cfg["models"]]
    prices = get_prices_for_models(models)
    catalog: Dict[str, Dict] = {}
    for m_entry in k10_cfg["models"]:
        mid = m_entry["model_id"]
        catalog[mid] = {
            "display": m_entry.get("display", mid.split("/")[-1]),
            **prices[mid],
        }
    return models, catalog

K10_MODELS, K10_CATALOG = _load_k10_portfolio()

REWARD_THEORETICAL_MIN: float = 0.0
REWARD_THEORETICAL_MAX: float = 1.0


# ============================================================================
# Data loading (mirrors run_prequential.py)
# ============================================================================


def _req_cost(inp: float, out: float) -> float:
    """Per-request cost assuming 100 input + 400 output tokens."""
    return (100 * inp + 400 * out) / 1_000_000


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


_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}


def embed_dataset(
    data: List[Dict],
    encoder: "SentenceTransformer",
    pca: Any,
) -> List[np.ndarray]:
    """Embed all prompts, using the pre-computed cache when available."""
    return embed_dataset_cached(data, _EMBEDDING_CACHE, encoder, pca)


# ============================================================================
# Core evaluation
# ============================================================================


def _set_exploit_mode(router: Any, *, enable: bool) -> Dict[str, Any]:
    """Switch to greedy exploitation on a frozen router (expert + meta level)."""
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
    """Restore expert alpha values and meta exploit mode after evaluation."""
    if not saved:
        return
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts") and saved.get("expert_alphas"):
        for expert, (a_s, a_e) in zip(cr.experts, saved["expert_alphas"]):
            expert.alpha_start = a_s
            expert.alpha_end = a_e
        cr.exploit_mode = saved.get("meta_exploit", False)


def train_and_evaluate(
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
    n_seeds: int,
    use_corralling: bool = True,
) -> Dict[str, Any]:
    """Train-then-freeze evaluation for a single hyperparameter configuration.

    Args:
        models: Candidate model IDs.
        catalog: Model metadata catalog.
        train_data: Dev-set prompts with rewards.
        eval_data: Holdout prompts with rewards.
        train_emb: Pre-computed feature vectors for dev set.
        eval_emb: Pre-computed feature vectors for holdout set.
        warmup_path: Path to warmup priors file (None for tabula rasa).
        costs: Per-model cost dict.
        alpha: Exploration coefficient.
        n_eff: Prior effective sample size.
        gamma: Forgetting factor (1.0 = stationary).
        n_seeds: Number of random seeds.
        use_corralling: Enable Corralling meta-learner (False for tabula rasa).

    Returns:
        Dict with mean/std reward across seeds and per-seed values.
    """
    dim = train_emb[0].shape[0]
    r_min = REWARD_THEORETICAL_MIN
    r_range = REWARD_THEORETICAL_MAX - REWARD_THEORETICAL_MIN
    burn_in = len(train_data)
    registry = build_model_registry(models, catalog)

    trial_rewards: List[float] = []
    trial_costs: List[float] = []

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
            cost_penalty=0.0,
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
        trial_rewards.append(r_total / n)
        trial_costs.append(c_total / n)

    mean_r = float(np.mean(trial_rewards))
    std_r = float(np.std(trial_rewards, ddof=1)) if n_seeds > 1 else 0.0
    t_crit = float(sp_stats.t.ppf(0.975, df=max(n_seeds - 1, 1)))
    hw = t_crit * std_r / np.sqrt(n_seeds)

    return {
        "alpha": alpha,
        "n_eff": n_eff,
        "gamma": gamma,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "ci_lower": mean_r - hw,
        "ci_upper": mean_r + hw,
        "mean_cost": float(np.mean(trial_costs)),
        "per_seed_rewards": [float(x) for x in trial_rewards],
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
    n_seeds: int,
    use_corralling: bool = True,
) -> List[Dict[str, Any]]:
    """Sweep the 3D grid and return results for every configuration."""
    total = len(alpha_values) * len(neff_values) * len(gamma_values)
    results: List[Dict[str, Any]] = []
    idx = 0

    for gamma in gamma_values:
        for n_eff in neff_values:
            for alpha in alpha_values:
                idx += 1
                res = train_and_evaluate(
                    models, catalog,
                    train_data, eval_data, train_emb, eval_emb,
                    warmup_path, costs,
                    alpha=alpha, n_eff=n_eff, gamma=gamma,
                    n_seeds=n_seeds,
                    use_corralling=use_corralling,
                )
                results.append(res)
                logger.info(
                    f"  [{idx:3d}/{total}] "
                    f"alpha={alpha:<5} n_eff={n_eff:<7} gamma={gamma:<7} "
                    f"| R={res['mean_reward']:.4f} "
                    f"+/-{res['std_reward']:.4f} "
                    f"CI=[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
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
    oracle_reward: float,
    k_label: int = 2,
    filename: str = "alpha_neff_gamma_grid_figure.png",
    supervised_peak: Optional[float] = None,
) -> None:
    """Generate a 2x2 heatmap figure (one panel per forgetting factor).

    Each panel shows the alpha x n_eff grid of mean dev-val rewards.
    The global best cell is starred.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    lookup: Dict[Tuple[float, float, float], float] = {}
    for r in results:
        lookup[(r["alpha"], r["n_eff"], r["gamma"])] = r["mean_reward"]

    all_rewards = [r["mean_reward"] for r in results]
    vmin = min(all_rewards) - 0.002
    vmax = max(all_rewards) + 0.002
    norm = Normalize(vmin=vmin, vmax=vmax)

    n_panels = len(gamma_values)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11, 5 * nrows),
        constrained_layout=True,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for panel_idx, gamma in enumerate(gamma_values):
        ax = axes_flat[panel_idx]
        grid = np.zeros((len(neff_values), len(alpha_values)))
        for i, n_eff in enumerate(neff_values):
            for j, alpha in enumerate(alpha_values):
                grid[i, j] = lookup.get((alpha, n_eff, gamma), np.nan)

        im = ax.imshow(
            grid, aspect="auto", origin="lower", norm=norm,
            cmap="YlOrRd",
        )
        ax.set_xticks(range(len(alpha_values)))
        ax.set_xticklabels([str(a) for a in alpha_values], fontsize=9)
        ax.set_yticks(range(len(neff_values)))
        ax.set_yticklabels([str(int(n)) for n in neff_values], fontsize=9)
        ax.set_xlabel(r"$\alpha$ (exploration)", fontsize=10)
        ax.set_ylabel(r"$n_{\mathrm{eff}}$ (prior strength)", fontsize=10)

        gamma_label = f"{gamma}" if gamma < 1.0 else "1.0 (stationary)"
        ax.set_title(
            rf"$\gamma = {gamma_label}$",
            fontsize=11, fontweight="bold",
        )

        for i in range(len(neff_values)):
            for j in range(len(alpha_values)):
                val = grid[i, j]
                is_best = (
                    alpha_values[j] == best_config["alpha"]
                    and neff_values[i] == best_config["n_eff"]
                    and gamma == best_config["gamma"]
                )
                text = f"{val:.3f}"
                if is_best:
                    text += "\n\u2605"
                color = "white" if val > (vmin + vmax) / 2 else "black"
                ax.text(
                    j, i, text, ha="center", va="center",
                    fontsize=8, fontweight="bold" if is_best else "normal",
                    color=color,
                )

    for panel_idx in range(n_panels, len(axes_flat)):
        axes_flat[panel_idx].set_visible(False)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd"),
        ax=axes_flat[:n_panels], shrink=0.8, label="Mean Dev-Val Reward",
    )

    fig.suptitle(
        r"3D Ablation: $\alpha \times n_{\mathrm{eff}} \times \gamma$ "
        f"(K={k_label}, {results[0]['n_seeds']} seeds)\n"
        f"Best: alpha={best_config['alpha']}, "
        f"n_eff={int(best_config['n_eff'])}, "
        f"gamma={best_config['gamma']} "
        f"-> R_val={best_config['mean_reward']:.4f} "
        f"[{best_config['ci_lower']:.4f}, {best_config['ci_upper']:.4f}]",
        fontsize=12, fontweight="bold",
    )

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
    encoder: "SentenceTransformer",
    pca: Any,
    warmup_path: Optional[str],
    output_dir: Path,
    *,
    k_label: int,
    json_filename: str,
    figure_filename: str,
    main_results_key: str,
    use_corralling: bool = True,
    variant_label: str = "BanditGPT",
) -> Dict[str, Any]:
    """Run the full 3D grid ablation for a single portfolio.

    Args:
        models: Candidate model IDs for this portfolio.
        catalog: Model metadata catalog.
        train_data: Dev/online-learn prompts with rewards.
        holdout_data: Holdout prompts with rewards.
        encoder: SentenceTransformer encoder.
        pca: Fitted PCA transform.
        warmup_path: Path to the warmup priors file (None for tabula rasa).
        output_dir: Directory for output artifacts.
        k_label: Portfolio size (2 or 10), used in labels.
        json_filename: Filename for the JSON results.
        figure_filename: Filename for the heatmap figure.
        main_results_key: Top-level key in prequential_results.json
            (e.g. "K2" or "K10").
        use_corralling: Enable Corralling meta-learner (False for tabula rasa).
        variant_label: Human-readable label for this variant (for logging).

    Returns:
        Dict with best configuration and summary statistics.
    """
    costs = {
        m: _req_cost(
            catalog[m]["input_cost_per_m"],
            catalog[m]["output_cost_per_m"],
        )
        for m in models
    }

    logger.info(
        f"  K={k_label}: {len(train_data)} dev (online-learn pool), "
        f"{len(holdout_data)} holdout prompts"
    )

    logger.info(f"  Embedding K={k_label} prompts ...")
    dev_emb = embed_dataset(train_data, encoder, pca)
    holdout_emb = embed_dataset(holdout_data, encoder, pca)
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
    logger.info(
        f"\n  3D grid: {total_configs} configs x {N_SEEDS} seeds "
        f"= {total_configs * N_SEEDS} trials"
    )

    # Selection happens on dev-val only (holdout remains untouched).
    results = run_grid(
        models, catalog,
        dev_train, dev_val, dev_train_emb, dev_val_emb,
        warmup_path, costs,
        alpha_values=ALPHA_VALUES,
        neff_values=NEFF_VALUES,
        gamma_values=GAMMA_VALUES,
        n_seeds=N_SEEDS,
        use_corralling=use_corralling,
    )

    ranked = sorted(results, key=lambda r: r["mean_reward"], reverse=True)
    best = ranked[0]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"K={k_label} {variant_label} TOP-10 CONFIGURATIONS (by mean dev-val reward)")
    logger.info(f"{'=' * 70}")
    for i, r in enumerate(ranked[:10]):
        logger.info(
            f"  #{i + 1:2d}  alpha={r['alpha']:<5} n_eff={r['n_eff']:<7} "
            f"gamma={r['gamma']:<7} "
            f"| R_val={r['mean_reward']:.4f} +/-{r['std_reward']:.4f} "
            f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        )
    logger.info(f"\n  Oracle (per-prompt best): {oracle_reward:.4f}")
    if supervised_peak is not None:
        logger.info(f"  Supervised peak:         {supervised_peak:.4f}")
    logger.info(
        f"\n  BEST (dev-val): alpha={best['alpha']}, n_eff={int(best['n_eff'])}, "
        f"gamma={best['gamma']} -> R_val={best['mean_reward']:.4f}"
    )

    # Final report: evaluate the selected config on holdout (not used for selection).
    logger.info("\n  Evaluating selected config once on holdout ...")
    best_holdout = train_and_evaluate(
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
        n_seeds=N_SEEDS,
        use_corralling=use_corralling,
    )
    logger.info(
        f"  Holdout reward for selected config: {best_holdout['mean_reward']:.4f} "
        f"+/-{best_holdout['std_reward']:.4f}"
    )

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": (
                "3D hyperparameter ablation (alpha x n_eff x forgetting_factor) "
                f"for {variant_label} "
                f"({'Corralling + warmup priors' if use_corralling else 'single LinUCB, no priors'}). "
                f"Train-then-freeze on K={k_label}, lambda=0 (quality-only)."
            ),
        },
        "config": {
            "alpha_values": ALPHA_VALUES,
            "neff_values": NEFF_VALUES,
            "gamma_values": GAMMA_VALUES,
            "n_seeds": N_SEEDS,
            "corralling_lr": CORRALLING_LR,
            "corralling_gamma": CORRALLING_GAMMA,
            "cost_penalty": 0.0,
            "dev_val_fraction": DEV_VAL_FRACTION,
            "dev_val_seed": DEV_VAL_SEED,
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
            "dev_val_mean_reward": best["mean_reward"],
            "dev_val_ci_lower": best["ci_lower"],
            "dev_val_ci_upper": best["ci_upper"],
            "holdout_mean_reward": best_holdout["mean_reward"],
            "holdout_ci_lower": best_holdout["ci_lower"],
            "holdout_ci_upper": best_holdout["ci_upper"],
        },
        "ranked_top10": [
            {k: v for k, v in r.items() if k != "per_seed_rewards"}
            for r in ranked[:10]
        ],
        "full_grid": results,
    }

    out_path = output_dir / json_filename
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults -> {out_path}")

    plot_grid(
        results, ALPHA_VALUES, NEFF_VALUES, GAMMA_VALUES,
        output_dir,
        best_config=best,
        oracle_reward=oracle_reward,
        k_label=k_label,
        filename=figure_filename,
        supervised_peak=supervised_peak,
    )

    return {
        "alpha": float(best["alpha"]),
        "n_eff": float(best["n_eff"]),
        "gamma": float(best["gamma"]),
        "dev_val_mean_reward": float(best["mean_reward"]),
        "holdout_mean_reward": float(best_holdout["mean_reward"]),
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

    global _EMBEDDING_CACHE  # noqa: PLW0603
    _EMBEDDING_CACHE = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    total_configs = (
        len(ALPHA_VALUES) * len(NEFF_VALUES) * len(GAMMA_VALUES)
    )
    logger.info(
        f"Grid: {total_configs} configs x {N_SEEDS} seeds "
        f"= {total_configs * N_SEEDS} trials per portfolio"
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

    dev_k2 = load_rewards_from_file(CANONICAL_DEV_DATA_PATH, K2_MODELS)
    holdout_k2 = load_rewards_from_file(
        CANONICAL_HOLDOUT_DATA_PATH, K2_MODELS,
    )

    best_k2 = run_portfolio_ablation(
        K2_MODELS, K2_CATALOG,
        dev_k2, holdout_k2, encoder, pca,
        str(K2_WARMUP_FROM_MULTIMODEL_PATH),
        output_dir,
        k_label=2,
        json_filename="alpha_neff_gamma_grid_results.json",
        figure_filename="alpha_neff_gamma_grid_figure.png",
        main_results_key="K2",
    )

    # ── K=10 ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("K=10 Portfolio Ablation")
    logger.info("=" * 70)

    prior_train_prompts: Set[str] = set()
    if THREE_WAY_SPLITS_PATH.exists():
        with open(THREE_WAY_SPLITS_PATH) as f:
            splits_3way = json.load(f)
        prior_train_prompts = set(splits_3way.get("prior_train_pool", []))

    all_dev_k10 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K10_MODELS)
    train_k10 = [
        d for d in all_dev_k10 if d["prompt"] not in prior_train_prompts
    ]
    holdout_k10 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K10_MODELS,
    )

    best_k10 = run_portfolio_ablation(
        K10_MODELS, K10_CATALOG,
        train_k10, holdout_k10, encoder, pca,
        str(MULTIMODEL_WARMUP_PRIORS_PATH),
        output_dir,
        k_label=10,
        json_filename="alpha_neff_gamma_grid_k10_results.json",
        figure_filename="alpha_neff_gamma_grid_k10_figure.png",
        main_results_key="K10",
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
        k_label=2,
        json_filename="alpha_neff_gamma_grid_tabula_rasa_results.json",
        figure_filename="alpha_neff_gamma_grid_tabula_rasa_figure.png",
        main_results_key="K2",
        use_corralling=False,
        variant_label="Tabula Rasa",
    )

    # ── K=10 Tabula Rasa ────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("K=10 Tabula Rasa Portfolio Ablation")
    logger.info("=" * 70)

    best_k10_tr = run_portfolio_ablation(
        K10_MODELS, K10_CATALOG,
        train_k10, holdout_k10, encoder, pca,
        None,
        output_dir,
        k_label=10,
        json_filename="alpha_neff_gamma_grid_tabula_rasa_k10_results.json",
        figure_filename="alpha_neff_gamma_grid_tabula_rasa_k10_figure.png",
        main_results_key="K10",
        use_corralling=False,
        variant_label="Tabula Rasa",
    )

    # Persist the selected hyperparameters for downstream experiments.
    # These are dev-val-selected (holdout not used for selection).
    (output_dir / "best_hparams_k2.json").write_text(
        json.dumps({"K2": best_k2}, indent=2)
    )
    (output_dir / "best_hparams_k10.json").write_text(
        json.dumps({"K10": best_k10}, indent=2)
    )
    (output_dir / "best_hparams_k2_tabula_rasa.json").write_text(
        json.dumps({"K2": best_k2_tr}, indent=2)
    )
    (output_dir / "best_hparams_k10_tabula_rasa.json").write_text(
        json.dumps({"K10": best_k10_tr}, indent=2)
    )
    logger.info(f"\nSaved best hyperparams to {output_dir / 'best_hparams_k2.json'}")
    logger.info(f"Saved best hyperparams to {output_dir / 'best_hparams_k10.json'}")
    logger.info(f"Saved best hyperparams to {output_dir / 'best_hparams_k2_tabula_rasa.json'}")
    logger.info(f"Saved best hyperparams to {output_dir / 'best_hparams_k10_tabula_rasa.json'}")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY — Hyperparameter Sensitivity Analysis")
    logger.info(f"{'=' * 70}")
    logger.info(
        "  Each variant is tuned independently.  Final reward similarity"
    )
    logger.info(
        "  is expected — the warmup advantage is in sample efficiency,"
    )
    logger.info(
        "  not asymptotic performance.  See Appendix J for learning curves."
    )
    logger.info("")
    logger.info(
        f"  K=2  BanditGPT best: alpha={best_k2['alpha']}, "
        f"n_eff={int(best_k2['n_eff'])}, "
        f"gamma={best_k2['gamma']} "
        f"-> R_val={best_k2['dev_val_mean_reward']:.4f} "
        f"(holdout={best_k2['holdout_mean_reward']:.4f})"
    )
    logger.info(
        f"  K=2  Tabula Rasa best: alpha={best_k2_tr['alpha']}, "
        f"n_eff={int(best_k2_tr['n_eff'])}, "
        f"gamma={best_k2_tr['gamma']} "
        f"-> R_val={best_k2_tr['dev_val_mean_reward']:.4f} "
        f"(holdout={best_k2_tr['holdout_mean_reward']:.4f})"
    )
    logger.info(
        f"  K=10 BanditGPT best: alpha={best_k10['alpha']}, "
        f"n_eff={int(best_k10['n_eff'])}, "
        f"gamma={best_k10['gamma']} "
        f"-> R_val={best_k10['dev_val_mean_reward']:.4f} "
        f"(holdout={best_k10['holdout_mean_reward']:.4f})"
    )
    logger.info(
        f"  K=10 Tabula Rasa best: alpha={best_k10_tr['alpha']}, "
        f"n_eff={int(best_k10_tr['n_eff'])}, "
        f"gamma={best_k10_tr['gamma']} "
        f"-> R_val={best_k10_tr['dev_val_mean_reward']:.4f} "
        f"(holdout={best_k10_tr['holdout_mean_reward']:.4f})"
    )
    logger.info(f"\nElapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
