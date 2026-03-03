#!/usr/bin/env python3
"""
Alpha (exploration) sensitivity analysis for BanditGPT learning curves.

Runs the K=2 learning curve at multiple exploration coefficients to show:
  - How alpha affects the early-step peak/dip artifact
  - Whether converged (step 1000+) performance is robust to alpha choice
  - The optimal exploration coefficient for this setup

This is a standard ablation for any UCB-based method and addresses a
common KDD reviewer concern: "Is the result sensitive to the exploration
parameter?"

No new API calls are required -- all evaluation uses pre-computed data.

Outputs (``results/``):
    alpha_ablation_results.json   — per-alpha learning curves
    figure_alpha_ablation.png     — publication-ready ablation figure
"""

import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from scipy import stats as sp_stats
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
)
from utils.rewards import extract_reward
from utils.router_factory import create_experiment_router
from utils.model_pricing import get_prices_for_models

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

ALPHA_VALUES: List[float] = [0.5, 1.0, 2.0, 4.0]

N_SEEDS: int = 20
SEED_OFFSET: int = 42
TARGET_NEFF: float = 10.0
CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05

CHECKPOINTS: List[int] = [
    0, 10, 25, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
]

DEV_VAL_FRACTION: float = 0.2
DEV_VAL_SEED: int = 7

K2_MODELS: List[str] = [
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-4-turbo",
]

_PRICES_K2 = get_prices_for_models(K2_MODELS)

K2_CATALOG: Dict[str, Dict] = {
    "mistralai/mixtral-8x7b-instruct": {
        "display": "Mixtral-8x7B",
        # Prices are loaded from experiments/config/model_prices.json.
        **_PRICES_K2["mistralai/mixtral-8x7b-instruct"],
    },
    "openai/gpt-4-turbo": {
        "display": "GPT-4-Turbo",
        **_PRICES_K2["openai/gpt-4-turbo"],
    },
}


def _req_cost(inp: float, out: float) -> float:
    """Per-request cost assuming 100 input + 400 output tokens."""
    return (100 * inp + 400 * out) / 1_000_000


# ============================================================================
# Data loading (mirrors run_prequential.py)
# ============================================================================


def load_rewards_from_file(
    data_path: Path,
    models: List[str],
) -> List[Dict]:
    """Load rewards for specific models from gzipped JSONL."""
    model_set = set(models)
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)

    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            prompt = entry["prompt"]
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


def embed_dataset(
    data: List[Dict],
    encoder: "SentenceTransformer",
    pca,
) -> List[np.ndarray]:
    """Embed all prompts, returning aligned feature vectors."""
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


def _split_dev_train_val(
    data: List[Dict],
    emb: List[np.ndarray],
    val_fraction: float = DEV_VAL_FRACTION,
    seed: int = DEV_VAL_SEED,
) -> Tuple[List[Dict], List[np.ndarray], List[Dict], List[np.ndarray]]:
    """Deterministically split (data, emb) into train and val portions.

    Uses the same split seed and fraction as ``run_prequential.py`` so
    the learning curves trained here are directly comparable to the
    Pareto sweep in the main experiment.

    Returns:
        (train_data, train_emb, val_data, val_emb).
    """
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
# Core evaluation
# ============================================================================


def _compute_reward_normalization(
    data: List[Dict], models: List[str],
) -> Tuple[float, float]:
    """Return theoretical reward bounds for normalisation.

    ``extract_reward()`` returns mean(vote × confidence) ∈ [0, 1].
    Using theoretical bounds avoids information leakage from the
    counterfactual reward matrix.
    """
    return 0.0, 1.0


def _set_exploit_mode(router, *, enable: bool) -> List[Tuple[float, float]]:
    """Zero-out UCB alpha on all Corralling experts for greedy eval."""
    if not enable:
        return []
    saved: List[Tuple[float, float]] = []
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts"):
        for expert in cr.experts:
            saved.append((expert.alpha_start, expert.alpha_end))
            expert.alpha_start = 0.0
            expert.alpha_end = 0.0
    return saved


def _restore_exploit_mode(
    router, saved: List[Tuple[float, float]],
) -> None:
    """Restore expert alpha values after greedy evaluation."""
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts") and saved:
        for expert, (a_s, a_e) in zip(cr.experts, saved):
            expert.alpha_start = a_s
            expert.alpha_end = a_e


def evaluate_frozen(
    router,
    eval_data: List[Dict],
    eval_embeddings: List[np.ndarray],
    costs: Dict[str, float],
    total_steps: int,
) -> Tuple[float, float]:
    """Evaluate a frozen router on the holdout set (no learning).

    Uses greedy exploitation (alpha=0) during evaluation.

    Returns:
        (mean_reward, mean_cost).
    """
    saved = _set_exploit_mode(router, enable=True)
    rng_state = np.random.get_state()
    r_total = c_total = 0.0

    for p, x in zip(eval_data, eval_embeddings):
        model, _log = router.route(x, total_steps=total_steps)
        r_total += p["rewards"][model]
        c_total += costs[model]

    np.random.set_state(rng_state)
    _restore_exploit_mode(router, saved)
    n = len(eval_data)
    return r_total / n, c_total / n


def run_learning_curve(
    *,
    models: List[str],
    catalog: Dict[str, Dict],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_path: str,
    costs: Dict[str, float],
    n_trials: int,
    checkpoints: List[int],
    alpha: float,
    label: str,
) -> List[Dict]:
    """Learning curve: holdout quality as a function of online training steps.

    At each checkpoint, the router is frozen and evaluated on the full
    holdout set.  Step 0 evaluates with priors only (no online data).

    Args:
        models: Candidate model IDs.
        catalog: Model metadata catalog.
        train_data: Dev-set prompts with rewards.
        eval_data: Holdout-set prompts with rewards.
        train_emb: Pre-computed embeddings for dev set.
        eval_emb: Pre-computed embeddings for holdout set.
        warmup_path: Path to warmup priors file.
        costs: Per-model cost dict.
        n_trials: Number of random seeds.
        checkpoints: Training steps at which to evaluate.
        alpha: Exploration coefficient passed to router creation.
        label: Label for the curve in output data.

    Returns:
        List of dicts, one per checkpoint, with mean/std reward.
    """
    dim = train_emb[0].shape[0]
    r_min, r_max = _compute_reward_normalization(train_data, models)
    r_range = r_max - r_min
    burn_in = len(train_data)
    checkpoint_set = set(checkpoints)

    by_step: Dict[int, Dict[str, List[float]]] = {
        s: {"rewards": [], "costs": []} for s in checkpoints
    }

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=build_model_registry(models, catalog),
            feature_dim=dim,
            prior_n_effective=TARGET_NEFF,
            alpha=alpha,
            warmup_path=warmup_path,
            use_corralling=True,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
            cost_penalty=0.0,
        )

        if 0 in checkpoint_set:
            r, c = evaluate_frozen(
                router, eval_data, eval_emb, costs, burn_in,
            )
            by_step[0]["rewards"].append(r)
            by_step[0]["costs"].append(c)

        order = np.random.permutation(len(train_data))
        for step_idx, idx in enumerate(order):
            p, x = train_data[idx], train_emb[idx]
            model, log = router.route(x, total_steps=burn_in)
            raw_reward = p["rewards"][model]
            norm_reward = (
                (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
            )
            router.process_feedback(log.request_id, norm_reward)
            current = step_idx + 1
            if current in checkpoint_set:
                r, c = evaluate_frozen(
                    router, eval_data, eval_emb, costs, burn_in,
                )
                by_step[current]["rewards"].append(r)
                by_step[current]["costs"].append(c)

        if (trial + 1) % 5 == 0:
            logger.info(
                f"      alpha={alpha} trial {trial + 1}/{n_trials}"
            )

    curve = []
    for s in sorted(checkpoints):
        rr = by_step[s]["rewards"]
        if rr:
            curve.append({
                "step": s,
                "mean_reward": float(np.mean(rr)),
                "std_reward": (
                    float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
                ),
                "n_trials": len(rr),
                "label": label,
            })
    return curve


# ============================================================================
# Plotting
# ============================================================================

COLORS = {
    0.5: "#009E73",   # green
    1.0: "#E69F00",   # orange
    2.0: "#0072B2",   # blue (default, matches main figure)
    4.0: "#CC79A7",   # purple
}
MARKERS = {0.5: "s", 1.0: "^", 2.0: "D", 4.0: "o"}


def plot_ablation(
    results: Dict[str, Any],
    out: Path,
) -> None:
    """Generate a single-panel alpha ablation figure.

    Shows learning curves for each alpha value with 95% CI bands,
    plus the RouteLLM peak reference line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_seeds = results["n_seeds"]
    t_crit = float(sp_stats.t.ppf(0.975, df=n_seeds - 1))

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    rl_peak = results.get("routellm_peak")
    if rl_peak is not None:
        ax.axhline(
            y=rl_peak, color="#D55E00", ls="--", lw=2, alpha=0.8,
            zorder=3, label=f"RouteLLM peak ({rl_peak:.3f})",
        )

    weak_r = results.get("weak_model_reward")
    if weak_r is not None:
        ax.axhline(
            y=weak_r, color="#999999", ls=":", lw=1.5, alpha=0.6,
            zorder=3, label=f"Weak model ({weak_r:.3f})",
        )

    for alpha_str, curve in sorted(results["curves"].items(),
                                    key=lambda kv: float(kv[0])):
        alpha_val = float(alpha_str)
        steps = [d["step"] for d in curve]
        rewards = [d["mean_reward"] for d in curve]
        stds = [d["std_reward"] for d in curve]

        ci_hi = [
            r + t_crit * s / np.sqrt(n_seeds)
            for r, s in zip(rewards, stds)
        ]
        ci_lo = [
            r - t_crit * s / np.sqrt(n_seeds)
            for r, s in zip(rewards, stds)
        ]

        color = COLORS.get(alpha_val, "#333333")
        marker = MARKERS.get(alpha_val, ".")
        is_default = alpha_val == 0.5
        lw = 2.5 if is_default else 1.8
        label_suffix = " (default)" if is_default else ""

        ax.plot(
            steps, rewards, f"{marker}-", color=color, lw=lw, ms=4,
            zorder=5 if is_default else 4,
            label=fr"$\alpha$={alpha_val}{label_suffix}",
        )
        ax.fill_between(
            steps, ci_lo, ci_hi, color=color,
            alpha=0.10 if is_default else 0.07, zorder=2,
        )

    ax.set_xlabel(
        "Online Learning Steps (dev prompts seen)",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel(
        f"Holdout Quality (frozen eval, n={results.get('n_holdout', '?')})",
        fontsize=11, fontweight="bold",
    )
    ax.set_title(
        r"Exploration Coefficient ($\alpha$) Sensitivity — K=2",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.15, ls="--")
    max_step = max(
        d["step"]
        for curve in results["curves"].values()
        for d in curve
    )
    ax.set_xlim(-30, max_step + 50)
    ax.tick_params(labelsize=9)

    path = out / "figure_alpha_ablation.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Load shared resources ---
    logger.info("Loading encoder and PCA ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    costs = {
        m: _req_cost(
            K2_CATALOG[m]["input_cost_per_m"],
            K2_CATALOG[m]["output_cost_per_m"],
        )
        for m in K2_MODELS
    }

    logger.info("Loading K=2 dev and holdout data ...")
    dev_data = load_rewards_from_file(CANONICAL_DEV_DATA_PATH, K2_MODELS)
    holdout_data = load_rewards_from_file(
        CANONICAL_HOLDOUT_DATA_PATH, K2_MODELS,
    )
    logger.info(f"  Dev: {len(dev_data)} prompts")
    logger.info(f"  Holdout: {len(holdout_data)} prompts")

    logger.info("Embedding prompts ...")
    dev_emb = embed_dataset(dev_data, encoder, pca)
    holdout_emb = embed_dataset(holdout_data, encoder, pca)

    # Split dev into train/val (same split as run_prequential.py) so that
    # the alpha ablation learning curves are trained on the same dev-train
    # subset used for the Pareto sweep, ensuring a fair comparison against
    # the RouteLLM peak reference line.
    logger.info(
        f"  Splitting dev into train/val "
        f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ..."
    )
    dev_train, dev_train_emb, dev_val, dev_val_emb = _split_dev_train_val(
        dev_data, dev_emb,
    )
    logger.info(f"    Dev-train: {len(dev_train)}  Dev-val: {len(dev_val)}")

    max_step = len(dev_train)
    checkpoints = [s for s in CHECKPOINTS if s <= max_step]
    if max_step not in checkpoints:
        checkpoints.append(max_step)

    # --- Load RouteLLM peak from main results (if available) ---
    main_results_path = output_dir / "prequential_results.json"
    rl_peak = None
    weak_r = None
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_res = json.load(f)
        k2 = main_res.get("K2", {})
        pareto = k2.get("routellm", {}).get("pareto", [])
        if pareto:
            rl_peak = max(p["avg_reward"] for p in pareto)
        static = k2.get("static", {})
        if static:
            weak_r = min(s["reward"] for s in static.values())
        logger.info(
            f"  Loaded baselines from main results: "
            f"RouteLLM peak={rl_peak}, weak={weak_r}"
        )

    # --- Run ablation ---
    logger.info(
        f"\nAlpha ablation: {len(ALPHA_VALUES)} values x "
        f"{N_SEEDS} seeds x {len(checkpoints)} checkpoints"
    )
    curves: Dict[str, List[Dict]] = {}
    for alpha_val in ALPHA_VALUES:
        logger.info(f"\n  alpha={alpha_val} ...")
        curve = run_learning_curve(
            models=K2_MODELS,
            catalog=K2_CATALOG,
            train_data=dev_train,
            eval_data=holdout_data,
            train_emb=dev_train_emb,
            eval_emb=holdout_emb,
            warmup_path=str(DEFAULT_WARMUP_PRIORS_PATH),
            costs=costs,
            n_trials=N_SEEDS,
            checkpoints=checkpoints,
            alpha=alpha_val,
            label=f"alpha={alpha_val}",
        )
        curves[str(alpha_val)] = curve

        final = curve[-1] if curve else {}
        logger.info(
            f"    Final (step {final.get('step', '?')}): "
            f"R={final.get('mean_reward', 0):.4f} "
            f"+/- {final.get('std_reward', 0):.4f}"
        )

    # --- Assemble & save results ---
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": (
                "Alpha (exploration coefficient) sensitivity analysis "
                "for BanditGPT Corralling router with warmup priors. "
                "Each curve shows holdout quality after N online learning "
                "steps on the dev-train split (80% of dev), averaged "
                "over N_SEEDS random presentation orders.  The dev-train "
                "split matches run_prequential.py for symmetric comparison."
            ),
        },
        "config": {
            "alpha_values": ALPHA_VALUES,
            "n_seeds": N_SEEDS,
            "checkpoints": checkpoints,
            "corralling_lr": CORRALLING_LR,
            "corralling_gamma": CORRALLING_GAMMA,
            "target_neff": TARGET_NEFF,
            "dev_val_fraction": DEV_VAL_FRACTION,
            "dev_val_seed": DEV_VAL_SEED,
        },
        "n_seeds": N_SEEDS,
        "n_dev": len(dev_data),
        "n_dev_train": len(dev_train),
        "n_holdout": len(holdout_data),
        "routellm_peak": rl_peak,
        "weak_model_reward": weak_r,
        "curves": curves,
    }

    out_path = output_dir / "alpha_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults -> {out_path}")

    # --- Generate figure ---
    plot_ablation(results, output_dir)

    elapsed = time.time() - t0
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
