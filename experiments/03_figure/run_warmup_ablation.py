#!/usr/bin/env python3
"""Experiment 3 / RQ2: Cold-Start Prior Value — Learning Curves.

Measures how warmup priors eliminate cold-start regret by comparing two
conditions that start from scratch on held-out test data:

- **BanditGPT (warm priors):** Disjoint LinUCB initialised with offline
  priors from the training split.  Uses alpha=0.5, N_eff=5000 — the same
  hyperparameters used in Figures 1 and 4 for cross-figure consistency.
- **Tabula Rasa (no priors):** Same Disjoint LinUCB architecture but
  starting from scratch (A=lambda*I, b=0).  Uses alpha=0.50 — matching
  BEST_K2_TABULA_RASA_HPARAMS for consistency across experiments.

Both conditions use the same:
  - alpha=0.5 exploration rate (isolates the effect of priors, not tuning)
  - Seeds, shuffle order, and number of seeds
  - Feature pipeline (PCA-25, all-MiniLM-L6-v2)
  - Cost penalty grid (default lambda=0.15)
  - No Corralling (isolating the effect of priors alone)

Protocol
--------
Evaluate on the held-out test split (n=1,824) from cold start — NO
pre-training phase.  The router is deployed fresh and must learn
purely from online partial feedback, prompt by prompt.  With warmup
priors the router starts with informative arm estimates from offline
full-information evaluation; without them it starts from scratch.

This directly measures the production value proposition of warmup
priors: when deploying a new router, how much cold-start cost do
priors eliminate?

Fairness notes
--------------
- **Priors source:** warmup priors are built offline from the training
  split (full-information ridge regression).  The test split used for
  evaluation was not used for prior construction or tuning.
- **Same exploration rate:** both conditions use alpha=0.5, the
  production value.  This isolates the effect of priors: the only
  difference is whether the bandit starts with informed arm estimates.
- **Shuffle parity:** both conditions share the same per-seed shuffle
  order (same ``rng`` initialised from the same seed), ensuring identical
  prompt presentation sequences.

Outputs
-------
``results/figure3_warmup_ablation.pdf``
    Two-panel figure: (a) cumulative regret showing cold-start gap,
    (b) windowed reward showing convergence dynamics.

``results/warmup_ablation_data.json``
    Machine-readable metrics including cold-start regret, convergence
    speed, and checkpoint-level statistics.

Usage
-----
    python experiments/03_figure/run_warmup_ablation.py
    python experiments/03_figure/run_warmup_ablation.py --n-seeds 5
    python experiments/03_figure/run_warmup_ablation.py --fast
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    HOLDOUT_DATA_PATH,
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.simulation import (
    SplitData,
    build_model_registry,
    load_split,
    CB_BLUE,
    CB_GRAY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

for _noisy in (
    "bandit_gpt.router",
    "bandit_gpt.router_v2",
    "bandit_gpt.feature_service",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"

ARM_ORDER = K2_ARM_ORDER
ARM_LABELS = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
}

# ── Hyperparameters: same as Figures 1 & 4 for cross-figure consistency ─
# Both conditions use the same alpha=0.5 to isolate the effect of priors.
WARMUP_HPARAMS: Dict[str, Any] = {
    "alpha": 0.5,
    "prior_n_effective": 5000.0,
    "policy": "disjoint",
    "use_corralling": False,
    "forgetting_factor": 1.0,
}

TABULA_RASA_HPARAMS: Dict[str, Any] = {
    "alpha": 0.50,
    "prior_n_effective": 1.0,
    "policy": "tabula_rasa",
    "use_corralling": False,
    "forgetting_factor": 1.0,
}

CHECKPOINT_INTERVAL = 25
WINDOW_SIZE = 100
SEED_OFFSET = 1000

CONVERGENCE_THRESHOLD = 0.005


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Checkpoint:
    """Snapshot of routing statistics at a given prompt count."""

    n_seen: int
    cumulative_reward: float
    cumulative_cost: float
    windowed_reward: float
    routing_mix: Dict[str, float]
    oracle_cumulative_reward: float


@dataclass
class AblationResult:
    """Full results for a single seed of one experimental condition."""

    condition: str
    seed: int
    checkpoints: List[Checkpoint]
    rewards: np.ndarray
    costs: np.ndarray
    choices: np.ndarray
    oracle_rewards: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════
# Router Construction
# ═══════════════════════════════════════════════════════════════════════════


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    hparams: Dict[str, Any],
    warmup_path: str,
    cost_penalty: float,
    *,
    use_priors: bool,
) -> BanditRouter:
    """Create a BanditRouter with or without warmup priors.

    Args:
        registry: Model registry (K=2 arms).
        feature_dim: Dimensionality of feature vectors.
        hparams: Tuned hyperparameters for this condition.
        warmup_path: Path to warmup priors joblib file.
        cost_penalty: Cost penalty weight lambda.
        use_priors: Whether to initialise with warmup priors.

    Returns:
        Fully initialised router.
    """
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if use_priors else "none",
        warmup_path=warmup_path if use_priors else None,
        prior_n_effective=hparams["prior_n_effective"],
        alpha=hparams["alpha"],
        use_corralling=False,
        cost_penalty=cost_penalty,
        forgetting_factor=hparams["forgetting_factor"],
        policy="disjoint",
    )
    return router


# ═══════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════


def simulate_cold_start(
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    hparams: Dict[str, Any],
    warmup_path: str,
    cost_penalty: float,
    seed: int,
    use_priors: bool,
    condition_name: str,
) -> AblationResult:
    """Run a cold-start simulation on the test split.

    The router is created fresh (with or without priors) and immediately
    deployed on the held-out test split.  No pre-training phase — this
    measures the router's out-of-the-box routing quality.

    Args:
        test: Test split (K=2).
        registry: Model registry.
        feature_dim: Feature dimensionality.
        hparams: Condition-specific hyperparameters.
        warmup_path: Warmup priors path.
        cost_penalty: Cost penalty lambda.
        seed: Random seed for shuffle order.
        use_priors: Whether to use warmup priors.
        condition_name: Label for this condition.

    Returns:
        Complete per-prompt results with periodic checkpoints.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    router = _create_router(
        registry, feature_dim, hparams, warmup_path,
        cost_penalty, use_priors=use_priors,
    )

    arm_to_idx = {arm: i for i, arm in enumerate(ARM_ORDER)}
    n_test = test.n
    rewards = np.zeros(n_test)
    costs = np.zeros(n_test)
    choices = np.zeros(n_test, dtype=np.int32)
    oracle_rewards = np.zeros(n_test)

    checkpoints: List[Checkpoint] = []
    arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
    cum_reward = 0.0
    cum_cost = 0.0
    cum_oracle = 0.0
    recent_rewards: deque[float] = deque(maxlen=WINDOW_SIZE)

    eval_idx = rng.permutation(n_test)
    for j, i in enumerate(eval_idx):
        emb = test.embeddings[i]
        model, log = router.route(emb)
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        oracle_r = max(float(test.rewards[a][i]) for a in ARM_ORDER)
        router.process_feedback(log.request_id, reward=reward)

        rewards[j] = reward
        costs[j] = cost
        choices[j] = arm_to_idx[model]
        oracle_rewards[j] = oracle_r

        arm_counts[model] += 1
        cum_reward += reward
        cum_cost += cost
        cum_oracle += oracle_r
        recent_rewards.append(reward)
        step = j + 1

        if step % CHECKPOINT_INTERVAL == 0 or step == n_test:
            mix = {a: arm_counts[a] / step for a in ARM_ORDER}
            checkpoints.append(Checkpoint(
                n_seen=step,
                cumulative_reward=cum_reward / step,
                cumulative_cost=cum_cost / step,
                windowed_reward=float(np.mean(recent_rewards)),
                routing_mix=dict(mix),
                oracle_cumulative_reward=cum_oracle / step,
            ))

    return AblationResult(
        condition=condition_name,
        seed=seed,
        checkpoints=checkpoints,
        rewards=rewards,
        costs=costs,
        choices=choices,
        oracle_rewards=oracle_rewards,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_curves(
    results: List[AblationResult],
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    window_size: int = WINDOW_SIZE,
) -> Dict[str, Dict[str, Any]]:
    """Compute cumulative regret and windowed reward curves per condition.

    Args:
        results: All seed results for all conditions.
        checkpoint_interval: Spacing between curve checkpoints.
        window_size: Window size for windowed reward.

    Returns:
        ``{condition: {"xs": array, "cum_regret_mean": array, ...}}``
    """
    by_cond: Dict[str, List[AblationResult]] = {}
    for r in results:
        by_cond.setdefault(r.condition, []).append(r)

    curves: Dict[str, Dict[str, Any]] = {}
    for cond, runs in by_cond.items():
        n_test = len(runs[0].rewards)
        n_seeds = len(runs)
        cp_indices = list(range(
            checkpoint_interval - 1, n_test, checkpoint_interval,
        ))
        if (n_test - 1) not in cp_indices:
            cp_indices.append(n_test - 1)
        xs = np.array([i + 1 for i in cp_indices])

        all_cum_regret = np.zeros((n_seeds, len(cp_indices)))
        all_win_reward = np.zeros((n_seeds, len(cp_indices)))

        for s, r in enumerate(runs):
            per_step_regret = r.oracle_rewards - r.rewards
            cum_regret = np.cumsum(per_step_regret)
            for ci, idx in enumerate(cp_indices):
                all_cum_regret[s, ci] = cum_regret[idx]
                win_start = max(0, idx + 1 - window_size)
                all_win_reward[s, ci] = r.rewards[win_start:idx + 1].mean()

        def _se(arr: np.ndarray) -> np.ndarray:
            if n_seeds > 1:
                return np.std(arr, axis=0, ddof=1) / np.sqrt(n_seeds)
            return np.zeros(arr.shape[1])

        curves[cond] = {
            "xs": xs,
            "cum_regret_mean": all_cum_regret.mean(axis=0),
            "cum_regret_se": _se(all_cum_regret),
            "win_reward_mean": all_win_reward.mean(axis=0),
            "win_reward_se": _se(all_win_reward),
        }
    return curves


def compute_costsave_at_quality(
    rewards: np.ndarray,
    costs: np.ndarray,
    strong_reward: float,
    strong_cost: float,
    threshold: float = 0.95,
) -> Optional[float]:
    """CostSave at a quality threshold.

    Args:
        rewards: Per-prompt rewards.
        costs: Per-prompt costs.
        strong_reward: Mean reward of the strong model.
        strong_cost: Mean cost of the strong model.
        threshold: Quality threshold.

    Returns:
        CostSave percentage, or None if unreachable.
    """
    target_r = threshold * strong_reward
    mean_r = float(rewards.mean())
    mean_c = float(costs.mean())
    if mean_r >= target_r and strong_cost > 0:
        return (1.0 - mean_c / strong_cost) * 100.0
    return None


def compute_convergence_prompt(
    rewards: np.ndarray,
    oracle_rewards: np.ndarray,
    window: int = WINDOW_SIZE,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> int:
    """First prompt index where the windowed regret rate drops below threshold.

    Measures when the router's per-prompt regret stabilises near its
    final (converged) level.

    Args:
        rewards: Per-prompt rewards.
        oracle_rewards: Per-prompt oracle rewards.
        window: Smoothing window.
        threshold: Absolute proximity to final windowed regret.

    Returns:
        Prompt count at convergence.
    """
    n = len(rewards)
    per_step_regret = oracle_rewards - rewards
    if n < window:
        return n
    final_rate = per_step_regret[-window:].mean()
    for i in range(window, n):
        win_rate = per_step_regret[i - window:i].mean()
        if abs(win_rate - final_rate) <= threshold:
            return i
    return n


def compute_all_metrics(
    all_results: List[AblationResult],
    strong_reward: float,
    strong_cost: float,
) -> Dict[str, Dict[str, Any]]:
    """Compute summary metrics per condition.

    Args:
        all_results: Raw per-seed results.
        strong_reward: Mean reward of the strong model.
        strong_cost: Mean cost of the strong model.

    Returns:
        Nested metrics dict ``{condition: {metric: {mean, se, n}}}``.
    """
    by_cond: Dict[str, List[AblationResult]] = {}
    for r in all_results:
        by_cond.setdefault(r.condition, []).append(r)

    def _agg(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
        valid = [v for v in values if v is not None]
        if not valid:
            return {"mean": None, "se": None, "n": 0}
        n = len(valid)
        return {
            "mean": float(np.mean(valid)),
            "se": float(np.std(valid, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n": n,
        }

    metrics: Dict[str, Dict[str, Any]] = {}
    for cond, runs in by_cond.items():
        cum_regrets = [
            float((r.oracle_rewards - r.rewards).sum()) for r in runs
        ]
        full_cs = [
            compute_costsave_at_quality(
                r.rewards, r.costs,
                strong_reward, strong_cost, 0.95,
            )
            for r in runs
        ]
        final_rewards = [float(r.rewards.mean()) for r in runs]
        convergence = [
            float(compute_convergence_prompt(r.rewards, r.oracle_rewards))
            for r in runs
        ]
        early_regrets = [
            float((r.oracle_rewards[:100] - r.rewards[:100]).sum())
            for r in runs
        ]

        metrics[cond] = {
            "test_reward": _agg(final_rewards),
            "cumulative_regret": _agg(cum_regrets),
            "costsave_95": _agg(full_cs),
            "convergence_prompt": _agg(convergence),
            "early_regret_100": _agg(early_regrets),
        }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Windowed CostSave Checkpoints
# ═══════════════════════════════════════════════════════════════════════════


def compute_costsave_checkpoints(
    all_results: List[AblationResult],
    strong_reward: float,
    strong_cost: float,
    window_prompts: List[int],
) -> Dict[str, Dict[int, Dict[str, Optional[float]]]]:
    """CostSave@95% computed on the first N prompts for various N.

    Args:
        all_results: Raw per-seed results.
        strong_reward: Strong model mean reward.
        strong_cost: Strong model mean cost.
        window_prompts: List of prompt counts at which to measure.

    Returns:
        ``{condition: {N: {mean, se, n}}}``.
    """
    by_cond: Dict[str, List[AblationResult]] = {}
    for r in all_results:
        by_cond.setdefault(r.condition, []).append(r)

    out: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    for cond, runs in by_cond.items():
        out[cond] = {}
        for wp in window_prompts:
            cs_list: List[Optional[float]] = []
            for r in runs:
                n = min(wp, len(r.rewards))
                cs = compute_costsave_at_quality(
                    r.rewards[:n], r.costs[:n],
                    strong_reward, strong_cost, 0.95,
                )
                cs_list.append(cs)
            valid = [v for v in cs_list if v is not None]
            if valid:
                nv = len(valid)
                out[cond][wp] = {
                    "mean": float(np.mean(valid)),
                    "se": float(np.std(valid, ddof=1) / np.sqrt(nv)) if nv > 1 else 0.0,
                    "n": nv,
                }
            else:
                out[cond][wp] = {"mean": None, "se": None, "n": 0}
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
CONDITION_COLORS = {
    "BanditGPT": CB_BLUE,
    "Tabula Rasa": CB_GRAY,
}

CONDITION_STYLES = {
    "BanditGPT": {"lw": 2.2, "ls": "-"},
    "Tabula Rasa": {"lw": 2.0, "ls": "--"},
}


def plot_learning_curves(
    curves: Dict[str, Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    n_test: int,
    out_dir: Path,
) -> Path:
    """Two-panel cold-start ablation figure.

    (a) Cumulative regret from prompt 1 — shows the cold-start penalty
        that warmup priors eliminate.
    (b) Windowed average reward — shows convergence dynamics.

    Both panels use exclusively held-out test data (no training phase).

    Args:
        curves: Output of ``aggregate_curves``.
        metrics: Output of ``compute_all_metrics``.
        n_test: Number of test prompts.
        out_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    fig, (ax_reg, ax_rew) = plt.subplots(
        1, 2, figsize=(12, 4.2),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # ── Panel (a): Cumulative Regret ─────────────────────────────────
    for cond in ["BanditGPT", "Tabula Rasa"]:
        if cond not in curves:
            continue
        c = curves[cond]
        color = CONDITION_COLORS[cond]
        style = CONDITION_STYLES[cond]

        ax_reg.plot(c["xs"], c["cum_regret_mean"],
                    color=color, label=cond, **style)
        ax_reg.fill_between(
            c["xs"],
            c["cum_regret_mean"] - c["cum_regret_se"],
            c["cum_regret_mean"] + c["cum_regret_se"],
            alpha=0.15, color=color,
        )

    bg_regret = metrics.get("BanditGPT", {}).get("cumulative_regret", {})
    tr_regret = metrics.get("Tabula Rasa", {}).get("cumulative_regret", {})
    bg_val = bg_regret.get("mean")
    tr_val = tr_regret.get("mean")
    if bg_val is not None and tr_val is not None and tr_val > 0:
        reduction_pct = (1.0 - bg_val / tr_val) * 100
        saved_prompts = tr_val - bg_val
        ax_reg.annotate(
            f"$\\Delta = {saved_prompts:.0f}$\n({reduction_pct:.0f}% reduction)",
            xy=(n_test * 0.65, (bg_val + tr_val) / 2),
            fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.9),
        )

    ax_reg.set_ylabel("Cumulative regret", fontsize=10)
    ax_reg.set_xlabel("Prompts routed", fontsize=10)
    ax_reg.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax_reg.grid(axis="y", alpha=0.3, ls=":")
    ax_reg.set_title("(a) Cold-Start Cumulative Regret",
                     fontsize=11, fontweight="bold")

    # ── Panel (b): Windowed Reward ───────────────────────────────────
    for cond in ["BanditGPT", "Tabula Rasa"]:
        if cond not in curves:
            continue
        c = curves[cond]
        color = CONDITION_COLORS[cond]
        style = CONDITION_STYLES[cond]

        ax_rew.plot(c["xs"], c["win_reward_mean"],
                    color=color, label=cond, **style)
        ax_rew.fill_between(
            c["xs"],
            c["win_reward_mean"] - c["win_reward_se"],
            c["win_reward_mean"] + c["win_reward_se"],
            alpha=0.15, color=color,
        )

    ax_rew.set_xlabel("Prompts routed", fontsize=10)
    ax_rew.set_ylabel(f"Windowed avg. reward (w={WINDOW_SIZE})",
                      fontsize=10)
    ax_rew.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax_rew.grid(axis="y", alpha=0.3, ls=":")
    ax_rew.set_title(
        f"(b) Windowed Reward (w={WINDOW_SIZE})",
        fontsize=11, fontweight="bold",
    )

    fig.tight_layout(w_pad=3.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "figure3_warmup_ablation.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", fig_path)
    return fig_path


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════


def export_results(
    curves: Dict[str, Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    costsave_checkpoints: Dict[str, Dict[int, Dict[str, Optional[float]]]],
    cost_penalty: float,
    strong_reward: float,
    strong_cost: float,
    elapsed_s: float,
    out_dir: Path,
) -> Path:
    """Write machine-readable JSON with all metrics and checkpoints.

    Args:
        curves: Checkpoint curve data per condition.
        metrics: Summary metrics per condition.
        costsave_checkpoints: CostSave@95% at various prompt counts.
        cost_penalty: Cost penalty used.
        strong_reward: Strong model mean reward.
        strong_cost: Strong model mean cost.
        elapsed_s: Wall-clock time.
        out_dir: Output directory.

    Returns:
        Path to the saved JSON.
    """
    serialisable_curves = {}
    for cond, c in curves.items():
        serialisable_curves[cond] = {
            "xs": c["xs"].tolist(),
            "cum_regret_mean": c["cum_regret_mean"].tolist(),
            "cum_regret_se": c["cum_regret_se"].tolist(),
            "win_reward_mean": c["win_reward_mean"].tolist(),
            "win_reward_se": c["win_reward_se"].tolist(),
        }

    payload = {
        "experiment": "cold_start_prior_value",
        "protocol": "cold_start_no_pretraining",
        "conditions": {
            "BanditGPT": {
                "hparams": WARMUP_HPARAMS,
                "use_priors": True,
            },
            "Tabula Rasa": {
                "hparams": TABULA_RASA_HPARAMS,
                "use_priors": False,
            },
        },
        "cost_penalty": cost_penalty,
        "strong_model_reward": strong_reward,
        "strong_model_cost": strong_cost,
        "metrics": metrics,
        "costsave_checkpoints": {
            cond: {str(k): v for k, v in cps.items()}
            for cond, cps in costsave_checkpoints.items()
        },
        "curves": serialisable_curves,
        "wall_time_s": round(elapsed_s, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "warmup_ablation_data.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Data saved to %s", json_path)
    return json_path


# ═══════════════════════════════════════════════════════════════════════════
# Console Summary
# ═══════════════════════════════════════════════════════════════════════════


def print_summary(
    metrics: Dict[str, Dict[str, Any]],
    costsave_checkpoints: Dict[str, Dict[int, Dict[str, Optional[float]]]],
    strong_reward: float,
    test_n: int,
) -> None:
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 72)
    print("RQ2 — COLD-START PRIOR VALUE")
    print("=" * 72)

    def _fmt(
        info: Dict[str, Any],
        unit: str = "",
        mult: float = 1.0,
        precision: int = 1,
    ) -> str:
        m = info.get("mean")
        se = info.get("se")
        if m is None:
            return "N/A"
        val = m * mult
        if se is not None and se > 0:
            return f"{val:.{precision}f}{unit} ± {se * mult:.{precision}f}"
        return f"{val:.{precision}f}{unit}"

    for cond in ["BanditGPT", "Tabula Rasa"]:
        m = metrics.get(cond, {})
        print(f"\n{cond}")
        print("-" * 50)
        print(f"  Mean reward:           {_fmt(m.get('test_reward', {}), precision=4)}")
        print(f"  Cumulative regret:     {_fmt(m.get('cumulative_regret', {}))}")
        print(f"  Early regret (N≤100):  {_fmt(m.get('early_regret_100', {}))}")
        print(f"  CostSave@95%:          {_fmt(m.get('costsave_95', {}), '%')}")
        print(f"  Convergence:           {_fmt(m.get('convergence_prompt', {}), ' prompts')}")

        cps = costsave_checkpoints.get(cond, {})
        if cps:
            print("  CostSave@95% by prompt count:")
            for wp, info in sorted(cps.items()):
                print(f"    N={wp:5d}: {_fmt(info, '%')}")

    bg = metrics.get("BanditGPT", {})
    tr = metrics.get("Tabula Rasa", {})
    bg_regret = bg.get("cumulative_regret", {}).get("mean")
    tr_regret = tr.get("cumulative_regret", {}).get("mean")
    if bg_regret is not None and tr_regret is not None:
        regret_reduction = (1.0 - bg_regret / tr_regret) * 100 if tr_regret > 0 else 0
        print(f"\n  Cold-start regret reduction: {regret_reduction:.1f}%")

    bg_early = bg.get("early_regret_100", {}).get("mean")
    tr_early = tr.get("early_regret_100", {}).get("mean")
    if bg_early is not None and tr_early is not None:
        early_reduction = (1.0 - bg_early / tr_early) * 100 if tr_early > 0 else 0
        print(f"  Early regret reduction (first 100): {early_reduction:.1f}%")

    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Cold-Start Prior Value (K=2)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=20,
        help="Number of seeds per condition (default: 20)",
    )
    parser.add_argument(
        "--cost-penalty", type=float, default=0.15,
        help="Cost penalty lambda (default: 0.15).",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick run with 2 seeds for debugging",
    )
    args = parser.parse_args()

    n_seeds = 2 if args.fast else args.n_seeds
    cost_penalty = args.cost_penalty
    warmup_path = str(K2_WARMUP_PRIORS_PATH)

    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading data ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    logger.info("  Feature dim: %d", feature_dim)

    test = load_split(HOLDOUT_DATA_PATH, fs, K2_ARM_ORDER)
    logger.info("  Test: %d prompts", test.n)

    registry = build_model_registry(K2_ARM_ORDER)
    logger.info("  Registry: %s", list(registry.keys()))

    strong_model = "google/gemini-2.5-pro"
    strong_reward = float(test.rewards[strong_model].mean())
    strong_cost = float(test.costs[strong_model].mean())
    logger.info(
        "  Strong model: %s (reward=%.4f, cost=$%.6f)",
        strong_model, strong_reward, strong_cost,
    )

    # ── Run simulations ───────────────────────────────────────────────
    conditions = [
        ("BanditGPT", WARMUP_HPARAMS, True),
        ("Tabula Rasa", TABULA_RASA_HPARAMS, False),
    ]
    all_results: List[AblationResult] = []

    for cond_name, hparams, use_priors in conditions:
        logger.info(
            "\n%s (alpha=%.2f, prior_n_eff=%.0f): %d seeds",
            cond_name, hparams["alpha"], hparams["prior_n_effective"], n_seeds,
        )
        for s in range(n_seeds):
            seed = SEED_OFFSET + s
            logger.info("  Seed %d/%d (seed=%d)", s + 1, n_seeds, seed)
            result = simulate_cold_start(
                test, registry, feature_dim,
                hparams=hparams,
                warmup_path=warmup_path,
                cost_penalty=cost_penalty,
                seed=seed,
                use_priors=use_priors,
                condition_name=cond_name,
            )
            all_results.append(result)
            final_r = float(result.rewards.mean())
            regret = float((result.oracle_rewards - result.rewards).sum())
            logger.info("    Reward: %.4f, Regret: %.1f", final_r, regret)

    elapsed = time.time() - t0

    # ── Aggregate and report ──────────────────────────────────────────
    curves = aggregate_curves(all_results)
    metrics = compute_all_metrics(all_results, strong_reward, strong_cost)

    costsave_windows = [50, 100, 250, 500, 1000, 1824]
    costsave_cps = compute_costsave_checkpoints(
        all_results, strong_reward, strong_cost, costsave_windows,
    )

    fig_path = plot_learning_curves(curves, metrics, test.n, RESULTS_DIR)
    json_path = export_results(
        curves, metrics, costsave_cps,
        cost_penalty, strong_reward, strong_cost,
        elapsed, RESULTS_DIR,
    )
    print_summary(metrics, costsave_cps, strong_reward, test.n)

    logger.info("\nTotal wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
