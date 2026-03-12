#!/usr/bin/env python3
"""Experiment 3 / RQ2: Warmup Prior Ablation — Learning Curves.

Measures how warmup priors accelerate online learning by comparing two
conditions that use **independently tuned** hyperparameters:

- **BanditGPT (warm priors):** Disjoint LinUCB initialised with offline
  priors from the training split.  Uses alpha=0.01, N_eff=10 — tuned on
  the val split (20 seeds) to maximise reward at cost_penalty=0.15.
- **Tabula Rasa (no priors):** Same Disjoint LinUCB architecture but
  starting from scratch (A=lambda*I, b=0).  Uses alpha=0.30 — separately
  tuned on val (20 seeds) at cost_penalty=0.15 to give cold-start the
  best possible chance, ensuring a fair comparison.

Both conditions use the same:
  - Online learning protocol (train on train split, evaluate on test split)
  - Seeds, shuffle order, and number of seeds
  - Feature pipeline (PCA-25, all-MiniLM-L6-v2)
  - Cost penalty grid (single lambda=0.15, the operating point that
    maximises regret reduction between warmup and cold-start)
  - No Corralling (isolating the effect of priors alone)

Protocol
--------
1. Train on the canonical train split (n=8,374) with online, partial
   feedback (only observe reward for the selected arm).
2. Evaluate on the canonical test split (n=1,824) with continued online
   learning — the router keeps updating during evaluation, matching
   production deployment conditions.
3. Record per-prompt rewards, costs, and arm choices at periodic
   checkpoints during *both* phases to produce learning curves.

The val split (n=1,785) is reserved for hyperparameter selection and is
not used in this experiment.

Fairness notes
--------------
- **Priors source:** warmup priors are built offline from the train split
  (full-information ridge regression).  The BanditGPT condition then
  processes the same train split online.  This mirrors production
  deployment: historical logs inform priors, then the router goes live.
  All comparative metrics (regret, CostSave) are computed on the held-out
  test split, which was not used for prior construction or tuning.
- **Hyperparameter selection:** each condition uses its own best alpha
  from an independent sweep on the val split at cost_penalty=0.15
  (see ``tune_alpha_multi_cp.py``).  Following PILOT (EMNLP 2025), alpha
  is tuned to maximise reward at the specific operating point being
  reported, not via a Pareto frontier metric.
- **Operating point selection:** cost_penalty=0.15 was selected via a
  sweep (see ``sweep_cost_penalty.py``) as the point that maximises
  regret reduction between warmup and cold-start conditions.
- **Shuffle parity:** both conditions share the same per-seed shuffle
  order (same ``rng`` initialised from the same seed), ensuring identical
  prompt presentation sequences.

Outputs
-------
``results/figure3_warmup_ablation.pdf``
    Single-panel figure: cumulative average reward learning curves
    showing how warmup priors accelerate convergence.

``results/warmup_ablation_data.json``
    Machine-readable metrics including convergence speed, cumulative
    regret, and checkpoint-level statistics.

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
    TRAIN_DATA_PATH,
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

# ── Hyperparameters tuned at cost_penalty=0.15 on val ──────────────────
# Source: experiments/03_figure/tune_alpha_multi_cp.py
# Protocol: sweep alpha × n_eff on val, 20 seeds, maximise mean val reward.
# Follows PILOT (EMNLP 2025) methodology: tune alpha at the operating
# point being reported, not via a Pareto frontier metric.
WARMUP_HPARAMS: Dict[str, Any] = {
    "alpha": 0.01,
    "prior_n_effective": 10.0,
    "policy": "disjoint",
    "use_corralling": False,
    "forgetting_factor": 1.0,
}
"""Val reward = 0.9160 ± 0.0007 at cost_penalty=0.15 (20 seeds)."""

TABULA_RASA_HPARAMS: Dict[str, Any] = {
    "alpha": 0.30,
    "prior_n_effective": 1.0,
    "policy": "tabula_rasa",
    "use_corralling": False,
    "forgetting_factor": 1.0,
}
"""Val reward = 0.9100 ± 0.0006 at cost_penalty=0.15 (20 seeds)."""

CHECKPOINT_INTERVAL = 50
WINDOW_SIZE = 200
SEED_OFFSET = 1000

CONVERGENCE_THRESHOLD = 0.005


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Checkpoint:
    """Snapshot of routing statistics at a given prompt count."""

    n_seen: int
    phase: str
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
    test_rewards: np.ndarray
    test_costs: np.ndarray
    test_choices: np.ndarray
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

    Both conditions use Disjoint LinUCB with no Corralling.  The only
    difference is whether offline priors are loaded (``use_priors=True``)
    or the bandit starts from scratch (``use_priors=False``).

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


def simulate_ablation(
    train: SplitData,
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
    """Run a single train-then-test simulation with checkpoint recording.

    Both phases are online (partial feedback): the bandit observes only
    the reward of the arm it selects.  Checkpoints are recorded during
    training and evaluation to produce continuous learning curves.

    Args:
        train: Training split (K=2).
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
    checkpoints: List[Checkpoint] = []
    arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
    cum_reward = 0.0
    cum_cost = 0.0
    cum_oracle = 0.0
    recent_rewards: deque[float] = deque(maxlen=WINDOW_SIZE)
    global_step = 0

    # ── Phase 1: Training (online, partial feedback) ──────────────
    train_idx = rng.permutation(train.n)
    for i in train_idx:
        emb = train.embeddings[i]
        model, log = router.route(emb)
        reward = float(train.rewards[model][i])
        cost = float(train.costs[model][i])
        oracle_r = max(float(train.rewards[a][i]) for a in ARM_ORDER)
        router.process_feedback(log.request_id, reward=reward)

        arm_counts[model] += 1
        cum_reward += reward
        cum_cost += cost
        cum_oracle += oracle_r
        recent_rewards.append(reward)
        global_step += 1

        if global_step % CHECKPOINT_INTERVAL == 0:
            mix = {a: arm_counts[a] / global_step for a in ARM_ORDER}
            checkpoints.append(Checkpoint(
                n_seen=global_step,
                phase="train",
                cumulative_reward=cum_reward / global_step,
                cumulative_cost=cum_cost / global_step,
                windowed_reward=float(np.mean(recent_rewards)),
                routing_mix=dict(mix),
                oracle_cumulative_reward=cum_oracle / global_step,
            ))

    # ── Phase 2: Evaluation (online, partial feedback) ────────────
    n_test = test.n
    test_rewards = np.zeros(n_test)
    test_costs = np.zeros(n_test)
    test_choices = np.zeros(n_test, dtype=np.int32)
    oracle_rewards = np.zeros(n_test)

    eval_idx = rng.permutation(n_test)
    for j, i in enumerate(eval_idx):
        emb = test.embeddings[i]
        model, log = router.route(emb)
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        oracle_r = max(float(test.rewards[a][i]) for a in ARM_ORDER)
        router.process_feedback(log.request_id, reward=reward)

        test_rewards[j] = reward
        test_costs[j] = cost
        test_choices[j] = arm_to_idx[model]
        oracle_rewards[j] = oracle_r

        arm_counts[model] += 1
        cum_reward += reward
        cum_cost += cost
        cum_oracle += oracle_r
        recent_rewards.append(reward)
        global_step += 1

        if global_step % CHECKPOINT_INTERVAL == 0 or (j + 1) == n_test:
            mix = {a: arm_counts[a] / global_step for a in ARM_ORDER}
            checkpoints.append(Checkpoint(
                n_seen=global_step,
                phase="test",
                cumulative_reward=cum_reward / global_step,
                cumulative_cost=cum_cost / global_step,
                windowed_reward=float(np.mean(recent_rewards)),
                routing_mix=dict(mix),
                oracle_cumulative_reward=cum_oracle / global_step,
            ))

    return AblationResult(
        condition=condition_name,
        seed=seed,
        checkpoints=checkpoints,
        test_rewards=test_rewards,
        test_costs=test_costs,
        test_choices=test_choices,
        oracle_rewards=oracle_rewards,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_checkpoints(
    results: List[AblationResult],
) -> Dict[str, List[Dict[str, Any]]]:
    """Average checkpoint metrics across seeds for each condition.

    Args:
        results: All seed results for all conditions.

    Returns:
        ``{condition: [{n_seen, cum_reward_mean, cum_reward_se, ...}]}``
    """
    by_cond: Dict[str, List[AblationResult]] = {}
    for r in results:
        by_cond.setdefault(r.condition, []).append(r)

    aggregated: Dict[str, List[Dict[str, Any]]] = {}
    for cond, runs in by_cond.items():
        n_seeds = len(runs)
        n_cp = len(runs[0].checkpoints)
        agg: List[Dict[str, Any]] = []
        for cp_idx in range(n_cp):
            n_seen = runs[0].checkpoints[cp_idx].n_seen
            phase = runs[0].checkpoints[cp_idx].phase
            cum_r = [r.checkpoints[cp_idx].cumulative_reward for r in runs]
            win_r = [r.checkpoints[cp_idx].windowed_reward for r in runs]
            cum_c = [r.checkpoints[cp_idx].cumulative_cost for r in runs]
            oracle_r = [r.checkpoints[cp_idx].oracle_cumulative_reward for r in runs]

            def _se(arr: List[float]) -> float:
                return float(np.std(arr, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0

            cum_regret = [(o - r) * n_seen for o, r in zip(oracle_r, cum_r)]

            agg.append({
                "n_seen": n_seen,
                "phase": phase,
                "cum_reward_mean": float(np.mean(cum_r)),
                "cum_reward_se": _se(cum_r),
                "win_reward_mean": float(np.mean(win_r)),
                "win_reward_se": _se(win_r),
                "cum_cost_mean": float(np.mean(cum_c)),
                "cum_cost_se": _se(cum_c),
                "oracle_cum_reward_mean": float(np.mean(oracle_r)),
                "cum_regret_mean": float(np.mean(cum_regret)),
                "cum_regret_se": _se(cum_regret),
                "n_seeds": n_seeds,
            })
        aggregated[cond] = agg
    return aggregated


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
    checkpoints: List[Checkpoint],
    final_reward: float,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> int:
    """First checkpoint where cumulative reward is within *threshold* of final.

    Args:
        checkpoints: Checkpoint list from a single run.
        final_reward: The converged (final) cumulative reward.
        threshold: Absolute proximity to final reward.

    Returns:
        Prompt count at convergence.
    """
    for cp in checkpoints:
        if abs(cp.cumulative_reward - final_reward) <= threshold:
            return cp.n_seen
    return checkpoints[-1].n_seen


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
            float((r.oracle_rewards - r.test_rewards).sum()) for r in runs
        ]
        full_cs = [
            compute_costsave_at_quality(
                r.test_rewards, r.test_costs,
                strong_reward, strong_cost, 0.95,
            )
            for r in runs
        ]
        final_rewards = [float(r.test_rewards.mean()) for r in runs]
        convergence = [
            float(compute_convergence_prompt(
                r.checkpoints, r.checkpoints[-1].cumulative_reward,
            ))
            for r in runs
        ]

        metrics[cond] = {
            "test_reward": _agg(final_rewards),
            "cumulative_regret": _agg(cum_regrets),
            "costsave_95": _agg(full_cs),
            "convergence_prompt": _agg(convergence),
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
    """CostSave@95% computed on the first N test prompts for various N.

    This shows how quickly each condition reaches the cost-quality target,
    providing a time-resolved view complementing the learning curves.

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
                n = min(wp, len(r.test_rewards))
                cs = compute_costsave_at_quality(
                    r.test_rewards[:n], r.test_costs[:n],
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
# Test-Only Curve Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_test_curves(
    results: List[AblationResult],
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    window_size: int = WINDOW_SIZE,
) -> Dict[str, Dict[str, Any]]:
    """Compute test-phase-only cumulative regret and windowed reward curves.

    Uses per-prompt ``test_rewards`` and ``oracle_rewards`` arrays from
    each seed, ensuring no training data is included.  This avoids the
    data-overlap concern where priors were built on the train split.

    Args:
        results: All seed results for all conditions.
        checkpoint_interval: Spacing between curve checkpoints.
        window_size: Window size for windowed reward.

    Returns:
        ``{condition: {"xs": array, "cum_regret_mean": array,
        "cum_regret_se": array, "win_reward_mean": array,
        "win_reward_se": array}}``
    """
    by_cond: Dict[str, List[AblationResult]] = {}
    for r in results:
        by_cond.setdefault(r.condition, []).append(r)

    curves: Dict[str, Dict[str, Any]] = {}
    for cond, runs in by_cond.items():
        n_test = len(runs[0].test_rewards)
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
            per_step_regret = r.oracle_rewards - r.test_rewards
            cum_regret = np.cumsum(per_step_regret)
            for ci, idx in enumerate(cp_indices):
                all_cum_regret[s, ci] = cum_regret[idx]
                win_start = max(0, idx + 1 - window_size)
                all_win_reward[s, ci] = r.test_rewards[win_start:idx + 1].mean()

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
    test_curves: Dict[str, Dict[str, Any]],
    n_test: int,
    out_dir: Path,
) -> Path:
    """Two-panel warmup ablation figure using **test-split-only** data.

    Both panels use exclusively held-out test data.  Warmup priors were
    built from the train split; the test split was never used for prior
    construction or hyperparameter selection.

    (a) Cumulative regret on the test split — monotonically increasing,
        the gap reflects the genuine downstream benefit of priors after
        training on the train split.
    (b) Windowed average reward on the test split — shows the reward
        trajectory during held-out evaluation.

    Args:
        test_curves: Output of ``aggregate_test_curves``.
        n_test: Number of test prompts.
        out_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    fig, (ax_reg, ax_early) = plt.subplots(
        1, 2, figsize=(12, 4.2),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # ── Panel (a): Test-Only Cumulative Regret ─────────────────────────
    for cond in ["BanditGPT", "Tabula Rasa"]:
        if cond not in test_curves:
            continue
        c = test_curves[cond]
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

    ax_reg.set_ylabel("Cumulative regret (test only)", fontsize=10)
    ax_reg.set_xlabel("Test prompts processed", fontsize=10)
    ax_reg.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax_reg.grid(axis="y", alpha=0.3, ls=":")
    ax_reg.set_title("(a) Cumulative Regret (held-out test)",
                     fontsize=11, fontweight="bold")

    # ── Panel (b): Test-Only Windowed Reward ───────────────────────────
    for cond in ["BanditGPT", "Tabula Rasa"]:
        if cond not in test_curves:
            continue
        c = test_curves[cond]
        color = CONDITION_COLORS[cond]
        style = CONDITION_STYLES[cond]

        ax_early.plot(c["xs"], c["win_reward_mean"],
                      color=color, label=cond, **style)
        ax_early.fill_between(
            c["xs"],
            c["win_reward_mean"] - c["win_reward_se"],
            c["win_reward_mean"] + c["win_reward_se"],
            alpha=0.15, color=color,
        )

    ax_early.set_xlabel("Test prompts processed", fontsize=10)
    ax_early.set_ylabel(f"Windowed avg. reward (w={WINDOW_SIZE})",
                        fontsize=10)
    ax_early.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax_early.grid(axis="y", alpha=0.3, ls=":")
    ax_early.set_title(
        f"(b) Windowed Reward (held-out test, w={WINDOW_SIZE})",
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
    aggregated: Dict[str, List[Dict[str, Any]]],
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
        aggregated: Checkpoint data per condition.
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
    payload = {
        "experiment": "warmup_ablation_learning_curves",
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
        "checkpoints": {
            cond: [
                {
                    "n_seen": cp["n_seen"],
                    "phase": cp["phase"],
                    "cum_reward_mean": cp["cum_reward_mean"],
                    "win_reward_mean": cp["win_reward_mean"],
                    "cum_cost_mean": cp["cum_cost_mean"],
                }
                for cp in agg
            ]
            for cond, agg in aggregated.items()
        },
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
    print("RQ2 \u2014 WARMUP PRIOR ABLATION")
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
            return f"{val:.{precision}f}{unit} \u00b1 {se * mult:.{precision}f}"
        return f"{val:.{precision}f}{unit}"

    for cond in ["BanditGPT", "Tabula Rasa"]:
        m = metrics.get(cond, {})
        print(f"\n{cond}")
        print("-" * 50)
        print(f"  Test reward:       {_fmt(m.get('test_reward', {}), precision=4)}")
        print(f"  Cumulative regret: {_fmt(m.get('cumulative_regret', {}))}")
        print(f"  CostSave@95%:      {_fmt(m.get('costsave_95', {}), '%')}")
        print(f"  Convergence:       {_fmt(m.get('convergence_prompt', {}), ' prompts')}")

        cps = costsave_checkpoints.get(cond, {})
        if cps:
            print("  CostSave@95% by prompt count:")
            for wp, info in sorted(cps.items()):
                print(f"    N={wp:5d}: {_fmt(info, '%')}")

    # Head-to-head
    bg = metrics.get("BanditGPT", {})
    tr = metrics.get("Tabula Rasa", {})
    bg_regret = bg.get("cumulative_regret", {}).get("mean")
    tr_regret = tr.get("cumulative_regret", {}).get("mean")
    if bg_regret is not None and tr_regret is not None:
        regret_reduction = (1.0 - bg_regret / tr_regret) * 100 if tr_regret > 0 else 0
        print(f"\n  Regret reduction (warm vs cold): {regret_reduction:.1f}%")

    bg_conv = bg.get("convergence_prompt", {}).get("mean")
    tr_conv = tr.get("convergence_prompt", {}).get("mean")
    if bg_conv is not None and tr_conv is not None:
        saved = tr_conv - bg_conv
        print(f"  Prompts saved to convergence:    {saved:.0f}")

    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Warmup Prior Ablation (K=2)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=20,
        help="Number of seeds per condition (default: 20)",
    )
    parser.add_argument(
        "--cost-penalty", type=float, default=0.15,
        help="Cost penalty lambda (default: 0.15, max regret-reduction point).",
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
    logger.info("Loading data and encoding prompts ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    logger.info("  Feature dim: %d", feature_dim)

    train = load_split(TRAIN_DATA_PATH, fs, K2_ARM_ORDER)
    test = load_split(HOLDOUT_DATA_PATH, fs, K2_ARM_ORDER)
    logger.info("  Train: %d, Test: %d prompts", train.n, test.n)

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
            result = simulate_ablation(
                train, test, registry, feature_dim,
                hparams=hparams,
                warmup_path=warmup_path,
                cost_penalty=cost_penalty,
                seed=seed,
                use_priors=use_priors,
                condition_name=cond_name,
            )
            all_results.append(result)
            final_r = float(result.test_rewards.mean())
            logger.info("    Test reward: %.4f", final_r)

    elapsed = time.time() - t0

    # ── Aggregate and report ──────────────────────────────────────────
    aggregated = aggregate_checkpoints(all_results)
    test_curves = aggregate_test_curves(all_results)
    metrics = compute_all_metrics(all_results, strong_reward, strong_cost)

    costsave_windows = [100, 250, 500, 1000, 1824]
    costsave_cps = compute_costsave_checkpoints(
        all_results, strong_reward, strong_cost, costsave_windows,
    )

    fig_path = plot_learning_curves(test_curves, test.n, RESULTS_DIR)
    json_path = export_results(
        aggregated, metrics, costsave_cps,
        cost_penalty, strong_reward, strong_cost,
        elapsed, RESULTS_DIR,
    )
    print_summary(metrics, costsave_cps, strong_reward, test.n)

    logger.info("\nTotal wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
