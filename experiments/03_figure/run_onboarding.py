#!/usr/bin/env python3
"""Experiment 3 / RQ2: Model Onboarding — K=2 to K=3 via register_model().

Demonstrates BanditGPT's Hybrid LinUCB enabling zero-shot warm start for a
newly added model (Mistral-Large-2512) by transferring shared-beta knowledge
from K=2 training.  The router autonomously discovers that the expensive
Gemini-2.5-Pro becomes largely redundant once the mid-tier newcomer joins.

Protocol
--------
1. **Phase 1 (K=2 pre-training):** Train a BanditGPT router on the canonical
   train split (8,374 prompts) with K=2 arms (Llama + Gemini).
2. **Phase 2 (Onboarding):** Call ``register_model()`` to add Mistral-Large
   as a third arm.  Hybrid LinUCB's shared beta transfers automatically.
3. **Phase 3 (K=3 evaluation):** Evaluate on the test split (1,824 prompts),
   recording per-prompt arm choices, rewards, and costs at checkpoints.

No artificial exploration boost is applied.  Hybrid LinUCB's shared beta
gives the newcomer a meaningful initial quality estimate (the mean of the
incumbents' calibrated theta_a), enabling zero-shot discovery at the
production exploration rate (α=0.05).

Two conditions are compared (all else equal — same K=2 warmup priors for
Llama/Gemini, same Corralling, same alpha/n_eff):
- **Hybrid LinUCB:** shared beta from K=2 transfers to the newcomer,
  giving it meaningful quality predictions from the first request.
- **Disjoint LinUCB:** independent per-arm parameters; the newcomer's
  arm-specific A/b start from scratch with no cross-arm knowledge.
  Falls into a cold-start trap — never discovered at production alpha.

Outputs
-------
``results/onboarding_data.json``
    Machine-readable results including routing mix evolution, CostSave,
    and convergence metrics.

``results/figure3_onboarding.pdf``
    Three-panel figure: routing mix evolution for Hybrid (left),
    Disjoint (center), and Mistral adoption overlay (right).

Usage
-----
    python experiments/03_figure/run_onboarding.py
    python experiments/03_figure/run_onboarding.py --n-seeds 20
    python experiments/03_figure/run_onboarding.py --fast
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
    BEST_K2_CORRALLING_HPARAMS,
    HOLDOUT_DATA_PATH,
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
    K3_ARM_ORDER,
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
    CB_ORANGE,
    CB_RED,
    CB_TEAL,
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

MISTRAL_ID = "mistralai/mistral-large-2512"
MISTRAL_INPUT_COST_PER_M = 0.5
MISTRAL_OUTPUT_COST_PER_M = 1.5
MISTRAL_BLENDED_COST_PER_M = (MISTRAL_INPUT_COST_PER_M + MISTRAL_OUTPUT_COST_PER_M) / 2.0

ARM_LABELS = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
}

CHECKPOINT_INTERVAL = 50
WINDOW_SIZE = 100
SEED_OFFSET = 1000

GEMINI_ID = "google/gemini-2.5-pro"
LATE_STAGE_N = 500
EARLY_STAGE_N = 500
CONVERGENCE_TOLERANCE = 0.05
GEMINI_DISPLACEMENT_THRESHOLD = 0.10

# Model colors for routing-mix plots
ARM_COLORS = {
    "meta-llama/llama-3.1-8b-instruct": CB_TEAL,
    "mistralai/mistral-large-2512": CB_ORANGE,
    "google/gemini-2.5-pro": CB_RED,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


PRETRAIN_DISPLAY = 2000  # show last N Phase-1 prompts in the figure


@dataclass
class Checkpoint:
    """Routing statistics at a given prompt count."""

    n_seen: int
    routing_mix: Dict[str, float]
    windowed_mix: Dict[str, float]
    cumulative_reward: float
    cumulative_cost: float


@dataclass
class OnboardingResult:
    """Full results for a single seed of one experimental condition."""

    condition: str
    seed: int
    n_pretrain: int
    pretrain_checkpoints: List[Checkpoint]
    checkpoints: List[Checkpoint]
    final_rewards: np.ndarray
    final_costs: np.ndarray
    final_choices: np.ndarray
    oracle_rewards: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    hparams: Dict[str, Any],
    warmup_path: str,
    cost_penalty: float,
    policy_override: Optional[str] = None,
) -> BanditRouter:
    """Create a BanditRouter with the given config.

    Args:
        registry: Model registry (initially K=2 arms).
        feature_dim: Dimensionality of feature vectors.
        hparams: Hyperparameters (alpha, policy, etc.).
        warmup_path: Path to warmup priors joblib file.
        cost_penalty: Cost penalty weight for routing.
        policy_override: If set, overrides hparams["policy"].

    Returns:
        Fully initialised router.
    """
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    policy = policy_override or hparams["policy"]

    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup",
        warmup_path=warmup_path,
        prior_n_effective=hparams["prior_n_effective"],
        alpha=hparams["alpha"],
        use_corralling=True,
        cost_penalty=cost_penalty,
        forgetting_factor=hparams["forgetting_factor"],
        policy=policy,
    )
    return router


def simulate_onboarding(
    train_k2: SplitData,
    test_k3: SplitData,
    registry_k2: Dict[str, Any],
    feature_dim: int,
    *,
    hparams: Dict[str, Any],
    warmup_path: str,
    cost_penalty: float,
    seed: int,
    policy_override: Optional[str] = None,
    condition_name: str = "hybrid",
) -> OnboardingResult:
    """Run the full K=2 pre-train → onboard → K=3 evaluation pipeline.

    No artificial exploration boost is applied after onboarding.
    Hybrid LinUCB's shared beta provides a meaningful initial quality
    estimate for the newcomer, enabling discovery at the production
    exploration rate (α=0.05).

    Args:
        train_k2: Training split (K=2 arms).
        test_k3: Test split (K=3 arms, includes Mistral rewards).
        registry_k2: Model registry for the initial K=2 portfolio.
        feature_dim: Dimensionality of feature vectors.
        hparams: BanditGPT hyperparameters.
        warmup_path: Path to K=2 warmup priors.
        cost_penalty: Cost penalty weight λ.
        seed: Random seed for shuffle order.
        policy_override: Override policy (e.g. "disjoint" for control).
        condition_name: Label for this condition.

    Returns:
        Full per-prompt results with periodic checkpoints.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Phase 1: K=2 pre-training (with checkpoint recording)
    router = _create_router(
        registry_k2, feature_dim, hparams, warmup_path,
        cost_penalty, policy_override,
    )

    pretrain_arm_counts: Dict[str, int] = {a: 0 for a in K3_ARM_ORDER}
    pretrain_cum_reward = 0.0
    pretrain_cum_cost = 0.0
    pretrain_recent: deque[str] = deque(maxlen=WINDOW_SIZE)
    pretrain_checkpoints: List[Checkpoint] = []
    n_pretrain = train_k2.n

    train_idx = rng.permutation(n_pretrain)
    for step, i in enumerate(train_idx):
        emb = train_k2.embeddings[i]
        model, log = router.route(emb)
        reward = float(train_k2.rewards[model][i])
        cost = float(train_k2.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

        pretrain_arm_counts[model] += 1
        pretrain_recent.append(model)
        pretrain_cum_reward += reward
        pretrain_cum_cost += cost

        n = step + 1
        if n % CHECKPOINT_INTERVAL == 0 or n == n_pretrain:
            mix = {a: pretrain_arm_counts[a] / n for a in K3_ARM_ORDER}
            w_total = len(pretrain_recent)
            w_mix = {
                a: sum(1 for c in pretrain_recent if c == a) / w_total
                for a in K3_ARM_ORDER
            }
            pretrain_checkpoints.append(Checkpoint(
                n_seen=n,
                routing_mix=dict(mix),
                windowed_mix=dict(w_mix),
                cumulative_reward=pretrain_cum_reward / n,
                cumulative_cost=pretrain_cum_cost / n,
            ))

    # Phase 2: Onboard Mistral-Large
    router.register_model(
        MISTRAL_ID,
        speed="balanced",
        cost_usd=MISTRAL_INPUT_COST_PER_M,
        blended_cost_per_m=MISTRAL_BLENDED_COST_PER_M,
    )

    n_test = test_k3.n

    # Phase 3: K=3 evaluation on test split (online / interleaved)
    arm_to_idx = {arm: i for i, arm in enumerate(K3_ARM_ORDER)}
    eval_rewards = np.zeros(n_test)
    eval_costs = np.zeros(n_test)
    eval_choices = np.zeros(n_test, dtype=np.int32)
    oracle_rewards = np.zeros(n_test)

    arm_counts: Dict[str, int] = {a: 0 for a in K3_ARM_ORDER}
    cum_reward = 0.0
    cum_cost = 0.0
    checkpoints: List[Checkpoint] = []
    recent_choices: deque[str] = deque(maxlen=WINDOW_SIZE)

    eval_idx = rng.permutation(n_test)
    for j, i in enumerate(eval_idx):
        emb = test_k3.embeddings[i]
        model, log = router.route(emb)
        reward = float(test_k3.rewards[model][i])
        cost = float(test_k3.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

        eval_rewards[j] = reward
        eval_costs[j] = cost
        eval_choices[j] = arm_to_idx[model]
        oracle_rewards[j] = max(
            float(test_k3.rewards[a][i]) for a in K3_ARM_ORDER
        )
        arm_counts[model] += 1
        recent_choices.append(model)
        cum_reward += reward
        cum_cost += cost

        n_seen = j + 1
        if n_seen % CHECKPOINT_INTERVAL == 0 or n_seen == n_test:
            mix = {a: arm_counts[a] / n_seen for a in K3_ARM_ORDER}
            w_total = len(recent_choices)
            w_mix = {
                a: sum(1 for c in recent_choices if c == a) / w_total
                for a in K3_ARM_ORDER
            }
            checkpoints.append(Checkpoint(
                n_seen=n_seen,
                routing_mix=dict(mix),
                windowed_mix=dict(w_mix),
                cumulative_reward=cum_reward / n_seen,
                cumulative_cost=cum_cost / n_seen,
            ))

    return OnboardingResult(
        condition=condition_name,
        seed=seed,
        n_pretrain=n_pretrain,
        pretrain_checkpoints=pretrain_checkpoints,
        checkpoints=checkpoints,
        final_rewards=eval_rewards,
        final_costs=eval_costs,
        final_choices=eval_choices,
        oracle_rewards=oracle_rewards,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_costsave_at_quality(
    rewards: np.ndarray,
    costs: np.ndarray,
    strong_reward: float,
    strong_cost: float,
    threshold: float = 0.95,
) -> Optional[float]:
    """CostSave at a quality threshold for a single run.

    Finds the cheapest cost that achieves ``threshold * strong_reward``
    quality on the given rewards/costs, then computes the percentage
    saving vs. always using the strong model.

    Args:
        rewards: Per-prompt rewards from the router.
        costs: Per-prompt costs from the router.
        strong_reward: Mean reward of the strong model.
        strong_cost: Mean cost of the strong model.
        threshold: Quality threshold (e.g. 0.95).

    Returns:
        CostSave percentage, or None if threshold is unreachable.
    """
    target_r = threshold * strong_reward
    mean_r = float(rewards.mean())
    mean_c = float(costs.mean())
    if mean_r >= target_r and strong_cost > 0:
        return (1.0 - mean_c / strong_cost) * 100.0
    return None


def aggregate_checkpoints(
    results: List[OnboardingResult],
) -> Dict[str, List[Dict[str, Any]]]:
    """Average checkpoint metrics across seeds for each condition.

    Args:
        results: All seed results for all conditions.

    Returns:
        ``{condition: [{n_seen, routing_mix_mean, ...}, ...]}``
    """
    by_condition: Dict[str, List[OnboardingResult]] = {}
    for r in results:
        by_condition.setdefault(r.condition, []).append(r)

    def _mean_and_se(
        arr: List[float], n: int,
    ) -> tuple[float, float]:
        m = float(np.mean(arr))
        se = (float(np.std(arr, ddof=1) / np.sqrt(n))
              if n > 1 else 0.0)
        return m, se

    aggregated: Dict[str, List[Dict[str, Any]]] = {}
    for cond, runs in by_condition.items():
        n_seeds = len(runs)
        n_checkpoints = len(runs[0].checkpoints)
        agg: List[Dict[str, Any]] = []
        for cp_idx in range(n_checkpoints):
            n_seen = runs[0].checkpoints[cp_idx].n_seen
            mix_arrays: Dict[str, List[float]] = {
                a: [] for a in K3_ARM_ORDER
            }
            wmix_arrays: Dict[str, List[float]] = {
                a: [] for a in K3_ARM_ORDER
            }
            rewards_list: List[float] = []
            costs_list: List[float] = []
            for run in runs:
                cp = run.checkpoints[cp_idx]
                for a in K3_ARM_ORDER:
                    mix_arrays[a].append(cp.routing_mix[a])
                    wmix_arrays[a].append(cp.windowed_mix[a])
                rewards_list.append(cp.cumulative_reward)
                costs_list.append(cp.cumulative_cost)

            mix_mean = {}
            mix_se = {}
            wmix_mean = {}
            wmix_se = {}
            for a in K3_ARM_ORDER:
                mix_mean[a], mix_se[a] = _mean_and_se(mix_arrays[a], n_seeds)
                wmix_mean[a], wmix_se[a] = _mean_and_se(wmix_arrays[a], n_seeds)

            r_mean, r_se = _mean_and_se(rewards_list, n_seeds)
            c_mean, c_se = _mean_and_se(costs_list, n_seeds)
            agg.append({
                "n_seen": n_seen,
                "routing_mix_mean": mix_mean,
                "routing_mix_se": mix_se,
                "windowed_mix_mean": wmix_mean,
                "windowed_mix_se": wmix_se,
                "mean_reward": r_mean,
                "reward_se": r_se,
                "mean_cost": c_mean,
                "cost_se": c_se,
                "n_seeds": n_seeds,
            })
        aggregated[cond] = agg
    return aggregated


def compute_convergence_speed(
    checkpoints: List[Checkpoint],
    model: str,
    tolerance: float = CONVERGENCE_TOLERANCE,
) -> int:
    """Number of prompts until *model*'s windowed share stabilises.

    Walks backwards from the final checkpoint; finds the last point where
    the windowed share deviates from its final value by more than
    ``tolerance`` (absolute).  Convergence is the next checkpoint.

    Args:
        checkpoints: Checkpoint list from a single simulation run.
        model: Model ID to track.
        tolerance: Maximum absolute deviation from final share.

    Returns:
        Number of prompts at which convergence is reached.
    """
    if len(checkpoints) <= 1:
        return checkpoints[0].n_seen if checkpoints else 0
    final_share = checkpoints[-1].windowed_mix[model]
    for i in range(len(checkpoints) - 2, -1, -1):
        if abs(checkpoints[i].windowed_mix[model] - final_share) > tolerance:
            return checkpoints[i + 1].n_seen
    return checkpoints[0].n_seen


def compute_gemini_displacement(
    checkpoints: List[Checkpoint],
    threshold: float = GEMINI_DISPLACEMENT_THRESHOLD,
) -> Optional[int]:
    """First checkpoint where Gemini's windowed share drops below *threshold*.

    Args:
        checkpoints: Checkpoint list from a single simulation run.
        threshold: Share below which Gemini is considered displaced.

    Returns:
        Number of prompts at the displacement point, or ``None`` if
        Gemini never drops below *threshold*.
    """
    for cp in checkpoints:
        if cp.windowed_mix[GEMINI_ID] < threshold:
            return cp.n_seen
    return None


def compute_late_stage_costsave(
    result: OnboardingResult,
    strong_reward: float,
    strong_cost: float,
    late_n: int = LATE_STAGE_N,
    threshold: float = 0.95,
) -> Optional[float]:
    """CostSave@95% computed on only the last *late_n* evaluation prompts.

    By restricting to late-stage data where both policies have had time
    to converge, this metric isolates the *steady-state* cost-quality
    trade-off from the transient onboarding phase.

    Args:
        result: Single-seed simulation result.
        strong_reward: Mean reward of the strong model.
        strong_cost: Mean cost of the strong model.
        late_n: Number of trailing prompts to use.
        threshold: Quality threshold (e.g. 0.95).

    Returns:
        CostSave percentage, or ``None`` if quality is unreachable.
    """
    start = max(0, len(result.final_rewards) - late_n)
    return compute_costsave_at_quality(
        result.final_rewards[start:],
        result.final_costs[start:],
        strong_reward, strong_cost, threshold,
    )


def compute_early_stage_cost(
    result: OnboardingResult,
    early_n: int = EARLY_STAGE_N,
) -> float:
    """Mean per-prompt cost during the first *early_n* evaluation prompts.

    Args:
        result: Single-seed simulation result.
        early_n: Number of leading prompts to use.

    Returns:
        Mean cost in the early window.
    """
    n = min(early_n, len(result.final_costs))
    return float(result.final_costs[:n].mean())


def compute_all_onboarding_metrics(
    all_results: List[OnboardingResult],
    aggregated: Dict[str, List[Dict[str, Any]]],
    strong_reward: float,
    strong_cost: float,
) -> Dict[str, Dict[str, Any]]:
    """Compute all onboarding metrics per condition.

    **Primary (full-sequence) metrics** — standard online-learning measures
    that account for the cost of exploration, not just the asymptote:

    - ``cumulative_regret``: Σ(oracle_reward − actual_reward) over all
      evaluation prompts.
    - ``full_costsave_95``: CostSave@95% computed on the *entire*
      evaluation sequence.

    **Supplementary metrics** — useful for understanding the *dynamics*
    of convergence and cost/quality trade-offs:

    - ``convergence_speed``: prompts to Mistral stabilisation.
    - ``gemini_displacement``: prompts to Gemini < 10%.
    - ``late_costsave_95``: CostSave on last 500 prompts (steady-state).
    - ``early_cost``: per-prompt cost on first 500 prompts (transition).

    Args:
        all_results: Raw per-seed results.
        aggregated: Aggregated checkpoint data (unused here but kept
            for API consistency).
        strong_reward: Mean reward of the strong model.
        strong_cost: Mean cost of the strong model.

    Returns:
        Nested dict of metrics per condition.
    """
    by_cond: Dict[str, List[OnboardingResult]] = {}
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
            float((r.oracle_rewards - r.final_rewards).sum()) for r in runs
        ]
        full_cs: List[Optional[float]] = [
            compute_costsave_at_quality(
                r.final_rewards, r.final_costs,
                strong_reward, strong_cost, threshold=0.95,
            )
            for r in runs
        ]
        conv_speeds = [
            float(compute_convergence_speed(r.checkpoints, MISTRAL_ID))
            for r in runs
        ]
        gem_disps: List[Optional[float]] = [
            float(v) if (v := compute_gemini_displacement(r.checkpoints))
            is not None else None
            for r in runs
        ]
        late_cs: List[Optional[float]] = [
            compute_late_stage_costsave(r, strong_reward, strong_cost)
            for r in runs
        ]
        early_costs = [compute_early_stage_cost(r) for r in runs]

        metrics[cond] = {
            "cumulative_regret": _agg(cum_regrets),
            "full_costsave_95": _agg(full_cs),
            "convergence_speed": _agg(conv_speeds),
            "gemini_displacement": _agg(gem_disps),
            "late_costsave_95": _agg(late_cs),
            "early_cost": _agg(early_costs),
        }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


CONDITION_COLORS = {
    "Hybrid LinUCB": CB_BLUE,
    "Disjoint LinUCB": CB_GRAY,
}


def _build_combined_timeline(
    pretrain_agg: List[Dict[str, Any]],
    eval_agg: List[Dict[str, Any]],
    n_pretrain: int,
) -> List[Dict[str, Any]]:
    """Merge Phase-1 tail and Phase-3 checkpoints into a single timeline.

    Phase-1 checkpoints keep their original ``n_seen``.
    Phase-3 checkpoints are offset by ``n_pretrain`` so the x-axis
    is continuous.

    Args:
        pretrain_agg: Aggregated Phase-1 checkpoints.
        eval_agg: Aggregated Phase-3 checkpoints.
        n_pretrain: Total Phase-1 prompts (onboarding boundary).

    Returns:
        Combined list sorted by the unified ``n_seen``.
    """
    cutoff = n_pretrain - PRETRAIN_DISPLAY
    tail = [d for d in pretrain_agg if d["n_seen"] >= cutoff]
    shifted = [{**d, "n_seen": d["n_seen"] + n_pretrain} for d in eval_agg]
    return tail + shifted


def plot_routing_mix(
    aggregated: Dict[str, List[Dict[str, Any]]],
    aggregated_pretrain: Dict[str, List[Dict[str, Any]]],
    metrics: Dict[str, Dict[str, Any]],
    n_pretrain: int,
    out_dir: Path,
) -> Path:
    """Two-row, two-column onboarding figure (Hybrid | Disjoint).

    Each column is one condition.  Each column has two vertically
    stacked panels sharing the same x-axis (prompts processed):

    * **Top row** — stacked-area routing mix (windowed) spanning the
      K=2 steady state, the onboarding event, and the K=3 transition.
    * **Bottom row** — cumulative average reward (mean ± SE across seeds)
      on the same timeline, showing the reward dip at onboarding and
      the subsequent recovery.

    A vertical dashed line marks the onboarding point in every panel.

    Args:
        aggregated: Aggregated Phase-3 checkpoint data.
        aggregated_pretrain: Aggregated Phase-1 checkpoint data.
        metrics: Output of :func:`compute_all_onboarding_metrics`.
        n_pretrain: Total Phase-1 prompts (for the onboarding marker).
        out_dir: Output directory for the figure.

    Returns:
        Path to the saved figure.
    """
    conditions = ["Hybrid LinUCB", "Disjoint LinUCB"]

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.3, 1], hspace=0.25, wspace=0.12,
    )
    ax_h = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1], sharey=ax_h)
    ax_cmp = fig.add_subplot(gs[1, :])

    ax_mix_panels = {"Hybrid LinUCB": ax_h, "Disjoint LinUCB": ax_d}
    combined_data: Dict[str, tuple] = {}

    # ── Top row: per-model routing share lines ────────────────────
    for cond in conditions:
        ax = ax_mix_panels[cond]
        if cond not in aggregated or cond not in aggregated_pretrain:
            ax.set_title(f"{cond} (no data)")
            continue

        combined = _build_combined_timeline(
            aggregated_pretrain[cond], aggregated[cond], n_pretrain,
        )
        xs = np.array([d["n_seen"] for d in combined])
        combined_data[cond] = (xs, combined)

        for arm in K3_ARM_ORDER:
            ys = np.array([d["windowed_mix_mean"][arm] for d in combined])
            se = np.array([d["windowed_mix_se"].get(arm, 0.0)
                           for d in combined])
            ax.plot(xs, ys, color=ARM_COLORS[arm], lw=2,
                    label=ARM_LABELS[arm])
            ax.fill_between(xs, ys - se, ys + se,
                            alpha=0.15, color=ARM_COLORS[arm])

        ax.axvline(n_pretrain, color="k", ls="--", lw=1.2, alpha=0.7)
        ax.annotate(
            "onboard Mistral", xy=(n_pretrain, 0.50),
            xycoords=("data", "axes fraction"),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color="k", lw=0.8),
        )
        ax.set_title(cond, fontsize=12, fontweight="bold")
        ax.set_xlabel("Prompts processed", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(xs[0], xs[-1])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(loc="center right", fontsize=8, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3, ls=":")

    ax_h.set_ylabel(
        f"Routing share\n(window = {WINDOW_SIZE})", fontsize=10,
    )
    plt.setp(ax_d.get_yticklabels(), visible=False)

    # ── Bottom: cumulative avg reward overlay (Hybrid vs Disjoint) ─
    for cond in conditions:
        if cond not in combined_data:
            continue
        xs, combined = combined_data[cond]
        r_mean = np.array([d["mean_reward"] for d in combined])
        r_se = np.array([d.get("reward_se", 0.0) for d in combined])

        color = CONDITION_COLORS[cond]
        ax_cmp.plot(xs, r_mean, color=color, lw=2, label=cond)
        ax_cmp.fill_between(xs, r_mean - r_se, r_mean + r_se,
                            alpha=0.15, color=color)

    ax_cmp.axvline(n_pretrain, color="k", ls="--", lw=1.2, alpha=0.7)
    ax_cmp.annotate(
        "onboard Mistral", xy=(n_pretrain, 0.50),
        xycoords=("data", "axes fraction"),
        xytext=(8, 0), textcoords="offset points",
        fontsize=8, va="center", ha="left",
        arrowprops=dict(arrowstyle="-", color="k", lw=0.8),
    )
    ax_cmp.set_xlabel("Prompts processed", fontsize=10)
    ax_cmp.set_ylabel("Cumulative avg. reward", fontsize=10)
    ax_cmp.legend(fontsize=9, framealpha=0.9)
    ax_cmp.grid(axis="y", alpha=0.3, ls=":")
    if combined_data:
        xs0 = next(iter(combined_data.values()))[0]
        ax_cmp.set_xlim(xs0[0], xs0[-1])

    fig.suptitle(
        "Model Onboarding: K=2 → K=3 (Mistral-Large added mid-stream)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "figure3_onboarding.pdf"
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
    all_results: List[OnboardingResult],
    metrics: Dict[str, Dict[str, Any]],
    strong_reward: float,
    strong_cost: float,
    cost_penalty: float,
    elapsed_s: float,
    out_dir: Path,
) -> Path:
    """Write machine-readable JSON with all metrics.

    Args:
        aggregated: Checkpoint data per condition.
        all_results: Raw per-seed results.
        metrics: Output of :func:`compute_all_onboarding_metrics`.
        strong_reward: Mean reward of the strong model (Gemini).
        strong_cost: Mean cost of the strong model (Gemini).
        cost_penalty: Cost penalty λ used for this run.
        elapsed_s: Wall-clock time.
        out_dir: Output directory.

    Returns:
        Path to the saved JSON.
    """
    costsave_by_cond: Dict[str, List[Optional[float]]] = {}
    for r in all_results:
        cs = compute_costsave_at_quality(
            r.final_rewards, r.final_costs,
            strong_reward, strong_cost, threshold=0.95,
        )
        costsave_by_cond.setdefault(r.condition, []).append(cs)

    costsave_overall: Dict[str, Any] = {}
    for cond, cs_list in costsave_by_cond.items():
        valid = [v for v in cs_list if v is not None]
        costsave_overall[cond] = {
            "costsave_95_mean": float(np.mean(valid)) if valid else None,
            "costsave_95_se": (
                float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
                if len(valid) > 1 else None
            ),
            "n_seeds": len(cs_list),
        }

    final_mix: Dict[str, Dict[str, float]] = {}
    for cond, agg in aggregated.items():
        final_mix[cond] = agg[-1]["routing_mix_mean"]

    hybrid_early = metrics.get("Hybrid LinUCB", {}).get("early_cost", {})
    disjoint_early = metrics.get("Disjoint LinUCB", {}).get("early_cost", {})
    h_mean = hybrid_early.get("mean")
    d_mean = disjoint_early.get("mean")
    transition_overhead_pct = (
        ((d_mean - h_mean) / h_mean * 100.0)
        if h_mean is not None and d_mean is not None and h_mean > 0
        else None
    )

    payload = {
        "experiment": "model_onboarding_k2_to_k3",
        "hparams": BEST_K2_CORRALLING_HPARAMS,
        "cost_penalty": cost_penalty,
        "exploration_boost": "none (shared β handles cold-start)",
        "gamma_ramp_steps": 500,
        "newcomer": MISTRAL_ID,
        "strong_model_reward": strong_reward,
        "strong_model_cost": strong_cost,
        "costsave_95_overall": costsave_overall,
        "onboarding_metrics": metrics,
        "transition_overhead_pct": transition_overhead_pct,
        "final_routing_mix": final_mix,
        "window_size": WINDOW_SIZE,
        "late_stage_n": LATE_STAGE_N,
        "early_stage_n": EARLY_STAGE_N,
        "convergence_tolerance": CONVERGENCE_TOLERANCE,
        "checkpoints": {
            cond: [
                {
                    "n_seen": cp["n_seen"],
                    "routing_mix_cumulative": cp["routing_mix_mean"],
                    "routing_mix_windowed": cp["windowed_mix_mean"],
                    "mean_reward": cp["mean_reward"],
                    "mean_cost": cp["mean_cost"],
                }
                for cp in agg
            ]
            for cond, agg in aggregated.items()
        },
        "wall_time_s": round(elapsed_s, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "onboarding_data.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Data saved to %s", json_path)
    return json_path


# ═══════════════════════════════════════════════════════════════════════════
# Console Summary
# ═══════════════════════════════════════════════════════════════════════════


def print_summary(
    aggregated: Dict[str, List[Dict[str, Any]]],
    all_results: List[OnboardingResult],
    metrics: Dict[str, Dict[str, Any]],
    strong_reward: float,
    strong_cost: float,
    test_n: int,
) -> None:
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 72)
    print("RQ2 — MODEL ONBOARDING: K=2 → K=3")
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

    for cond in ["Hybrid LinUCB", "Disjoint LinUCB"]:
        if cond not in aggregated:
            continue
        agg = aggregated[cond]
        final = agg[-1]
        m = metrics.get(cond, {})

        print(f"\n{cond}")
        print("-" * 50)

        # Primary metrics (full-sequence, standard online-learning)
        print("  [PRIMARY — full sequence]")
        cr_info = m.get("cumulative_regret", {})
        print(f"  Cumulative regret: {_fmt(cr_info)}")
        fc_info = m.get("full_costsave_95", {})
        print(f"  CostSave@95% (all {test_n} prompts): {_fmt(fc_info, '%')}")

        # Routing summary
        print("  [Routing summary]")
        print("  Final routing mix (cumulative):")
        for arm in K3_ARM_ORDER:
            pct = final["routing_mix_mean"][arm] * 100
            print(f"    {ARM_LABELS[arm]:20s}  {pct:5.1f}%")
        print(f"  Mean reward: {final['mean_reward']:.4f}")
        print(f"  Mean cost:   ${final['mean_cost']:.6f}")

        # Supplementary metrics (dynamics)
        print("  [Supplementary — dynamics]")
        cs_info = m.get("convergence_speed", {})
        print(f"  Mistral convergence speed: {_fmt(cs_info, ' prompts')}")

        gd_info = m.get("gemini_displacement", {})
        print(f"  Gemini displacement (<10%): {_fmt(gd_info, ' prompts')}")

        lc_info = m.get("late_costsave_95", {})
        print(f"  Late-stage CostSave@95% (last {LATE_STAGE_N}): "
              f"{_fmt(lc_info, '%')}")

        ec_info = m.get("early_cost", {})
        print(f"  Early-stage mean cost (first {EARLY_STAGE_N}): "
              f"${_fmt(ec_info, precision=4)}")

    # ── Cross-condition: transition overhead ──────────────────────────
    h_early = metrics.get("Hybrid LinUCB", {}).get("early_cost", {})
    d_early = metrics.get("Disjoint LinUCB", {}).get("early_cost", {})
    h_mean = h_early.get("mean")
    d_mean = d_early.get("mean")
    if h_mean is not None and d_mean is not None and h_mean > 0:
        overhead = (d_mean - h_mean) / h_mean * 100
        print(f"\n  Transition overhead (Disjoint vs Hybrid, "
              f"first {EARLY_STAGE_N}): {overhead:+.1f}%")

    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Model Onboarding (K=2 → K=3)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=20,
        help="Number of seeds per condition (default: 20)",
    )
    parser.add_argument(
        "--cost-penalty", type=float, default=0.20,
        help=(
            "Cost penalty λ for all conditions (default: 0.20). "
            "Matches the median λ from the K=2 Pareto sweep "
            "(see Experiment 1)."
        ),
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick run with 2 seeds for debugging",
    )
    args = parser.parse_args()

    n_seeds = 2 if args.fast else args.n_seeds
    cost_penalty = args.cost_penalty
    warmup_path = str(K2_WARMUP_PRIORS_PATH)
    hparams = dict(BEST_K2_CORRALLING_HPARAMS)

    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading data and encoding prompts ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    logger.info("  Feature dim: %d", feature_dim)

    train_k2 = load_split(TRAIN_DATA_PATH, fs, K2_ARM_ORDER)
    test_k3 = load_split(HOLDOUT_DATA_PATH, fs, K3_ARM_ORDER)
    logger.info(
        "  Train (K=2): %d, Test (K=3): %d prompts",
        train_k2.n, test_k3.n,
    )

    registry_k2 = build_model_registry(K2_ARM_ORDER)
    logger.info("  K=2 registry: %s", list(registry_k2.keys()))
    logger.info(
        "  α=%.2f (no boost — shared β handles onboarding)",
        hparams["alpha"],
    )

    strong_model = "google/gemini-2.5-pro"
    strong_reward = float(test_k3.rewards[strong_model].mean())
    strong_cost = float(test_k3.costs[strong_model].mean())
    logger.info(
        "  Strong model: %s (reward=%.4f, cost=$%.6f)",
        strong_model, strong_reward, strong_cost,
    )

    # ── Run simulations ───────────────────────────────────────────────
    all_results: List[OnboardingResult] = []

    for cond_name, policy_override in [
        ("Hybrid LinUCB", None),
        ("Disjoint LinUCB", "disjoint"),
    ]:
        logger.info("\n%s: %d seeds", cond_name, n_seeds)
        for s in range(n_seeds):
            seed = SEED_OFFSET + s
            logger.info("  Seed %d/%d (seed=%d)", s + 1, n_seeds, seed)
            result = simulate_onboarding(
                train_k2, test_k3, registry_k2, feature_dim,
                hparams=hparams,
                warmup_path=warmup_path,
                cost_penalty=cost_penalty,
                seed=seed,
                policy_override=policy_override,
                condition_name=cond_name,
            )
            all_results.append(result)
            final_cp = result.checkpoints[-1]
            mistral_pct = final_cp.routing_mix[MISTRAL_ID] * 100
            logger.info(
                "    Final Mistral share: %.1f%%, reward: %.4f",
                mistral_pct, final_cp.cumulative_reward,
            )

    elapsed = time.time() - t0

    # ── Aggregate and report ──────────────────────────────────────────
    aggregated = aggregate_checkpoints(all_results)

    pretrain_proxy = [
        OnboardingResult(
            condition=r.condition,
            seed=r.seed,
            n_pretrain=r.n_pretrain,
            pretrain_checkpoints=[],
            checkpoints=r.pretrain_checkpoints,
            final_rewards=r.final_rewards,
            final_costs=r.final_costs,
            final_choices=r.final_choices,
            oracle_rewards=r.oracle_rewards,
        )
        for r in all_results
    ]
    aggregated_pretrain = aggregate_checkpoints(pretrain_proxy)

    n_pretrain = all_results[0].n_pretrain
    onboarding_metrics = compute_all_onboarding_metrics(
        all_results, aggregated, strong_reward, strong_cost,
    )

    fig_path = plot_routing_mix(
        aggregated, aggregated_pretrain, onboarding_metrics,
        n_pretrain, RESULTS_DIR,
    )
    json_path = export_results(
        aggregated, all_results, onboarding_metrics,
        strong_reward, strong_cost, cost_penalty,
        elapsed, RESULTS_DIR,
    )
    print_summary(
        aggregated, all_results, onboarding_metrics,
        strong_reward, strong_cost, test_k3.n,
    )

    logger.info("\nTotal wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
