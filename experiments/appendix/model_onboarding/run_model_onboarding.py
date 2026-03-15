#!/usr/bin/env python3
"""Appendix: Model Onboarding Under Budget Constraints (K=3 → K=4).

Demonstrates that BanditGPT can incorporate a newly released model
(Gemini-2.5-Flash) into a running K=3 portfolio while the BudgetPacer
maintains cost compliance.

Experimental setup
------------------
  **Phase 1** (K=3 online learning):  The router learns on the val split
  (1,785 prompts) using the canonical K=3 portfolio (Llama-8B,
  Mistral-Large, Gemini-Pro) with warmup priors and a BudgetPacer at
  three budget targets (tight / moderate / loose).

  **Phase 2** (K=4 onboarding + evaluation):  ``register_model()`` adds
  Gemini-2.5-Flash as a fourth arm.  The router continues online learning
  on the K=4 test split, where Flash rewards are available.

Key questions
-------------
  1. Does Flash get explored despite the budget constraint?
  2. Which arm does Flash displace (if any)?
  3. Does the pacer maintain budget compliance after onboarding?
  4. Does BanditGPT onboarding outperform simpler strategies?

Conditions (per budget target)
------------------------------
  - **Fixed Policy (uniform 1/4)**: No routing intelligence — equal
    allocation across all K=4 arms.  Simplest possible onboarding
    strategy.
  - **BanditGPT (transfer)**: Phase 1 posteriors carry over; Flash gets
    a T-shirt prior via ``register_model()``.
  - **BanditGPT (unconstrained)**: Same as transfer but without budget
    pacer — quality ceiling reference.

Hyperparameter note
-------------------
  ``alpha=0.1`` and ``prior_n_effective=10.0`` are simulation-tuned
  values, consistent with Experiments 02-03.  The production-tuned
  values (alpha=0.01, n_eff=5000 from the hparam sweep appendix) are
  designed for long-horizon deployment; in a 1.8K-prompt simulation,
  the weaker priors allow the bandit to adapt within the available
  time horizon.

Output: ``results/model_onboarding_results.json``

Usage
-----
    python experiments/appendix/model_onboarding/run_model_onboarding.py
    python experiments/appendix/model_onboarding/run_model_onboarding.py --fast
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.budget_pacer import BudgetPacer, PacingMode
from bandit_gpt.config import (
    K3_ARM_ORDER,
    K3_WARMUP_PRIORS_PATH,
    K4_MODELS_PATH,
    OFFLINE_DATASET_DIR,
    VAL_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.simulation import SplitData, build_model_registry, load_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service", "bandit_gpt.policy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Constants
# ======================================================================

K3_ARMS: List[str] = K3_ARM_ORDER
K4_ARMS: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]
FLASH_ID = "google/gemini-2.5-flash"

ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-flash": "Flash",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

N_SEEDS: int = 20
SEED_OFFSET: int = 9000
RESULTS_DIR = Path(__file__).parent / "results"

CHECKPOINT_INTERVAL: int = 25
WINDOW_SIZE: int = 100

# Simulation-tuned hparams (see docstring for rationale).
PRIOR_N_EFFECTIVE: float = 10.0
ALPHA: float = 0.1
FORGETTING_FACTOR: float = 0.997

PACER_LR: float = 0.05
PACER_LAMBDA_MAX: float = 5.0
PACER_EMA_ALPHA: float = 0.05

BUDGET_TARGETS: List[float] = [2.34e-4, 6.62e-4, 1.87e-3]
BUDGET_LABELS: List[str] = ["tight", "moderate", "loose"]

FLASH_INPUT_COST_PER_M: float = 0.3
FLASH_OUTPUT_COST_PER_M: float = 2.5
FLASH_BLENDED_COST_PER_M: float = (
    (FLASH_INPUT_COST_PER_M + FLASH_OUTPUT_COST_PER_M) / 2.0
)

# K=4 data paths (produced by merge_flash_into_splits.py)
VAL_K4_PATH = OFFLINE_DATASET_DIR / "val_k4.jsonl"
TEST_K4_PATH = OFFLINE_DATASET_DIR / "test_k4.jsonl"

# Sustained-adoption thresholds
SUSTAINED_THRESHOLD: float = 0.10
SUSTAINED_HOLD_STEPS: int = 50


# ======================================================================
# Phase 2 strategies
# ======================================================================

STRATEGY_BANDITGPT_TRANSFER = "banditgpt_transfer"
STRATEGY_FIXED_UNIFORM = "fixed_uniform"


# ======================================================================
# Data types
# ======================================================================


@dataclass
class Checkpoint:
    """Routing statistics snapshot at a given step count."""

    step: int
    phase: str
    routing_mix: Dict[str, float]
    windowed_mix: Dict[str, float]
    cumulative_reward: float
    cumulative_cost: float
    cumulative_regret: float
    lambda_t: float


@dataclass
class TrialResult:
    """Full result for a single (condition, seed) trial."""

    condition: str
    strategy: str
    budget_label: str
    budget_target: float
    seed: int
    n_phase1: int
    n_phase2: int
    phase1_reward: float
    phase1_cost: float
    phase1_regret: float
    phase2_reward: float
    phase2_cost: float
    phase2_regret: float
    overall_reward: float
    overall_cost: float
    overall_regret: float
    final_model_fractions: Dict[str, float]
    phase2_model_fractions: Dict[str, float]
    flash_first_selected: Optional[int]
    flash_sustained_step: Optional[int]
    flash_final_share: float
    displacement: Dict[str, float]
    checkpoints: List[Checkpoint] = field(default_factory=list)


# ======================================================================
# Data loading
# ======================================================================


def _load_k3(fs: FeatureService) -> SplitData:
    """Load the K=3 val split for Phase 1."""
    return load_split(VAL_DATA_PATH, fs, K3_ARMS)


def _load_k4_eval(fs: FeatureService) -> SplitData:
    """Load the K=4 test split for Phase 2.

    Falls back to val_k4 if test_k4 doesn't exist yet, so the experiment
    structure can be tested before full data collection.
    """
    path = TEST_K4_PATH if TEST_K4_PATH.exists() else VAL_K4_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"K=4 data not found at {TEST_K4_PATH} or {VAL_K4_PATH}. "
            "Run collect_flash_canonical.py + merge_flash_into_splits.py first."
        )
    return load_split(path, fs, K4_ARMS)


# ======================================================================
# Router factory
# ======================================================================


def _make_pacer(budget_target: float) -> Optional[BudgetPacer]:
    """Create a BudgetPacer if budget_target > 0, else None."""
    if budget_target <= 0:
        return None
    return BudgetPacer(
        target_avg_spend_usd=budget_target,
        mode=PacingMode.ADAPTIVE,
        lr=PACER_LR,
        lambda_max=PACER_LAMBDA_MAX,
        ema_alpha=PACER_EMA_ALPHA,
    )


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    cost_penalty: float = 0.0,
    budget_pacer: Optional[BudgetPacer] = None,
    warmup: bool = True,
) -> BanditRouter:
    """Build a router with optional warmup priors and BudgetPacer."""
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    return BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if warmup else "none",
        warmup_path=str(K3_WARMUP_PRIORS_PATH) if warmup else None,
        prior_n_effective=PRIOR_N_EFFECTIVE,
        alpha=ALPHA,
        use_corralling=False,
        cost_penalty=cost_penalty,
        forgetting_factor=FORGETTING_FACTOR,
        policy="disjoint",
        budget_pacer=budget_pacer,
        adaptive_gamma=False,
    )


# ======================================================================
# Sustained-adoption detection
# ======================================================================


def _compute_sustained_step(
    windowed_history: List[float],
    threshold: float = SUSTAINED_THRESHOLD,
    hold: int = SUSTAINED_HOLD_STEPS,
) -> Optional[int]:
    """Return the 1-indexed step where Flash share first exceeds *threshold*
    and remains above it for *hold* consecutive checkpoints.

    Args:
        windowed_history: Flash windowed-mix value at each step (not just
            checkpoints — one entry per Phase 2 step).
        threshold: Minimum share to qualify as "adopted".
        hold: Number of consecutive steps above threshold.

    Returns:
        The 1-indexed step of the first qualifying window, or ``None``.
    """
    run = 0
    for i, share in enumerate(windowed_history):
        if share >= threshold:
            run += 1
            if run >= hold:
                return i - hold + 2  # 1-indexed start of the run
        else:
            run = 0
    return None


def _compute_displacement(
    early_mix: Dict[str, float],
    late_mix: Dict[str, float],
) -> Dict[str, float]:
    """Share change per arm: late minus early (positive = gained share)."""
    return {a: late_mix.get(a, 0.0) - early_mix.get(a, 0.0) for a in K4_ARMS}


# ======================================================================
# Simulation
# ======================================================================


def _run_trial(
    train_k3: SplitData,
    eval_k4: SplitData,
    registry_k3: Dict[str, Any],
    feature_dim: int,
    *,
    condition: str,
    strategy: str,
    budget_label: str,
    budget_target: float,
    seed: int,
) -> TrialResult:
    """Run one K=3 train → onboard Flash → K=4 eval trial.

    Args:
        train_k3: K=3 val split for Phase 1 online learning.
        eval_k4: K=4 test split for Phase 2 online evaluation.
        registry_k3: K=3 model registry.
        feature_dim: Embedding dimensionality.
        condition: Human-readable label for this condition.
        strategy: One of ``STRATEGY_*`` constants controlling Phase 2
            model selection and learning behaviour.
        budget_label: Budget tier name.
        budget_target: Dollar budget target per request (0 = unconstrained).
        seed: Random seed.

    Returns:
        Complete trial result with per-step checkpoints and adoption
        metrics.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    pacer = _make_pacer(budget_target)

    router = _create_router(
        registry_k3, feature_dim,
        cost_penalty=0.0,
        budget_pacer=pacer,
    )

    # ── Phase 1: K=3 online learning (identical for all strategies) ───
    checkpoints: List[Checkpoint] = []
    arm_counts: Dict[str, int] = {a: 0 for a in K4_ARMS}
    recent: deque = deque(maxlen=WINDOW_SIZE)
    cum_reward, cum_cost, cum_regret = 0.0, 0.0, 0.0

    train_order = rng.permutation(train_k3.n)
    for step_idx, i in enumerate(train_order):
        model, log = router.route(train_k3.embeddings[i])
        reward = float(train_k3.rewards[model][i])
        cost = float(train_k3.costs[model][i])
        oracle = max(float(train_k3.rewards[a][i]) for a in K3_ARMS)
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

        arm_counts[model] += 1
        recent.append(model)
        cum_reward += reward
        cum_cost += cost
        cum_regret += (oracle - reward)

        step = step_idx + 1
        if step % CHECKPOINT_INTERVAL == 0 or step == train_k3.n:
            w_total = len(recent)
            lam = pacer.lambda_t if pacer is not None else 0.0
            checkpoints.append(Checkpoint(
                step=step,
                phase="phase1",
                routing_mix={a: arm_counts[a] / step for a in K4_ARMS},
                windowed_mix={
                    a: sum(1 for c in recent if c == a) / w_total
                    for a in K4_ARMS
                },
                cumulative_reward=cum_reward / step,
                cumulative_cost=cum_cost / step,
                cumulative_regret=cum_regret,
                lambda_t=lam,
            ))

    phase1_reward = cum_reward / train_k3.n
    phase1_cost = cum_cost / train_k3.n
    phase1_regret = cum_regret

    # ── Onboard Gemini Flash ──────────────────────────────────────────
    if strategy == STRATEGY_BANDITGPT_TRANSFER:
        router.register_model(
            FLASH_ID,
            speed="fast",
            cost_usd=FLASH_INPUT_COST_PER_M,
            blended_cost_per_m=FLASH_BLENDED_COST_PER_M,
        )

    # ── Phase 2: K=4 evaluation ───────────────────────────────────────
    p2_cum_reward, p2_cum_cost, p2_cum_regret = 0.0, 0.0, 0.0
    p2_arm_counts: Dict[str, int] = {a: 0 for a in K4_ARMS}
    p2_recent: deque = deque(maxlen=WINDOW_SIZE)

    flash_first_selected: Optional[int] = None
    flash_windowed_history: List[float] = []

    # For displacement: track early (first 200) and late (last 200) arm mix
    early_window_size = 200
    late_window_size = 200
    early_arm_counts: Dict[str, int] = {a: 0 for a in K4_ARMS}
    late_arm_counts: Dict[str, int] = {a: 0 for a in K4_ARMS}

    eval_order = rng.permutation(eval_k4.n)
    n2 = eval_k4.n

    for step_idx, i in enumerate(eval_order):
        # ── Model selection depends on strategy ───────────────────────
        if strategy == STRATEGY_FIXED_UNIFORM:
            model = rng.choice(K4_ARMS)
        else:
            model, log = router.route(eval_k4.embeddings[i])
            log.cost_usd = float(eval_k4.costs[model][i])
            router.process_feedback(log.request_id, reward=float(eval_k4.rewards[model][i]))

        reward = float(eval_k4.rewards[model][i])
        cost = float(eval_k4.costs[model][i])
        oracle = max(float(eval_k4.rewards[a][i]) for a in K4_ARMS)

        p2_arm_counts[model] += 1
        p2_recent.append(model)
        p2_cum_reward += reward
        p2_cum_cost += cost
        p2_cum_regret += (oracle - reward)

        # Flash tracking
        if model == FLASH_ID and flash_first_selected is None:
            flash_first_selected = step_idx + 1

        w_total = len(p2_recent)
        flash_share = sum(1 for c in p2_recent if c == FLASH_ID) / w_total
        flash_windowed_history.append(flash_share)

        # Displacement windows
        if step_idx < early_window_size:
            early_arm_counts[model] += 1
        if step_idx >= n2 - late_window_size:
            late_arm_counts[model] += 1

        # Checkpoints
        p2_step = step_idx + 1
        global_step = train_k3.n + p2_step
        if p2_step % CHECKPOINT_INTERVAL == 0 or p2_step == n2:
            total_step = train_k3.n + p2_step
            total_counts = {
                a: arm_counts[a] + p2_arm_counts[a] for a in K4_ARMS
            }
            lam = pacer.lambda_t if pacer is not None else 0.0
            checkpoints.append(Checkpoint(
                step=global_step,
                phase="phase2",
                routing_mix={
                    a: total_counts[a] / total_step for a in K4_ARMS
                },
                windowed_mix={
                    a: sum(1 for c in p2_recent if c == a) / w_total
                    for a in K4_ARMS
                },
                cumulative_reward=(cum_reward + p2_cum_reward) / total_step,
                cumulative_cost=(cum_cost + p2_cum_cost) / total_step,
                cumulative_regret=cum_regret + p2_cum_regret,
                lambda_t=lam,
            ))

    phase2_reward = p2_cum_reward / n2 if n2 > 0 else 0.0
    phase2_cost = p2_cum_cost / n2 if n2 > 0 else 0.0
    phase2_regret = p2_cum_regret

    total_n = train_k3.n + n2
    overall_reward = (cum_reward + p2_cum_reward) / total_n
    overall_cost = (cum_cost + p2_cum_cost) / total_n
    overall_regret = cum_regret + p2_cum_regret

    total_counts = {a: arm_counts[a] + p2_arm_counts[a] for a in K4_ARMS}
    final_fractions = {a: total_counts[a] / total_n for a in K4_ARMS}
    p2_fractions = (
        {a: p2_arm_counts[a] / n2 for a in K4_ARMS} if n2 > 0 else {}
    )

    # Sustained adoption
    flash_sustained = _compute_sustained_step(flash_windowed_history)

    # Flash final share: last *late_window_size* steps
    late_total = sum(late_arm_counts.values()) or 1
    flash_final_share = late_arm_counts.get(FLASH_ID, 0) / late_total

    # Displacement: change from early to late window
    early_total = sum(early_arm_counts.values()) or 1
    early_mix = {a: early_arm_counts[a] / early_total for a in K4_ARMS}
    late_mix = {a: late_arm_counts[a] / late_total for a in K4_ARMS}
    displacement = _compute_displacement(early_mix, late_mix)

    return TrialResult(
        condition=condition,
        strategy=strategy,
        budget_label=budget_label,
        budget_target=budget_target,
        seed=seed,
        n_phase1=train_k3.n,
        n_phase2=n2,
        phase1_reward=phase1_reward,
        phase1_cost=phase1_cost,
        phase1_regret=phase1_regret,
        phase2_reward=phase2_reward,
        phase2_cost=phase2_cost,
        phase2_regret=phase2_regret,
        overall_reward=overall_reward,
        overall_cost=overall_cost,
        overall_regret=overall_regret,
        final_model_fractions=final_fractions,
        phase2_model_fractions=p2_fractions,
        flash_first_selected=flash_first_selected,
        flash_sustained_step=flash_sustained,
        flash_final_share=flash_final_share,
        displacement=displacement,
        checkpoints=checkpoints,
    )


# ======================================================================
# Aggregation
# ======================================================================


def _stat(values: List[float]) -> Dict[str, float]:
    """Mean, standard error, and std for a list of scalars."""
    arr = np.array(values)
    n = len(arr)
    return {
        "mean": float(arr.mean()),
        "se": float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
    }


def _aggregate_trials(
    trials: List[TrialResult],
) -> Dict[str, Any]:
    """Aggregate per-seed trials into summary statistics."""
    n = len(trials)
    if n == 0:
        return {}

    p2_fracs: Dict[str, List[float]] = {a: [] for a in K4_ARMS}
    displacements: Dict[str, List[float]] = {a: [] for a in K4_ARMS}
    for t in trials:
        for a in K4_ARMS:
            p2_fracs[a].append(t.phase2_model_fractions.get(a, 0.0))
            displacements[a].append(t.displacement.get(a, 0.0))

    first_steps = [
        t.flash_first_selected
        for t in trials
        if t.flash_first_selected is not None
    ]
    sustained_steps = [
        t.flash_sustained_step
        for t in trials
        if t.flash_sustained_step is not None
    ]

    return {
        "condition": trials[0].condition,
        "strategy": trials[0].strategy,
        "budget_label": trials[0].budget_label,
        "budget_target": trials[0].budget_target,
        "n_seeds": n,
        "phase1_reward": _stat([t.phase1_reward for t in trials]),
        "phase1_cost": _stat([t.phase1_cost for t in trials]),
        "phase1_regret": _stat([t.phase1_regret for t in trials]),
        "phase2_reward": _stat([t.phase2_reward for t in trials]),
        "phase2_cost": _stat([t.phase2_cost for t in trials]),
        "phase2_regret": _stat([t.phase2_regret for t in trials]),
        "overall_reward": _stat([t.overall_reward for t in trials]),
        "overall_cost": _stat([t.overall_cost for t in trials]),
        "overall_regret": _stat([t.overall_regret for t in trials]),
        "phase2_model_fractions": {
            a: _stat(p2_fracs[a]) for a in K4_ARMS
        },
        "displacement": {
            a: _stat(displacements[a]) for a in K4_ARMS
        },
        "flash_adoption": {
            "n_first_selected": len(first_steps),
            "mean_first_step": (
                float(np.mean(first_steps)) if first_steps else None
            ),
            "n_sustained": len(sustained_steps),
            "mean_sustained_step": (
                float(np.mean(sustained_steps)) if sustained_steps else None
            ),
            "flash_final_share": _stat(
                [t.flash_final_share for t in trials]
            ),
        },
    }


def _aggregate_checkpoints(
    trials: List[TrialResult],
) -> List[Dict[str, Any]]:
    """Average checkpoint traces across seeds for plotting."""
    if not trials:
        return []

    ref = trials[0].checkpoints
    n_cp = len(ref)
    n_seeds = len(trials)

    agg: List[Dict[str, Any]] = []
    for cp_idx in range(n_cp):
        step = ref[cp_idx].step
        phase = ref[cp_idx].phase

        mix_arrays: Dict[str, List[float]] = {a: [] for a in K4_ARMS}
        wmix_arrays: Dict[str, List[float]] = {a: [] for a in K4_ARMS}
        rewards, costs, regrets, lambdas = [], [], [], []

        for trial in trials:
            if cp_idx >= len(trial.checkpoints):
                continue
            cp = trial.checkpoints[cp_idx]
            for a in K4_ARMS:
                mix_arrays[a].append(cp.routing_mix.get(a, 0.0))
                wmix_arrays[a].append(cp.windowed_mix.get(a, 0.0))
            rewards.append(cp.cumulative_reward)
            costs.append(cp.cumulative_cost)
            regrets.append(cp.cumulative_regret)
            lambdas.append(cp.lambda_t)

        def _se(vals: List[float]) -> float:
            if n_seeds > 1:
                return float(np.std(vals, ddof=1) / np.sqrt(n_seeds))
            return 0.0

        agg.append({
            "step": step,
            "phase": phase,
            "routing_mix_mean": {
                a: float(np.mean(mix_arrays[a])) for a in K4_ARMS
            },
            "routing_mix_se": {
                a: _se(mix_arrays[a]) for a in K4_ARMS
            },
            "windowed_mix_mean": {
                a: float(np.mean(wmix_arrays[a])) for a in K4_ARMS
            },
            "windowed_mix_se": {
                a: _se(wmix_arrays[a]) for a in K4_ARMS
            },
            "cumulative_reward": float(np.mean(rewards)),
            "cumulative_cost": float(np.mean(costs)),
            "cumulative_regret": float(np.mean(regrets)),
            "lambda_t_mean": float(np.mean(lambdas)),
            "lambda_t_se": _se(lambdas),
        })
    return agg


# ======================================================================
# Console summary
# ======================================================================


def _print_summary(all_summaries: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("APPENDIX: MODEL ONBOARDING UNDER BUDGET CONSTRAINTS (K=3 → K=4)")
    print("=" * 90)

    for key, summary in all_summaries.items():
        label = summary.get("budget_label", key)
        target = summary.get("budget_target", 0)
        cond = summary.get("condition", "")
        strat = summary.get("strategy", "")
        print(f"\n  [{label.upper()}] {cond}  (target=${target:.2e}/req)")
        print(f"  {'':4s}Strategy: {strat}")
        print(
            f"  {'':4s}Phase 2 reward: "
            f"{summary['phase2_reward']['mean']:.4f} "
            f"± {summary['phase2_reward']['se']:.4f}"
        )
        print(
            f"  {'':4s}Phase 2 cost:   "
            f"${summary['phase2_cost']['mean']:.6f}"
        )
        print(f"  {'':4s}Phase 2 model fractions:")
        for arm in K4_ARMS:
            frac = summary["phase2_model_fractions"][arm]
            short = ARM_SHORT[arm]
            print(
                f"  {'':8s}{short:<16s} "
                f"{frac['mean']*100:5.1f}% ± {frac['se']*100:.1f}%"
            )

        adoption = summary.get("flash_adoption", {})
        n_seeds = summary.get("n_seeds", 0)

        n_first = adoption.get("n_first_selected", 0)
        mean_first = adoption.get("mean_first_step")
        if mean_first is not None:
            print(
                f"  {'':4s}Flash first selected: step {mean_first:.0f} "
                f"(in {n_first}/{n_seeds} seeds)"
            )

        n_sust = adoption.get("n_sustained", 0)
        mean_sust = adoption.get("mean_sustained_step")
        if mean_sust is not None:
            print(
                f"  {'':4s}Flash sustained (>{SUSTAINED_THRESHOLD*100:.0f}% "
                f"for {SUSTAINED_HOLD_STEPS} steps): "
                f"step {mean_sust:.0f} ({n_sust}/{n_seeds} seeds)"
            )
        else:
            print(
                f"  {'':4s}Flash never sustained "
                f">{SUSTAINED_THRESHOLD*100:.0f}% in any seed"
            )

        final_share = adoption.get("flash_final_share", {})
        if final_share:
            print(
                f"  {'':4s}Flash final share (last 200): "
                f"{final_share['mean']*100:.1f}% "
                f"± {final_share['se']*100:.1f}%"
            )

        disp = summary.get("displacement", {})
        if disp:
            print(f"  {'':4s}Displacement (late − early):")
            for arm in K4_ARMS:
                d = disp[arm]
                short = ARM_SHORT[arm]
                sign = "+" if d["mean"] >= 0 else ""
                print(
                    f"  {'':8s}{short:<16s} {sign}"
                    f"{d['mean']*100:.1f}pp"
                )

    print("\n" + "=" * 90)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-seeds", type=int, default=N_SEEDS,
        help=f"Number of seeds (default: {N_SEEDS}).",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick run with 3 seeds for debugging.",
    )
    args = parser.parse_args()

    n_seeds = 3 if args.fast else args.n_seeds
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading data...")
    fs = FeatureService()
    feature_dim = fs.dimension

    train_k3 = _load_k3(fs)
    eval_k4 = _load_k4_eval(fs)
    logger.info("  K=3 val (Phase 1): %d prompts", train_k3.n)
    logger.info("  K=4 eval (Phase 2): %d prompts", eval_k4.n)

    registry_k3 = build_model_registry(K3_ARMS)

    # ── Build condition matrix ─────────────────────────────────────────
    #
    # For each budget tier we run:
    #   1. Fixed Policy (uniform 1/4)  — simplest onboarding baseline
    #   2. BanditGPT (transfer)        — the system under test
    # Plus one unconstrained BanditGPT run as a quality ceiling.
    conditions: List[Dict[str, Any]] = []

    for budget_label, budget_target in zip(BUDGET_LABELS, BUDGET_TARGETS):
        conditions.append({
            "condition": f"Fixed Policy ({budget_label})",
            "strategy": STRATEGY_FIXED_UNIFORM,
            "budget_label": budget_label,
            "budget_target": budget_target,
        })
        conditions.append({
            "condition": f"BanditGPT ({budget_label})",
            "strategy": STRATEGY_BANDITGPT_TRANSFER,
            "budget_label": budget_label,
            "budget_target": budget_target,
        })

    conditions.append({
        "condition": "BanditGPT (unconstrained)",
        "strategy": STRATEGY_BANDITGPT_TRANSFER,
        "budget_label": "unconstrained",
        "budget_target": 0.0,
    })

    all_results: Dict[str, List[TrialResult]] = {}
    all_summaries: Dict[str, Dict[str, Any]] = {}
    all_checkpoint_traces: Dict[str, List[Dict[str, Any]]] = {}

    for cond in conditions:
        key = f"{cond['strategy']}_{cond['budget_label']}"
        trials: List[TrialResult] = []

        logger.info(
            "\n%s [%s] (%d seeds)",
            cond["condition"], cond["strategy"], n_seeds,
        )
        for s in range(n_seeds):
            seed = SEED_OFFSET + s
            logger.info("  Seed %d/%d (seed=%d)", s + 1, n_seeds, seed)
            result = _run_trial(
                train_k3, eval_k4, registry_k3, feature_dim,
                condition=cond["condition"],
                strategy=cond["strategy"],
                budget_label=cond["budget_label"],
                budget_target=cond["budget_target"],
                seed=seed,
            )
            trials.append(result)

            flash_pct = result.phase2_model_fractions.get(FLASH_ID, 0) * 100
            logger.info(
                "    P2 Flash=%.1f%%, reward=%.4f, cost=$%.6f",
                flash_pct, result.phase2_reward, result.phase2_cost,
            )

        all_results[key] = trials
        all_summaries[key] = _aggregate_trials(trials)
        all_checkpoint_traces[key] = _aggregate_checkpoints(trials)

    elapsed = time.time() - t0

    # ── Export ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "model_onboarding_k3_to_k4",
        "flash_model": FLASH_ID,
        "n_seeds": n_seeds,
        "phase1_n": train_k3.n,
        "phase2_n": eval_k4.n,
        "k3_arms": K3_ARMS,
        "k4_arms": K4_ARMS,
        "hparams": {
            "alpha": ALPHA,
            "prior_n_effective": PRIOR_N_EFFECTIVE,
            "forgetting_factor": FORGETTING_FACTOR,
            "pacer_lr": PACER_LR,
            "pacer_lambda_max": PACER_LAMBDA_MAX,
        },
        "budget_targets": dict(zip(BUDGET_LABELS, BUDGET_TARGETS)),
        "strategies": [STRATEGY_FIXED_UNIFORM, STRATEGY_BANDITGPT_TRANSFER],
        "summaries": all_summaries,
        "checkpoint_traces": all_checkpoint_traces,
        "wall_time_s": round(elapsed, 1),
    }

    json_path = RESULTS_DIR / "model_onboarding_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Results saved to %s", json_path)

    _print_summary(all_summaries)
    logger.info("\nTotal wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
