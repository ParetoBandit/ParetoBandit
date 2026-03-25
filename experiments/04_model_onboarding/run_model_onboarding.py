#!/usr/bin/env python3
"""Appendix: Model Onboarding Under Budget Constraints (K=3 → K=4).

Demonstrates that ParetoBandit can incorporate a newly released model
(Gemini-2.5-Flash) into a running K=3 portfolio while the BudgetPacer
maintains cost compliance — and correctly *rejects* inferior models.

Experimental setup
------------------
  **Phase 1** (K=3 online learning):  The router learns on the val split
  (1,785 prompts) using the canonical K=3 portfolio (Llama-8B,
  Mistral-Large, Gemini-Pro) with warmup priors and a BudgetPacer at
  three budget targets (tight / moderate / loose).

  **Phase 2** (K=4 onboarding + evaluation):  ``register_model()`` adds
  Gemini-2.5-Flash as a fourth arm.  The router continues online learning
  on the K=4 test split, where Flash rewards are available.

Forced exploration (burn-in)
----------------------------
  With ``alpha={alpha}`` and strong K=3 warmup priors
  (``prior_n_effective={n_eff}``), the UCB exploration bonus for a
  cold-start arm is too small to trigger natural exploration.  Flash
  starts with ``A=I, b=0`` (``theta=0``), and ``alpha * sqrt(x' A_inv x)``
  cannot compete with K=3 arms' learned predictions (~0.5–0.7).

  To address this, the first ``BURNIN_PULLS`` prompts of Phase 2 are
  routed unconditionally to Flash (forced exploration).  This guarantees
  the bandit collects real observations before UCB selection takes over.
  After burn-in, the algorithm has enough data to discriminate:

  - Good model → ``theta_flash`` is competitive → UCB adopts
  - Bad model → ``theta_flash`` is low → UCB rejects

  The exploration cost is bounded at O(N) regret regardless of Flash
  quality.  Forced exploration is a standard strategy in the contextual
  bandit literature (cf. "forced exploration at decaying rate" in LLM
  routing, arXiv 2602.02061).

Onboarding scenarios
--------------------
  Three scenarios test whether the router *discriminates* rather than
  blindly adopting every new model:

  - **good_cheap** (default): Flash as collected — high quality, low cost.
    Expected: adopted and displaces expensive alternatives.
  - **bad_cheap**: Flash rewards scaled to 0.5×.  Expected: explored
    during burn-in then suppressed once the bandit learns the low reward.
  - **good_expensive**: Flash costs scaled to 10×.  Expected: suppressed
    under tight budgets by the pacer, partially adopted when unconstrained.

Key questions
-------------
  1. Does Flash get explored despite the budget constraint?
  2. Which arm does Flash displace (if any)?
  3. Does the pacer maintain budget compliance after onboarding?
  4. Does ParetoBandit onboarding outperform simpler strategies?
  5. Does the router correctly *reject* a bad or expensive new model?

Conditions (per budget target, per scenario)
--------------------------------------------
  - **Fixed Policy (uniform 1/4)**: No routing intelligence — equal
    allocation across all K=4 arms.  Simplest possible onboarding
    strategy.
  - **ParetoBandit (transfer)**: Phase 1 posteriors carry over; Flash gets
    a T-shirt prior via ``register_model()`` plus forced burn-in.
  - **ParetoBandit (unconstrained)**: Same as transfer but without budget
    pacer — quality ceiling reference.

Hyperparameter note
-------------------
  ``alpha``, ``prior_n_effective``, and ``forgetting_factor`` are taken
  from ``BEST_K3_HPARAMS`` (Experiment 05 epsilon-constraint selection),
  ensuring consistency with the main paper's hyperparameter choices.

Output: ``results/model_onboarding_results.json``

Usage
-----
    python experiments/04_model_onboarding/run_model_onboarding.py
    python experiments/04_model_onboarding/run_model_onboarding.py --fast
    python experiments/04_model_onboarding/run_model_onboarding.py --scenario bad_cheap
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
from pareto_bandit.config import (
    BEST_K3_HPARAMS,
    DEFAULT_PACER_EMA_ALPHA,
    DEFAULT_PACER_LAMBDA_MAX,
    DEFAULT_PACER_LR,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_BUDGET_LABELS,
    K3_BUDGET_TARGETS,
    K3_WARMUP_PRIORS_PATH,
    K4_MODELS_PATH,
    N_SEEDS,
    OFFLINE_DATASET_DIR,
    VAL_DATA_PATH,
)
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from utils.simulation import SplitData, build_model_registry, load_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("pareto_bandit.router", "pareto_bandit.feature_service", "pareto_bandit.policy"):
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

ARM_SHORT: Dict[str, str] = {**K3_ARM_SHORT, FLASH_ID: "Flash"}

SEED_OFFSET: int = 9000
RESULTS_DIR = Path(__file__).parent / "results"

CHECKPOINT_INTERVAL: int = 25
WINDOW_SIZE: int = 100

PRIOR_N_EFFECTIVE: float = BEST_K3_HPARAMS["prior_n_effective"]
ALPHA: float = BEST_K3_HPARAMS["alpha"]
FORGETTING_FACTOR: float = BEST_K3_HPARAMS["forgetting_factor"]

__doc__ = __doc__.format(alpha=ALPHA, n_eff=PRIOR_N_EFFECTIVE)

PACER_LR: float = DEFAULT_PACER_LR
PACER_LAMBDA_MAX: float = DEFAULT_PACER_LAMBDA_MAX
PACER_EMA_ALPHA: float = DEFAULT_PACER_EMA_ALPHA

BUDGET_TARGETS: List[float] = K3_BUDGET_TARGETS
BUDGET_LABELS: List[str] = K3_BUDGET_LABELS

FLASH_INPUT_COST_PER_M: float = 0.3
FLASH_OUTPUT_COST_PER_M: float = 2.5
FLASH_BLENDED_COST_PER_M: float = (
    (FLASH_INPUT_COST_PER_M + FLASH_OUTPUT_COST_PER_M) / 2.0
)

# K=4 data paths (produced by merge_flash_into_splits.py)
VAL_K4_PATH = OFFLINE_DATASET_DIR / "val_k4.jsonl"
TEST_K4_PATH = OFFLINE_DATASET_DIR / "test_k4.jsonl"

# Forced exploration burn-in for newly onboarded arms.
# Routes the first N Phase 2 prompts unconditionally to Flash so the
# bandit collects real observations before UCB selection kicks in.
# With alpha=ALPHA and strong K=3 priors (n_eff=PRIOR_N_EFFECTIVE), the
# UCB exploration bonus alone is too small to trigger natural exploration
# of a cold-start arm.  20 pulls is ~4% of the test set (~460 prompts).
BURNIN_PULLS: int = 20

# Sustained-adoption thresholds
SUSTAINED_THRESHOLD: float = 0.10
SUSTAINED_HOLD_STEPS: int = 50

# ======================================================================
# Onboarding scenarios (reward / cost scaling for Flash in Phase 2)
# ======================================================================

ONBOARDING_SCENARIOS: Dict[str, Dict[str, float]] = {
    "good_cheap": {"reward_scale": 1.0, "cost_scale": 1.0},
    "bad_cheap": {"reward_scale": 0.5, "cost_scale": 1.0},
    "good_expensive": {"reward_scale": 1.0, "cost_scale": 10.0},
}

SCENARIO_LABELS: Dict[str, str] = {
    "good_cheap": "Good & Cheap (baseline)",
    "bad_cheap": "Bad & Cheap (negative)",
    "good_expensive": "Good & Expensive (cost negative)",
}


def _apply_scenario(
    eval_k4: SplitData,
    *,
    reward_scale: float = 1.0,
    cost_scale: float = 1.0,
) -> SplitData:
    """Return a view of *eval_k4* with Flash rewards/costs scaled.

    K=3 arm arrays are shared (not copied).  Only the Flash arrays
    are duplicated and scaled, keeping memory overhead minimal.

    Args:
        eval_k4: Original K=4 evaluation split.
        reward_scale: Multiplicative factor for Flash rewards.
        cost_scale: Multiplicative factor for Flash costs.

    Returns:
        New ``SplitData`` with modified Flash data.
    """
    if reward_scale == 1.0 and cost_scale == 1.0:
        return eval_k4

    new_rewards = dict(eval_k4.rewards)
    new_costs = dict(eval_k4.costs)
    new_rewards[FLASH_ID] = eval_k4.rewards[FLASH_ID] * reward_scale
    new_costs[FLASH_ID] = eval_k4.costs[FLASH_ID] * cost_scale
    return SplitData(
        prompts=eval_k4.prompts,
        rewards=new_rewards,
        costs=new_costs,
        embeddings=eval_k4.embeddings,
    )


def _snapshot_trace_A_inv(router: "BanditRouter", arms: List[str]) -> Dict[str, float]:
    """Return tr(A_inv) for each arm — a scalar summary of uncertainty."""
    traces: Dict[str, float] = {}
    for arm in arms:
        if arm in router.bandit.A_inv:
            traces[arm] = float(np.trace(router.bandit.A_inv[arm]))
        else:
            traces[arm] = 0.0
    return traces


# ======================================================================
# Phase 2 strategies
# ======================================================================

STRATEGY_BANDITGPT_TRANSFER = "paretobandit_transfer"
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
    trace_A_inv: Dict[str, float] = field(default_factory=dict)


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
    n_burnin: int
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

    Raises:
        FileNotFoundError: If ``test_k4.jsonl`` does not exist.  We
            intentionally do *not* fall back to ``val_k4.jsonl`` because
            Phase 1 uses ``val.jsonl`` — sharing the same prompt pool
            across phases would constitute data leakage.
    """
    if not TEST_K4_PATH.exists():
        raise FileNotFoundError(
            f"K=4 test split not found at {TEST_K4_PATH}. "
            "Run collect_flash_canonical.py + merge_flash_into_splits.py first. "
            "Note: val_k4.jsonl is NOT a valid substitute because Phase 1 "
            "uses val.jsonl (same prompt pool → data leakage)."
        )
    return load_split(TEST_K4_PATH, fs, K4_ARMS)


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
        cost_penalty=cost_penalty,
        forgetting_factor=FORGETTING_FACTOR,
        budget_pacer=budget_pacer,
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
    and remains above it for *hold* consecutive steps.

    When forced exploration (burn-in) is used, callers should pre-slice
    ``windowed_history`` to exclude the burn-in period **and** the
    subsequent washout window (``n_burnin + WINDOW_SIZE`` entries) so
    the metric reflects organic UCB-driven adoption only.

    Args:
        windowed_history: Flash windowed-mix value at each step (not just
            checkpoints — one entry per Phase 2 step).
        threshold: Minimum share to qualify as "adopted".
        hold: Number of consecutive steps above threshold.

    Returns:
        The 1-indexed step within the provided history of the first
        qualifying window, or ``None``.
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
    cost_scale: float = 1.0,
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
        cost_scale: Multiplicative factor for Flash's registry cost.
            Ensures the router's hard ceiling filter and soft cost
            penalties see the scenario-scaled price, not the original.

    Returns:
        Complete trial result with per-step checkpoints and adoption
        metrics.
    """
    rng = np.random.default_rng(seed)

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
                trace_A_inv=_snapshot_trace_A_inv(router, K4_ARMS),
            ))

    phase1_reward = cum_reward / train_k3.n
    phase1_cost = cum_cost / train_k3.n
    phase1_regret = cum_regret

    # ── Onboard Gemini Flash ──────────────────────────────────────────
    if strategy == STRATEGY_BANDITGPT_TRANSFER:
        scaled_input = FLASH_INPUT_COST_PER_M * cost_scale
        scaled_output = FLASH_OUTPUT_COST_PER_M * cost_scale
        scaled_blended = FLASH_BLENDED_COST_PER_M * cost_scale
        router.register_model(
            FLASH_ID,
            speed="fast",
            cost_usd=scaled_input,
            blended_cost_per_m=scaled_blended,
        )
        if cost_scale != 1.0:
            router.update_model_pricing(
                FLASH_ID,
                input_cost_per_m=scaled_input,
                output_cost_per_m=scaled_output,
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

    n_burnin = BURNIN_PULLS if strategy == STRATEGY_BANDITGPT_TRANSFER else 0

    for step_idx, i in enumerate(eval_order):
        # ── Model selection depends on strategy ───────────────────────
        if strategy == STRATEGY_FIXED_UNIFORM:
            model = rng.choice(K4_ARMS)
        elif strategy == STRATEGY_BANDITGPT_TRANSFER and step_idx < n_burnin:
            # Forced exploration: guarantee Flash receives real observations
            # before UCB selection takes over.
            model = FLASH_ID
            x = eval_k4.embeddings[i]
            reward_val = float(eval_k4.rewards[FLASH_ID][i])
            cost_val = float(eval_k4.costs[FLASH_ID][i])
            router.update(FLASH_ID, x, reward_val, advance_time=True)
            if router.budget_pacer is not None:
                router.budget_pacer.observe(cost_val)
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
                trace_A_inv=_snapshot_trace_A_inv(router, K4_ARMS),
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

    # Sustained adoption — skip burn-in + window washout so the metric
    # only reflects organic UCB-driven adoption, not the forced exploration
    # residue that lingers in the sliding window for WINDOW_SIZE steps.
    washout = n_burnin + WINDOW_SIZE
    if washout < len(flash_windowed_history):
        clean_history = flash_windowed_history[washout:]
        flash_sustained = _compute_sustained_step(clean_history)
        if flash_sustained is not None:
            flash_sustained += washout  # convert back to Phase 2 step (1-indexed)
    else:
        flash_sustained = None

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
        n_burnin=n_burnin,
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
        "n_burnin": trials[0].n_burnin,
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
        trace_arrays: Dict[str, List[float]] = {a: [] for a in K4_ARMS}
        rewards, costs, regrets, lambdas = [], [], [], []

        for trial in trials:
            if cp_idx >= len(trial.checkpoints):
                continue
            cp = trial.checkpoints[cp_idx]
            for a in K4_ARMS:
                mix_arrays[a].append(cp.routing_mix.get(a, 0.0))
                wmix_arrays[a].append(cp.windowed_mix.get(a, 0.0))
                trace_arrays[a].append(cp.trace_A_inv.get(a, 0.0))
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
            "per_seed_windowed_mix": {
                a: [float(v) for v in wmix_arrays[a]] for a in K4_ARMS
            },
            "per_seed_cumulative_cost": [float(v) for v in costs],
            "trace_A_inv_mean": {
                a: float(np.mean(trace_arrays[a])) for a in K4_ARMS
            },
            "trace_A_inv_se": {
                a: _se(trace_arrays[a]) for a in K4_ARMS
            },
            "per_seed_trace_A_inv": {
                a: [float(v) for v in trace_arrays[a]] for a in K4_ARMS
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


def _print_summary(
    all_summaries: Dict[str, Dict[str, Any]],
    scenario_name: str = "",
) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 90)
    header = "APPENDIX: MODEL ONBOARDING UNDER BUDGET CONSTRAINTS (K=3 → K=4)"
    if scenario_name:
        header += f"  [{scenario_name.upper()}]"
    print(header)
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


def _build_conditions() -> List[Dict[str, Any]]:
    """Build the condition matrix (budget tiers x strategies + unconstrained)."""
    conditions: List[Dict[str, Any]] = []
    for budget_label, budget_target in zip(BUDGET_LABELS, BUDGET_TARGETS):
        conditions.append({
            "condition": f"Fixed Policy ({budget_label})",
            "strategy": STRATEGY_FIXED_UNIFORM,
            "budget_label": budget_label,
            "budget_target": budget_target,
        })
        conditions.append({
            "condition": f"ParetoBandit ({budget_label})",
            "strategy": STRATEGY_BANDITGPT_TRANSFER,
            "budget_label": budget_label,
            "budget_target": budget_target,
        })
    conditions.append({
        "condition": "ParetoBandit (unconstrained)",
        "strategy": STRATEGY_BANDITGPT_TRANSFER,
        "budget_label": "unconstrained",
        "budget_target": 0.0,
    })
    return conditions


def _run_scenario(
    scenario_name: str,
    scenario_cfg: Dict[str, float],
    conditions: List[Dict[str, Any]],
    train_k3: SplitData,
    eval_k4_base: SplitData,
    registry_k3: Dict[str, Any],
    feature_dim: int,
    n_seeds: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Run all conditions under one onboarding scenario.

    Returns:
        Tuple of (summaries_dict, checkpoint_traces_dict).
    """
    eval_k4 = _apply_scenario(eval_k4_base, **scenario_cfg)
    all_summaries: Dict[str, Dict[str, Any]] = {}
    all_checkpoint_traces: Dict[str, List[Dict[str, Any]]] = {}

    for cond in conditions:
        key = f"{cond['strategy']}_{cond['budget_label']}"
        trials: List[TrialResult] = []

        logger.info(
            "\n[%s] %s [%s] (%d seeds)",
            scenario_name, cond["condition"], cond["strategy"], n_seeds,
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
                cost_scale=scenario_cfg.get("cost_scale", 1.0),
            )
            trials.append(result)

            flash_pct = result.phase2_model_fractions.get(FLASH_ID, 0) * 100
            logger.info(
                "    P2 Flash=%.1f%%, reward=%.4f, cost=$%.6f",
                flash_pct, result.phase2_reward, result.phase2_cost,
            )

        all_summaries[key] = _aggregate_trials(trials)
        all_checkpoint_traces[key] = _aggregate_checkpoints(trials)

    return all_summaries, all_checkpoint_traces


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
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=list(ONBOARDING_SCENARIOS.keys()),
        help="Run a single scenario (default: all).",
    )
    args = parser.parse_args()

    n_seeds = 3 if args.fast else args.n_seeds
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading data...")
    fs = FeatureService()
    feature_dim = fs.dimension

    train_k3 = _load_k3(fs)
    eval_k4_base = _load_k4_eval(fs)
    logger.info("  K=3 val (Phase 1): %d prompts", train_k3.n)
    logger.info("  K=4 eval (Phase 2): %d prompts", eval_k4_base.n)

    registry_k3 = build_model_registry(K3_ARMS)
    conditions = _build_conditions()

    # ── Determine which scenarios to run ───────────────────────────────
    if args.scenario is not None:
        scenarios_to_run = {args.scenario: ONBOARDING_SCENARIOS[args.scenario]}
    else:
        scenarios_to_run = ONBOARDING_SCENARIOS

    # ── Run all scenarios ──────────────────────────────────────────────
    all_scenario_results: Dict[str, Dict[str, Any]] = {}

    for scenario_name, scenario_cfg in scenarios_to_run.items():
        logger.info(
            "\n" + "=" * 70 + "\nSCENARIO: %s  (reward_scale=%.1f, cost_scale=%.1f)\n" + "=" * 70,
            scenario_name, scenario_cfg["reward_scale"], scenario_cfg["cost_scale"],
        )
        summaries, traces = _run_scenario(
            scenario_name, scenario_cfg, conditions,
            train_k3, eval_k4_base, registry_k3, feature_dim, n_seeds,
        )
        all_scenario_results[scenario_name] = {
            "summaries": summaries,
            "checkpoint_traces": traces,
        }
        _print_summary(summaries, scenario_name)

    elapsed = time.time() - t0

    # ── Export ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "experiment": "model_onboarding_k3_to_k4",
        "flash_model": FLASH_ID,
        "n_seeds": n_seeds,
        "phase1_n": train_k3.n,
        "phase2_n": eval_k4_base.n,
        "burnin_pulls": BURNIN_PULLS,
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
        "onboarding_scenarios": {
            name: {"reward_scale": cfg["reward_scale"], "cost_scale": cfg["cost_scale"]}
            for name, cfg in scenarios_to_run.items()
        },
        "scenarios": all_scenario_results,
        "wall_time_s": round(elapsed, 1),
    }

    # Backward compatibility: expose good_cheap at the top level so
    # existing figure scripts continue to work without modification.
    if "good_cheap" in all_scenario_results:
        payload["summaries"] = all_scenario_results["good_cheap"]["summaries"]
        payload["checkpoint_traces"] = all_scenario_results["good_cheap"]["checkpoint_traces"]

    json_path = RESULTS_DIR / "model_onboarding_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Results saved to %s", json_path)

    logger.info("\nTotal wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
