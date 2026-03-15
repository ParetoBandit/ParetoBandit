#!/usr/bin/env python3
"""Experiment 03: Budget Pacing Under Cost Drift.

Demonstrates that the BudgetPacer automatically exploits a mid-stream
model pricing change to improve routing quality while maintaining budget
compliance --- the most production-relevant scenario for cost-constrained
LLM routing.

Experimental setup
------------------
The pipeline follows a **train-then-evaluate** design (matching Exp 01):

  **Train phase** (val split, 1,785 prompts): The router online-learns
  under normal pricing.  No evaluation metrics are recorded.  This uses
  the validation split (``VAL_DATA_PATH``), the same data Exp 04 uses
  for hyperparameter tuning --- but here it serves only as a learning
  stream, not an evaluation surface.

  **Evaluation phase** (holdout split, 1,824 prompts): The router is
  evaluated on held-out data (``HOLDOUT_DATA_PATH``) that was never
  used for hyperparameter selection:

    **Phase 1** (steps 1--912): Normal pricing.  Gemini-Pro is expensive
    (normalized cost 0.67); the BudgetPacer enforces the dollar budget
    target by raising lambda_t, which suppresses Gemini selection.

    **Phase 2** (steps 913--1824): **Gemini price drop** — pricing falls
    to $0.10/$0.10 per million tokens (normalized cost ~0.0).  The router
    registry is updated at the boundary.  The cost EMA should decline,
    driving lambda_t downward and allowing Gemini routing.

Three budget targets span the constraint regime:

  - **Tight**    ($2.3 × 10⁻⁴ $/req): lambda high, Llama-heavy Phase 1
  - **Moderate** ($6.6 × 10⁻⁴ $/req): mixed routing Phase 1
  - **Loose**    ($1.9 × 10⁻³ $/req): light constraint Phase 1

Per target, four conditions are compared, representing increasing
levels of routing sophistication:

  1. **Fixed Policy (offline)** — Warmup priors with a matched static
     cost penalty but no online learning.  The dominant production
     pattern: train offline, deploy, never update.
  2. **Naive Bandit** — LinUCB with warmup priors, infinite memory
     (γ=1.0), and a matched static cost penalty.  The obvious first
     attempt at online routing — adapts, but Phase 1 inertia dilutes
     Phase 2 signal, and has no principled budget mechanism.
  3. **Recalibrated Bandit** — Same as Naive Bandit, but at the Phase 2
     boundary the static cost penalty is re-tuned offline using the
     validation split with Phase 2 pricing.  This isolates the value
     of continuous online tracking (BanditGPT) vs. stepwise offline
     recalibration.
  4. **BanditGPT** — Warmup priors + geometric forgetting (γ=0.995) +
     primal-dual BudgetPacer.  The full system.

Plus one unconstrained baseline (cp=0, no pacer) for quality ceiling.

Usage:
    python experiments/03_budget_plus_drift/run_budget_cost_drift.py
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.budget_pacer import BudgetPacer, PacingMode
from bandit_gpt.config import (
    BEST_K3_HPARAMS,
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_WARMUP_PRIORS_PATH,
    VAL_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.simulation import SplitData, build_model_registry

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

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

GEMINI_ID: str = "google/gemini-2.5-pro"
GEMINI_NEW_INPUT_COST: float = 0.10
GEMINI_NEW_OUTPUT_COST: float = 0.10

N_SEEDS: int = 50
SEED_OFFSET: int = 7000
RESULTS_DIR = Path(__file__).parent / "results"

PHASE1_N: int = 912
PHASE2_N: int = 912
CHECKPOINT_INTERVAL: int = 25

PRIOR_N_EFFECTIVE: float = BEST_K3_HPARAMS["prior_n_effective"]
ALPHA: float = BEST_K3_HPARAMS["alpha"]

PACER_LR: float = 0.05
PACER_LAMBDA_MAX: float = 5.0
PACER_EMA_ALPHA: float = 0.05

BUDGET_TARGETS: List[float] = [2.34e-4, 6.62e-4, 1.87e-3]
BUDGET_LABELS: List[str] = ["tight", "moderate", "loose"]

MATCHED_STATIC_CPS: Dict[str, float] = {
    "tight": 0.50,
    "moderate": 0.30,
    "loose": 0.10,
}

# Calibration sweep for the Recalibrated Bandit baseline.
CAL_N_SEEDS: int = 10
CAL_SEED_OFFSET: int = 9000
CAL_LAMBDA_CANDIDATES: List[float] = [
    0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0,
]


# ======================================================================
# Data types
# ======================================================================


@dataclass
class StepRecord:
    """Per-step metrics recorded during the two-phase stream."""

    step: int
    phase: int
    model: str
    reward: float
    cost: float
    lambda_t: float
    cost_ema: float
    gamma: float


@dataclass
class SeedResult:
    """Aggregate metrics for one (condition, seed) trial."""

    condition: str
    seed: int
    steps: List[StepRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.steps)

    def phase_metrics(self, phase: int) -> Dict[str, Any]:
        """Compute aggregate metrics for a single phase."""
        phase_steps = [s for s in self.steps if s.phase == phase]
        if not phase_steps:
            return {}
        rewards = [s.reward for s in phase_steps]
        costs = [s.cost for s in phase_steps]
        arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
        for s in phase_steps:
            arm_counts[s.model] += 1
        n = len(phase_steps)
        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_cost": float(np.mean(costs)),
            "arm_fractions": {a: cnt / n for a, cnt in arm_counts.items()},
            "mean_lambda": float(np.mean([s.lambda_t for s in phase_steps])),
            "mean_cost_ema": float(np.mean([s.cost_ema for s in phase_steps])),
            "n_steps": n,
        }


# ======================================================================
# Data Loading (reused from Exp02b)
# ======================================================================


def _load_all(
    path: Path,
    fs: FeatureService,
    arm_order: List[str],
) -> SplitData:
    """Load all prompts from a JSONL file into a ``SplitData``."""
    prompts: List[str] = []
    rewards: Dict[str, List[float]] = {a: [] for a in arm_order}
    costs: Dict[str, List[float]] = {a: [] for a in arm_order}

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            for arm_id in arm_order:
                info = rec["arms"][arm_id]
                rewards[arm_id].append(info["reward"])
                costs[arm_id].append(info["cost"])

    logger.info("  Encoding %d prompts from %s ...", len(prompts), path.name)
    embeddings = fs.extract_features_batch(prompts)

    return SplitData(
        prompts=prompts,
        rewards={a: np.array(v) for a, v in rewards.items()},
        costs={a: np.array(v) for a, v in costs.items()},
        embeddings=embeddings,
    )


def _apply_gemini_cost_reduction(
    split: SplitData,
    gemini_id: str,
    old_input: float,
    old_output: float,
    new_input: float,
    new_output: float,
) -> SplitData:
    """Return a new ``SplitData`` with Gemini's costs scaled to new pricing."""
    old_avg = (old_input + old_output) / 2.0
    new_avg = (new_input + new_output) / 2.0
    scale = new_avg / old_avg

    new_costs = dict(split.costs)
    new_costs[gemini_id] = split.costs[gemini_id] * scale

    return SplitData(
        prompts=split.prompts,
        rewards=split.rewards,
        costs=new_costs,
        embeddings=split.embeddings,
    )


# ======================================================================
# Router Factory
# ======================================================================


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup: bool = True,
    forgetting_factor: float = 1.0,
    cost_penalty: float = 0.0,
    budget_pacer: Optional[BudgetPacer] = None,
) -> BanditRouter:
    """Build a K=3 router with optional warmup priors and budget pacer."""
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
        forgetting_factor=forgetting_factor,
        budget_pacer=budget_pacer,
    )


# ======================================================================
# Two-Phase Learning Curve
# ======================================================================


def _run_two_phase_trial(
    *,
    condition_label: str,
    train_data: SplitData,
    phase1: SplitData,
    phase2: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    cost_penalty: float,
    warmup: bool = True,
    forgetting_factor: float = 1.0,
    online_learn: bool = True,
    budget_pacer: Optional[BudgetPacer] = None,
    phase2_cost_penalty: Optional[float] = None,
    seed: int,
) -> SeedResult:
    """Run one seed through the train-then-evaluate cost-drift scenario.

    The trial has three stages:

    1. **Train** (val split, normal pricing): online-learn without
       recording metrics.  Skipped when ``online_learn=False``.
    2. **Phase 1** (holdout split, normal pricing): evaluate and record.
    3. **Phase 2** (holdout split, Gemini price drop): the registry is
       updated at the boundary, then evaluate and record.

    Parameters
    ----------
    condition_label : str
        Human-readable condition name.
    train_data : SplitData
        Validation split used for online learning (no metrics recorded).
    phase1, phase2 : SplitData
        Holdout evaluation data for each phase.
    registry : dict
        Original model registry (Phase 1 pricing).
    feature_dim : int
        Context vector dimensionality.
    cost_penalty : float
        Static cost penalty weight (0.0 for pacer conditions).
    warmup : bool
        Whether to load warmup priors.
    forgetting_factor : float
        Fixed forgetting factor.
    online_learn : bool
        If False, the policy is frozen at deployment — ``process_feedback``
        is never called and the train phase is skipped.
    budget_pacer : BudgetPacer or None
        Budget pacer instance (reset before each seed).
    phase2_cost_penalty : float or None
        If provided, the router's ``cost_penalty`` is updated to this value
        at the Phase 2 boundary — simulating periodic offline recalibration.
    seed : int
        Random seed for prompt ordering.

    Returns
    -------
    SeedResult
        Per-step metrics for this seed (eval phases only).
    """
    rng = np.random.default_rng(seed)

    if budget_pacer is not None:
        budget_pacer.reset()

    router = _create_router(
        registry,
        feature_dim,
        warmup=warmup,
        forgetting_factor=forgetting_factor,
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
    )

    # --- Train phase (val split, normal pricing, no metrics) ---
    if online_learn:
        train_order = rng.permutation(train_data.n)
        for i in train_order:
            model, log = router.route(train_data.embeddings[i])
            reward = float(train_data.rewards[model][i])
            log.cost_usd = float(train_data.costs[model][i])
            router.process_feedback(log.request_id, reward=reward)

    # --- Eval phases (holdout split) ---
    n_p1 = phase1.n
    n_p2 = phase2.n

    p1_order = rng.permutation(n_p1)
    p2_order = rng.permutation(n_p2)

    all_emb = np.concatenate([
        phase1.embeddings[p1_order],
        phase2.embeddings[p2_order],
    ], axis=0)

    all_rewards: Dict[str, np.ndarray] = {}
    all_costs: Dict[str, np.ndarray] = {}
    for arm in ARM_ORDER:
        all_rewards[arm] = np.concatenate([
            phase1.rewards[arm][p1_order],
            phase2.rewards[arm][p2_order],
        ])
        all_costs[arm] = np.concatenate([
            phase1.costs[arm][p1_order],
            phase2.costs[arm][p2_order],
        ])

    result = SeedResult(condition=condition_label, seed=seed)
    registry_updated = False

    for t in range(n_p1 + n_p2):
        if t == n_p1 and not registry_updated:
            gemini_reg = router.registry[GEMINI_ID]
            gemini_reg["input_cost_per_m"] = GEMINI_NEW_INPUT_COST
            gemini_reg["output_cost_per_m"] = GEMINI_NEW_OUTPUT_COST
            gemini_reg.pop("blended_cost_per_m", None)
            router._resolve_registry_costs()
            registry_updated = True
            if phase2_cost_penalty is not None:
                router.cost_penalty = phase2_cost_penalty

        phase = 1 if t < n_p1 else 2

        emb = all_emb[t]
        model, log = router.route(emb)
        reward = float(all_rewards[model][t])
        cost = float(all_costs[model][t])

        log.cost_usd = cost
        if online_learn:
            router.process_feedback(log.request_id, reward=reward)

        lam = budget_pacer.lambda_t if budget_pacer is not None else 0.0
        ema = budget_pacer.cost_ema if budget_pacer is not None else 0.0
        gamma = router.bandit.gamma

        result.steps.append(StepRecord(
            step=t + 1,
            phase=phase,
            model=model,
            reward=reward,
            cost=cost,
            lambda_t=lam,
            cost_ema=ema,
            gamma=gamma,
        ))

    return result


# ======================================================================
# Phase 2 cost-penalty calibration (Recalibrated Bandit)
# ======================================================================


def _calibrate_phase2_cp(
    train_data: SplitData,
    cal_data_phase2: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    original_cp: float,
    budget_target: float,
    n_cal_seeds: int = CAL_N_SEEDS,
    candidates: Optional[List[float]] = None,
) -> float:
    """Find the static cost penalty that best tracks *budget_target* under Phase 2 pricing.

    Simulates periodic offline recalibration: after observing a price change
    an operator re-tunes the static cost penalty on a dev set with updated
    pricing.  The candidate that minimises ``|mean_cost − target|`` wins.

    The calibration trains a Naive Bandit (γ=1.0) on the validation split
    under normal pricing, then — for each candidate λ — deep-copies the
    trained router, applies the price drop, sets ``cost_penalty = λ``, and
    evaluates on the **same** validation split with Phase 2 pricing.
    Using the validation split for both training and calibration is
    standard dev-set practice and avoids any leakage from the holdout set.

    Parameters
    ----------
    train_data : SplitData
        Validation split under **original** pricing (used for online training).
    cal_data_phase2 : SplitData
        Validation split with **Phase 2** Gemini pricing applied.
    registry : dict
        Original model registry (Phase 1 pricing).
    feature_dim : int
        Context vector dimensionality.
    original_cp : float
        Static cost penalty used during Phase 1 training.
    budget_target : float
        Dollar budget target per request.
    n_cal_seeds : int
        Number of calibration seeds.
    candidates : list[float] or None
        Candidate λ values to sweep.

    Returns
    -------
    float
        Cost penalty for Phase 2 that best tracks the budget target.
    """
    if candidates is None:
        candidates = CAL_LAMBDA_CANDIDATES

    candidate_costs: Dict[float, List[float]] = {c: [] for c in candidates}

    for s in range(n_cal_seeds):
        seed = CAL_SEED_OFFSET + s
        rng = np.random.default_rng(seed)

        base_router = _create_router(
            registry,
            feature_dim,
            warmup=True,
            forgetting_factor=1.0,
            cost_penalty=original_cp,
        )

        train_order = rng.permutation(train_data.n)
        for i in train_order:
            model, log = base_router.route(train_data.embeddings[i])
            log.cost_usd = float(train_data.costs[model][i])
            base_router.process_feedback(
                log.request_id, reward=float(train_data.rewards[model][i])
            )

        cal_order = rng.permutation(cal_data_phase2.n)

        for cp_candidate in candidates:
            router = copy.deepcopy(base_router)

            gemini_reg = router.registry[GEMINI_ID]
            gemini_reg["input_cost_per_m"] = GEMINI_NEW_INPUT_COST
            gemini_reg["output_cost_per_m"] = GEMINI_NEW_OUTPUT_COST
            gemini_reg.pop("blended_cost_per_m", None)
            router._resolve_registry_costs()
            router.cost_penalty = cp_candidate

            costs: List[float] = []
            for i in cal_order:
                model, log = router.route(cal_data_phase2.embeddings[i])
                cost = float(cal_data_phase2.costs[model][i])
                log.cost_usd = cost
                router.process_feedback(
                    log.request_id,
                    reward=float(cal_data_phase2.rewards[model][i]),
                )
                costs.append(cost)

            candidate_costs[cp_candidate].append(float(np.mean(costs)))

    best_cp = original_cp
    best_gap = float("inf")
    for cp_candidate in candidates:
        avg_cost = float(np.mean(candidate_costs[cp_candidate]))
        gap = abs(avg_cost - budget_target)
        logger.info(
            "    λ=%.3f → $%.6f/req (gap=$%.2e)", cp_candidate, avg_cost, gap,
        )
        if gap < best_gap:
            best_gap = gap
            best_cp = cp_candidate

    logger.info("  → Best Phase 2 λ = %.3f (gap=$%.2e)", best_cp, best_gap)
    return best_cp


# ======================================================================
# Condition definitions
# ======================================================================


def _build_conditions(
    budget_target: float,
    budget_label: str,
    matched_cp: float,
    recalibrated_phase2_cp: float,
) -> List[Dict[str, Any]]:
    """Build four conditions for a given budget target.

    The conditions represent increasing routing sophistication:
    Fixed Policy → Naive Bandit → Recalibrated Bandit → BanditGPT.

    Parameters
    ----------
    budget_target : float
        Dollar budget target per request.
    budget_label : str
        Human-readable budget label (tight/moderate/loose).
    matched_cp : float
        Static cost penalty that produces similar Phase 1 spend.
    recalibrated_phase2_cp : float
        Cost penalty recalibrated offline for Phase 2 pricing.

    Returns
    -------
    list[dict]
        Condition definitions.
    """
    return [
        {
            "label": f"Fixed Policy ({budget_label})",
            "budget_target": None,
            "cost_penalty": matched_cp,
            "warmup": True,
            "forgetting_factor": 1.0,
            "online_learn": False,
            "phase2_cost_penalty": None,
        },
        {
            "label": f"Naive Bandit ({budget_label})",
            "budget_target": None,
            "cost_penalty": matched_cp,
            "warmup": True,
            "forgetting_factor": 1.0,
            "online_learn": True,
            "phase2_cost_penalty": None,
        },
        {
            "label": f"Recalibrated ({budget_label})",
            "budget_target": None,
            "cost_penalty": matched_cp,
            "warmup": True,
            "forgetting_factor": 1.0,
            "online_learn": True,
            "phase2_cost_penalty": recalibrated_phase2_cp,
        },
        {
            "label": f"BanditGPT ({budget_label})",
            "budget_target": budget_target,
            "cost_penalty": 0.0,
            "warmup": True,
            "forgetting_factor": BEST_K3_HPARAMS["forgetting_factor"],
            "online_learn": True,
            "phase2_cost_penalty": None,
        },
    ]


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_seeds(
    seed_results: List[SeedResult],
    n_p1: int,
) -> Dict[str, Any]:
    """Aggregate per-seed results into checkpoint curves and phase summaries.

    Parameters
    ----------
    seed_results : list[SeedResult]
        Results from each seed for one condition.
    n_p1 : int
        Number of steps in Phase 1 (for phase boundary).

    Returns
    -------
    dict
        Aggregated metrics including checkpoint curves and phase summaries.
    """
    n_seeds = len(seed_results)
    n_total = seed_results[0].n

    checkpoints = sorted(set(
        [1]
        + list(range(CHECKPOINT_INTERVAL, n_total + 1, CHECKPOINT_INTERVAL))
        + [n_total]
    ))

    curves: List[Dict[str, Any]] = []
    for cp_step in checkpoints:
        lambdas, cost_emas, gammas = [], [], []
        arm_frac_lists: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
        rewards_agg, costs_agg = [], []

        avg_costs: List[float] = []

        for sr in seed_results:
            steps_so_far = sr.steps[:cp_step]
            last = steps_so_far[-1]
            lambdas.append(last.lambda_t)
            cost_emas.append(last.cost_ema)
            gammas.append(last.gamma)
            rewards_agg.append(last.reward)
            costs_agg.append(last.cost)
            avg_costs.append(float(np.mean([s.cost for s in steps_so_far])))

            arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
            window = steps_so_far[-min(50, len(steps_so_far)):]
            for s in window:
                arm_counts[s.model] += 1
            wn = len(window)
            for a in ARM_ORDER:
                arm_frac_lists[a].append(arm_counts[a] / wn)

        arm_fracs = {
            ARM_SHORT[a]: float(np.mean(arm_frac_lists[a]))
            for a in ARM_ORDER
        }
        arm_fracs_std = {
            ARM_SHORT[a]: float(np.std(arm_frac_lists[a]))
            for a in ARM_ORDER
        }

        curves.append({
            "step": cp_step,
            "phase": "normal" if cp_step <= n_p1 else "price-drop",
            "phase_boundary": n_p1,
            "mean_lambda": float(np.mean(lambdas)),
            "std_lambda": float(np.std(lambdas)),
            "mean_cost_ema": float(np.mean(cost_emas)),
            "std_cost_ema": float(np.std(cost_emas)),
            "mean_gamma": float(np.mean(gammas)),
            "std_gamma": float(np.std(gammas)),
            "mean_avg_cost": float(np.mean(avg_costs)),
            "std_avg_cost": float(np.std(avg_costs)),
            "arm_fractions": arm_fracs,
            "arm_fractions_std": arm_fracs_std,
            "n_seeds": n_seeds,
        })

    phase1_metrics = [sr.phase_metrics(1) for sr in seed_results]
    phase2_metrics = [sr.phase_metrics(2) for sr in seed_results]

    return {
        "label": seed_results[0].condition,
        "curves": curves,
        "phase1_summary": {
            "mean_reward": float(np.mean([m["mean_reward"] for m in phase1_metrics])),
            "mean_cost": float(np.mean([m["mean_cost"] for m in phase1_metrics])),
            "mean_lambda": float(np.mean([m["mean_lambda"] for m in phase1_metrics])),
            "arm_fractions": {
                ARM_SHORT[a]: float(np.mean([m["arm_fractions"][a] for m in phase1_metrics]))
                for a in ARM_ORDER
            },
        },
        "phase2_summary": {
            "mean_reward": float(np.mean([m["mean_reward"] for m in phase2_metrics])),
            "mean_cost": float(np.mean([m["mean_cost"] for m in phase2_metrics])),
            "mean_lambda": float(np.mean([m["mean_lambda"] for m in phase2_metrics])),
            "arm_fractions": {
                ARM_SHORT[a]: float(np.mean([m["arm_fractions"][a] for m in phase2_metrics]))
                for a in ARM_ORDER
            },
        },
        "per_seed_phase1_reward": [m["mean_reward"] for m in phase1_metrics],
        "per_seed_phase2_reward": [m["mean_reward"] for m in phase2_metrics],
        "per_seed_phase1_cost": [m["mean_cost"] for m in phase1_metrics],
        "per_seed_phase2_cost": [m["mean_cost"] for m in phase2_metrics],
    }


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading K=3 data ...")
    # PCA projection is pre-fitted on ~46K disjoint LMSYS prompts and frozen;
    # only .transform() is called during evaluation (no leakage).
    fs = FeatureService()
    feature_dim = fs.dimension

    train_all = _load_all(VAL_DATA_PATH, fs, ARM_ORDER)
    test_all = _load_all(HOLDOUT_DATA_PATH, fs, ARM_ORDER)

    logger.info("  Train (val): %d prompts — online learning, no eval", train_all.n)
    logger.info("  Eval (holdout): %d prompts — Phase 1 + Phase 2", test_all.n)

    rng_global = np.random.default_rng(42)
    all_indices = rng_global.permutation(test_all.n)
    p1_indices = all_indices[:PHASE1_N]
    p2_indices = all_indices[PHASE1_N : PHASE1_N + PHASE2_N]

    phase1 = SplitData(
        prompts=[test_all.prompts[i] for i in p1_indices],
        rewards={a: test_all.rewards[a][p1_indices] for a in ARM_ORDER},
        costs={a: test_all.costs[a][p1_indices] for a in ARM_ORDER},
        embeddings=test_all.embeddings[p1_indices],
    )

    phase2_raw = SplitData(
        prompts=[test_all.prompts[i] for i in p2_indices],
        rewards={a: test_all.rewards[a][p2_indices] for a in ARM_ORDER},
        costs={a: test_all.costs[a][p2_indices] for a in ARM_ORDER},
        embeddings=test_all.embeddings[p2_indices],
    )

    registry = build_model_registry(ARM_ORDER)
    gemini_meta = registry[GEMINI_ID]
    old_input = gemini_meta["input_cost_per_m"]
    old_output = gemini_meta["output_cost_per_m"]

    phase2 = _apply_gemini_cost_reduction(
        phase2_raw, GEMINI_ID,
        old_input, old_output,
        GEMINI_NEW_INPUT_COST, GEMINI_NEW_OUTPUT_COST,
    )

    # ------------------------------------------------------------------
    # Calibrate Phase 2 cost penalties for the Recalibrated Bandit
    # ------------------------------------------------------------------
    train_phase2 = _apply_gemini_cost_reduction(
        train_all, GEMINI_ID,
        old_input, old_output,
        GEMINI_NEW_INPUT_COST, GEMINI_NEW_OUTPUT_COST,
    )

    recalibrated_cps: Dict[str, float] = {}
    for target, blabel in zip(BUDGET_TARGETS, BUDGET_LABELS):
        matched_cp = MATCHED_STATIC_CPS[blabel]
        logger.info(
            "\nCalibrating Phase 2 λ for %s (target=$%.2e) ...", blabel, target,
        )
        recalibrated_cps[blabel] = _calibrate_phase2_cp(
            train_data=train_all,
            cal_data_phase2=train_phase2,
            registry=registry,
            feature_dim=feature_dim,
            original_cp=matched_cp,
            budget_target=target,
        )

    logger.info("\nRecalibrated Phase 2 cost penalties: %s", recalibrated_cps)

    # ------------------------------------------------------------------
    # Run all conditions
    # ------------------------------------------------------------------
    all_condition_results: Dict[str, Dict[str, Any]] = {}

    for target, blabel in zip(BUDGET_TARGETS, BUDGET_LABELS):
        matched_cp = MATCHED_STATIC_CPS[blabel]
        conditions = _build_conditions(
            target, blabel, matched_cp, recalibrated_cps[blabel],
        )

        for cond in conditions:
            label = cond["label"]
            logger.info("\n=== %s ===", label)

            pacer: Optional[BudgetPacer] = None
            if cond["budget_target"] is not None:
                pacer = BudgetPacer(
                    target_avg_spend_usd=cond["budget_target"],
                    mode=PacingMode.ADAPTIVE,
                    lr=PACER_LR,
                    ema_alpha=PACER_EMA_ALPHA,
                    lambda_max=PACER_LAMBDA_MAX,
                )

            seed_results: List[SeedResult] = []
            for s in range(N_SEEDS):
                seed = SEED_OFFSET + s
                sr = _run_two_phase_trial(
                    condition_label=label,
                    train_data=train_all,
                    phase1=phase1,
                    phase2=phase2,
                    registry=registry,
                    feature_dim=feature_dim,
                    cost_penalty=cond["cost_penalty"],
                    warmup=cond["warmup"],
                    forgetting_factor=cond["forgetting_factor"],
                    online_learn=cond.get("online_learn", True),
                    budget_pacer=pacer,
                    phase2_cost_penalty=cond.get("phase2_cost_penalty"),
                    seed=seed,
                )
                seed_results.append(sr)

            agg = _aggregate_seeds(seed_results, PHASE1_N)
            all_condition_results[label] = agg

            logger.info(
                "  Phase 1: reward=%.4f  cost=$%.6f  λ=%.3f  arm=%s",
                agg["phase1_summary"]["mean_reward"],
                agg["phase1_summary"]["mean_cost"],
                agg["phase1_summary"]["mean_lambda"],
                agg["phase1_summary"]["arm_fractions"],
            )
            logger.info(
                "  Phase 2: reward=%.4f  cost=$%.6f  λ=%.3f  arm=%s",
                agg["phase2_summary"]["mean_reward"],
                agg["phase2_summary"]["mean_cost"],
                agg["phase2_summary"]["mean_lambda"],
                agg["phase2_summary"]["arm_fractions"],
            )

    # Unconstrained baseline
    logger.info("\n=== Unconstrained (cp=0) ===")
    unconstrained_seeds: List[SeedResult] = []
    for s in range(N_SEEDS):
        seed = SEED_OFFSET + s
        sr = _run_two_phase_trial(
            condition_label="Unconstrained",
            train_data=train_all,
            phase1=phase1,
            phase2=phase2,
            registry=registry,
            feature_dim=feature_dim,
            cost_penalty=0.0,
            warmup=True,
            forgetting_factor=1.0,
            budget_pacer=None,
            seed=seed,
        )
        unconstrained_seeds.append(sr)

    unconstrained_agg = _aggregate_seeds(unconstrained_seeds, PHASE1_N)
    all_condition_results["Unconstrained"] = unconstrained_agg

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "experiment": "03_budget_plus_drift",
        "arm_order": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "cost_shift_model": GEMINI_ID,
        "gemini_new_input_cost": GEMINI_NEW_INPUT_COST,
        "gemini_new_output_cost": GEMINI_NEW_OUTPUT_COST,
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "phase1_n": PHASE1_N,
        "phase2_n": PHASE2_N,
        "budget_targets": BUDGET_TARGETS,
        "budget_labels": BUDGET_LABELS,
        "matched_static_cps": MATCHED_STATIC_CPS,
        "recalibrated_phase2_cps": recalibrated_cps,
        "cal_n_seeds": CAL_N_SEEDS,
        "cal_lambda_candidates": CAL_LAMBDA_CANDIDATES,
        "pacer_lr": PACER_LR,
        "pacer_lambda_max": PACER_LAMBDA_MAX,
        "pacer_ema_alpha": PACER_EMA_ALPHA,
        "prior_n_effective": PRIOR_N_EFFECTIVE,
        "alpha": ALPHA,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "train_n": train_all.n,
        "eval_n": test_all.n,
        "conditions": all_condition_results,
    }

    out_path = RESULTS_DIR / "budget_cost_drift_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nSaved results to %s", out_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 100)
    logger.info("EXPERIMENT 03: BUDGET + COST DRIFT — Summary")
    logger.info("=" * 100)
    logger.info(
        "  %-35s  %8s  %8s  %8s  %8s",
        "Condition", "P1 Rwd", "P2 Rwd", "P1 λ", "P2 λ",
    )
    logger.info("  " + "-" * 80)
    for label, agg in all_condition_results.items():
        p1 = agg["phase1_summary"]
        p2 = agg["phase2_summary"]
        logger.info(
            "  %-35s  %8.4f  %8.4f  %8.3f  %8.3f",
            label,
            p1["mean_reward"], p2["mean_reward"],
            p1["mean_lambda"], p2["mean_lambda"],
        )
    logger.info("=" * 100)
    logger.info("Wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
