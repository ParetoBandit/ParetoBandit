#!/usr/bin/env python3
"""Experiment 03: Budget Pacing Under Cost Drift (Three-Phase).

A three-phase experiment that tests whether the BudgetPacer adapts its
routing behavior in both directions when model pricing changes and is
subsequently restored.

Experimental setup
------------------
The pipeline follows the same train-then-evaluate design as earlier experiments.

  **Train phase** (val split, 1,785 prompts): Online learning under normal
  pricing.  No evaluation metrics recorded.

  **Evaluation phase** (holdout split, 1,824 prompts split into 3 × 608):

    **Phase 1** (steps 1–608): Normal pricing.  Gemini-Pro is expensive;
    the BudgetPacer raises λ_t to enforce the dollar budget.

    **Phase 2** (steps 609–1216): **Gemini price drop** — pricing falls
    to $0.10/$0.10 per million tokens.  λ_t should decline, allowing
    more Gemini routing.

    **Phase 3** (steps 1217–1824): **Price correction** — Gemini pricing
    is restored to its original level.  The key question: does the pacer
    re-raise λ_t and return to Phase 1-like routing?

Three budget targets (tight, moderate, loose) and four conditions
(Fixed Policy, Naive Bandit, Recalibrated Bandit, ParetoBandit) plus
an unconstrained baseline are tested.

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

from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
from pareto_bandit.config import (
    BEST_K3_HPARAMS,
    DEFAULT_PACER_EMA_ALPHA,
    DEFAULT_PACER_LAMBDA_MAX,
    DEFAULT_PACER_LR,
    GEMINI_COST_DROP,
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_BUDGET_LABELS,
    K3_BUDGET_TARGETS,
    K3_WARMUP_PRIORS_PATH,
    VAL_DATA_PATH,
)
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from utils.simulation import SplitData, build_model_registry

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

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = K3_ARM_SHORT

GEMINI_ID: str = GEMINI_COST_DROP["model_id"]
GEMINI_NEW_INPUT_COST: float = GEMINI_COST_DROP["new_input_cost_per_m"]
GEMINI_NEW_OUTPUT_COST: float = GEMINI_COST_DROP["new_output_cost_per_m"]

N_SEEDS: int = 50
SEED_OFFSET: int = 8000
RESULTS_DIR = Path(__file__).parent / "results"

PHASE_N: int = 608
N_PHASES: int = 3
CHECKPOINT_INTERVAL: int = 20

PRIOR_N_EFFECTIVE: float = BEST_K3_HPARAMS["prior_n_effective"]
ALPHA: float = BEST_K3_HPARAMS["alpha"]

PACER_LR: float = DEFAULT_PACER_LR
PACER_LAMBDA_MAX: float = DEFAULT_PACER_LAMBDA_MAX
PACER_EMA_ALPHA: float = DEFAULT_PACER_EMA_ALPHA

BUDGET_TARGETS: List[float] = K3_BUDGET_TARGETS
BUDGET_LABELS: List[str] = K3_BUDGET_LABELS

MATCHED_STATIC_CPS: Dict[str, float] = {
    "tight": 0.40,
    "moderate": 0.30,
    "loose": 0.10,
}

CAL_N_SEEDS: int = 10
CAL_SEED_OFFSET: int = 9500
CAL_LAMBDA_CANDIDATES: List[float] = [
    0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0,
]


# ======================================================================
# Data types
# ======================================================================


@dataclass
class StepRecord:
    """Per-step metrics recorded during the three-phase stream."""

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
# Data Loading
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
# Three-Phase Learning Curve
# ======================================================================


def _run_three_phase_trial(
    *,
    condition_label: str,
    train_data: SplitData,
    phase1: SplitData,
    phase2: SplitData,
    phase3: SplitData,
    registry: Dict[str, Any],
    original_gemini_input: float,
    original_gemini_output: float,
    feature_dim: int,
    cost_penalty: float,
    warmup: bool = True,
    forgetting_factor: float = 1.0,
    online_learn: bool = True,
    budget_pacer: Optional[BudgetPacer] = None,
    phase2_cost_penalty: Optional[float] = None,
    phase3_cost_penalty: Optional[float] = None,
    seed: int,
) -> SeedResult:
    """Run one seed through the three-phase cost correction scenario.

    Stages:
      1. **Train** (val split, normal pricing): online-learn, no metrics.
      2. **Phase 1** (holdout, normal pricing): evaluate and record.
      3. **Phase 2** (holdout, Gemini price drop): registry updated.
      4. **Phase 3** (holdout, price correction): registry restored.

    Parameters
    ----------
    condition_label : str
        Human-readable condition name.
    train_data : SplitData
        Validation split for online learning (no metrics recorded).
    phase1, phase2, phase3 : SplitData
        Holdout evaluation data for each phase.
    registry : dict
        Original model registry (Phase 1 / Phase 3 pricing).
    original_gemini_input, original_gemini_output : float
        Original Gemini pricing ($/M tokens) to restore in Phase 3.
    feature_dim : int
        Context vector dimensionality.
    cost_penalty : float
        Static cost penalty weight (0.0 for pacer conditions).
    warmup : bool
        Whether to load warmup priors.
    forgetting_factor : float
        Fixed forgetting factor.
    online_learn : bool
        If False, the policy is frozen (no ``process_feedback``).
    budget_pacer : BudgetPacer or None
        Budget pacer instance (reset before each seed).
    phase2_cost_penalty : float or None
        If provided, router's ``cost_penalty`` is updated at Phase 2 boundary.
    phase3_cost_penalty : float or None
        If provided, router's ``cost_penalty`` is updated at Phase 3 boundary.
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

    if online_learn:
        train_order = rng.permutation(train_data.n)
        for i in train_order:
            model, log = router.route(train_data.embeddings[i])
            reward = float(train_data.rewards[model][i])
            log.cost_usd = float(train_data.costs[model][i])
            router.process_feedback(log.request_id, reward=reward)

    n_p1 = phase1.n
    n_p2 = phase2.n
    n_p3 = phase3.n

    p1_order = rng.permutation(n_p1)
    p2_order = rng.permutation(n_p2)
    p3_order = rng.permutation(n_p3)

    all_emb = np.concatenate([
        phase1.embeddings[p1_order],
        phase2.embeddings[p2_order],
        phase3.embeddings[p3_order],
    ], axis=0)

    all_rewards: Dict[str, np.ndarray] = {}
    all_costs: Dict[str, np.ndarray] = {}
    for arm in ARM_ORDER:
        all_rewards[arm] = np.concatenate([
            phase1.rewards[arm][p1_order],
            phase2.rewards[arm][p2_order],
            phase3.rewards[arm][p3_order],
        ])
        all_costs[arm] = np.concatenate([
            phase1.costs[arm][p1_order],
            phase2.costs[arm][p2_order],
            phase3.costs[arm][p3_order],
        ])

    result = SeedResult(condition=condition_label, seed=seed)
    p2_boundary = n_p1
    p3_boundary = n_p1 + n_p2
    n_total = n_p1 + n_p2 + n_p3

    registry_phase2_applied = False
    registry_phase3_applied = False

    for t in range(n_total):
        # Phase 2 boundary: apply price drop
        if t == p2_boundary and not registry_phase2_applied:
            gemini_reg = router.registry[GEMINI_ID]
            gemini_reg["input_cost_per_m"] = GEMINI_NEW_INPUT_COST
            gemini_reg["output_cost_per_m"] = GEMINI_NEW_OUTPUT_COST
            gemini_reg.pop("blended_cost_per_m", None)
            router._resolve_registry_costs()
            registry_phase2_applied = True
            if phase2_cost_penalty is not None:
                router.cost_penalty = phase2_cost_penalty

        # Phase 3 boundary: restore original pricing
        if t == p3_boundary and not registry_phase3_applied:
            gemini_reg = router.registry[GEMINI_ID]
            gemini_reg["input_cost_per_m"] = original_gemini_input
            gemini_reg["output_cost_per_m"] = original_gemini_output
            gemini_reg.pop("blended_cost_per_m", None)
            router._resolve_registry_costs()
            registry_phase3_applied = True
            if phase3_cost_penalty is not None:
                router.cost_penalty = phase3_cost_penalty

        if t < p2_boundary:
            phase = 1
        elif t < p3_boundary:
            phase = 2
        else:
            phase = 3

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


def _calibrate_phase_cp(
    train_data: SplitData,
    cal_data: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    original_cp: float,
    budget_target: float,
    gemini_input: float,
    gemini_output: float,
    n_cal_seeds: int = CAL_N_SEEDS,
    candidates: Optional[List[float]] = None,
) -> float:
    """Find the static cost penalty that best tracks *budget_target* under given pricing.

    Generalizes the Phase 2 calibration from Exp 03 to work with arbitrary
    target pricing (used for both Phase 2 and Phase 3 recalibration).

    Parameters
    ----------
    train_data : SplitData
        Validation split under **original** pricing (for online training).
    cal_data : SplitData
        Validation split with **target** pricing applied.
    registry : dict
        Original model registry (Phase 1 pricing).
    feature_dim : int
        Context vector dimensionality.
    original_cp : float
        Static cost penalty used during Phase 1 training.
    budget_target : float
        Dollar budget target per request.
    gemini_input, gemini_output : float
        Gemini pricing to apply in the calibration registry.
    n_cal_seeds : int
        Number of calibration seeds.
    candidates : list[float] or None
        Candidate cost penalty values to sweep.

    Returns
    -------
    float
        Cost penalty for the target phase that best tracks the budget.
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

        cal_order = rng.permutation(cal_data.n)

        for cp_candidate in candidates:
            router = copy.deepcopy(base_router)

            gemini_reg = router.registry[GEMINI_ID]
            gemini_reg["input_cost_per_m"] = gemini_input
            gemini_reg["output_cost_per_m"] = gemini_output
            gemini_reg.pop("blended_cost_per_m", None)
            router._resolve_registry_costs()
            router.cost_penalty = cp_candidate

            costs: List[float] = []
            for i in cal_order:
                model, log = router.route(cal_data.embeddings[i])
                cost = float(cal_data.costs[model][i])
                log.cost_usd = cost
                router.process_feedback(
                    log.request_id,
                    reward=float(cal_data.rewards[model][i]),
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

    logger.info("  → Best λ = %.3f (gap=$%.2e)", best_cp, best_gap)
    return best_cp


# ======================================================================
# Condition definitions
# ======================================================================


def _build_conditions(
    budget_target: float,
    budget_label: str,
    matched_cp: float,
    recalibrated_phase2_cp: float,
    recalibrated_phase3_cp: float,
) -> List[Dict[str, Any]]:
    """Build four conditions for a given budget target.

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
    recalibrated_phase3_cp : float
        Cost penalty recalibrated offline for Phase 3 pricing (restored).

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
            "phase3_cost_penalty": None,
        },
        {
            "label": f"Naive Bandit ({budget_label})",
            "budget_target": None,
            "cost_penalty": matched_cp,
            "warmup": True,
            "forgetting_factor": 1.0,
            "online_learn": True,
            "phase2_cost_penalty": None,
            "phase3_cost_penalty": None,
        },
        {
            "label": f"Recalibrated ({budget_label})",
            "budget_target": None,
            "cost_penalty": matched_cp,
            "warmup": True,
            "forgetting_factor": 1.0,
            "online_learn": True,
            "phase2_cost_penalty": recalibrated_phase2_cp,
            "phase3_cost_penalty": recalibrated_phase3_cp,
        },
        {
            "label": f"ParetoBandit ({budget_label})",
            "budget_target": budget_target,
            "cost_penalty": 0.0,
            "warmup": True,
            "forgetting_factor": BEST_K3_HPARAMS["forgetting_factor"],
            "online_learn": True,
            "phase2_cost_penalty": None,
            "phase3_cost_penalty": None,
        },
    ]


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_seeds(
    seed_results: List[SeedResult],
    phase_boundaries: List[int],
) -> Dict[str, Any]:
    """Aggregate per-seed results into checkpoint curves and phase summaries.

    Parameters
    ----------
    seed_results : list[SeedResult]
        Results from each seed for one condition.
    phase_boundaries : list[int]
        Step counts where each phase ends (cumulative).
        E.g. [608, 1216, 1824] for three 608-step phases.

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

    def _step_to_phase_label(step: int) -> str:
        if step <= phase_boundaries[0]:
            return "normal"
        elif step <= phase_boundaries[1]:
            return "price-drop"
        else:
            return "price-restored"

    curves: List[Dict[str, Any]] = []
    for cp_step in checkpoints:
        lambdas, cost_emas, gammas = [], [], []
        arm_frac_lists: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
        avg_costs: List[float] = []

        for sr in seed_results:
            steps_so_far = sr.steps[:cp_step]
            last = steps_so_far[-1]
            lambdas.append(last.lambda_t)
            cost_emas.append(last.cost_ema)
            gammas.append(last.gamma)
            cost_window = steps_so_far[-min(50, len(steps_so_far)):]
            avg_costs.append(float(np.mean([s.cost for s in cost_window])))

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
        per_seed_arm_fracs = {
            ARM_SHORT[a]: [float(f) for f in arm_frac_lists[a]]
            for a in ARM_ORDER
        }

        curves.append({
            "step": cp_step,
            "phase": _step_to_phase_label(cp_step),
            "phase_boundaries": phase_boundaries,
            "mean_lambda": float(np.mean(lambdas)),
            "std_lambda": float(np.std(lambdas)),
            "per_seed_lambda": [float(l) for l in lambdas],
            "mean_cost_ema": float(np.mean(cost_emas)),
            "std_cost_ema": float(np.std(cost_emas)),
            "mean_gamma": float(np.mean(gammas)),
            "std_gamma": float(np.std(gammas)),
            "mean_window_cost": float(np.mean(avg_costs)),
            "std_window_cost": float(np.std(avg_costs)),
            "per_seed_window_cost": [float(c) for c in avg_costs],
            "arm_fractions": arm_fracs,
            "arm_fractions_std": arm_fracs_std,
            "per_seed_arm_fractions": per_seed_arm_fracs,
            "n_seeds": n_seeds,
        })

    phase_summaries: Dict[str, Dict[str, Any]] = {}
    for p in (1, 2, 3):
        p_metrics = [sr.phase_metrics(p) for sr in seed_results]
        if not p_metrics or not p_metrics[0]:
            continue
        phase_summaries[f"phase{p}_summary"] = {
            "mean_reward": float(np.mean([m["mean_reward"] for m in p_metrics])),
            "mean_cost": float(np.mean([m["mean_cost"] for m in p_metrics])),
            "mean_lambda": float(np.mean([m["mean_lambda"] for m in p_metrics])),
            "arm_fractions": {
                ARM_SHORT[a]: float(np.mean([m["arm_fractions"][a] for m in p_metrics]))
                for a in ARM_ORDER
            },
        }

    per_seed_data: Dict[str, List[float]] = {}
    for p in (1, 2, 3):
        p_metrics = [sr.phase_metrics(p) for sr in seed_results]
        if not p_metrics or not p_metrics[0]:
            continue
        per_seed_data[f"per_seed_phase{p}_reward"] = [m["mean_reward"] for m in p_metrics]
        per_seed_data[f"per_seed_phase{p}_cost"] = [m["mean_cost"] for m in p_metrics]

    return {
        "label": seed_results[0].condition,
        "curves": curves,
        **phase_summaries,
        **per_seed_data,
    }


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading K=3 data ...")
    fs = FeatureService()
    feature_dim = fs.dimension

    train_all = _load_all(VAL_DATA_PATH, fs, ARM_ORDER)
    test_all = _load_all(HOLDOUT_DATA_PATH, fs, ARM_ORDER)

    logger.info("  Train (val): %d prompts — online learning, no eval", train_all.n)
    logger.info("  Eval (holdout): %d prompts — 3 phases × %d", test_all.n, PHASE_N)

    rng_global = np.random.default_rng(42)
    all_indices = rng_global.permutation(test_all.n)
    p1_idx = all_indices[:PHASE_N]
    p2_idx = all_indices[PHASE_N:2 * PHASE_N]
    p3_idx = all_indices[2 * PHASE_N:3 * PHASE_N]

    phase1 = SplitData(
        prompts=[test_all.prompts[i] for i in p1_idx],
        rewards={a: test_all.rewards[a][p1_idx] for a in ARM_ORDER},
        costs={a: test_all.costs[a][p1_idx] for a in ARM_ORDER},
        embeddings=test_all.embeddings[p1_idx],
    )

    registry = build_model_registry(ARM_ORDER)
    gemini_meta = registry[GEMINI_ID]
    old_input = gemini_meta["input_cost_per_m"]
    old_output = gemini_meta["output_cost_per_m"]

    phase2_raw = SplitData(
        prompts=[test_all.prompts[i] for i in p2_idx],
        rewards={a: test_all.rewards[a][p2_idx] for a in ARM_ORDER},
        costs={a: test_all.costs[a][p2_idx] for a in ARM_ORDER},
        embeddings=test_all.embeddings[p2_idx],
    )
    phase2 = _apply_gemini_cost_reduction(
        phase2_raw, GEMINI_ID,
        old_input, old_output,
        GEMINI_NEW_INPUT_COST, GEMINI_NEW_OUTPUT_COST,
    )

    # Phase 3 uses original pricing (same as Phase 1)
    phase3 = SplitData(
        prompts=[test_all.prompts[i] for i in p3_idx],
        rewards={a: test_all.rewards[a][p3_idx] for a in ARM_ORDER},
        costs={a: test_all.costs[a][p3_idx] for a in ARM_ORDER},
        embeddings=test_all.embeddings[p3_idx],
    )

    phase_boundaries = [PHASE_N, 2 * PHASE_N, 3 * PHASE_N]

    # ------------------------------------------------------------------
    # Calibrate cost penalties for Recalibrated Bandit
    # ------------------------------------------------------------------
    train_phase2 = _apply_gemini_cost_reduction(
        train_all, GEMINI_ID,
        old_input, old_output,
        GEMINI_NEW_INPUT_COST, GEMINI_NEW_OUTPUT_COST,
    )

    recalibrated_phase2_cps: Dict[str, float] = {}
    recalibrated_phase3_cps: Dict[str, float] = {}
    for target, blabel in zip(BUDGET_TARGETS, BUDGET_LABELS):
        matched_cp = MATCHED_STATIC_CPS[blabel]

        logger.info(
            "\nCalibrating Phase 2 λ for %s (target=$%.2e) ...", blabel, target,
        )
        recalibrated_phase2_cps[blabel] = _calibrate_phase_cp(
            train_data=train_all,
            cal_data=train_phase2,
            registry=registry,
            feature_dim=feature_dim,
            original_cp=matched_cp,
            budget_target=target,
            gemini_input=GEMINI_NEW_INPUT_COST,
            gemini_output=GEMINI_NEW_OUTPUT_COST,
        )

        logger.info(
            "\nCalibrating Phase 3 λ for %s (target=$%.2e) ...", blabel, target,
        )
        recalibrated_phase3_cps[blabel] = _calibrate_phase_cp(
            train_data=train_all,
            cal_data=train_all,
            registry=registry,
            feature_dim=feature_dim,
            original_cp=matched_cp,
            budget_target=target,
            gemini_input=old_input,
            gemini_output=old_output,
        )

    logger.info("\nRecalibrated Phase 2 cost penalties: %s", recalibrated_phase2_cps)
    logger.info("Recalibrated Phase 3 cost penalties: %s", recalibrated_phase3_cps)

    # ------------------------------------------------------------------
    # Run all conditions
    # ------------------------------------------------------------------
    all_condition_results: Dict[str, Dict[str, Any]] = {}

    for target, blabel in zip(BUDGET_TARGETS, BUDGET_LABELS):
        matched_cp = MATCHED_STATIC_CPS[blabel]
        conditions = _build_conditions(
            target, blabel, matched_cp,
            recalibrated_phase2_cps[blabel],
            recalibrated_phase3_cps[blabel],
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
                sr = _run_three_phase_trial(
                    condition_label=label,
                    train_data=train_all,
                    phase1=phase1,
                    phase2=phase2,
                    phase3=phase3,
                    registry=registry,
                    original_gemini_input=old_input,
                    original_gemini_output=old_output,
                    feature_dim=feature_dim,
                    cost_penalty=cond["cost_penalty"],
                    warmup=cond["warmup"],
                    forgetting_factor=cond["forgetting_factor"],
                    online_learn=cond.get("online_learn", True),
                    budget_pacer=pacer,
                    phase2_cost_penalty=cond.get("phase2_cost_penalty"),
                    phase3_cost_penalty=cond.get("phase3_cost_penalty"),
                    seed=seed,
                )
                seed_results.append(sr)

            agg = _aggregate_seeds(seed_results, phase_boundaries)
            all_condition_results[label] = agg

            for p in (1, 2, 3):
                key = f"phase{p}_summary"
                if key in agg:
                    summ = agg[key]
                    logger.info(
                        "  Phase %d: reward=%.4f  cost=$%.6f  λ=%.3f  arm=%s",
                        p, summ["mean_reward"], summ["mean_cost"],
                        summ["mean_lambda"], summ["arm_fractions"],
                    )

    # Unconstrained baseline
    logger.info("\n=== Unconstrained (cp=0) ===")
    unconstrained_seeds: List[SeedResult] = []
    for s in range(N_SEEDS):
        seed = SEED_OFFSET + s
        sr = _run_three_phase_trial(
            condition_label="Unconstrained",
            train_data=train_all,
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            registry=registry,
            original_gemini_input=old_input,
            original_gemini_output=old_output,
            feature_dim=feature_dim,
            cost_penalty=0.0,
            warmup=True,
            forgetting_factor=BEST_K3_HPARAMS["forgetting_factor"],
            budget_pacer=None,
            seed=seed,
        )
        unconstrained_seeds.append(sr)

    unconstrained_agg = _aggregate_seeds(unconstrained_seeds, phase_boundaries)
    all_condition_results["Unconstrained"] = unconstrained_agg

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "experiment": "03_budget_plus_drift",
        "description": (
            "Three-phase cost drift: normal pricing → Gemini price drop → "
            "pricing restored. Tests budget pacing adaptation in both directions."
        ),
        "arm_order": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "cost_shift_model": GEMINI_ID,
        "gemini_original_input_cost": old_input,
        "gemini_original_output_cost": old_output,
        "gemini_drop_input_cost": GEMINI_NEW_INPUT_COST,
        "gemini_drop_output_cost": GEMINI_NEW_OUTPUT_COST,
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "phase_n": PHASE_N,
        "n_phases": N_PHASES,
        "phase_boundaries": phase_boundaries,
        "budget_targets": BUDGET_TARGETS,
        "budget_labels": BUDGET_LABELS,
        "matched_static_cps": MATCHED_STATIC_CPS,
        "recalibrated_phase2_cps": recalibrated_phase2_cps,
        "recalibrated_phase3_cps": recalibrated_phase3_cps,
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
    logger.info("\n" + "=" * 120)
    logger.info("EXPERIMENT 03: BUDGET + COST DRIFT — Summary (3 phases: normal → price-drop → restored)")
    logger.info("=" * 120)
    logger.info(
        "  %-35s  %8s  %8s  %8s  %8s  %8s  %8s",
        "Condition", "P1 Rwd", "P2 Rwd", "P3 Rwd", "P1 λ", "P2 λ", "P3 λ",
    )
    logger.info("  " + "-" * 100)
    for label, agg in all_condition_results.items():
        vals = []
        for p in (1, 2, 3):
            key = f"phase{p}_summary"
            if key in agg:
                vals.extend([agg[key]["mean_reward"], agg[key]["mean_lambda"]])
            else:
                vals.extend([0.0, 0.0])
        logger.info(
            "  %-35s  %8.4f  %8.4f  %8.4f  %8.3f  %8.3f  %8.3f",
            label, vals[0], vals[2], vals[4], vals[1], vals[3], vals[5],
        )
    logger.info("=" * 120)
    logger.info("Wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
