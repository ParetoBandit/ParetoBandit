#!/usr/bin/env python3
"""Experiment 02: Catastrophic Model Failure (Three-Phase).

Demonstrates that ParetoBandit's BudgetPacer maintains cost-invariance
($/request) through a catastrophic model failure while the bandit
automatically redistributes traffic across the remaining K>2 models.

Experimental setup
------------------
The pipeline follows the train-then-evaluate design of Experiment 03.

  **Train phase** (val split, 1,785 prompts): Online learning under
  normal conditions.  No evaluation metrics are recorded.  The router
  and BudgetPacer calibrate their internal state.

  **Evaluation phase** (holdout split, 1,824 prompts → 608 per phase):

    **Phase 1** (steps 1–608): Normal.  All models are healthy.  The
    router has converged to a Mistral-heavy allocation via warmup
    priors and online training.

    **Phase 2** (steps 609–1216): **Catastrophic failure** —
    Mistral-Large's reward drops to ~0.05 and per-request cost drops
    to $0 (the API returns errors/garbage and the provider does not
    charge).  The router must detect the quality collapse through its
    reward signal and redistribute traffic to Llama-8B and Gemini-Pro.

    **Phase 3** (steps 1217–1824): **Recovery** — Mistral-Large is
    restored to normal quality and pricing.  The router must
    re-discover Mistral via UCB exploration as geometric forgetting
    erases the Phase 2 failure memory.

  Phase 3 deliberately reuses Phase 1 prompts so that the P1-vs-P3
  comparison is a controlled within-subject design: any difference
  between the two phases is attributable solely to the router's
  internal state, not prompt-sampling variability.

Registry handling
-----------------
The model registry is NOT mutated at phase boundaries.  Unlike the
cost-drift scenario (Experiment 03), where pricing changes are
publicly announced, a model failure is an emergent event discovered
through observed rewards.  The router's internal cost penalty
continues to reflect Mistral's normal pricing; only the reward signal
drives adaptation.

Conditions (per budget level)
-----------------------------
Three algorithm variants at each budget target, plus an unconstrained
baseline:

  - **Fixed Policy**: warmup priors frozen, static cost penalty matched
    to the budget target, no online learning, no BudgetPacer.  Keeps
    routing to dead Mistral — the counterfactual.
  - **Naive Bandit (γ=1.0)**: infinite memory, static cost penalty,
    online learning.  Detects failure but Phase 1 inertia slows both
    detection and recovery.
  - **ParetoBandit (γ=0.995)**: geometric forgetting with BudgetPacer
    active.  Fast detection, redistribution, and budget maintenance.
  - **Unconstrained**: ParetoBandit without budget constraint (λ=0).

Usage::

    python experiments/02_catastrophic_failure/run_catastrophic_failure.py
"""

from __future__ import annotations

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
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_BUDGET_LABELS,
    K3_BUDGET_TARGETS,
    K3_FAILURE_ARM,
    K3_FAILURE_REWARD,
    K3_WARMUP_PRIORS_PATH,
    N_SEEDS,
    VAL_DATA_PATH,
)
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from utils.simulation import (
    SplitData,
    apply_catastrophic_failure,
    build_model_registry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in (
    "pareto_bandit.router",
    "pareto_bandit.feature_service",
    "pareto_bandit.policy",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Constants
# ======================================================================

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = K3_ARM_SHORT

FAILURE_ARM: str = K3_FAILURE_ARM
FAILURE_REWARD: float = K3_FAILURE_REWARD

SEED_OFFSET: int = 6000
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
# Three-Phase Trial
# ======================================================================


def _run_three_phase_trial(
    *,
    condition_label: str,
    train_data: SplitData,
    phase1: SplitData,
    phase2: SplitData,
    phase3: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    cost_penalty: float,
    warmup: bool = True,
    forgetting_factor: float = 1.0,
    online_learn: bool = True,
    budget_pacer: Optional[BudgetPacer] = None,
    seed: int,
) -> SeedResult:
    """Run one seed through the three-phase catastrophic failure scenario.

    Stages:
      1. **Train** (val split, normal conditions): online-learn, no metrics.
      2. **Phase 1** (holdout, normal): evaluate and record.
      3. **Phase 2** (holdout, Mistral failed): evaluate and record.
      4. **Phase 3** (holdout, Mistral recovered): evaluate and record.

    The registry is NOT mutated at phase boundaries.  Adaptation is
    driven purely by the reward signal embedded in the phase data.

    Parameters
    ----------
    condition_label : str
        Human-readable condition name.
    train_data : SplitData
        Validation split for online learning (no metrics recorded).
    phase1, phase2, phase3 : SplitData
        Holdout evaluation data for each phase.  Phase 2 has Mistral's
        rewards degraded and costs zeroed via ``apply_catastrophic_failure``.
        Phase 3 uses the same prompts as Phase 1 (normal rewards/costs).
    registry : dict
        Model registry (unchanged throughout all phases).
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

    for t in range(n_total):
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
# Condition Builder
# ======================================================================


def _build_conditions(
    budget_target: float,
    budget_label: str,
    matched_cp: float,
) -> List[Dict[str, Any]]:
    """Build three conditions for a given budget target.

    Parameters
    ----------
    budget_target : float
        Dollar budget target per request.
    budget_label : str
        Human-readable budget label (tight/moderate/loose).
    matched_cp : float
        Static cost penalty that produces similar Phase 1 spend.

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
        },
        {
            "label": f"Naive Bandit ({budget_label})",
            "budget_target": None,
            "cost_penalty": matched_cp,
            "warmup": True,
            "forgetting_factor": 1.0,
            "online_learn": True,
        },
        {
            "label": f"ParetoBandit ({budget_label})",
            "budget_target": budget_target,
            "cost_penalty": 0.0,
            "warmup": True,
            "forgetting_factor": BEST_K3_HPARAMS["forgetting_factor"],
            "online_learn": True,
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
            return "failure"
        else:
            return "recovered"

    curves: List[Dict[str, Any]] = []
    for cp_step in checkpoints:
        lambdas, cost_emas, gammas = [], [], []
        arm_frac_lists: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
        avg_costs: List[float] = []
        avg_rewards: List[float] = []

        for sr in seed_results:
            steps_so_far = sr.steps[:cp_step]
            last = steps_so_far[-1]
            lambdas.append(last.lambda_t)
            cost_emas.append(last.cost_ema)
            gammas.append(last.gamma)
            cost_window = steps_so_far[-min(50, len(steps_so_far)):]
            avg_costs.append(float(np.mean([s.cost for s in cost_window])))
            avg_rewards.append(float(np.mean([s.reward for s in cost_window])))

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
            "mean_window_reward": float(np.mean(avg_rewards)),
            "std_window_reward": float(np.std(avg_rewards)),
            "per_seed_window_reward": [float(r) for r in avg_rewards],
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
                ARM_SHORT[a]: float(np.mean([
                    m["arm_fractions"][a] for m in p_metrics
                ]))
                for a in ARM_ORDER
            },
        }

    per_seed_data: Dict[str, List[float]] = {}
    for p in (1, 2, 3):
        p_metrics = [sr.phase_metrics(p) for sr in seed_results]
        if not p_metrics or not p_metrics[0]:
            continue
        per_seed_data[f"per_seed_phase{p}_reward"] = [
            m["mean_reward"] for m in p_metrics
        ]
        per_seed_data[f"per_seed_phase{p}_cost"] = [
            m["mean_cost"] for m in p_metrics
        ]

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

    logger.info(
        "  Train (val): %d prompts — online learning, no eval", train_all.n,
    )
    logger.info(
        "  Eval (holdout): %d prompts — P1/P3 share %d, P2 uses %d different",
        test_all.n, PHASE_N, PHASE_N,
    )

    rng_global = np.random.default_rng(42)
    all_indices = rng_global.permutation(test_all.n)
    p1_idx = all_indices[:PHASE_N]
    p2_idx = all_indices[PHASE_N:2 * PHASE_N]

    phase1 = SplitData(
        prompts=[test_all.prompts[i] for i in p1_idx],
        rewards={a: test_all.rewards[a][p1_idx] for a in ARM_ORDER},
        costs={a: test_all.costs[a][p1_idx] for a in ARM_ORDER},
        embeddings=test_all.embeddings[p1_idx],
    )

    phase2 = apply_catastrophic_failure(
        SplitData(
            prompts=[test_all.prompts[i] for i in p2_idx],
            rewards={a: test_all.rewards[a][p2_idx] for a in ARM_ORDER},
            costs={a: test_all.costs[a][p2_idx] for a in ARM_ORDER},
            embeddings=test_all.embeddings[p2_idx],
        ),
        failed_arm=FAILURE_ARM,
        failure_reward=FAILURE_REWARD,
    )

    phase3 = phase1

    phase_boundaries = [PHASE_N, 2 * PHASE_N, 3 * PHASE_N]

    registry = build_model_registry(ARM_ORDER)

    logger.info("\nFailure scenario:")
    logger.info("  Failed model: %s", ARM_SHORT[FAILURE_ARM])
    logger.info("  Failure reward: %.2f", FAILURE_REWARD)
    logger.info("  Failure cost: $0 (API failure, no charge)")
    logger.info("  Phase sizes: %d / %d / %d", phase1.n, phase2.n, phase3.n)
    logger.info("  Phase 3 reuses Phase 1 prompts (within-subject design)")

    mistral_normal_reward = float(np.mean(phase1.rewards[FAILURE_ARM]))
    mistral_failure_reward = float(np.mean(phase2.rewards[FAILURE_ARM]))
    logger.info(
        "  Mistral mean reward: normal=%.3f  failure=%.3f",
        mistral_normal_reward, mistral_failure_reward,
    )

    # ------------------------------------------------------------------
    # Run all conditions
    # ------------------------------------------------------------------
    all_condition_results: Dict[str, Dict[str, Any]] = {}

    for target, blabel in zip(BUDGET_TARGETS, BUDGET_LABELS):
        matched_cp = MATCHED_STATIC_CPS[blabel]
        conditions = _build_conditions(target, blabel, matched_cp)

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
                    feature_dim=feature_dim,
                    cost_penalty=cond["cost_penalty"],
                    warmup=cond["warmup"],
                    forgetting_factor=cond["forgetting_factor"],
                    online_learn=cond.get("online_learn", True),
                    budget_pacer=pacer,
                    seed=seed,
                )
                seed_results.append(sr)

            agg = _aggregate_seeds(seed_results, phase_boundaries)
            all_condition_results[label] = agg

            for p in (1, 2, 3):
                key = f"phase{p}_summary"
                if key in agg:
                    summ = agg[key]
                    arm_str = "  ".join(
                        f"{k}={v:.0%}"
                        for k, v in summ["arm_fractions"].items()
                    )
                    logger.info(
                        "  Phase %d: reward=%.4f  cost=$%.6f  λ=%.3f  %s",
                        p, summ["mean_reward"], summ["mean_cost"],
                        summ["mean_lambda"], arm_str,
                    )

    # Unconstrained baseline
    logger.info("\n=== Unconstrained (λ=0) ===")
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

    for p in (1, 2, 3):
        key = f"phase{p}_summary"
        if key in unconstrained_agg:
            summ = unconstrained_agg[key]
            logger.info(
                "  Phase %d: reward=%.4f  cost=$%.6f",
                p, summ["mean_reward"], summ["mean_cost"],
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "experiment": "02_catastrophic_failure",
        "description": (
            "Three-phase catastrophic failure: normal → Mistral failure "
            "(reward→0.05, cost→$0) → recovery.  Phase 3 reuses Phase 1 "
            "prompts for within-subject P1-vs-P3 comparison.  Registry "
            "is NOT mutated; adaptation is reward-driven."
        ),
        "arm_order": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "failure_arm": FAILURE_ARM,
        "failure_arm_short": ARM_SHORT[FAILURE_ARM],
        "failure_reward": FAILURE_REWARD,
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "phase_n": PHASE_N,
        "n_phases": N_PHASES,
        "phase_boundaries": phase_boundaries,
        "budget_targets": BUDGET_TARGETS,
        "budget_labels": BUDGET_LABELS,
        "matched_static_cps": MATCHED_STATIC_CPS,
        "pacer_lr": PACER_LR,
        "pacer_lambda_max": PACER_LAMBDA_MAX,
        "pacer_ema_alpha": PACER_EMA_ALPHA,
        "prior_n_effective": PRIOR_N_EFFECTIVE,
        "alpha": ALPHA,
        "forgetting_factor": BEST_K3_HPARAMS["forgetting_factor"],
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "train_n": train_all.n,
        "eval_n": test_all.n,
        "conditions": all_condition_results,
    }

    out_path = RESULTS_DIR / "catastrophic_failure_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nSaved results to %s", out_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 120)
    logger.info(
        "EXPERIMENT 02: CATASTROPHIC FAILURE — Summary "
        "(3 phases: normal → failure → recovered)"
    )
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
