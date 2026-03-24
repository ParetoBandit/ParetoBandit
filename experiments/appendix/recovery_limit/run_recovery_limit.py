#!/usr/bin/env python3
"""Appendix: Recovery Limit under Quality Degradation.

Characterises the degradation severity from which ParetoBandit can fully
recover within a fixed evaluation horizon, and demonstrates that deeper
degradations converge given a longer Phase 3.

Protocol
--------
For each failure reward in a sweep from 0.05 (catastrophic) to 0.85 (mild):

  1. Train on the validation split (normal conditions).
  2. Phase 1 (608 steps, holdout, normal) — baseline.
  3. Phase 2 (608 steps, holdout, quality-only degradation: reward drops,
     cost unchanged) — the silent regression.
  4. Phase 3 (608 or 1800 steps, holdout, normal) — recovery.

Phase 3 reuses Phase 1 prompts (cycled for extended runs) for a
within-subject P1-vs-P3 comparison.  All runs use the moderate budget
target ($6.62 × 10⁻⁴) with ParetoBandit's tuned hyperparameters.

Extended Phase 3
~~~~~~~~~~~~~~~~
For a subset of degradation levels, Phase 3 is extended to 1800 steps
(~3× standard) by cycling the Phase 1 prompt pool.  This demonstrates
that moderate degradations (30–70%) DO recover given sufficient time,
supporting the paper's claim that the 608-step horizon is the binding
constraint, not an inherent algorithmic limitation.

Analytical bound
~~~~~~~~~~~~~~~~
At the Phase 2/3 boundary, the script extracts the policy's base
variance (x^T A_inv x) and computes the maximum mean-estimate gap
recoverable by staleness-driven exploration:

    Δ_max = α √(V_max · σ²) − λ · Δc̃

where α is the exploration rate, V_max=200 is the variance inflation
cap, σ² is the empirical base variance, λ is the budget pacer's dual
variable, and Δc̃ is the normalised cost differential between Mistral
and the cheapest arm.

Usage::

    python experiments/appendix/recovery_limit/run_recovery_limit.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
    K3_FAILURE_ARM,
    K3_WARMUP_PRIORS_PATH,
    N_SEEDS,
    VAL_DATA_PATH,
)
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from utils.simulation import (
    SplitData,
    apply_quality_degradation,
    build_model_registry,
    compute_normalized_costs,
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

SEED_OFFSET: int = 7000
RESULTS_DIR = Path(__file__).parent / "results"

PHASE_N: int = 608
EXTENDED_PHASE3_N: int = 1800
CHECKPOINT_INTERVAL: int = 20

BUDGET_TARGET: float = 6.62e-4  # moderate
BUDGET_LABEL: str = "moderate"

ALPHA: float = BEST_K3_HPARAMS["alpha"]
PRIOR_N_EFFECTIVE: float = BEST_K3_HPARAMS["prior_n_effective"]
FORGETTING_FACTOR: float = BEST_K3_HPARAMS["forgetting_factor"]

FAILURE_REWARDS: List[float] = [
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85,
]
EXTENDED_FAILURE_REWARDS: List[float] = [
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85,
]

MAX_VAR_INFLATION: float = 200.0


# ======================================================================
# Data types
# ======================================================================


@dataclass
class StepRecord:
    """Per-step metrics."""

    step: int
    phase: int
    model: str
    reward: float
    cost: float


@dataclass
class TrialResult:
    """Results from one (failure_reward, seed) trial."""

    failure_reward: float
    seed: int
    extended: bool
    steps: List[StepRecord] = field(default_factory=list)
    base_variance: Optional[float] = None
    lambda_at_boundary: Optional[float] = None

    def phase_metrics(self, phase: int) -> Dict[str, Any]:
        """Aggregate metrics for a single phase."""
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
            "n_steps": n,
        }

    def checkpoint_curve(
        self, interval: int = CHECKPOINT_INTERVAL,
    ) -> List[Dict[str, Any]]:
        """Windowed reward at each checkpoint for time-series plotting."""
        n_total = len(self.steps)
        checkpoints = sorted(set(
            [1]
            + list(range(interval, n_total + 1, interval))
            + [n_total]
        ))
        window = 50
        curves = []
        for cp in checkpoints:
            recent = self.steps[max(0, cp - window):cp]
            curves.append({
                "step": cp,
                "mean_window_reward": float(np.mean([s.reward for s in recent])),
                "mean_window_cost": float(np.mean([s.cost for s in recent])),
            })
        return curves


# ======================================================================
# Data loading
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
# Trial runner
# ======================================================================


def _run_trial(
    *,
    train_data: SplitData,
    phase1: SplitData,
    phase2: SplitData,
    phase3: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    failure_reward: float,
    extended: bool,
    seed: int,
) -> TrialResult:
    """Run one seed of the three-phase degradation/recovery trial."""
    rng = np.random.default_rng(seed)

    pacer = BudgetPacer(
        target_avg_spend_usd=BUDGET_TARGET,
        mode=PacingMode.ADAPTIVE,
        lr=DEFAULT_PACER_LR,
        ema_alpha=DEFAULT_PACER_EMA_ALPHA,
        lambda_max=DEFAULT_PACER_LAMBDA_MAX,
    )

    fserv = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fserv,
        context_store=store,
        priors="warmup",
        warmup_path=str(K3_WARMUP_PRIORS_PATH),
        prior_n_effective=PRIOR_N_EFFECTIVE,
        alpha=ALPHA,
        cost_penalty=0.0,
        forgetting_factor=FORGETTING_FACTOR,
        budget_pacer=pacer,
    )

    # Train on val split
    train_order = rng.permutation(train_data.n)
    for i in train_order:
        model, log = router.route(train_data.embeddings[i])
        log.cost_usd = float(train_data.costs[model][i])
        router.process_feedback(
            log.request_id, reward=float(train_data.rewards[model][i]),
        )

    result = TrialResult(
        failure_reward=failure_reward, seed=seed, extended=extended,
    )

    # Build Phase 3 data (cycle Phase 1 prompts for extended runs)
    n_p3 = EXTENDED_PHASE3_N if extended else PHASE_N
    if n_p3 > phase3.n:
        n_reps = (n_p3 // phase3.n) + 1
        idx_pool = np.tile(np.arange(phase3.n), n_reps)[:n_p3]
    else:
        idx_pool = np.arange(n_p3)

    phases = [
        (1, phase1, PHASE_N),
        (2, phase2, PHASE_N),
        (3, phase3, n_p3),
    ]

    global_step = 0
    for phase_num, phase_data, n_steps in phases:
        if phase_num == 3:
            order = rng.permutation(idx_pool)
        else:
            order = rng.permutation(phase_data.n)[:n_steps]

        # Capture diagnostics at the Phase 2→3 boundary
        if phase_num == 3:
            result.lambda_at_boundary = pacer.lambda_t
            sample_x = phase_data.embeddings[order[0]]
            if FAILURE_ARM in router.bandit.A_inv:
                A_inv = router.bandit.A_inv[FAILURE_ARM]
                result.base_variance = float(
                    sample_x.dot(A_inv).dot(sample_x)
                )

        for idx in order:
            idx = int(idx % phase_data.n)
            model, log = router.route(phase_data.embeddings[idx])
            reward = float(phase_data.rewards[model][idx])
            cost = float(phase_data.costs[model][idx])
            log.cost_usd = cost
            router.process_feedback(log.request_id, reward=reward)
            global_step += 1

            result.steps.append(StepRecord(
                step=global_step,
                phase=phase_num,
                model=model,
                reward=reward,
                cost=cost,
            ))

    return result


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_trials(
    trials: List[TrialResult],
) -> Dict[str, Any]:
    """Aggregate per-seed trials for one failure reward level."""
    fr = trials[0].failure_reward
    extended = trials[0].extended

    phase_summaries: Dict[str, Any] = {}
    for p in (1, 2, 3):
        p_metrics = [t.phase_metrics(p) for t in trials]
        if not p_metrics or not p_metrics[0]:
            continue
        phase_summaries[f"phase{p}"] = {
            "mean_reward": float(np.mean(
                [m["mean_reward"] for m in p_metrics],
            )),
            "std_reward": float(np.std(
                [m["mean_reward"] for m in p_metrics],
            )),
            "mean_cost": float(np.mean(
                [m["mean_cost"] for m in p_metrics],
            )),
            "arm_fractions": {
                ARM_SHORT[a]: float(np.mean(
                    [m["arm_fractions"][a] for m in p_metrics],
                ))
                for a in ARM_ORDER
            },
            "per_seed_reward": [m["mean_reward"] for m in p_metrics],
        }

    p1_rwd = phase_summaries.get("phase1", {}).get("mean_reward", 0)
    p3_rwd = phase_summaries.get("phase3", {}).get("mean_reward", 0)
    ratio = p3_rwd / p1_rwd if p1_rwd > 0 else 0

    normal_reward = float(np.mean(
        [t.phase_metrics(1).get("mean_reward", 0) for t in trials],
    ))
    degradation_pct = (
        (normal_reward - fr) / normal_reward * 100 if normal_reward > 0 else 0
    )

    base_vars = [t.base_variance for t in trials if t.base_variance is not None]
    lambdas = [
        t.lambda_at_boundary for t in trials
        if t.lambda_at_boundary is not None
    ]

    # Checkpoint curves (aggregate across seeds)
    all_curves = [t.checkpoint_curve() for t in trials]
    agg_curves: List[Dict[str, Any]] = []
    if all_curves:
        for i, cp in enumerate(all_curves[0]):
            rewards_at_cp = [c[i]["mean_window_reward"] for c in all_curves]
            agg_curves.append({
                "step": cp["step"],
                "mean_window_reward": float(np.mean(rewards_at_cp)),
                "std_window_reward": float(np.std(rewards_at_cp)),
                "per_seed_window_reward": [float(r) for r in rewards_at_cp],
            })

    return {
        "failure_reward": fr,
        "degradation_pct": degradation_pct,
        "extended": extended,
        "n_seeds": len(trials),
        "p3_p1_ratio": ratio,
        "phases": phase_summaries,
        "mean_base_variance": float(np.mean(base_vars)) if base_vars else None,
        "mean_lambda_at_boundary": float(np.mean(lambdas)) if lambdas else None,
        "curves": agg_curves,
    }


# ======================================================================
# Analytical bound
# ======================================================================


def _compute_analytical_bound(
    alpha: float,
    max_inflation: float,
    base_variance: float,
    lambda_t: float,
    delta_cost: float,
) -> float:
    """Maximum mean-estimate gap recoverable by staleness exploration.

    Returns:
        The max gap Δ_max = α √(V_max · σ²) − λ · Δc̃.
        Positive means the exploration bonus can bridge this gap;
        negative means even at maximum inflation the cost penalty dominates.
    """
    return alpha * np.sqrt(max_inflation * base_variance) - lambda_t * delta_cost


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

    rng_global = np.random.default_rng(42)
    all_idx = rng_global.permutation(test_all.n)
    p1_idx = all_idx[:PHASE_N]
    p2_idx = all_idx[PHASE_N:2 * PHASE_N]

    phase1 = SplitData(
        prompts=[test_all.prompts[i] for i in p1_idx],
        rewards={a: test_all.rewards[a][p1_idx] for a in ARM_ORDER},
        costs={a: test_all.costs[a][p1_idx] for a in ARM_ORDER},
        embeddings=test_all.embeddings[p1_idx],
    )
    phase3 = phase1

    registry = build_model_registry(ARM_ORDER)
    norm_costs = compute_normalized_costs(registry, ARM_ORDER)
    delta_cost = norm_costs[FAILURE_ARM] - min(norm_costs.values())

    mistral_normal_reward = float(np.mean(phase1.rewards[FAILURE_ARM]))
    logger.info("Mistral normal reward: %.4f", mistral_normal_reward)
    logger.info("Normalised cost differential (Mistral - cheapest): %.4f", delta_cost)

    # ------------------------------------------------------------------
    # Standard sweep (608-step Phase 3)
    # ------------------------------------------------------------------
    logger.info("\n=== Standard sweep (Phase 3 = %d steps) ===", PHASE_N)
    standard_results: List[Dict[str, Any]] = []

    for fr in FAILURE_REWARDS:
        degradation_pct = (mistral_normal_reward - fr) / mistral_normal_reward * 100
        logger.info(
            "  failure_reward=%.2f  (%.1f%% degradation)", fr, degradation_pct,
        )

        p2_base = SplitData(
            prompts=[test_all.prompts[i] for i in p2_idx],
            rewards={a: test_all.rewards[a][p2_idx] for a in ARM_ORDER},
            costs={a: test_all.costs[a][p2_idx] for a in ARM_ORDER},
            embeddings=test_all.embeddings[p2_idx],
        )
        phase2 = apply_quality_degradation(
            p2_base, degraded_arm=FAILURE_ARM, degraded_reward=fr,
        )

        trials: List[TrialResult] = []
        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            tr = _run_trial(
                train_data=train_all,
                phase1=phase1,
                phase2=phase2,
                phase3=phase3,
                registry=registry,
                feature_dim=feature_dim,
                failure_reward=fr,
                extended=False,
                seed=seed,
            )
            trials.append(tr)

        agg = _aggregate_trials(trials)
        standard_results.append(agg)

        logger.info(
            "    P1=%.4f  P2=%.4f  P3=%.4f  P3/P1=%.1f%%  Mistral_P3=%s",
            agg["phases"]["phase1"]["mean_reward"],
            agg["phases"]["phase2"]["mean_reward"],
            agg["phases"]["phase3"]["mean_reward"],
            agg["p3_p1_ratio"] * 100,
            f"{agg['phases']['phase3']['arm_fractions']['Mistral-Large']:.0%}",
        )

    # ------------------------------------------------------------------
    # Extended sweep (1800-step Phase 3)
    # ------------------------------------------------------------------
    logger.info(
        "\n=== Extended sweep (Phase 3 = %d steps) ===", EXTENDED_PHASE3_N,
    )
    extended_results: List[Dict[str, Any]] = []

    for fr in EXTENDED_FAILURE_REWARDS:
        degradation_pct = (mistral_normal_reward - fr) / mistral_normal_reward * 100
        logger.info(
            "  failure_reward=%.2f  (%.1f%% degradation)", fr, degradation_pct,
        )

        p2_base = SplitData(
            prompts=[test_all.prompts[i] for i in p2_idx],
            rewards={a: test_all.rewards[a][p2_idx] for a in ARM_ORDER},
            costs={a: test_all.costs[a][p2_idx] for a in ARM_ORDER},
            embeddings=test_all.embeddings[p2_idx],
        )
        phase2 = apply_quality_degradation(
            p2_base, degraded_arm=FAILURE_ARM, degraded_reward=fr,
        )

        trials = []
        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            tr = _run_trial(
                train_data=train_all,
                phase1=phase1,
                phase2=phase2,
                phase3=phase3,
                registry=registry,
                feature_dim=feature_dim,
                failure_reward=fr,
                extended=True,
                seed=seed,
            )
            trials.append(tr)

        agg = _aggregate_trials(trials)
        extended_results.append(agg)

        logger.info(
            "    P1=%.4f  P2=%.4f  P3=%.4f  P3/P1=%.1f%%",
            agg["phases"]["phase1"]["mean_reward"],
            agg["phases"]["phase2"]["mean_reward"],
            agg["phases"]["phase3"]["mean_reward"],
            agg["p3_p1_ratio"] * 100,
        )

    # ------------------------------------------------------------------
    # Analytical bound
    # ------------------------------------------------------------------
    ref = standard_results[-1]  # fr=0.85 (mildest, most data)
    mean_var = ref.get("mean_base_variance", 0.1)
    mean_lam = ref.get("mean_lambda_at_boundary", 0.3)
    bound = _compute_analytical_bound(
        ALPHA, MAX_VAR_INFLATION, mean_var, mean_lam, delta_cost,
    )
    logger.info("\nAnalytical recovery bound:")
    logger.info("  alpha=%.3f, V_max=%.0f, base_var=%.4f", ALPHA, MAX_VAR_INFLATION, mean_var)
    logger.info("  lambda=%.3f, delta_cost=%.4f", mean_lam, delta_cost)
    logger.info("  max_recoverable_gap = %.4f", bound)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "experiment": "appendix_recovery_limit",
        "description": (
            "Recovery limit study: sweeps degradation severity under "
            "quality-only degradation (cost unchanged) and measures "
            "Phase 3 recovery.  Standard (608-step) and extended "
            f"({EXTENDED_PHASE3_N}-step) Phase 3 horizons."
        ),
        "arm_order": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "failure_arm": FAILURE_ARM,
        "budget_target": BUDGET_TARGET,
        "budget_label": BUDGET_LABEL,
        "phase_n": PHASE_N,
        "extended_phase3_n": EXTENDED_PHASE3_N,
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "alpha": ALPHA,
        "forgetting_factor": FORGETTING_FACTOR,
        "prior_n_effective": PRIOR_N_EFFECTIVE,
        "max_var_inflation": MAX_VAR_INFLATION,
        "mistral_normal_reward": mistral_normal_reward,
        "delta_cost_normalized": delta_cost,
        "analytical_bound": {
            "mean_base_variance": mean_var,
            "mean_lambda_at_boundary": mean_lam,
            "delta_cost": delta_cost,
            "max_recoverable_gap": bound,
        },
        "standard_results": standard_results,
        "extended_results": extended_results,
    }

    out_path = RESULTS_DIR / "recovery_limit_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nSaved results to %s", out_path)
    logger.info("Wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
