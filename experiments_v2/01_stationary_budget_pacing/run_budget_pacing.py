#!/usr/bin/env python3
"""Experiment 01: Stationary Budget Pacing — Pareto frontier comparison.

Compares BudgetPacer (ADAPTIVE mode, various targets) against a static
cost_penalty sweep on the K=3 portfolio under stationary conditions.

**Methodological notes:**

- *Cost feedback*:  The pacer receives actual per-request costs from the
  offline dataset (the same costs used for Pareto metrics), not the router's
  heuristic token-count estimate.  This mirrors production, where the billing
  system provides exact costs.
- *Forgetting factor*:  ``ff = 1.0`` (no forgetting) — appropriate because
  this experiment evaluates stationary reward distributions.  Non-stationary
  conditions are addressed in Experiment 02.
- *Knob asymmetry*:  The static baseline sweeps a dimensionless
  ``cost_penalty`` weight, while the pacer sweeps a dollar-denominated
  ``target_avg_spend_usd``.  The Pareto frontier (mean_reward vs. mean_cost)
  is the common evaluation surface; both methods trace a curve on the same
  axes regardless of their internal parameterisation.
- *Pacer hyperparameters*:  ``lr`` and ``lambda_max`` are fixed to defaults
  chosen before inspecting results.  A sensitivity analysis over these
  is deferred to Experiment 04.

For each condition × seed the script:
  1. Creates a router with K=3 warmup priors.
  2. Online-learns on the train split (shuffled).
  3. Evaluates on the holdout/test split (shuffled).
  4. Tracks per-step reward, cost, model selection, lambda_t, and budget
     compliance diagnostics.

Produces a JSON results file consumed by ``generate_figure.py``.

Usage:
    python experiments_v2/01_stationary_budget_pacing/run_budget_pacing.py
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

from bandit_gpt.budget_pacer import BudgetPacer, PacingMode
from bandit_gpt.config import (
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
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
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Constants
# ======================================================================

ARM_ORDER: List[str] = K3_ARM_ORDER
N_SEEDS = 5
SEED_OFFSET = 3000
RESULTS_DIR = Path(__file__).parent / "results"

WARMUP_HPARAMS: Dict[str, Any] = {
    "alpha": 1.0,
    "prior_n_effective": 50.0,
    "forgetting_factor": 1.0,
    "policy": "disjoint",
}

STATIC_COST_PENALTIES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]

PACER_LR = 0.05
PACER_LAMBDA_MAX = 5.0


# ======================================================================
# Dataclasses for results
# ======================================================================


@dataclass
class StepRecord:
    """Per-step metrics recorded during online evaluation."""

    model: str
    reward: float
    cost: float
    oracle_reward: float
    lambda_t: float


@dataclass
class TrialResult:
    """Aggregate metrics for one (condition, seed) trial."""

    condition: str
    seed: int
    mean_reward: float
    mean_cost: float
    cumulative_regret: float
    model_fractions: Dict[str, float]
    trailing_100_cost: float
    final_lambda: float
    budget_utilization: float = 0.0
    lambda_trajectory_quartiles: Dict[str, float] = field(default_factory=dict)
    per_step: List[Dict[str, Any]] = field(default_factory=list)


# ======================================================================
# Simulation
# ======================================================================


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    cost_penalty: float = 0.0,
    budget_pacer: Optional[BudgetPacer] = None,
) -> BanditRouter:
    """Build a K=3 router with warmup priors."""
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    return BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup",
        warmup_path=str(K3_WARMUP_PRIORS_PATH),
        prior_n_effective=WARMUP_HPARAMS["prior_n_effective"],
        alpha=WARMUP_HPARAMS["alpha"],
        use_corralling=False,
        cost_penalty=cost_penalty,
        forgetting_factor=WARMUP_HPARAMS["forgetting_factor"],
        policy=WARMUP_HPARAMS["policy"],
        budget_pacer=budget_pacer,
    )


def _run_trial(
    train: SplitData,
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    condition: str,
    cost_penalty: float = 0.0,
    budget_pacer: Optional[BudgetPacer] = None,
    target_spend: float = 0.0,
    seed: int,
    record_per_step: bool = False,
) -> TrialResult:
    """Run one train->test trial and return aggregate metrics.

    The pacer receives **actual per-request costs** from the offline dataset
    (via ``log.cost_usd`` override) rather than the router's heuristic token
    estimate.  This ensures the dual-variable update reflects the true cost
    scale, consistent with production where billing provides exact costs.

    Args:
        train: Training split data.
        test: Test/holdout split data.
        registry: Model registry with pricing info.
        feature_dim: Embedding dimensionality.
        condition: Label for this experimental condition.
        cost_penalty: Static cost penalty weight (for baseline).
        budget_pacer: Optional BudgetPacer instance.
        target_spend: Budget target in $/req (for compliance metrics).
        seed: Random seed for reproducibility.
        record_per_step: If True, record full per-step trace for seed 0.

    Returns:
        TrialResult with aggregate and compliance metrics.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    if budget_pacer is not None:
        budget_pacer.reset()

    router = _create_router(
        registry, feature_dim,
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
    )

    # --- Train phase (online learning, no metrics) ---
    train_order = rng.permutation(train.n)
    for i in train_order:
        model, log = router.route(
            train.embeddings[i], total_steps=train.n,
        )
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # --- Test phase (metrics) ---
    test_order = rng.permutation(test.n)
    steps: List[StepRecord] = []
    model_counts: Dict[str, int] = {m: 0 for m in ARM_ORDER}

    for i in test_order:
        model, log = router.route(
            test.embeddings[i], total_steps=test.n,
        )
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        oracle_reward = max(float(test.rewards[a][i]) for a in ARM_ORDER)

        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

        lam = budget_pacer.lambda_t if budget_pacer is not None else 0.0
        steps.append(StepRecord(
            model=model, reward=reward, cost=cost,
            oracle_reward=oracle_reward, lambda_t=lam,
        ))
        model_counts[model] += 1

    rewards = np.array([s.reward for s in steps])
    costs = np.array([s.cost for s in steps])
    oracles = np.array([s.oracle_reward for s in steps])
    lambdas = np.array([s.lambda_t for s in steps])

    trailing_100_cost = float(np.mean(costs[-100:])) if len(costs) >= 100 else float(np.mean(costs))
    model_fractions = {m: cnt / len(steps) for m, cnt in model_counts.items()}

    budget_util = float(np.mean(costs) / target_spend) if target_spend > 0 else 0.0

    lambda_quarts: Dict[str, float] = {}
    if budget_pacer is not None and len(lambdas) > 0:
        lambda_quarts = {
            "q25": float(np.percentile(lambdas, 25)),
            "q50": float(np.percentile(lambdas, 50)),
            "q75": float(np.percentile(lambdas, 75)),
            "final": float(lambdas[-1]),
            "last_100_mean": float(np.mean(lambdas[-100:])) if len(lambdas) >= 100 else float(np.mean(lambdas)),
        }

    per_step_records: List[Dict[str, Any]] = []
    if record_per_step:
        per_step_records = [
            {
                "model": s.model, "reward": s.reward, "cost": s.cost,
                "oracle_reward": s.oracle_reward, "lambda_t": s.lambda_t,
            }
            for s in steps
        ]

    return TrialResult(
        condition=condition,
        seed=seed,
        mean_reward=float(np.mean(rewards)),
        mean_cost=float(np.mean(costs)),
        cumulative_regret=float(np.sum(oracles - rewards)),
        model_fractions=model_fractions,
        trailing_100_cost=trailing_100_cost,
        final_lambda=steps[-1].lambda_t if steps else 0.0,
        budget_utilization=budget_util,
        lambda_trajectory_quartiles=lambda_quarts,
        per_step=per_step_records,
    )


# ======================================================================
# Condition builders
# ======================================================================


def _compute_budget_targets(train: SplitData) -> List[float]:
    """Compute 7 log-spaced targets from empirical per-model mean costs.

    Uses the actual dataset cost distribution rather than synthetic
    token-count estimates, ensuring targets are on the same scale as the
    costs the pacer will observe during the trial.

    Args:
        train: Training split with per-model costs.

    Returns:
        Seven log-spaced budget targets spanning the cheapest to most
        expensive model's empirical mean cost per request.
    """
    per_model_means = []
    for m in ARM_ORDER:
        per_model_means.append(float(np.mean(train.costs[m])))

    lo = min(per_model_means)
    hi = max(per_model_means)
    return list(np.geomspace(lo, hi, num=7))


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()

    logger.info("Loading K=3 data ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    train = load_split(TRAIN_DATA_PATH, fs, ARM_ORDER)
    test = load_split(HOLDOUT_DATA_PATH, fs, ARM_ORDER)
    registry = build_model_registry(ARM_ORDER)
    logger.info("  Train=%d  Test=%d  dim=%d", train.n, test.n, feature_dim)
    logger.info("  Models: %s", ARM_ORDER)

    logger.info("  Empirical mean cost/req:")
    for m in ARM_ORDER:
        logger.info("    %s: $%.8f", m, float(np.mean(train.costs[m])))

    budget_targets = _compute_budget_targets(train)
    logger.info(
        "  Budget targets ($/req): %s",
        [f"${t:.6f}" for t in budget_targets],
    )

    all_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Condition A: Static cost_penalty sweep (no pacer)
    # ------------------------------------------------------------------
    logger.info("\n=== Static cost_penalty sweep ===")
    for cp in STATIC_COST_PENALTIES:
        seed_trials: List[TrialResult] = []
        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            trial = _run_trial(
                train, test, registry, feature_dim,
                condition=f"static_cp{cp:.2f}",
                cost_penalty=cp,
                seed=seed,
            )
            seed_trials.append(trial)

        mean_reward = float(np.mean([t.mean_reward for t in seed_trials]))
        mean_cost = float(np.mean([t.mean_cost for t in seed_trials]))
        mean_regret = float(np.mean([t.cumulative_regret for t in seed_trials]))
        se_reward = float(np.std([t.mean_reward for t in seed_trials], ddof=1) / np.sqrt(N_SEEDS))
        se_cost = float(np.std([t.mean_cost for t in seed_trials], ddof=1) / np.sqrt(N_SEEDS))

        avg_fracs = {}
        for m in ARM_ORDER:
            avg_fracs[m] = float(np.mean([t.model_fractions[m] for t in seed_trials]))

        row: Dict[str, Any] = {
            "method": "static",
            "cost_penalty": cp,
            "target_spend": None,
            "mean_reward": mean_reward,
            "se_reward": se_reward,
            "mean_cost": mean_cost,
            "se_cost": se_cost,
            "mean_regret": mean_regret,
            "model_fractions": avg_fracs,
        }
        all_results.append(row)
        logger.info(
            "  cp=%.2f  reward=%.4f±%.4f  cost=$%.6f±$%.6f  regret=%.1f",
            cp, mean_reward, se_reward, mean_cost, se_cost, mean_regret,
        )

    # ------------------------------------------------------------------
    # Condition B: BudgetPacer ADAPTIVE sweep
    # ------------------------------------------------------------------
    logger.info("\n=== BudgetPacer ADAPTIVE sweep ===")
    for target in budget_targets:
        pacer = BudgetPacer(
            target_avg_spend_usd=target,
            mode=PacingMode.ADAPTIVE,
            lr=PACER_LR,
            lambda_max=PACER_LAMBDA_MAX,
        )

        seed_trials = []
        record_first_seed = True
        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            trial = _run_trial(
                train, test, registry, feature_dim,
                condition=f"pacer_target{target:.8f}",
                budget_pacer=pacer,
                target_spend=target,
                seed=seed,
                record_per_step=record_first_seed,
            )
            seed_trials.append(trial)
            record_first_seed = False

        mean_reward = float(np.mean([t.mean_reward for t in seed_trials]))
        mean_cost = float(np.mean([t.mean_cost for t in seed_trials]))
        mean_regret = float(np.mean([t.cumulative_regret for t in seed_trials]))
        se_reward = float(np.std([t.mean_reward for t in seed_trials], ddof=1) / np.sqrt(N_SEEDS))
        se_cost = float(np.std([t.mean_cost for t in seed_trials], ddof=1) / np.sqrt(N_SEEDS))
        mean_util = float(np.mean([t.budget_utilization for t in seed_trials]))

        avg_fracs = {}
        for m in ARM_ORDER:
            avg_fracs[m] = float(np.mean([t.model_fractions[m] for t in seed_trials]))

        avg_lambda_quarts: Dict[str, float] = {}
        quart_keys = ["q25", "q50", "q75", "final", "last_100_mean"]
        for k in quart_keys:
            vals = [t.lambda_trajectory_quartiles.get(k, 0.0) for t in seed_trials]
            avg_lambda_quarts[k] = float(np.mean(vals))

        row = {
            "method": "pacer",
            "cost_penalty": 0.0,
            "target_spend": target,
            "mean_reward": mean_reward,
            "se_reward": se_reward,
            "mean_cost": mean_cost,
            "se_cost": se_cost,
            "mean_regret": mean_regret,
            "model_fractions": avg_fracs,
            "final_lambda": float(np.mean([t.final_lambda for t in seed_trials])),
            "trailing_100_cost": float(np.mean([t.trailing_100_cost for t in seed_trials])),
            "budget_utilization": mean_util,
            "lambda_quartiles": avg_lambda_quarts,
        }

        if seed_trials[0].per_step:
            row["per_step_seed0"] = seed_trials[0].per_step

        all_results.append(row)
        logger.info(
            "  target=$%.6f  reward=%.4f±%.4f  cost=$%.6f±$%.6f  "
            "regret=%.1f  λ_final=%.3f  util=%.2fx  trail100=$%.6f",
            target, mean_reward, se_reward, mean_cost, se_cost,
            mean_regret,
            row["final_lambda"],
            mean_util,
            row["trailing_100_cost"],
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "budget_pacing_results.json"
    output = {
        "experiment": "01_stationary_budget_pacing",
        "arm_order": ARM_ORDER,
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "warmup_hparams": WARMUP_HPARAMS,
        "pacer_lr": PACER_LR,
        "pacer_lambda_max": PACER_LAMBDA_MAX,
        "budget_targets": budget_targets,
        "static_cost_penalties": STATIC_COST_PENALTIES,
        "train_n": train.n,
        "test_n": test.n,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    logger.info("\nSaved results to %s", out_path)
    logger.info("Wall time: %.1fs", elapsed)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    hdr_w = 110
    print("\n" + "=" * hdr_w)
    print("STATIONARY BUDGET PACING — Pareto Summary")
    print("=" * hdr_w)
    print(
        f"  {'Method':<12s} {'Config':>16s}  {'Reward':>10s}  "
        f"{'Cost':>12s}  {'Regret':>10s}  {'Util':>7s}  {'λ_final':>8s}"
    )
    print(
        f"  {'-'*12}  {'-'*16}  {'-'*10}  "
        f"{'-'*12}  {'-'*10}  {'-'*7}  {'-'*8}"
    )

    for r in all_results:
        if r["method"] == "static":
            cfg = f"cp={r['cost_penalty']:.2f}"
            util_str = "—"
            lam_str = "—"
        else:
            cfg = f"t=${r['target_spend']:.6f}"
            util_str = f"{r['budget_utilization']:.2f}x"
            lam_str = f"{r['final_lambda']:.4f}"
        print(
            f"  {r['method']:<12s} {cfg:>16s}  {r['mean_reward']:10.4f}  "
            f"${r['mean_cost']:11.6f}  {r['mean_regret']:10.1f}  "
            f"{util_str:>7s}  {lam_str:>8s}"
        )
    print("=" * hdr_w)

    # Budget compliance summary for pacer conditions
    pacer_rows = [r for r in all_results if r["method"] == "pacer"]
    if pacer_rows:
        print(f"\n{'Budget Compliance Diagnostics':^{hdr_w}}")
        print("-" * hdr_w)
        print(
            f"  {'Target':>12s}  {'Actual':>12s}  {'Util':>7s}  "
            f"{'λ_q25':>7s}  {'λ_q50':>7s}  {'λ_q75':>7s}  "
            f"{'λ_last100':>9s}  {'Trail100':>12s}"
        )
        for r in pacer_rows:
            lq = r.get("lambda_quartiles", {})
            print(
                f"  ${r['target_spend']:11.6f}  ${r['mean_cost']:11.6f}  "
                f"{r['budget_utilization']:6.2f}x  "
                f"{lq.get('q25', 0):7.4f}  {lq.get('q50', 0):7.4f}  "
                f"{lq.get('q75', 0):7.4f}  {lq.get('last_100_mean', 0):9.4f}  "
                f"${r['trailing_100_cost']:11.6f}"
            )
        print("-" * hdr_w)


if __name__ == "__main__":
    main()
