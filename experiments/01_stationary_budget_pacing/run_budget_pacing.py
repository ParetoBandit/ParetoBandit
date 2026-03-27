#!/usr/bin/env python3
"""Experiment 01: Stationary Budget Pacing.

Evaluates BudgetPacer (ADAPTIVE mode) across a sweep of dollar-denominated
budget targets on the K=3 portfolio under stationary conditions.  Reference
points are three fixed single-model policies and a uniform-random router.

**Methodological notes:**

- *Cost feedback*:  The pacer receives actual per-request costs from the
  offline dataset (the same costs used for Pareto metrics), not the router's
  heuristic token-count estimate.  This mirrors production, where the billing
  system provides exact costs.
- *Forgetting factor*:  ``ff = 0.996`` — selected jointly with alpha and
  n_eff via the epsilon-constraint hyperparameter sweep.  Mild forgetting
  is applied consistently across all experiments for a fair comparison.
- *Sweep point selection*:  The pacer's budget targets are log-spaced
  between the cheapest and most expensive model's empirical mean costs
  (computed from the online/val split), giving the pacer sweep points
  that are optimally distributed across its operating range.
- *Pacer hyperparameters*:  ``lr`` and ``lambda_max`` are fixed to defaults
  chosen before inspecting results.  A sensitivity analysis over these
  is deferred to Experiment 04.

For each budget target × seed the script:
  1. Creates a router with K=3 warmup priors.
  2. Online-learns on the validation split (shuffled).
  3. Evaluates on the holdout/test split (shuffled).
  4. Tracks per-step reward, cost, model selection, lambda_t, and budget
     compliance diagnostics.

**Online evaluation protocol:**  The router continues learning during
the test phase (standard for bandit regret evaluation).  Each prompt
is routed *before* its reward is observed, so there is no look-ahead;
the bandit's LinUCB parameters and the pacer's dual variable update
after every routing decision, including test-phase decisions.  This
matches the deployment scenario where the router never stops learning.

Produces a JSON results file consumed by ``generate_figure.py``.

Usage:
    python experiments/01_stationary_budget_pacing/run_budget_pacing.py
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
    DEFAULT_PACER_LAMBDA_MAX,
    DEFAULT_PACER_LR,
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_WARMUP_PRIORS_PATH,
    N_SEEDS,
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

ARM_ORDER: List[str] = K3_ARM_ORDER
SEED_OFFSET = 3000
RESULTS_DIR = Path(__file__).parent / "results"

WARMUP_HPARAMS: Dict[str, Any] = BEST_K3_HPARAMS

PACER_LR = DEFAULT_PACER_LR
PACER_LAMBDA_MAX = DEFAULT_PACER_LAMBDA_MAX


# ======================================================================
# Dataclasses for results
# ======================================================================


@dataclass
class StepRecord:
    """Per-step metrics recorded during online evaluation."""

    model: str
    reward: float
    cost: float
    unconstrained_oracle: float
    lambda_t: float


@dataclass
class TrialResult:
    """Aggregate metrics for one (condition, seed) trial."""

    condition: str
    seed: int
    mean_reward: float
    mean_cost: float
    cumulative_quality_gap: float
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
        cost_penalty=cost_penalty,
        forgetting_factor=WARMUP_HPARAMS["forgetting_factor"],
        budget_pacer=budget_pacer,
    )


def _run_trial(
    online: SplitData,
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
    """Run one online-learning -> test trial and return aggregate metrics.

    The pacer receives **actual per-request costs** from the offline dataset
    (via ``log.cost_usd`` override) rather than the router's heuristic token
    estimate.  This ensures the dual-variable update reflects the true cost
    scale, consistent with production where billing provides exact costs.

    Args:
        online: Online-learning split (validation set, disjoint from warmup).
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

    if budget_pacer is not None:
        budget_pacer.reset()

    router = _create_router(
        registry, feature_dim,
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
    )

    # --- Online-learning phase (no metrics) ---
    online_order = rng.permutation(online.n)
    for i in online_order:
        model, log = router.route(online.embeddings[i])
        reward = float(online.rewards[model][i])
        log.cost_usd = float(online.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # --- Test phase (metrics) ---
    test_order = rng.permutation(test.n)
    steps: List[StepRecord] = []
    model_counts: Dict[str, int] = {m: 0 for m in ARM_ORDER}

    for i in test_order:
        model, log = router.route(test.embeddings[i])
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        unconstrained_oracle = max(float(test.rewards[a][i]) for a in ARM_ORDER)

        lam = budget_pacer.lambda_t if budget_pacer is not None else 0.0

        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

        steps.append(StepRecord(
            model=model, reward=reward, cost=cost,
            unconstrained_oracle=unconstrained_oracle, lambda_t=lam,
        ))
        model_counts[model] += 1

    rewards = np.array([s.reward for s in steps])
    costs = np.array([s.cost for s in steps])
    oracles = np.array([s.unconstrained_oracle for s in steps])
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
                "unconstrained_oracle": s.unconstrained_oracle,
                "lambda_t": s.lambda_t,
            }
            for s in steps
        ]

    return TrialResult(
        condition=condition,
        seed=seed,
        mean_reward=float(np.mean(rewards)),
        mean_cost=float(np.mean(costs)),
        cumulative_quality_gap=float(np.sum(oracles - rewards)),
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


def _compute_budget_targets(online: SplitData) -> List[float]:
    """Compute 7 log-spaced targets from empirical per-model mean costs.

    Uses the actual dataset cost distribution rather than synthetic
    token-count estimates, ensuring targets are on the same scale as the
    costs the pacer will observe during the trial.

    Args:
        online: Online-learning split with per-model costs.

    Returns:
        Seven log-spaced budget targets spanning the cheapest to most
        expensive model's empirical mean cost per request.
    """
    per_model_means = []
    for m in ARM_ORDER:
        per_model_means.append(float(np.mean(online.costs[m])))

    lo = min(per_model_means)
    hi = max(per_model_means)
    return list(np.geomspace(lo, hi, num=7))


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()

    logger.info("Loading K=3 data ...")
    # PCA projection is pre-fitted on ~46K LMSYS Arena prompts (train-split
    # excluded) and frozen; only .transform() is called during evaluation.
    fs = FeatureService()
    feature_dim = fs.dimension
    # Online learning uses val (unseen by warmup priors, which were
    # trained on train.jsonl).  Evaluation uses the held-out test split.
    online = load_split(VAL_DATA_PATH, fs, ARM_ORDER)
    test = load_split(HOLDOUT_DATA_PATH, fs, ARM_ORDER)
    registry = build_model_registry(ARM_ORDER)
    logger.info("  Online=%d  Test=%d  dim=%d", online.n, test.n, feature_dim)
    logger.info("  Models: %s", ARM_ORDER)

    logger.info("  Empirical mean cost/req:")
    for m in ARM_ORDER:
        logger.info("    %s: $%.8f", m, float(np.mean(online.costs[m])))

    budget_targets = _compute_budget_targets(online)
    logger.info(
        "  Budget targets ($/req): %s",
        [f"${t:.6f}" for t in budget_targets],
    )

    # Per-prompt oracle: always select the highest-quality model per prompt.
    oracle_rewards_arr = np.array([
        max(float(test.rewards[a][i]) for a in ARM_ORDER)
        for i in range(test.n)
    ])
    oracle_best_arms = [
        max(ARM_ORDER, key=lambda a, idx=i: float(test.rewards[a][idx]))
        for i in range(test.n)
    ]
    oracle_costs_arr = np.array([
        float(test.costs[oracle_best_arms[i]][i]) for i in range(test.n)
    ])
    oracle_mean_reward = float(np.mean(oracle_rewards_arr))
    oracle_mean_cost = float(np.mean(oracle_costs_arr))
    logger.info(
        "  Per-prompt oracle: reward=%.4f  cost=$%.6f",
        oracle_mean_reward, oracle_mean_cost,
    )

    all_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Baseline: Fixed single-model policies (always pick one model)
    # ------------------------------------------------------------------
    logger.info("\n=== Fixed single-model baselines ===")
    for m in ARM_ORDER:
        m_reward = float(np.mean(test.rewards[m]))
        m_cost = float(np.mean(test.costs[m]))
        per_seed_r = [float(np.mean(test.rewards[m])) for _ in range(N_SEEDS)]
        per_seed_c = [float(np.mean(test.costs[m])) for _ in range(N_SEEDS)]
        row: Dict[str, Any] = {
            "method": "fixed_model",
            "model_id": m,
            "cost_penalty": 0.0,
            "target_spend": None,
            "mean_reward": m_reward,
            "se_reward": 0.0,
            "mean_cost": m_cost,
            "se_cost": 0.0,
            "mean_quality_gap": float(np.mean(oracle_rewards_arr)) - m_reward,
            "model_fractions": {a: (1.0 if a == m else 0.0) for a in ARM_ORDER},
            "per_seed_rewards": per_seed_r,
            "per_seed_costs": per_seed_c,
        }
        all_results.append(row)
        logger.info(
            "  %-20s  reward=%.4f  cost=$%.6f",
            m.split("/")[-1], m_reward, m_cost,
        )

    # ------------------------------------------------------------------
    # Baseline: Random router (uniform 1/K, no learning)
    # ------------------------------------------------------------------
    logger.info("\n=== Random router (uniform 1/K) ===")
    per_seed_rand_rewards: List[float] = []
    per_seed_rand_costs: List[float] = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(SEED_OFFSET + s)
        choices = rng.choice(ARM_ORDER, size=test.n)
        rewards_s = np.array([
            float(test.rewards[choices[i]][i]) for i in range(test.n)
        ])
        costs_s = np.array([
            float(test.costs[choices[i]][i]) for i in range(test.n)
        ])
        per_seed_rand_rewards.append(float(np.mean(rewards_s)))
        per_seed_rand_costs.append(float(np.mean(costs_s)))

    rand_reward = float(np.mean(per_seed_rand_rewards))
    rand_cost = float(np.mean(per_seed_rand_costs))
    se_rand_r = float(np.std(per_seed_rand_rewards, ddof=1) / np.sqrt(N_SEEDS))
    se_rand_c = float(np.std(per_seed_rand_costs, ddof=1) / np.sqrt(N_SEEDS))
    rand_row: Dict[str, Any] = {
        "method": "random",
        "cost_penalty": 0.0,
        "target_spend": None,
        "mean_reward": rand_reward,
        "se_reward": se_rand_r,
        "mean_cost": rand_cost,
        "se_cost": se_rand_c,
        "mean_quality_gap": float(np.mean(oracle_rewards_arr)) - rand_reward,
        "model_fractions": {a: 1.0 / len(ARM_ORDER) for a in ARM_ORDER},
        "per_seed_rewards": [float(v) for v in per_seed_rand_rewards],
        "per_seed_costs": [float(v) for v in per_seed_rand_costs],
    }
    all_results.append(rand_row)
    logger.info(
        "  reward=%.4f±%.4f  cost=$%.6f±$%.6f",
        rand_reward, se_rand_r, rand_cost, se_rand_c,
    )

    # ------------------------------------------------------------------
    # BudgetPacer ADAPTIVE sweep
    # ------------------------------------------------------------------
    logger.info("\n=== BudgetPacer ADAPTIVE sweep ===")
    for target in budget_targets:
        seed_trials = []
        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            pacer = BudgetPacer(
                target_avg_spend_usd=target,
                mode=PacingMode.ADAPTIVE,
                lr=PACER_LR,
                lambda_max=PACER_LAMBDA_MAX,
            )
            trial = _run_trial(
                online, test, registry, feature_dim,
                condition=f"pacer_target{target:.8f}",
                budget_pacer=pacer,
                target_spend=target,
                seed=seed,
                record_per_step=(s == 0),
            )
            seed_trials.append(trial)

        per_seed_rewards = [t.mean_reward for t in seed_trials]
        per_seed_costs = [t.mean_cost for t in seed_trials]

        mean_reward = float(np.mean(per_seed_rewards))
        mean_cost = float(np.mean(per_seed_costs))
        mean_quality_gap = float(np.mean([t.cumulative_quality_gap for t in seed_trials]))
        se_reward = float(np.std(per_seed_rewards, ddof=1) / np.sqrt(N_SEEDS))
        se_cost = float(np.std(per_seed_costs, ddof=1) / np.sqrt(N_SEEDS))
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
            "mean_quality_gap": mean_quality_gap,
            "model_fractions": avg_fracs,
            "per_seed_rewards": [float(v) for v in per_seed_rewards],
            "per_seed_costs": [float(v) for v in per_seed_costs],
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
            "qgap=%.1f  λ_final=%.3f  util=%.2fx  trail100=$%.6f",
            target, mean_reward, se_reward, mean_cost, se_cost,
            mean_quality_gap,
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
        "online_n": online.n,
        "test_n": test.n,
        "oracle_mean_reward": oracle_mean_reward,
        "oracle_mean_cost": oracle_mean_cost,
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
        f"{'Cost':>12s}  {'QGap':>10s}  {'Util':>7s}  {'λ_final':>8s}"
    )
    print(
        f"  {'-'*12}  {'-'*16}  {'-'*10}  "
        f"{'-'*12}  {'-'*10}  {'-'*7}  {'-'*8}"
    )

    for r in all_results:
        method = r["method"]
        if method == "pacer":
            cfg = f"t=${r['target_spend']:.6f}"
            util_str = f"{r['budget_utilization']:.2f}x"
            lam_str = f"{r['final_lambda']:.4f}"
        elif method == "fixed_model":
            cfg = r.get("model_id", "").split("/")[-1][:16]
            util_str = "—"
            lam_str = "—"
        elif method == "random":
            cfg = "uniform 1/K"
            util_str = "—"
            lam_str = "—"
        else:
            cfg = "?"
            util_str = "—"
            lam_str = "—"
        print(
            f"  {method:<12s} {cfg:>16s}  {r['mean_reward']:10.4f}  "
            f"${r['mean_cost']:11.6f}  {r['mean_quality_gap']:10.1f}  "
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
