#!/usr/bin/env python3
"""Main paper: T_adapt-Constrained Hyperparameter Selection (Experiment 05).

Produces the results for Section~\\ref{sec:hparam_sweep} of the main
paper, which presents the adaptation-horizon-constrained tuning
framework and key insights.

Constrained 2D grid search over exploration parameter (alpha) and
forgetting factor (gamma) for ParetoBandit.  The prior strength
(n_eff) is **not** an independent hyperparameter — it is derived
from gamma via the adaptation horizon formula:

    n_eff = (gamma^{-T_adapt} - 1) / (1 - gamma)

This coupling ensures the bandit can override its prior within
T_adapt queries after a distributional shift.  T_adapt is anchored
to the catastrophic-failure phase length (N_phase2 of the validation
split), ensuring the router can fully adapt within the experimental
measurement window.

**Full-information approximation.**  The coupling formula assumes the
arm receives one observation per timestep (full-information feedback).
In a K-armed bandit, arm *a* is selected with probability p_a, so the
effective adaptation time scales as T_adapt / p_a.  This approximation
is benign for our protocol because: (a) the catastrophic-failure arm
(Mistral) is the dominant arm before failure, so p_a >> 1/K during
the critical detection period, and (b) the final config selection is
purely empirical (Pareto knee-point on observed Phase-2 reward), so
any formula imprecision is corrected by the data.

We fix PCA dimensionality to d=25 (~28.5% cumulative variance).

Data protocol
-------------
Strict disjoint splits to avoid prior/online-stream overlap:

- **train.jsonl** — used *only* for offline prior generation and to
  anchor budget-target ranges.  Never seen during online learning.
- **val.jsonl** — split into two disjoint portions at load time
  (1/3 burn-in, 2/3 eval):

  - ``val_burnin`` (first 1/3, ~595 prompts): online warm-up — the
    router learns but no metrics are recorded.
  - ``val_eval`` (remaining 2/3, ~1190 prompts): selection surface —
    metrics are recorded here.  No prompt appears in both portions.

  The asymmetric split preserves more data for evaluation (giving
  stable metric estimates and ~595-step non-stationary phases) while
  still providing adequate burn-in on top of the offline priors.

- **test.jsonl** — held-out evaluation (burn-in on full val, eval on
  test; never used for selection).

Scoring protocol
----------------
Each (alpha, gamma) config — with n_eff derived from gamma — is scored
on two objectives:

1. **Budget-Paced Pareto AUC** (primary) — sweep budget targets on the
   val split with ``BudgetPacer`` active, build per-seed Pareto
   frontiers (with fixed-model endpoints), and average AUC across
   seeds.  Higher = better budget-aware routing.

2. **Catastrophic-failure Phase-2 reward** (secondary) — two-phase
   simulation on the val split with BudgetPacer active.
   Phase 1 (first half): normal rewards.  Phase 2 (second half):
   one model's reward drops to near-zero and its cost drops to $0
   (simulating catastrophic failure).  Mean Phase-2 reward is the
   metric.  Higher = faster detection and reallocation away from the
   failed arm.  Averaged over budget targets.

   **Single-arm tuning rationale.**  Only Mistral failure is used for
   the tuning objective (not an all-arms average) for three reasons:
   (i) exponential discounting is arm-agnostic — ``A *= gamma^dt``
   applies identically to every arm, so detection speed is governed
   by gamma regardless of which arm fails; (ii) Mistral is the most
   operationally stressful failure (dominant arm across the widest
   budget range), making this a conservative worst-case selection;
   (iii) an all-arms average would 3x the compute and inject the
   noisy Llama signal (cross-arm std=0.14 vs 0.02 for Mistral) into
   the selection, potentially destabilising the Pareto frontier.
   Cross-arm validation (step 5b on val, step 6b on held-out test)
   empirically confirms generalisation to all K arms.

Selection uses the **Pareto knee-point method** from multi-objective
optimization:

1. Evaluate every (alpha, gamma) config on both objectives.
2. Build the Pareto frontier of non-dominated (AUC, P2_reward) configs.
3. Identify the **knee point** — the config on the frontier with
   maximum perpendicular distance from the line connecting the two
   extreme endpoints.  This is the inflection where improving one
   metric starts requiring disproportionate sacrifice of the other.

The knee-point requires no weight specification and is parameter-free.
The winning config is validated against catastrophic failure of all K
arms (not just the tuning arm) to confirm the forgetting mechanism
generalises, then re-evaluated on the held-out test split.

Usage::

    python experiments/05_hparam_optimization/run_hparam_sweep.py
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.config import (
    DEFAULT_PACER_LAMBDA_MAX,
    DEFAULT_PACER_LR,
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_BUDGET_TARGETS,
    K3_FAILURE_ARM,
    K3_FAILURE_REWARD,
    K3_WARMUP_PRIORS_PATH,
    N_SEEDS,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from utils.pareto import pareto_auc
from utils.simulation import SplitData, build_model_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("pareto_bandit.router", "pareto_bandit.feature_service", "pareto_bandit.policy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Sweep Grid
# ======================================================================

ALPHA_VALUES: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
BUDGET_TARGET_COUNT: int = 7
PACER_LR: float = DEFAULT_PACER_LR
PACER_LAMBDA_MAX: float = DEFAULT_PACER_LAMBDA_MAX
VARIANTS: List[str] = ["paretobandit", "tabula_rasa"]

PCA_DIM: int = 25
GAMMA_VALUES: List[float] = [0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1.0]
SEED_OFFSET_VAL: int = 0
SEED_OFFSET_FAILURE_VAL: int = 2000
SEED_OFFSET_TEST: int = 1000

T_ADAPT: int = 500
"""Adaptation horizon — number of online queries for online evidence to
match the discounted prior.  Anchored to the catastrophic-failure phase
length (N_phase2 ≈ 595), ensuring the bandit can override its prior
within the measurement window.  Practitioners should set T_adapt based
on the shortest acceptable reaction time in their deployment."""

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = K3_ARM_SHORT
RESULTS_DIR = Path(__file__).parent / "results"

FAILURE_ARMS: List[str] = [K3_FAILURE_ARM]
"""Models whose failure is used for the tuning objective.

Only Mistral is used (not an all-arms average) because:
(i) exponential discounting is arm-agnostic — detection speed is governed
by gamma regardless of which arm fails, so the optimal (alpha, gamma) is
approximately arm-independent; (ii) Mistral is the most operationally
stressful failure (dominant arm across the broadest range of budget targets,
yielding the lowest Phase-2 reward in cross-arm validation), making this
a conservative worst-case selection; (iii) including noisier arms (e.g.
Llama, whose cross-arm std=0.14 vs 0.02 for Mistral) would inject
measurement noise that could destabilise the Pareto frontier.

Cross-arm validation (step 5b on val, step 6b on held-out test) evaluates
all K arms post-hoc to confirm generalisation."""

FAILURE_REWARD: float = K3_FAILURE_REWARD
FAILURE_BUDGET_TARGETS: List[float] = K3_BUDGET_TARGETS


# ======================================================================
# Config generation
# ======================================================================


def derive_n_eff(gamma: float, t_adapt: int) -> float:
    """Compute n_eff from gamma and adaptation horizon T_adapt.

    Uses the coupling formula derived from discounted LinUCB sufficient
    statistics.  At time T_adapt the discounted online evidence equals
    the discounted prior, i.e. the prior has been effectively overridden:

        n_eff = (gamma^{-T_adapt} - 1) / (1 - gamma)

    For gamma = 1.0 (no forgetting), the online sample count grows
    linearly, so n_eff = T_adapt by L'Hôpital's rule.

    **Full-information approximation.**  This formula assumes the arm
    receives one observation per timestep (full information).  In a
    K-armed bandit, arm *a* is selected with probability p_a per round,
    so the correct parity condition gives
    ``n_eff = p_a * (gamma^{-T} - 1) / (1 - gamma)``.  The formula
    above corresponds to p_a = 1.  For the catastrophic-failure tuning
    objective this is approximately correct: the failed arm (Mistral)
    is the dominant arm before failure (p_a >> 1/K), and the final
    selection is empirical (Pareto knee-point on observed Phase-2
    reward), so any formula imprecision is corrected by the data.

    Args:
        gamma: Forgetting factor in (0, 1].
        t_adapt: Target adaptation horizon (number of queries).

    Returns:
        Derived prior effective sample size.
    """
    if gamma <= 0.0 or gamma > 1.0:
        raise ValueError(
            f"gamma must be in (0, 1], got {gamma}"
        )
    if gamma == 1.0:
        return float(t_adapt)
    return (gamma ** (-t_adapt) - 1.0) / (1.0 - gamma)


def _build_configs() -> List[Dict[str, Any]]:
    """Generate all hyperparameter configurations for active variants.

    For ParetoBandit: alpha x gamma grid, with n_eff derived from gamma
    via :func:`derive_n_eff` and T_ADAPT.
    For Tabula Rasa: alpha x gamma (n_eff=1.0, no priors).

    Only variants listed in :data:`VARIANTS` are generated.
    """
    configs: List[Dict[str, Any]] = []
    if "paretobandit" in VARIANTS:
        for alpha, gamma in itertools.product(ALPHA_VALUES, GAMMA_VALUES):
            n_eff = derive_n_eff(gamma, T_ADAPT)
            configs.append({
                "variant": "paretobandit",
                "alpha": alpha,
                "n_eff": round(n_eff, 1),
                "gamma": gamma,
            })
    if "tabula_rasa" in VARIANTS:
        for alpha, gamma in itertools.product(ALPHA_VALUES, GAMMA_VALUES):
            configs.append({
                "variant": "tabula_rasa",
                "alpha": alpha,
                "n_eff": 1.0,
                "gamma": gamma,
            })
    return configs


# ======================================================================
# Data Loading
# ======================================================================


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file returning a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def parse_and_embed(
    records: List[Dict[str, Any]],
    fs: FeatureService,
) -> SplitData:
    """Extract prompts, rewards, costs, and embed via FeatureService.

    Args:
        records: JSONL records with ``prompt`` and ``arms`` fields.
        fs: Feature service configured with the target PCA.

    Returns:
        Fully loaded and embedded split data.
    """
    prompts = [r["prompt"] for r in records]
    rewards: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    costs: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    for r in records:
        for arm_id in ARM_ORDER:
            info = r["arms"][arm_id]
            rewards[arm_id].append(info["reward"])
            costs[arm_id].append(info["cost"])

    embeddings = fs.extract_features_batch(prompts)

    return SplitData(
        prompts=prompts,
        rewards={a: np.array(v) for a, v in rewards.items()},
        costs={a: np.array(v) for a, v in costs.items()},
        embeddings=embeddings,
    )


def split_data(
    data: SplitData,
    arm_order: List[str],
    burnin_frac: float = 1 / 3,
) -> Tuple[SplitData, SplitData]:
    """Split data into burn-in and eval portions.

    Used to create ``val_burnin`` (first ``burnin_frac``) and
    ``val_eval`` (remainder) so that the burn-in phase never sees
    evaluation prompts.  The default 1/3 burn-in preserves 2/3 of the
    data for evaluation, giving more statistical power for metric
    estimation while still providing adequate online warm-up on top of
    the offline priors.

    Args:
        data: Loaded split data as produced by :func:`parse_and_embed`.
        arm_order: Model identifiers for slicing reward/cost arrays.
        burnin_frac: Fraction of data allocated to burn-in (default 1/3).

    Returns:
        ``(burnin_portion, eval_portion)`` — two :class:`SplitData`
        instances, disjoint by construction.
    """
    split_idx = int(data.n * burnin_frac)
    first = SplitData(
        prompts=data.prompts[:split_idx],
        rewards={a: data.rewards[a][:split_idx] for a in arm_order},
        costs={a: data.costs[a][:split_idx] for a in arm_order},
        embeddings=data.embeddings[:split_idx],
    )
    second = SplitData(
        prompts=data.prompts[split_idx:],
        rewards={a: data.rewards[a][split_idx:] for a in arm_order},
        costs={a: data.costs[a][split_idx:] for a in arm_order},
        embeddings=data.embeddings[split_idx:],
    )
    return first, second


# ======================================================================
# Simulation
# ======================================================================


# ======================================================================
# Budget-Paced Simulation
# ======================================================================


def _simulate_budget_paced(
    burnin_data: SplitData,
    eval_data: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup_path: Optional[str],
    alpha: float,
    n_eff: float,
    gamma: float,
    budget_target: float,
    seed: int,
) -> Tuple[float, float]:
    """Run burn-in then budget-paced eval, return (mean_reward, mean_cost).

    The router and BudgetPacer first burn in on ``burnin_data`` (disjoint
    from both priors and eval), then record metrics on ``eval_data``.

    Args:
        burnin_data: Burn-in split (bandit + pacer learn, results not
            recorded).
        eval_data: Evaluation split (per-prompt metrics recorded).  Must
            be disjoint from ``burnin_data`` to avoid look-ahead bias.
        registry: Filtered model registry.
        feature_dim: Context vector dimensionality.
        warmup_path: Path to warmup priors, or ``None`` for tabula rasa.
        alpha: LinUCB exploration coefficient.
        n_eff: Prior effective sample size (ignored when warmup_path is None).
        gamma: Forgetting factor (1.0 = no forgetting).
        budget_target: Target average spend per request in USD.
        seed: Random seed for data shuffling.

    Returns:
        ``(mean_reward, mean_cost)`` on the eval split.
    """
    rng = np.random.default_rng(seed)

    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    use_warmup = warmup_path is not None
    pacer = BudgetPacer(
        target_avg_spend_usd=budget_target,
        mode=PacingMode.ADAPTIVE,
        lr=PACER_LR,
        lambda_max=PACER_LAMBDA_MAX,
    )

    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if use_warmup else "none",
        warmup_path=warmup_path if use_warmup else None,
        prior_n_effective=n_eff if use_warmup else 1.0,
        alpha=alpha,
        cost_penalty=0.0,
        forgetting_factor=gamma,
        budget_pacer=pacer,
    )

    burnin_order = rng.permutation(burnin_data.n)
    for i in burnin_order:
        model, log = router.route(burnin_data.embeddings[i])
        reward = float(burnin_data.rewards[model][i])
        log.cost_usd = float(burnin_data.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    eval_order = rng.permutation(eval_data.n)
    eval_rewards: List[float] = []
    eval_costs: List[float] = []
    for i in eval_order:
        model, log = router.route(eval_data.embeddings[i])
        reward = float(eval_data.rewards[model][i])
        cost = float(eval_data.costs[model][i])
        eval_rewards.append(reward)
        eval_costs.append(cost)
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

    return float(np.mean(eval_rewards)), float(np.mean(eval_costs))


def compute_budget_paced_pareto_auc(
    burnin_data: SplitData,
    eval_data: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    budget_targets: List[float],
    *,
    warmup_path: Optional[str],
    alpha: float,
    n_eff: float,
    gamma: float,
    n_seeds: int,
    seed_offset: int,
) -> Tuple[float, float, List[Dict[str, Any]], List[float]]:
    """Sweep budget targets with BudgetPacer and compute per-seed Pareto AUC.

    For each seed independently:
      1. Burn in on ``burnin_data`` (disjoint from both priors and eval),
         then evaluate for every budget target with BudgetPacer.
      2. Build the Pareto frontier from the resulting (mean_cost,
         mean_reward) points.
      3. Compute the seed's Pareto AUC.

    Cost range is anchored to fixed-model extremes (same as the
    unbounded Pareto AUC for comparability).

    Args:
        burnin_data: Burn-in split (bandit + pacer learn, results not
            recorded).
        eval_data: Evaluation split (disjoint from ``burnin_data``).
        registry: Filtered model registry.
        feature_dim: Context vector dimensionality.
        budget_targets: Per-request USD budget targets to sweep.
        warmup_path: Path to warmup priors, or ``None`` for tabula rasa.
        alpha: LinUCB exploration coefficient.
        n_eff: Prior effective sample size.
        gamma: Forgetting factor (1.0 = no forgetting).
        n_seeds: Number of independent random seeds.
        seed_offset: Base offset added to each seed index.

    Returns:
        ``(mean_auc, std_auc, sweep_points, per_seed_aucs)`` where
        ``per_seed_aucs`` is a list of per-seed AUC values for
        bootstrap stability analysis.
    """
    fixed_costs = [float(eval_data.costs[a].mean()) for a in ARM_ORDER]
    fixed_rewards = [float(eval_data.rewards[a].mean()) for a in ARM_ORDER]
    cost_lo = min(fixed_costs)
    cost_hi = max(fixed_costs)

    per_seed_auc: List[float] = []
    bt_reward_accum: Dict[float, List[float]] = {bt: [] for bt in budget_targets}
    bt_cost_accum: Dict[float, List[float]] = {bt: [] for bt in budget_targets}

    for s in range(n_seeds):
        seed = seed_offset + s
        seed_costs: List[float] = []
        seed_rewards: List[float] = []

        for bt in budget_targets:
            mr, mc = _simulate_budget_paced(
                burnin_data, eval_data, registry, feature_dim,
                warmup_path=warmup_path,
                alpha=alpha,
                n_eff=n_eff,
                gamma=gamma,
                budget_target=bt,
                seed=seed,
            )
            seed_costs.append(mc)
            seed_rewards.append(mr)
            bt_reward_accum[bt].append(mr)
            bt_cost_accum[bt].append(mc)

        all_c = seed_costs
        all_r = seed_rewards
        per_seed_auc.append(pareto_auc(all_c, all_r, cost_lo, cost_hi))

    mean_auc = float(np.mean(per_seed_auc))
    std_auc = float(np.std(per_seed_auc, ddof=1)) if n_seeds > 1 else 0.0

    sweep_points: List[Dict[str, Any]] = []
    for bt in budget_targets:
        sweep_points.append({
            "budget_target": bt,
            "mean_reward": round(float(np.mean(bt_reward_accum[bt])), 6),
            "mean_cost": round(float(np.mean(bt_cost_accum[bt])), 6),
            "std_reward": round(float(np.std(bt_reward_accum[bt], ddof=1)), 6),
            "std_cost": round(float(np.std(bt_cost_accum[bt], ddof=1)), 6),
        })

    return mean_auc, std_auc, sweep_points, per_seed_auc


# ======================================================================
# Catastrophic-Failure Simulation
# ======================================================================


def _simulate_catastrophic_failure(
    burnin_data: SplitData,
    val_data: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup_path: Optional[str],
    alpha: float,
    n_eff: float,
    gamma: float,
    seed: int,
    failure_arm: str,
    failure_reward: float,
    budget_target: float,
) -> Tuple[float, float]:
    """Burn-in then two-phase simulation with catastrophic model failure.

    Phase 1 (first half of ``val_data``): normal rewards for all arms.
    Phase 2 (second half): ``failure_arm`` returns ``failure_reward``
    regardless of the prompt, and its cost drops to zero (the failed
    model returns garbage but doesn't charge).  The bandit must detect
    the quality collapse and reallocate traffic.

    BudgetPacer is active throughout, matching the deployment
    conditions tested in Experiment 03.

    Args:
        burnin_data: Burn-in split (bandit learns, results not recorded).
        val_data: Evaluation split (split into Phase 1 + Phase 2;
            disjoint from ``burnin_data``).
        registry: Filtered model registry.
        feature_dim: Context vector dimensionality.
        warmup_path: Path to warmup priors, or ``None`` for tabula rasa.
        alpha: LinUCB exploration coefficient.
        n_eff: Prior effective sample size.
        gamma: Forgetting factor (1.0 = no forgetting).
        seed: Random seed.
        failure_arm: Model ID whose reward collapses in Phase 2.
        failure_reward: Degraded reward value during failure.
        budget_target: Per-request USD target for the BudgetPacer.

    Returns:
        ``(phase1_mean_reward, phase2_mean_reward)`` — mean reward per
        phase.  Higher Phase 2 reward indicates faster failure detection
        and reallocation.
    """
    rng = np.random.default_rng(seed)

    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    use_warmup = warmup_path is not None

    pacer = BudgetPacer(
        target_avg_spend_usd=budget_target,
        mode=PacingMode.ADAPTIVE,
        lr=PACER_LR,
        lambda_max=PACER_LAMBDA_MAX,
    )

    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if use_warmup else "none",
        warmup_path=warmup_path if use_warmup else None,
        prior_n_effective=n_eff if use_warmup else 1.0,
        alpha=alpha,
        cost_penalty=0.0,
        forgetting_factor=gamma,
        budget_pacer=pacer,
    )

    burnin_order = rng.permutation(burnin_data.n)
    for i in burnin_order:
        model, log = router.route(burnin_data.embeddings[i])
        reward = float(burnin_data.rewards[model][i])
        log.cost_usd = float(burnin_data.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    n_val = val_data.n
    val_order = rng.permutation(n_val)
    mid = n_val // 2

    phase1_rewards: List[float] = []
    for idx in val_order[:mid]:
        model, log = router.route(val_data.embeddings[idx])
        reward = float(val_data.rewards[model][idx])
        log.cost_usd = float(val_data.costs[model][idx])
        router.process_feedback(log.request_id, reward=reward)
        phase1_rewards.append(reward)

    phase2_rewards: List[float] = []
    for idx in val_order[mid:]:
        model, log = router.route(val_data.embeddings[idx])
        if model == failure_arm:
            reward = failure_reward
            log.cost_usd = 0.0
        else:
            reward = float(val_data.rewards[model][idx])
            log.cost_usd = float(val_data.costs[model][idx])
        router.process_feedback(log.request_id, reward=reward)
        phase2_rewards.append(reward)

    return float(np.mean(phase1_rewards)), float(np.mean(phase2_rewards))


def compute_failure_resilience(
    burnin_data: SplitData,
    val_data: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup_path: Optional[str],
    alpha: float,
    n_eff: float,
    gamma: float,
    n_seeds: int,
    seed_offset: int,
    failure_arms: List[str],
    failure_reward: float,
    budget_targets: List[float],
) -> Tuple[float, float, float, List[float]]:
    """Phase 2 mean reward averaged over failure scenarios, budgets, and seeds.

    For each ``(failure_arm, budget_target, seed)`` triple, runs a
    two-phase catastrophic-failure simulation and records Phase 2 mean
    reward.  Higher values indicate the bandit detects the failure
    faster and reallocates traffic more effectively.

    Args:
        burnin_data: Burn-in split (bandit learns, results not recorded).
        val_data: Evaluation split (disjoint from ``burnin_data``).
        registry: Filtered model registry.
        feature_dim: Context vector dimensionality.
        warmup_path: Path to warmup priors, or ``None`` for tabula rasa.
        alpha: LinUCB exploration coefficient.
        n_eff: Prior effective sample size.
        gamma: Forgetting factor (1.0 = no forgetting).
        n_seeds: Number of independent random seeds.
        seed_offset: Base offset added to each seed index.
        failure_arms: Model IDs to simulate failure for (one scenario each).
        failure_reward: Degraded reward during failure.
        budget_targets: Per-request USD targets to sweep.

    Returns:
        ``(mean_phase2_reward, std_phase2_reward, mean_phase1_reward,
        per_seed_p2_means)`` where the first two drive selection, the
        third is diagnostic, and the fourth provides per-seed Phase-2
        means for bootstrap stability analysis.
    """
    p1_rewards: List[float] = []
    p2_rewards: List[float] = []
    per_seed_p2: List[List[float]] = [[] for _ in range(n_seeds)]
    for fail_arm in failure_arms:
        for bt in budget_targets:
            for s in range(n_seeds):
                p1, p2 = _simulate_catastrophic_failure(
                    burnin_data, val_data, registry, feature_dim,
                    warmup_path=warmup_path,
                    alpha=alpha,
                    n_eff=n_eff,
                    gamma=gamma,
                    seed=seed_offset + s,
                    failure_arm=fail_arm,
                    failure_reward=failure_reward,
                    budget_target=bt,
                )
                p1_rewards.append(p1)
                p2_rewards.append(p2)
                per_seed_p2[s].append(p2)

    per_seed_p2_means = [float(np.mean(vals)) for vals in per_seed_p2]

    return (
        float(np.mean(p2_rewards)),
        float(np.std(p2_rewards, ddof=1)) if len(p2_rewards) > 1 else 0.0,
        float(np.mean(p1_rewards)),
        per_seed_p2_means,
    )


# ======================================================================
# Pareto Frontier & Knee-Point Selection
# ======================================================================


def cfg_matches(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True when two result dicts share the same hyperparameter config."""
    return (
        a["alpha"] == b["alpha"]
        and a["n_eff"] == b["n_eff"]
        and a["gamma"] == b["gamma"]
    )


def find_pareto_frontier(
    aucs: np.ndarray,
    p2s: np.ndarray,
) -> List[int]:
    """Return indices of Pareto-optimal configs (both objectives maximized).

    Dominance is determined from point estimates (seed-averaged means)
    without formal significance testing.  With finite seeds, adjacent
    frontier configs whose AUC gap is smaller than ~2x the standard
    error of the mean may have overlapping confidence intervals, making
    their dominance relationship statistically ambiguous.  The frontier
    should be interpreted as an approximate structural property of the
    mean outcome surface rather than a precise boundary.
    """
    n = len(aucs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if (aucs[j] >= aucs[i] and p2s[j] >= p2s[i]
                    and (aucs[j] > aucs[i] or p2s[j] > p2s[i])):
                dominated[i] = True
                break
    return sorted(
        [i for i in range(n) if not dominated[i]],
        key=lambda i: aucs[i],
    )


def find_knee_point(
    aucs: np.ndarray,
    p2s: np.ndarray,
    pareto_indices: List[int],
) -> int:
    """Find the knee point on the Pareto frontier.

    The knee is the point with maximum perpendicular distance from
    the line connecting the two extreme endpoints of the frontier
    (after min-max normalization to ensure scale invariance).  This
    is the standard chord-distance definition from multi-objective
    optimization (Branke et al., 2004; Das, 1999), which identifies
    the inflection where improving one objective begins requiring
    disproportionate sacrifice of the other — i.e., the point of
    maximum curvature on the trade-off surface.

    **Why chord distance rather than ideal-point (utopian) distance.**
    Ideal-point distance minimizes absolute distance to ``(max_AUC,
    max_P2)`` and always selects the config closest to perfection.
    Chord distance instead selects the *marginal trade-off inflection*:
    the config where the cost of additional resilience in AUC units
    changes most sharply.  This is the deployment-relevant question
    ("where do diminishing returns begin?") and is invariant to
    monotone rescaling of the frontier endpoints.

    **Interpretation caveat.**  Adjacent frontier configs may have
    overlapping confidence intervals on the AUC axis (see
    :func:`find_pareto_frontier`).  The knee-point should be
    interpreted as identifying a *region* of favorable trade-offs
    rather than a uniquely optimal configuration.

    **Degenerate cases.**  For a single frontier point, that point
    is returned.  For exactly two frontier points there is no
    interior curvature — both points lie on the chord, so
    perpendicular distance is zero for both and the knee concept
    is undefined.  The fallback returns the point closest to the
    ideal point ``(max_AUC, max_P2)`` in normalised space, which
    is the only sensible tiebreaker when no inflection exists.
    In practice this branch never fires: the sweep produces
    frontiers with 9–13 points.

    Args:
        aucs: AUC values for all configs.
        p2s: Phase-2 reward values for all configs.
        pareto_indices: Sorted indices of Pareto-optimal configs.

    Returns:
        Index (into the original arrays) of the knee-point config.

    Raises:
        ValueError: If ``pareto_indices`` is empty.
    """
    if not pareto_indices:
        raise ValueError("pareto_indices is empty; cannot identify a knee point")
    if len(pareto_indices) == 1:
        return pareto_indices[0]

    if len(pareto_indices) == 2:
        p_aucs = np.array([aucs[i] for i in pareto_indices])
        p_p2s = np.array([p2s[i] for i in pareto_indices])
        auc_n = (p_aucs - p_aucs.min()) / (np.ptp(p_aucs) + 1e-12)
        p2_n = (p_p2s - p_p2s.min()) / (np.ptp(p_p2s) + 1e-12)
        dists = np.sqrt((1.0 - auc_n) ** 2 + (1.0 - p2_n) ** 2)
        return pareto_indices[int(np.argmin(dists))]

    p_aucs = np.array([aucs[i] for i in pareto_indices])
    p_p2s = np.array([p2s[i] for i in pareto_indices])

    auc_range = p_aucs.max() - p_aucs.min()
    p2_range = p_p2s.max() - p_p2s.min()
    p_aucs_n = (p_aucs - p_aucs.min()) / (auc_range + 1e-12)
    p_p2s_n = (p_p2s - p_p2s.min()) / (p2_range + 1e-12)

    x1, y1 = p_aucs_n[0], p_p2s_n[0]
    x2, y2 = p_aucs_n[-1], p_p2s_n[-1]
    line_len = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    best_dist = -1.0
    best_k = 0
    for k in range(len(pareto_indices)):
        x0, y0 = p_aucs_n[k], p_p2s_n[k]
        dist = abs(
            (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1
        ) / (line_len + 1e-12)
        if dist > best_dist:
            best_dist = dist
            best_k = k

    return pareto_indices[best_k]


# ======================================================================
# Bootstrap Knee-Point Stability
# ======================================================================

N_BOOTSTRAP: int = 2000
BOOTSTRAP_SEED: int = 42


def bootstrap_knee_stability(
    configs: List[Dict[str, Any]],
    per_seed_aucs: np.ndarray,
    per_seed_p2s: np.ndarray,
    original_knee_idx: int,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Assess stability of the Pareto knee-point via seed-level bootstrap.

    For each bootstrap iteration, resamples seed indices with replacement
    and re-runs the full Pareto frontier construction and knee-point
    selection.  Reports how often each config is selected as the knee
    point, providing a non-parametric measure of selection robustness.

    The bootstrap resamples paired (AUC_seed_i, P2_seed_i) tuples
    jointly, so each bootstrap sample rebuilds the full Pareto frontier
    from resampled config-level means.  AUC and P2 use different seed
    offsets (independent RNG streams), so the pairing is by index
    rather than by shared randomness.

    Args:
        configs: List of config dicts with ``alpha``, ``gamma``,
            ``n_eff`` keys (for labeling selected configs).
        per_seed_aucs: Shape ``(n_configs, n_seeds)`` array of per-seed
            budget-paced Pareto AUC values.
        per_seed_p2s: Shape ``(n_configs, n_seeds)`` array of per-seed
            Phase-2 reward values (each averaged over failure arms and
            budget targets within that seed).
        original_knee_idx: Index of the knee point from the original
            (non-bootstrap) analysis.
        n_bootstrap: Number of bootstrap iterations.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with bootstrap stability metrics including selection
        frequencies and neighborhood stability.
    """
    rng = np.random.default_rng(seed)
    n_configs, n_seeds = per_seed_aucs.shape

    knee_counts = np.zeros(n_configs, dtype=int)

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_seeds, size=n_seeds, replace=True)
        boot_aucs = per_seed_aucs[:, boot_idx].mean(axis=1)
        boot_p2s = per_seed_p2s[:, boot_idx].mean(axis=1)

        pareto_idx = find_pareto_frontier(boot_aucs, boot_p2s)
        knee_idx = find_knee_point(boot_aucs, boot_p2s, pareto_idx)
        knee_counts[knee_idx] += 1

    original_freq = float(knee_counts[original_knee_idx]) / n_bootstrap
    modal_idx = int(np.argmax(knee_counts))
    modal_freq = float(knee_counts[modal_idx]) / n_bootstrap

    selected_configs: List[Dict[str, Any]] = []
    for i in range(n_configs):
        if knee_counts[i] > 0:
            selected_configs.append({
                "config_idx": i,
                "alpha": configs[i]["alpha"],
                "gamma": configs[i]["gamma"],
                "n_eff": configs[i].get("n_eff"),
                "frequency": round(float(knee_counts[i]) / n_bootstrap, 4),
                "count": int(knee_counts[i]),
                "is_original_knee": i == original_knee_idx,
            })
    selected_configs.sort(key=lambda x: x["frequency"], reverse=True)

    original_gamma = configs[original_knee_idx]["gamma"]
    original_alpha = configs[original_knee_idx]["alpha"]
    neighbor_freq = 0.0
    for i in range(n_configs):
        if knee_counts[i] > 0:
            gamma_dist = abs(configs[i]["gamma"] - original_gamma)
            if gamma_dist <= 0.001 + 1e-9 and configs[i]["alpha"] == original_alpha:
                neighbor_freq += float(knee_counts[i]) / n_bootstrap

    return {
        "n_bootstrap": n_bootstrap,
        "original_knee_frequency": round(original_freq, 4),
        "modal_config_frequency": round(modal_freq, 4),
        "modal_is_original": modal_idx == original_knee_idx,
        "neighborhood_frequency": round(neighbor_freq, 4),
        "n_unique_selections": sum(1 for c in knee_counts if c > 0),
        "selected_configs": selected_configs,
    }


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load all three splits and compute embeddings
    # ------------------------------------------------------------------
    logger.info("Loading data records ...")
    train_records = load_jsonl(TRAIN_DATA_PATH)
    val_records = load_jsonl(VAL_DATA_PATH)
    test_records = load_jsonl(HOLDOUT_DATA_PATH)
    logger.info(
        "  train=%d  val=%d  test=%d",
        len(train_records), len(val_records), len(test_records),
    )

    # PCA projection is pre-fitted on ~46K disjoint LMSYS prompts and frozen;
    # only .transform() is called during evaluation (no leakage).
    logger.info("Initializing FeatureService (PCA-%d) ...", PCA_DIM)
    fs = FeatureService(pca_components=PCA_DIM)
    feature_dim = fs.dimension
    logger.info("  feature_dim=%d", feature_dim)

    logger.info("Encoding and embedding prompts (val + test) ...")
    val_data = parse_and_embed(val_records, fs)
    test_data = parse_and_embed(test_records, fs)

    val_burnin, val_eval = split_data(val_data, ARM_ORDER)
    logger.info(
        "  val split → val_burnin=%d  val_eval=%d  (disjoint)",
        val_burnin.n, val_eval.n,
    )

    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K3_WARMUP_PRIORS_PATH)

    # ------------------------------------------------------------------
    # 2. Fixed-model baselines (computed on val_eval only)
    # ------------------------------------------------------------------
    val_fixed_baselines: Dict[str, Dict[str, float]] = {}
    for arm_id in ARM_ORDER:
        val_fixed_baselines[ARM_SHORT[arm_id]] = {
            "mean_reward": round(float(val_eval.rewards[arm_id].mean()), 6),
            "mean_cost": round(float(val_eval.costs[arm_id].mean()), 6),
        }
    val_fixed_costs = [v["mean_cost"] for v in val_fixed_baselines.values()]
    val_fixed_rewards = [v["mean_reward"] for v in val_fixed_baselines.values()]
    val_fixed_auc = pareto_auc(
        val_fixed_costs, val_fixed_rewards,
        min(val_fixed_costs), max(val_fixed_costs),
    )

    logger.info("\nFixed-model baselines (val_eval):")
    logger.info("  Pareto AUC: %.6f", val_fixed_auc)
    for name, stats in val_fixed_baselines.items():
        logger.info(
            "    %-14s  reward=%.4f  cost=$%.6f",
            name, stats["mean_reward"], stats["mean_cost"],
        )

    configs = _build_configs()

    per_model_means = {
        a: float(np.mean([r["arms"][a]["cost"] for r in train_records]))
        for a in ARM_ORDER
    }
    budget_targets = list(np.geomspace(
        min(per_model_means.values()),
        max(per_model_means.values()),
        num=BUDGET_TARGET_COUNT,
    ))
    total_trials = len(configs) * len(budget_targets) * N_SEEDS
    logger.info(
        "\nSweep: %d configs (%s), %d budget targets x %d seeds = %d total val trials",
        len(configs), ", ".join(VARIANTS),
        len(budget_targets), N_SEEDS, total_trials,
    )
    logger.info("T_adapt = %d (adaptation horizon constraint)", T_ADAPT)
    if "paretobandit" in VARIANTS:
        pb_configs = [c for c in configs if c["variant"] == "paretobandit"]
        gamma_neff_pairs = sorted(
            {(c["gamma"], c["n_eff"]) for c in pb_configs},
        )
        logger.info("  gamma → n_eff (derived):")
        for g, ne in gamma_neff_pairs:
            logger.info("    γ=%.4f → n_eff=%.1f", g, ne)
    logger.info(
        "Budget targets ($/req): %s",
        [f"${t:.6f}" for t in budget_targets],
    )

    # ------------------------------------------------------------------
    # 3. Budget-paced Pareto AUC on val (primary metric)
    # ------------------------------------------------------------------
    val_results: List[Dict[str, Any]] = []
    current_variant = None

    for idx, cfg in enumerate(configs):
        variant = cfg["variant"]
        alpha = cfg["alpha"]
        n_eff = cfg["n_eff"]
        gamma = cfg["gamma"]

        if variant != current_variant:
            current_variant = variant
            logger.info("\n--- %s (budget-paced val selection) ---", variant)

        use_warmup = variant == "paretobandit"
        wp = warmup_path if use_warmup else None

        t_cfg = time.time()
        auc, auc_std, sweep, seed_aucs = compute_budget_paced_pareto_auc(
            val_burnin, val_eval, registry, feature_dim,
            budget_targets,
            warmup_path=wp,
            alpha=alpha,
            n_eff=n_eff,
            gamma=gamma,
            n_seeds=N_SEEDS,
            seed_offset=SEED_OFFSET_VAL,
        )
        elapsed = time.time() - t_cfg
        delta_pct = (auc - val_fixed_auc) / val_fixed_auc * 100

        val_results.append({
            "variant": variant,
            "alpha": alpha,
            "n_eff": n_eff,
            "gamma": gamma,
            "pca_dim": PCA_DIM,
            "val_pareto_auc": round(auc, 6),
            "val_pareto_auc_std": round(auc_std, 6),
            "val_fixed_auc": round(val_fixed_auc, 6),
            "val_delta_pct": round(delta_pct, 3),
            "sweep_points": sweep,
            "elapsed_s": round(elapsed, 1),
            "_per_seed_aucs": seed_aucs,
        })

        marker = ""
        best_so_far = max(
            (r for r in val_results if r["variant"] == variant),
            key=lambda x: x["val_pareto_auc"],
        )
        if (best_so_far["alpha"] == alpha
                and best_so_far["n_eff"] == n_eff
                and best_so_far["gamma"] == gamma):
            marker = " *** BEST ***"

        logger.info(
            "  [%3d/%d] alpha=%.3f n_eff=%7.0f γ=%.4f  BP_AUC=%.6f ± %.6f "
            "(Δ=%+.3f%%)  %.1fs%s",
            idx + 1, len(configs), alpha, n_eff, gamma,
            auc, auc_std, delta_pct, elapsed, marker,
        )

    # ------------------------------------------------------------------
    # 4. Failure-resilience evaluation on val (Phase 2 mean reward)
    # ------------------------------------------------------------------
    logger.info(
        "\nFailure-resilience eval: %d failure arm(s), %d budget targets, "
        "failure_reward=%.2f",
        len(FAILURE_ARMS), len(FAILURE_BUDGET_TARGETS), FAILURE_REWARD,
    )
    for fa in FAILURE_ARMS:
        logger.info("  fail: %s → reward=%.2f, cost=$0", ARM_SHORT[fa], FAILURE_REWARD)
    logger.info(
        "  budget targets: %s",
        [f"${t:.6f}" for t in FAILURE_BUDGET_TARGETS],
    )

    failure_results: List[Dict[str, Any]] = []
    current_variant = None

    for idx, cfg in enumerate(configs):
        variant = cfg["variant"]
        alpha = cfg["alpha"]
        n_eff = cfg["n_eff"]
        gamma = cfg["gamma"]

        if variant != current_variant:
            current_variant = variant
            logger.info("\n--- %s (failure-resilience eval) ---", variant)

        use_warmup = variant == "paretobandit"
        wp = warmup_path if use_warmup else None

        t_cfg = time.time()
        p2_rwd, p2_std, p1_rwd, seed_p2s = compute_failure_resilience(
            val_burnin, val_eval, registry, feature_dim,
            warmup_path=wp,
            alpha=alpha,
            n_eff=n_eff,
            gamma=gamma,
            n_seeds=N_SEEDS,
            seed_offset=SEED_OFFSET_FAILURE_VAL,
            failure_arms=FAILURE_ARMS,
            failure_reward=FAILURE_REWARD,
            budget_targets=FAILURE_BUDGET_TARGETS,
        )
        elapsed = time.time() - t_cfg

        failure_results.append({
            "variant": variant,
            "alpha": alpha,
            "n_eff": n_eff,
            "gamma": gamma,
            "phase2_reward": round(p2_rwd, 4),
            "phase2_reward_std": round(p2_std, 4),
            "phase1_reward_diag": round(p1_rwd, 4),
            "elapsed_s": round(elapsed, 1),
            "_per_seed_p2s": seed_p2s,
        })

        marker = ""
        best_so_far = max(
            (r for r in failure_results if r["variant"] == variant),
            key=lambda x: x["phase2_reward"],
        )
        if (best_so_far["alpha"] == alpha
                and best_so_far["n_eff"] == n_eff
                and best_so_far["gamma"] == gamma):
            marker = " *** BEST ***"

        logger.info(
            "  [%3d/%d] alpha=%.3f n_eff=%7.0f γ=%.4f  P2_reward=%.4f ± %.4f  "
            "(P1_diag=%.4f)  %.1fs%s",
            idx + 1, len(configs), alpha, n_eff, gamma,
            p2_rwd, p2_std, p1_rwd, elapsed, marker,
        )

    # ------------------------------------------------------------------
    # 5. Pareto knee-point selection
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PARETO KNEE-POINT SELECTION")
    logger.info("=" * 70)

    per_variant_best: Dict[str, Dict[str, Any]] = {}
    auc_only_best: Dict[str, Dict[str, Any]] = {}
    pareto_frontiers: Dict[str, List[Dict[str, Any]]] = {}
    bootstrap_results: Dict[str, Dict[str, Any]] = {}

    for variant in VARIANTS:
        var_stat = [r for r in val_results if r["variant"] == variant]
        var_fail = [r for r in failure_results if r["variant"] == variant]

        aucs = np.array([r["val_pareto_auc"] for r in var_stat])
        p2s = np.array([
            next(
                f["phase2_reward"]
                for f in var_fail if cfg_matches(f, r)
            )
            for r in var_stat
        ])

        auc_best_cfg = max(var_stat, key=lambda r: r["val_pareto_auc"])
        auc_only_best[variant] = {
            "alpha": auc_best_cfg["alpha"],
            "n_eff": auc_best_cfg["n_eff"],
            "gamma": auc_best_cfg["gamma"],
            "val_pareto_auc": auc_best_cfg["val_pareto_auc"],
        }

        logger.info("\n  --- %s ---", variant)

        logger.info(
            "\n  %-7s  %-7s  %-7s  %-10s  %-10s  %-8s",
            "alpha", "n_eff", "gamma", "BP_AUC", "P2_Reward", "Pareto?",
        )
        logger.info("  " + "-" * 65)

        pareto_idx = find_pareto_frontier(aucs, p2s)
        pareto_set = set(pareto_idx)

        for i in range(len(var_stat)):
            logger.info(
                "  %.3f  %7.0f  %.4f  %.6f  %8.4f    %s",
                var_stat[i]["alpha"], var_stat[i]["n_eff"],
                var_stat[i]["gamma"],
                aucs[i], p2s[i],
                "***" if i in pareto_set else "",
            )

        logger.info(
            "\n  Pareto frontier: %d / %d configs",
            len(pareto_idx), len(var_stat),
        )
        for i in pareto_idx:
            logger.info(
                "    alpha=%.3f  γ=%.4f  n_eff=%7.0f  "
                "AUC=%.6f  P2=%.4f",
                var_stat[i]["alpha"], var_stat[i]["gamma"],
                var_stat[i]["n_eff"], aucs[i], p2s[i],
            )

        knee_idx = find_knee_point(aucs, p2s, pareto_idx)
        knee_stat = var_stat[knee_idx]
        knee_fail = next(
            f for f in var_fail if cfg_matches(f, knee_stat)
        )

        pareto_frontiers[variant] = [
            {
                "alpha": var_stat[i]["alpha"],
                "n_eff": var_stat[i]["n_eff"],
                "gamma": var_stat[i]["gamma"],
                "val_pareto_auc": round(float(aucs[i]), 6),
                "phase2_reward": round(float(p2s[i]), 4),
                "is_knee": i == knee_idx,
            }
            for i in pareto_idx
        ]

        per_variant_best[variant] = {
            "alpha": knee_stat["alpha"],
            "n_eff": knee_stat["n_eff"],
            "gamma": knee_stat["gamma"],
            "pca_dim": PCA_DIM,
            "val_pareto_auc": knee_stat["val_pareto_auc"],
            "val_phase2_reward": knee_fail["phase2_reward"],
            "selection_method": "pareto_knee_point",
            "t_adapt": T_ADAPT,
        }

        logger.info(
            "\n  KNEE POINT: alpha=%.3f, n_eff=%.0f, γ=%.4f  "
            "BP_AUC=%.6f  P2_Reward=%.4f",
            knee_stat["alpha"], knee_stat["n_eff"],
            knee_stat["gamma"],
            knee_stat["val_pareto_auc"], knee_fail["phase2_reward"],
        )
        logger.info(
            "  AUC-only:   alpha=%.3f, n_eff=%.0f, γ=%.4f  "
            "BP_AUC=%.6f",
            auc_best_cfg["alpha"], auc_best_cfg["n_eff"],
            auc_best_cfg["gamma"], auc_best_cfg["val_pareto_auc"],
        )

        # Bootstrap stability analysis
        var_per_seed_aucs = np.array([r["_per_seed_aucs"] for r in var_stat])
        var_per_seed_p2s = np.array([
            next(
                f["_per_seed_p2s"]
                for f in var_fail if cfg_matches(f, r)
            )
            for r in var_stat
        ])

        boot = bootstrap_knee_stability(
            [{"alpha": r["alpha"], "gamma": r["gamma"], "n_eff": r["n_eff"]}
             for r in var_stat],
            var_per_seed_aucs,
            var_per_seed_p2s,
            knee_idx,
        )
        bootstrap_results[variant] = boot

        logger.info(
            "\n  BOOTSTRAP (%d iters): knee selected in %.1f%% of "
            "resamples, within 1 grid step in %.1f%% (%d unique configs)",
            boot["n_bootstrap"],
            boot["original_knee_frequency"] * 100,
            boot["neighborhood_frequency"] * 100,
            boot["n_unique_selections"],
        )
        for bc in boot["selected_configs"][:5]:
            tag = " ← original" if bc["is_original_knee"] else ""
            logger.info(
                "    α=%.3f γ=%.4f n_eff=%.0f  freq=%.1f%%%s",
                bc["alpha"], bc["gamma"], bc["n_eff"],
                bc["frequency"] * 100, tag,
            )

    # ------------------------------------------------------------------
    # 5b. Cross-arm catastrophic-failure validation (selected config only)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-ARM FAILURE VALIDATION (selected config only)")
    logger.info("=" * 70)
    logger.info(
        "  Evaluating knee-point config against failure of each arm "
        "independently."
    )
    logger.info(
        "  Tuning used %s only; this validates generalization to all %d arms.",
        ARM_SHORT[K3_FAILURE_ARM], len(ARM_ORDER),
    )

    cross_arm_validation: Dict[str, Dict[str, Any]] = {}

    for variant in VARIANTS:
        best = per_variant_best[variant]
        use_warmup = variant == "paretobandit"
        wp = warmup_path if use_warmup else None

        logger.info(
            "\n  %s (alpha=%.3f, n_eff=%.0f, γ=%.4f):",
            variant, best["alpha"], best["n_eff"], best["gamma"],
        )

        per_arm_results: Dict[str, Dict[str, float]] = {}
        for fail_arm in ARM_ORDER:
            t_arm = time.time()
            p2_rwd, p2_std, p1_rwd, _ca_seed_p2 = compute_failure_resilience(
                val_burnin, val_eval, registry, feature_dim,
                warmup_path=wp,
                alpha=best["alpha"],
                n_eff=best["n_eff"],
                gamma=best["gamma"],
                n_seeds=N_SEEDS,
                seed_offset=SEED_OFFSET_FAILURE_VAL,
                failure_arms=[fail_arm],
                failure_reward=FAILURE_REWARD,
                budget_targets=FAILURE_BUDGET_TARGETS,
            )
            elapsed_arm = time.time() - t_arm

            short = ARM_SHORT[fail_arm]
            tuned_tag = " (tuned on)" if fail_arm == K3_FAILURE_ARM else ""
            per_arm_results[short] = {
                "phase2_reward": round(p2_rwd, 4),
                "phase2_reward_std": round(p2_std, 4),
                "phase1_reward_diag": round(p1_rwd, 4),
            }
            logger.info(
                "    %-18s  P2=%.4f ± %.4f  (P1=%.4f)  %.1fs%s",
                short, p2_rwd, p2_std, p1_rwd, elapsed_arm, tuned_tag,
            )

        cross_arm_validation[variant] = per_arm_results

        stds = [v["phase2_reward_std"] for v in per_arm_results.values()]
        if min(stds) > 0:
            std_ratio = max(stds) / min(stds)
            max_arm = max(per_arm_results, key=lambda k: per_arm_results[k]["phase2_reward_std"])
            if std_ratio > 3.0:
                logger.info(
                    "\n  NOTE: %s failure has %.1fx higher P2 std than the "
                    "most stable arm (%.4f vs %.4f).  This is expected: "
                    "failure of the cheapest arm redistributes traffic to "
                    "expensive arms, and the interaction with tight vs. "
                    "loose budget targets creates high outcome variance.  "
                    "The tuning objective uses %s (lowest P2, lowest std) "
                    "to avoid injecting this noise into the Pareto frontier.",
                    max_arm, std_ratio, max(stds), min(stds),
                    ARM_SHORT[K3_FAILURE_ARM],
                )

    # ------------------------------------------------------------------
    # 6. Holdout evaluation (budget-paced AUC on test)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("HOLDOUT EVALUATION (test split)")
    logger.info("=" * 70)

    test_fixed_baselines: Dict[str, Dict[str, float]] = {}
    for arm_id in ARM_ORDER:
        test_fixed_baselines[ARM_SHORT[arm_id]] = {
            "mean_reward": round(float(test_data.rewards[arm_id].mean()), 6),
            "mean_cost": round(float(test_data.costs[arm_id].mean()), 6),
        }
    test_fixed_costs = [v["mean_cost"] for v in test_fixed_baselines.values()]
    test_fixed_rewards = [v["mean_reward"] for v in test_fixed_baselines.values()]
    test_fixed_auc = pareto_auc(
        test_fixed_costs, test_fixed_rewards,
        min(test_fixed_costs), max(test_fixed_costs),
    )

    logger.info("  Fixed-model Pareto AUC (test): %.6f", test_fixed_auc)
    for name, stats in test_fixed_baselines.items():
        logger.info(
            "    %-14s  reward=%.4f  cost=$%.6f",
            name, stats["mean_reward"], stats["mean_cost"],
        )

    test_results: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        use_warmup = variant == "paretobandit"
        wp = warmup_path if use_warmup else None
        best_alpha = per_variant_best[variant]["alpha"]
        best_n_eff = per_variant_best[variant]["n_eff"]
        best_gamma = per_variant_best[variant]["gamma"]

        logger.info(
            "\n  %s (alpha=%.3f, n_eff=%.0f, γ=%.4f) on test "
            "[knee-point] ...",
            variant, best_alpha, best_n_eff, best_gamma,
        )
        t_test = time.time()
        test_auc, test_std, test_sweep, _test_seed_aucs = compute_budget_paced_pareto_auc(
            val_data, test_data, registry, feature_dim,
            budget_targets,
            warmup_path=wp,
            alpha=best_alpha,
            n_eff=best_n_eff,
            gamma=best_gamma,
            n_seeds=N_SEEDS,
            seed_offset=SEED_OFFSET_TEST,
        )
        elapsed_test = time.time() - t_test
        test_delta_pct = (test_auc - test_fixed_auc) / test_fixed_auc * 100

        test_results[variant] = {
            "alpha": best_alpha,
            "n_eff": best_n_eff,
            "gamma": best_gamma,
            "selection_method": "pareto_knee_point",
            "test_pareto_auc": round(test_auc, 6),
            "test_pareto_auc_std": round(test_std, 6),
            "test_fixed_auc": round(test_fixed_auc, 6),
            "test_delta_pct": round(test_delta_pct, 3),
            "test_sweep": test_sweep,
        }

        logger.info(
            "    BP_AUC=%.6f ± %.6f (Δ=%+.3f%%)  %.1fs",
            test_auc, test_std, test_delta_pct, elapsed_test,
        )

    # ------------------------------------------------------------------
    # 6b. Cross-arm failure validation on HELD-OUT test split
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-ARM FAILURE VALIDATION — HELD-OUT TEST SPLIT")
    logger.info("=" * 70)
    logger.info(
        "  Same protocol as step 5b but on the test split (burn-in on "
        "full val, eval on test).  This provides a true held-out "
        "failure-resilience evaluation."
    )

    cross_arm_validation_test: Dict[str, Dict[str, Any]] = {}

    for variant in VARIANTS:
        best = per_variant_best[variant]
        use_warmup = variant == "paretobandit"
        wp = warmup_path if use_warmup else None

        logger.info(
            "\n  %s (alpha=%.3f, n_eff=%.0f, γ=%.4f):",
            variant, best["alpha"], best["n_eff"], best["gamma"],
        )

        per_arm_results_test: Dict[str, Dict[str, float]] = {}
        for fail_arm in ARM_ORDER:
            t_arm = time.time()
            p2_rwd, p2_std, p1_rwd, _test_seed_p2 = compute_failure_resilience(
                val_data, test_data, registry, feature_dim,
                warmup_path=wp,
                alpha=best["alpha"],
                n_eff=best["n_eff"],
                gamma=best["gamma"],
                n_seeds=N_SEEDS,
                seed_offset=SEED_OFFSET_TEST,
                failure_arms=[fail_arm],
                failure_reward=FAILURE_REWARD,
                budget_targets=FAILURE_BUDGET_TARGETS,
            )
            elapsed_arm = time.time() - t_arm

            short = ARM_SHORT[fail_arm]
            tuned_tag = " (tuned on)" if fail_arm == K3_FAILURE_ARM else ""
            per_arm_results_test[short] = {
                "phase2_reward": round(p2_rwd, 4),
                "phase2_reward_std": round(p2_std, 4),
                "phase1_reward_diag": round(p1_rwd, 4),
            }
            logger.info(
                "    %-18s  P2=%.4f ± %.4f  (P1=%.4f)  %.1fs%s",
                short, p2_rwd, p2_std, p1_rwd, elapsed_arm, tuned_tag,
            )

        cross_arm_validation_test[variant] = per_arm_results_test

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "experiment": "t_adapt_constrained_pareto_knee_hparam_selection",
        "protocol": (
            "T_adapt-constrained 2D search: n_eff is derived from gamma "
            f"via the adaptation-horizon formula with T_adapt={T_ADAPT}. "
            "3-split disjoint protocol: priors from train.jsonl, "
            "val.jsonl split into val_burnin (first 1/3) + val_eval "
            "(remaining 2/3) — burn-in on val_burnin, select on val_eval "
            "(no prompt overlap between burn-in and eval), "
            "report on test.jsonl (burn-in on full val, eval on test). "
            "Pareto knee-point selection: build Pareto frontier over "
            "(budget-paced AUC, catastrophic-failure Phase-2 reward), "
            "select the knee point (maximum perpendicular distance from "
            "the line connecting the two extreme endpoints)."
        ),
        "grid": {
            "variants": VARIANTS,
            "alpha_values": ALPHA_VALUES,
            "t_adapt": T_ADAPT,
            "gamma_values": GAMMA_VALUES,
            "pca_dim": PCA_DIM,
            "budget_targets": [round(t, 10) for t in budget_targets],
            "pacer_lr": PACER_LR,
            "pacer_lambda_max": PACER_LAMBDA_MAX,
            "failure_arms": [ARM_SHORT[a] for a in FAILURE_ARMS],
            "failure_reward": FAILURE_REWARD,
            "failure_budget_targets": FAILURE_BUDGET_TARGETS,
            "n_seeds": N_SEEDS,
            "seed_offset_val": SEED_OFFSET_VAL,
            "seed_offset_failure_val": SEED_OFFSET_FAILURE_VAL,
            "seed_offset_test": SEED_OFFSET_TEST,
        },
        "val_fixed_auc": round(val_fixed_auc, 6),
        "val_baselines": val_fixed_baselines,
        "test_fixed_auc": round(test_fixed_auc, 6),
        "test_baselines": test_fixed_baselines,
        "selection_method": "pareto_knee_point",
        "best_per_variant": per_variant_best,
        "auc_only_best": auc_only_best,
        "pareto_frontiers": pareto_frontiers,
        "test_per_variant": test_results,
        "val_budget_paced": [
            {k: v for k, v in r.items()
             if k != "sweep_points" and not k.startswith("_")}
            for r in val_results
        ],
        "val_budget_paced_full": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in val_results
        ],
        "val_failure_resilience": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in failure_results
        ],
        "cross_arm_validation": cross_arm_validation,
        "cross_arm_validation_test": cross_arm_validation_test,
        "bootstrap_knee_stability": bootstrap_results,
    }

    out_path = RESULTS_DIR / "hparam_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nResults written to %s", out_path)

    best_path = RESULTS_DIR / "best_hparams.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "selection_method": "pareto_knee_point",
                "t_adapt": T_ADAPT,
                "best_per_variant_val": per_variant_best,
                "auc_only_best": auc_only_best,
                "pareto_frontiers": pareto_frontiers,
                "cross_arm_validation": cross_arm_validation,
                "cross_arm_validation_test": cross_arm_validation_test,
                "test_per_variant": test_results,
                "bootstrap_knee_stability": bootstrap_results,
            },
            f,
            indent=2,
        )
    logger.info("Best hparams written to %s", best_path)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (Pareto knee-point, T_adapt=%d)", T_ADAPT)
    logger.info("=" * 70)
    for variant in VARIANTS:
        b = per_variant_best[variant]
        t = test_results[variant]
        auc_b = auc_only_best[variant]
        logger.info(
            "  %-12s  SELECTED (knee): alpha=%.3f  n_eff=%7.0f  γ=%.4f  "
            "val_BP_AUC=%.6f  val_P2_reward=%.4f  "
            "test_BP_AUC=%.6f (Δ=%+.3f%%)",
            variant, b["alpha"], b["n_eff"], b["gamma"],
            b["val_pareto_auc"], b["val_phase2_reward"],
            t["test_pareto_auc"], t["test_delta_pct"],
        )
        logger.info(
            "  %-12s  AUC-only:       alpha=%.3f  n_eff=%7.0f  γ=%.4f  "
            "val_BP_AUC=%.6f",
            "", auc_b["alpha"], auc_b["n_eff"], auc_b["gamma"],
            auc_b["val_pareto_auc"],
        )
    logger.info("=" * 70)
    logger.info("Wall time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
