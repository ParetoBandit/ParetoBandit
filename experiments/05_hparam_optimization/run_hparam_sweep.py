#!/usr/bin/env python3
"""Main paper: Epsilon-Constraint Hyperparameter Selection (Experiment 05).

Produces the results for Section~\\ref{sec:hparam_sweep} of the main
paper, which presents the multi-objective framework and key insights.

Grid search over exploration parameter (alpha), prior strength (n_eff),
forgetting factor (gamma), and window size for **three variants** —
ParetoBandit (warmup priors), Tabula Rasa (cold start with geometric
forgetting), and SW-UCB (cold start with sliding-window forgetting) —
using an epsilon-constraint method to balance budget-paced routing
quality and non-stationary adaptation ability.

We fix PCA dimensionality to d=25 (~28.5% cumulative variance).
For Tabula Rasa, n_eff is irrelevant (no priors), so only alpha x
gamma is swept with n_eff fixed at 1.0.
For SW-UCB, gamma is fixed at 1.0 (no geometric decay) and the
sliding-window size is swept alongside alpha.

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
Each (alpha, n_eff, gamma) config is scored on two objectives using
the **same** fixed gamma (no adaptive mechanism):

1. **Budget-Paced Pareto AUC** (primary) — sweep budget targets on the
   val split with ``BudgetPacer`` active, build per-seed Pareto
   frontiers (with fixed-model endpoints), and average AUC across
   seeds.  Higher = better budget-aware routing.

2. **Non-stationary Phase-2 regret** (secondary) — two-phase simulation
   on the val split with the candidate's fixed gamma.
   Phase 1 (first half): normal rewards.  Phase 2 (second half):
   reward/cost swap.  Only Phase-2 cumulative regret is used for
   selection.  The swap is averaged over **all** C(K,2) arm pairs.
   Lower = better adaptation.

Selection uses the **epsilon-constraint method**: among all configs
whose budget-paced AUC is within epsilon (0.25%) of the best, select
the one with lowest Phase-2 regret.  This gives budget-pacing
priority while allowing the secondary objective to break ties.

The winning config is re-evaluated on the held-out test split.

Usage::

    python experiments/05_hparam_optimization/run_hparam_sweep.py
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.config import (
    DEFAULT_NONSTAT_COST_PENALTY,
    DEFAULT_PACER_LAMBDA_MAX,
    DEFAULT_PACER_LR,
    HOLDOUT_DATA_PATH,
    K3_ALL_SWAP_PAIRS,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_WARMUP_PRIORS_PATH,
    N_SEEDS,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.policy import SlidingWindowLinUCBPolicy
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from utils.pareto import pareto_auc
from utils.simulation import build_model_registry, compute_normalized_costs

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
N_EFF_VALUES: List[float] = [1.0, 10.0, 50.0, 200.0, 1000.0, 5000.0]
WINDOW_SIZE_VALUES: List[int] = [50, 100, 200, 400, 800]
BUDGET_TARGET_COUNT: int = 7
PACER_LR: float = DEFAULT_PACER_LR
PACER_LAMBDA_MAX: float = DEFAULT_PACER_LAMBDA_MAX
EPSILON: float = 0.0025
VARIANTS: List[str] = ["paretobandit", "tabula_rasa", "sw_ucb"]

PCA_DIM: int = 25
GAMMA_VALUES: List[float] = [0.995, 0.997, 0.999, 1.0]
SEED_OFFSET_VAL: int = 0
SEED_OFFSET_NONSTAT_VAL: int = 2000
SEED_OFFSET_TEST: int = 1000

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = K3_ARM_SHORT
RESULTS_DIR = Path(__file__).parent / "results"

NONSTAT_COST_PENALTY: float = DEFAULT_NONSTAT_COST_PENALTY
NONSTAT_SWAP_PAIRS: List[Tuple[str, str]] = K3_ALL_SWAP_PAIRS


# ======================================================================
# Config generation
# ======================================================================


def _build_configs() -> List[Dict[str, Any]]:
    """Generate all hyperparameter configurations for every variant.

    For ParetoBandit: full alpha x n_eff x gamma grid (window_size=0).
    For Tabula Rasa: alpha x gamma (n_eff=1.0, window_size=0).
    For SW-UCB: alpha x window_size (gamma=1.0, n_eff=1.0).
    """
    configs: List[Dict[str, Any]] = []
    for alpha, n_eff, gamma in itertools.product(
        ALPHA_VALUES, N_EFF_VALUES, GAMMA_VALUES,
    ):
        configs.append({
            "variant": "paretobandit",
            "alpha": alpha,
            "n_eff": n_eff,
            "gamma": gamma,
            "window_size": 0,
        })
    for alpha, gamma in itertools.product(ALPHA_VALUES, GAMMA_VALUES):
        configs.append({
            "variant": "tabula_rasa",
            "alpha": alpha,
            "n_eff": 1.0,
            "gamma": gamma,
            "window_size": 0,
        })
    for alpha, ws in itertools.product(ALPHA_VALUES, WINDOW_SIZE_VALUES):
        configs.append({
            "variant": "sw_ucb",
            "alpha": alpha,
            "n_eff": 1.0,
            "gamma": 1.0,
            "window_size": ws,
        })
    return configs


# ======================================================================
# Data Loading
# ======================================================================


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file returning a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _parse_and_embed(
    records: List[Dict[str, Any]],
    fs: FeatureService,
) -> Dict[str, Any]:
    """Extract prompts, rewards, costs, and embed via FeatureService.

    Args:
        records: JSONL records with ``prompt`` and ``arms`` fields.
        fs: Feature service configured with the target PCA.

    Returns:
        Dict with ``prompts``, ``rewards``, ``costs``, ``embeddings``, ``n``.
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

    return {
        "prompts": prompts,
        "rewards": {a: np.array(v) for a, v in rewards.items()},
        "costs": {a: np.array(v) for a, v in costs.items()},
        "embeddings": embeddings,
        "n": len(prompts),
    }


def _split_data(
    data: Dict[str, Any],
    arm_order: List[str],
    burnin_frac: float = 1 / 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a data dict into burn-in and eval portions.

    Used to create ``val_burnin`` (first ``burnin_frac``) and
    ``val_eval`` (remainder) so that the burn-in phase never sees
    evaluation prompts.  The default 1/3 burn-in preserves 2/3 of the
    data for evaluation, giving more statistical power for metric
    estimation while still providing adequate online warm-up on top of
    the offline priors.

    Args:
        data: Dict with ``prompts``, ``rewards``, ``costs``,
            ``embeddings``, ``n`` as produced by :func:`_parse_and_embed`.
        arm_order: Model identifiers for slicing reward/cost arrays.
        burnin_frac: Fraction of data allocated to burn-in (default 1/3).

    Returns:
        ``(burnin_portion, eval_portion)`` — two dicts with the same
        schema, disjoint by construction.
    """
    split_idx = int(data["n"] * burnin_frac)
    first = {
        "prompts": data["prompts"][:split_idx],
        "rewards": {a: data["rewards"][a][:split_idx] for a in arm_order},
        "costs": {a: data["costs"][a][:split_idx] for a in arm_order},
        "embeddings": data["embeddings"][:split_idx],
        "n": split_idx,
    }
    second = {
        "prompts": data["prompts"][split_idx:],
        "rewards": {a: data["rewards"][a][split_idx:] for a in arm_order},
        "costs": {a: data["costs"][a][split_idx:] for a in arm_order},
        "embeddings": data["embeddings"][split_idx:],
        "n": data["n"] - split_idx,
    }
    return first, second


# ======================================================================
# Simulation
# ======================================================================


def _maybe_swap_to_sw_ucb(
    router: BanditRouter,
    window_size: int,
    alpha: float,
    feature_dim: int,
    seed: int,
) -> None:
    """Replace the router's policy with :class:`SlidingWindowLinUCBPolicy`.

    Called only when ``window_size > 0``, i.e. the current config is SW-UCB.
    Mirrors the pattern in Experiment 02's ``_create_router``.

    Args:
        router: Already-constructed router whose ``bandit`` field will be
            overwritten in place.
        window_size: Sliding-window length (number of retained observations).
        alpha: UCB exploration coefficient.
        feature_dim: Context vector dimensionality.
        seed: RNG seed for deterministic tie-breaking.
    """
    router.bandit = SlidingWindowLinUCBPolicy(
        model_names=ARM_ORDER,
        dim=feature_dim,
        alpha=alpha,
        window_size=window_size,
        seed=seed,
    )


def _simulate_bandit(
    burnin_data: Dict[str, Any],
    eval_data: Dict[str, Any],
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup_path: Optional[str],
    alpha: float,
    n_eff: float,
    gamma: float,
    cost_penalty: float,
    seed: int,
    window_size: int = 0,
) -> Tuple[float, float]:
    """Run burn-in then eval bandit simulation, return (mean_reward, mean_cost).

    The router first burns in on ``burnin_data`` (disjoint from both
    priors and eval), then records metrics on ``eval_data``.  For val
    selection, burn-in and eval are disjoint halves of val.jsonl; for
    holdout reporting, burn-in is full val and eval is test.

    Args:
        burnin_data: Burn-in split (bandit learns, results not recorded).
        eval_data: Evaluation split (per-prompt metrics recorded).  Must
            be disjoint from ``burnin_data`` to avoid look-ahead bias.
        registry: Filtered model registry.
        feature_dim: Context vector dimensionality.
        warmup_path: Path to warmup priors, or ``None`` for tabula rasa.
        alpha: LinUCB exploration coefficient.
        n_eff: Prior effective sample size (ignored when warmup_path is None).
        gamma: Forgetting factor (1.0 = no forgetting).
        cost_penalty: Cost penalty weight in the routing objective.
        seed: Random seed for data shuffling.
        window_size: If > 0, use SW-UCB policy with this window length.

    Returns:
        ``(mean_reward, mean_cost)`` on the eval split.
    """
    rng = np.random.default_rng(seed)

    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    use_warmup = warmup_path is not None
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if use_warmup else "none",
        warmup_path=warmup_path if use_warmup else None,
        prior_n_effective=n_eff if use_warmup else 1.0,
        alpha=alpha,
        cost_penalty=cost_penalty,
        forgetting_factor=gamma,
        budget_pacer=None,
    )
    if window_size > 0:
        _maybe_swap_to_sw_ucb(router, window_size, alpha, feature_dim, seed)

    burnin_order = rng.permutation(burnin_data["n"])
    for i in burnin_order:
        model, log = router.route(burnin_data["embeddings"][i])
        reward = float(burnin_data["rewards"][model][i])
        router.process_feedback(log.request_id, reward=reward)

    eval_order = rng.permutation(eval_data["n"])
    eval_rewards: List[float] = []
    eval_costs: List[float] = []
    for i in eval_order:
        model, log = router.route(eval_data["embeddings"][i])
        reward = float(eval_data["rewards"][model][i])
        cost = float(eval_data["costs"][model][i])
        eval_rewards.append(reward)
        eval_costs.append(cost)
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

    return float(np.mean(eval_rewards)), float(np.mean(eval_costs))


# ======================================================================
# Budget-Paced Simulation
# ======================================================================


def _simulate_budget_paced(
    burnin_data: Dict[str, Any],
    eval_data: Dict[str, Any],
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup_path: Optional[str],
    alpha: float,
    n_eff: float,
    gamma: float,
    budget_target: float,
    seed: int,
    window_size: int = 0,
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
        window_size: If > 0, use SW-UCB policy with this window length.

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
    if window_size > 0:
        _maybe_swap_to_sw_ucb(router, window_size, alpha, feature_dim, seed)

    burnin_order = rng.permutation(burnin_data["n"])
    for i in burnin_order:
        model, log = router.route(burnin_data["embeddings"][i])
        reward = float(burnin_data["rewards"][model][i])
        log.cost_usd = float(burnin_data["costs"][model][i])
        router.process_feedback(log.request_id, reward=reward)

    eval_order = rng.permutation(eval_data["n"])
    eval_rewards: List[float] = []
    eval_costs: List[float] = []
    for i in eval_order:
        model, log = router.route(eval_data["embeddings"][i])
        reward = float(eval_data["rewards"][model][i])
        cost = float(eval_data["costs"][model][i])
        eval_rewards.append(reward)
        eval_costs.append(cost)
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

    return float(np.mean(eval_rewards)), float(np.mean(eval_costs))


def compute_budget_paced_pareto_auc(
    burnin_data: Dict[str, Any],
    eval_data: Dict[str, Any],
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
    window_size: int = 0,
) -> Tuple[float, float, List[Dict[str, Any]]]:
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
        window_size: If > 0, use SW-UCB policy with this window length.

    Returns:
        ``(mean_auc, std_auc, sweep_points)``.
    """
    fixed_costs = [float(eval_data["costs"][a].mean()) for a in ARM_ORDER]
    fixed_rewards = [float(eval_data["rewards"][a].mean()) for a in ARM_ORDER]
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
                window_size=window_size,
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

    return mean_auc, std_auc, sweep_points


# ======================================================================
# Non-stationary Simulation
# ======================================================================


def _simulate_nonstationary_regret(
    burnin_data: Dict[str, Any],
    val_data: Dict[str, Any],
    registry: Dict[str, Any],
    feature_dim: int,
    normalized_costs: Dict[str, float],
    *,
    warmup_path: Optional[str],
    alpha: float,
    n_eff: float,
    gamma: float,
    cost_penalty: float,
    seed: int,
    swap_arms: Tuple[str, str],
    window_size: int = 0,
) -> Tuple[float, float]:
    """Burn-in then two-phase simulation with reward swap.

    The router first burns in on ``burnin_data`` (disjoint from both
    priors and eval), then runs Phase 1 + Phase 2 on ``val_data``.

    Phase 1 uses the first half of ``val_data`` with normal rewards.
    Phase 2 uses the second half with ``swap_arms`` rewards/costs
    exchanged.

    Args:
        burnin_data: Burn-in split (bandit learns, results not recorded).
        val_data: Evaluation split (split into Phase 1 + Phase 2;
            disjoint from ``burnin_data``).
        registry: Filtered model registry.
        feature_dim: Context vector dimensionality.
        normalized_costs: Per-model normalized costs for regret computation.
        warmup_path: Path to warmup priors, or ``None`` for tabula rasa.
        alpha: LinUCB exploration coefficient.
        n_eff: Prior effective sample size.
        gamma: Forgetting factor (1.0 = no forgetting).
        cost_penalty: Cost penalty weight.
        seed: Random seed.
        swap_arms: Pair of arm IDs whose rewards/costs are exchanged in Phase 2.
        window_size: If > 0, use SW-UCB policy with this window length.

    Returns:
        ``(phase1_regret, phase2_regret)`` — cumulative regret per phase.
    """
    rng = np.random.default_rng(seed)

    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    use_warmup = warmup_path is not None

    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if use_warmup else "none",
        warmup_path=warmup_path if use_warmup else None,
        prior_n_effective=n_eff if use_warmup else 1.0,
        alpha=alpha,
        cost_penalty=cost_penalty,
        forgetting_factor=gamma,
        budget_pacer=None,
    )
    if window_size > 0:
        _maybe_swap_to_sw_ucb(router, window_size, alpha, feature_dim, seed)

    burnin_order = rng.permutation(burnin_data["n"])
    for i in burnin_order:
        model, log = router.route(burnin_data["embeddings"][i])
        reward = float(burnin_data["rewards"][model][i])
        router.process_feedback(log.request_id, reward=reward)

    # --- Build swapped reward/cost arrays for Phase 2 ---
    a1, a2 = swap_arms
    swapped_rewards = dict(val_data["rewards"])
    swapped_rewards[a1] = val_data["rewards"][a2]
    swapped_rewards[a2] = val_data["rewards"][a1]
    swapped_costs = dict(val_data["costs"])
    swapped_costs[a1] = val_data["costs"][a2]
    swapped_costs[a2] = val_data["costs"][a1]

    # --- Phase 1 + Phase 2 ---
    n_val = val_data["n"]
    val_order = rng.permutation(n_val)
    mid = n_val // 2
    nc = normalized_costs

    phase1_regret = 0.0
    for idx in val_order[:mid]:
        model, log = router.route(val_data["embeddings"][idx])
        reward = float(val_data["rewards"][model][idx])
        log.cost_usd = float(val_data["costs"][model][idx])
        router.process_feedback(log.request_id, reward=reward)

        oracle = max(
            float(val_data["rewards"][a][idx]) - cost_penalty * nc[a]
            for a in ARM_ORDER
        )
        chosen = reward - cost_penalty * nc[model]
        phase1_regret += oracle - chosen

    # --- Swap registry costs so the router's internal cost penalty
    #     reflects the Phase-2 cost environment. ---
    reg_a1, reg_a2 = router.registry[a1], router.registry[a2]
    for field in ("input_cost_per_m", "output_cost_per_m"):
        reg_a1[field], reg_a2[field] = reg_a2[field], reg_a1[field]
    # Remove cached blended figures so _resolve_registry_costs re-derives them.
    reg_a1.pop("blended_cost_per_m", None)
    reg_a2.pop("blended_cost_per_m", None)
    router._resolve_registry_costs()

    nc_phase2 = compute_normalized_costs(router.registry, ARM_ORDER)

    phase2_regret = 0.0
    for idx in val_order[mid:]:
        model, log = router.route(val_data["embeddings"][idx])
        reward = float(swapped_rewards[model][idx])
        log.cost_usd = float(swapped_costs[model][idx])
        router.process_feedback(log.request_id, reward=reward)

        oracle = max(
            float(swapped_rewards[a][idx]) - cost_penalty * nc_phase2[a]
            for a in ARM_ORDER
        )
        chosen = reward - cost_penalty * nc_phase2[model]
        phase2_regret += oracle - chosen

    return phase1_regret, phase2_regret


def compute_nonstat_metric(
    burnin_data: Dict[str, Any],
    val_data: Dict[str, Any],
    registry: Dict[str, Any],
    feature_dim: int,
    normalized_costs: Dict[str, float],
    *,
    warmup_path: Optional[str],
    alpha: float,
    n_eff: float,
    gamma: float,
    n_seeds: int,
    seed_offset: int,
    window_size: int = 0,
) -> Tuple[float, float, float]:
    """Phase 2 regret averaged over all pairwise swaps and seeds.

    The router burns in on ``burnin_data`` (disjoint from both priors
    and eval), then runs Phase 1 + Phase 2 on ``val_data``.  Phase 1
    regret is discarded to avoid double-counting with the stationary
    AUC metric.

    Returns:
        ``(mean_phase2_regret, std_phase2_regret, mean_phase1_regret)``
        where the first two are the selection-relevant metrics and the
        third is reported for diagnostic purposes only.
    """
    p1_regs: List[float] = []
    p2_regs: List[float] = []
    for swap_pair in NONSTAT_SWAP_PAIRS:
        for s in range(n_seeds):
            p1, p2 = _simulate_nonstationary_regret(
                burnin_data, val_data, registry, feature_dim,
                normalized_costs,
                warmup_path=warmup_path,
                alpha=alpha,
                n_eff=n_eff,
                gamma=gamma,
                cost_penalty=NONSTAT_COST_PENALTY,
                seed=seed_offset + s,
                swap_arms=swap_pair,
                window_size=window_size,
            )
            p1_regs.append(p1)
            p2_regs.append(p2)

    return (
        float(np.mean(p2_regs)),
        float(np.std(p2_regs, ddof=1)) if len(p2_regs) > 1 else 0.0,
        float(np.mean(p1_regs)),
    )


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
    train_records = _load_jsonl(TRAIN_DATA_PATH)
    val_records = _load_jsonl(VAL_DATA_PATH)
    test_records = _load_jsonl(HOLDOUT_DATA_PATH)
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
    val_data = _parse_and_embed(val_records, fs)
    test_data = _parse_and_embed(test_records, fs)

    val_burnin, val_eval = _split_data(val_data, ARM_ORDER)
    logger.info(
        "  val split → val_burnin=%d  val_eval=%d  (disjoint)",
        val_burnin["n"], val_eval["n"],
    )

    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K3_WARMUP_PRIORS_PATH)

    # ------------------------------------------------------------------
    # 2. Fixed-model baselines (computed on val_eval only)
    # ------------------------------------------------------------------
    val_fixed_baselines: Dict[str, Dict[str, float]] = {}
    for arm_id in ARM_ORDER:
        val_fixed_baselines[ARM_SHORT[arm_id]] = {
            "mean_reward": round(float(val_eval["rewards"][arm_id].mean()), 6),
            "mean_cost": round(float(val_eval["costs"][arm_id].mean()), 6),
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
    n_paretobandit = len(ALPHA_VALUES) * len(N_EFF_VALUES) * len(GAMMA_VALUES)
    n_tabula = len(ALPHA_VALUES) * len(GAMMA_VALUES)
    n_sw_ucb = len(ALPHA_VALUES) * len(WINDOW_SIZE_VALUES)

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
        "\nSweep: %d paretobandit (alpha x n_eff x gamma) + %d tabula_rasa "
        "(alpha x gamma) + %d sw_ucb (alpha x window_size) = %d configs, "
        "%d budget targets x %d seeds = %d total val trials",
        n_paretobandit, n_tabula, n_sw_ucb, len(configs),
        len(budget_targets), N_SEEDS, total_trials,
    )
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
        ws = cfg["window_size"]

        if variant != current_variant:
            current_variant = variant
            logger.info("\n--- %s (budget-paced val selection) ---", variant)

        use_warmup = variant == "paretobandit"
        wp = warmup_path if use_warmup else None

        t_cfg = time.time()
        auc, auc_std, sweep = compute_budget_paced_pareto_auc(
            val_burnin, val_eval, registry, feature_dim,
            budget_targets,
            warmup_path=wp,
            alpha=alpha,
            n_eff=n_eff,
            gamma=gamma,
            n_seeds=N_SEEDS,
            seed_offset=SEED_OFFSET_VAL,
            window_size=ws,
        )
        elapsed = time.time() - t_cfg
        delta_pct = (auc - val_fixed_auc) / val_fixed_auc * 100

        val_results.append({
            "variant": variant,
            "alpha": alpha,
            "n_eff": n_eff,
            "gamma": gamma,
            "window_size": ws,
            "pca_dim": PCA_DIM,
            "val_pareto_auc": round(auc, 6),
            "val_pareto_auc_std": round(auc_std, 6),
            "val_fixed_auc": round(val_fixed_auc, 6),
            "val_delta_pct": round(delta_pct, 3),
            "sweep_points": sweep,
            "elapsed_s": round(elapsed, 1),
        })

        marker = ""
        best_so_far = max(
            (r for r in val_results if r["variant"] == variant),
            key=lambda x: x["val_pareto_auc"],
        )
        if (best_so_far["alpha"] == alpha
                and best_so_far["n_eff"] == n_eff
                and best_so_far["gamma"] == gamma
                and best_so_far["window_size"] == ws):
            marker = " *** BEST ***"

        if variant == "sw_ucb":
            logger.info(
                "  [%3d/%d] alpha=%.3f W=%4d  BP_AUC=%.6f ± %.6f "
                "(Δ=%+.3f%%)  %.1fs%s",
                idx + 1, len(configs), alpha, ws,
                auc, auc_std, delta_pct, elapsed, marker,
            )
        else:
            logger.info(
                "  [%3d/%d] alpha=%.3f n_eff=%7.0f γ=%.4f  BP_AUC=%.6f ± %.6f "
                "(Δ=%+.3f%%)  %.1fs%s",
                idx + 1, len(configs), alpha, n_eff, gamma,
                auc, auc_std, delta_pct, elapsed, marker,
            )

    # ------------------------------------------------------------------
    # 4. Non-stationary evaluation on val (Phase 2 regret)
    # ------------------------------------------------------------------
    normalized_costs = compute_normalized_costs(registry, ARM_ORDER)
    logger.info(
        "\nNormalized costs: %s",
        {ARM_SHORT[a]: f"{v:.4f}" for a, v in normalized_costs.items()},
    )
    logger.info(
        "Non-stationary eval: %d swap pairs, cp=%.2f, fixed gamma per config",
        len(NONSTAT_SWAP_PAIRS), NONSTAT_COST_PENALTY,
    )
    for a1, a2 in NONSTAT_SWAP_PAIRS:
        logger.info("  swap: %s ↔ %s", ARM_SHORT[a1], ARM_SHORT[a2])

    nonstat_results: List[Dict[str, Any]] = []
    current_variant = None

    for idx, cfg in enumerate(configs):
        variant = cfg["variant"]
        alpha = cfg["alpha"]
        n_eff = cfg["n_eff"]
        gamma = cfg["gamma"]
        ws = cfg["window_size"]

        if variant != current_variant:
            current_variant = variant
            logger.info("\n--- %s (non-stationary eval) ---", variant)

        use_warmup = variant == "paretobandit"
        wp = warmup_path if use_warmup else None

        t_cfg = time.time()
        p2_reg, p2_std, p1_reg = compute_nonstat_metric(
            val_burnin, val_eval, registry, feature_dim, normalized_costs,
            warmup_path=wp,
            alpha=alpha,
            n_eff=n_eff,
            gamma=gamma,
            n_seeds=N_SEEDS,
            seed_offset=SEED_OFFSET_NONSTAT_VAL,
            window_size=ws,
        )
        elapsed = time.time() - t_cfg

        nonstat_results.append({
            "variant": variant,
            "alpha": alpha,
            "n_eff": n_eff,
            "gamma": gamma,
            "window_size": ws,
            "phase2_regret": round(p2_reg, 2),
            "phase2_regret_std": round(p2_std, 2),
            "phase1_regret_diag": round(p1_reg, 2),
            "elapsed_s": round(elapsed, 1),
        })

        if variant == "sw_ucb":
            logger.info(
                "  [%3d/%d] alpha=%.3f W=%4d  P2_regret=%.1f ± %.1f  "
                "(P1_diag=%.1f)  %.1fs",
                idx + 1, len(configs), alpha, ws,
                p2_reg, p2_std, p1_reg, elapsed,
            )
        else:
            logger.info(
                "  [%3d/%d] alpha=%.3f n_eff=%7.0f γ=%.4f  P2_regret=%.1f ± %.1f  "
                "(P1_diag=%.1f)  %.1fs",
                idx + 1, len(configs), alpha, n_eff, gamma,
                p2_reg, p2_std, p1_reg, elapsed,
            )

    # ------------------------------------------------------------------
    # 5. Epsilon-constraint selection
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("EPSILON-CONSTRAINT SELECTION (epsilon=%.2f)", EPSILON)
    logger.info("=" * 70)

    per_variant_best: Dict[str, Dict[str, Any]] = {}
    auc_only_best: Dict[str, Dict[str, Any]] = {}

    def _cfg_matches(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """True when two result dicts share the same hyperparameter config."""
        return (
            a["alpha"] == b["alpha"]
            and a["n_eff"] == b["n_eff"]
            and a["gamma"] == b["gamma"]
            and a["window_size"] == b["window_size"]
        )

    for variant in VARIANTS:
        var_stat = [r for r in val_results if r["variant"] == variant]
        var_nonstat = [r for r in nonstat_results if r["variant"] == variant]

        best_auc = max(r["val_pareto_auc"] for r in var_stat)
        threshold = best_auc * (1.0 - EPSILON)

        auc_best_cfg = max(var_stat, key=lambda r: r["val_pareto_auc"])
        auc_only_best[variant] = {
            "alpha": auc_best_cfg["alpha"],
            "n_eff": auc_best_cfg["n_eff"],
            "gamma": auc_best_cfg["gamma"],
            "window_size": auc_best_cfg["window_size"],
            "val_pareto_auc": auc_best_cfg["val_pareto_auc"],
        }

        logger.info("\n  --- %s ---", variant)
        logger.info("  Best BP AUC: %.6f  Threshold (%.0f%%): %.6f",
                     best_auc, (1 - EPSILON) * 100, threshold)

        if variant == "sw_ucb":
            logger.info(
                "  AUC-only winner: alpha=%.3f, W=%d, AUC=%.6f",
                auc_best_cfg["alpha"], auc_best_cfg["window_size"],
                auc_best_cfg["val_pareto_auc"],
            )
            logger.info(
                "\n  %-7s  %-7s  %-10s  %-9s  %-10s  %-8s",
                "alpha", "W", "BP_AUC", "Δ%", "P2_Regret", "Feasible",
            )
            logger.info("  " + "-" * 65)
        else:
            logger.info(
                "  AUC-only winner: alpha=%.3f, n_eff=%.0f, γ=%.4f, AUC=%.6f",
                auc_best_cfg["alpha"], auc_best_cfg["n_eff"],
                auc_best_cfg["gamma"], auc_best_cfg["val_pareto_auc"],
            )
            logger.info(
                "\n  %-7s  %-7s  %-7s  %-10s  %-9s  %-10s  %-8s",
                "alpha", "n_eff", "gamma", "BP_AUC", "Δ%",
                "P2_Regret", "Feasible",
            )
            logger.info("  " + "-" * 75)

        feasible_indices: List[int] = []
        for i in range(len(var_stat)):
            auc_val = var_stat[i]["val_pareto_auc"]
            ns_match = [
                r for r in var_nonstat if _cfg_matches(r, var_stat[i])
            ]
            p2_reg = ns_match[0]["phase2_regret"] if ns_match else float("nan")
            is_feasible = auc_val >= threshold
            if is_feasible:
                feasible_indices.append(i)

            delta = (auc_val - best_auc) / best_auc * 100
            if variant == "sw_ucb":
                logger.info(
                    "  %.3f  %7d  %.6f  %+6.2f%%  %8.1f    %s",
                    var_stat[i]["alpha"], var_stat[i]["window_size"],
                    auc_val, delta, p2_reg,
                    "YES" if is_feasible else "",
                )
            else:
                logger.info(
                    "  %.3f  %7.0f  %.4f  %.6f  %+6.2f%%  %8.1f    %s",
                    var_stat[i]["alpha"], var_stat[i]["n_eff"],
                    var_stat[i]["gamma"],
                    auc_val, delta, p2_reg,
                    "YES" if is_feasible else "",
                )

        logger.info("\n  Feasible set: %d / %d configs",
                     len(feasible_indices), len(var_stat))

        best_idx = min(
            feasible_indices,
            key=lambda i: next(
                r["phase2_regret"]
                for r in var_nonstat
                if _cfg_matches(r, var_stat[i])
            ),
        )
        winner_stat = var_stat[best_idx]
        winner_ns = next(
            r for r in var_nonstat if _cfg_matches(r, winner_stat)
        )

        per_variant_best[variant] = {
            "alpha": winner_stat["alpha"],
            "n_eff": winner_stat["n_eff"],
            "gamma": winner_stat["gamma"],
            "window_size": winner_stat["window_size"],
            "pca_dim": PCA_DIM,
            "val_pareto_auc": winner_stat["val_pareto_auc"],
            "val_phase2_regret": winner_ns["phase2_regret"],
            "selection_method": "epsilon_constraint",
            "epsilon": EPSILON,
        }

        if variant == "sw_ucb":
            logger.info(
                "\n  SELECTED: alpha=%.3f, W=%d  "
                "BP_AUC=%.6f  P2_Regret=%.1f",
                winner_stat["alpha"], winner_stat["window_size"],
                winner_stat["val_pareto_auc"], winner_ns["phase2_regret"],
            )
        else:
            logger.info(
                "\n  SELECTED: alpha=%.3f, n_eff=%.0f, γ=%.4f  "
                "BP_AUC=%.6f  P2_Regret=%.1f",
                winner_stat["alpha"], winner_stat["n_eff"],
                winner_stat["gamma"],
                winner_stat["val_pareto_auc"], winner_ns["phase2_regret"],
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
            "mean_reward": round(float(test_data["rewards"][arm_id].mean()), 6),
            "mean_cost": round(float(test_data["costs"][arm_id].mean()), 6),
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
        best_ws = per_variant_best[variant]["window_size"]

        if variant == "sw_ucb":
            logger.info(
                "\n  %s (alpha=%.3f, W=%d) on test "
                "[epsilon-constraint] ...",
                variant, best_alpha, best_ws,
            )
        else:
            logger.info(
                "\n  %s (alpha=%.3f, n_eff=%.0f, γ=%.4f) on test "
                "[epsilon-constraint] ...",
                variant, best_alpha, best_n_eff, best_gamma,
            )
        t_test = time.time()
        test_auc, test_std, test_sweep = compute_budget_paced_pareto_auc(
            val_data, test_data, registry, feature_dim,
            budget_targets,
            warmup_path=wp,
            alpha=best_alpha,
            n_eff=best_n_eff,
            gamma=best_gamma,
            n_seeds=N_SEEDS,
            seed_offset=SEED_OFFSET_TEST,
            window_size=best_ws,
        )
        elapsed_test = time.time() - t_test
        test_delta_pct = (test_auc - test_fixed_auc) / test_fixed_auc * 100

        test_results[variant] = {
            "alpha": best_alpha,
            "n_eff": best_n_eff,
            "gamma": best_gamma,
            "window_size": best_ws,
            "selection_method": "epsilon_constraint",
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
    # 7. Save results
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "experiment": "appendix_epsilon_constraint_hparam_selection",
        "protocol": (
            "3-split disjoint protocol: priors from train.jsonl, "
            "val.jsonl split into val_burnin (first 1/3) + val_eval "
            "(remaining 2/3) — burn-in on val_burnin, select on val_eval "
            "(no prompt overlap between burn-in and eval), "
            "report on test.jsonl (burn-in on full val, eval on test). "
            "Epsilon-constraint selection: "
            "(1) budget-paced Pareto AUC as primary metric, "
            "(2) non-stationary Phase-2 regret as secondary. "
            f"All configs within {EPSILON:.0%} of best budget-paced AUC "
            "form the feasible set; among those, lowest Phase-2 regret wins."
        ),
        "grid": {
            "variants": VARIANTS,
            "alpha_values": ALPHA_VALUES,
            "n_eff_values": N_EFF_VALUES,
            "gamma_values": GAMMA_VALUES,
            "window_size_values": WINDOW_SIZE_VALUES,
            "pca_dim": PCA_DIM,
            "budget_targets": [round(t, 10) for t in budget_targets],
            "pacer_lr": PACER_LR,
            "pacer_lambda_max": PACER_LAMBDA_MAX,
            "epsilon": EPSILON,
            "nonstat_cost_penalty": NONSTAT_COST_PENALTY,
            "nonstat_swap_pairs": [
                [ARM_SHORT[a] for a in pair] for pair in NONSTAT_SWAP_PAIRS
            ],
            "n_seeds": N_SEEDS,
            "seed_offset_val": SEED_OFFSET_VAL,
            "seed_offset_nonstat_val": SEED_OFFSET_NONSTAT_VAL,
            "seed_offset_test": SEED_OFFSET_TEST,
        },
        "val_fixed_auc": round(val_fixed_auc, 6),
        "val_baselines": val_fixed_baselines,
        "test_fixed_auc": round(test_fixed_auc, 6),
        "test_baselines": test_fixed_baselines,
        "selection_method": "epsilon_constraint",
        "best_per_variant": per_variant_best,
        "auc_only_best": auc_only_best,
        "test_per_variant": test_results,
        "val_budget_paced": [
            {k: v for k, v in r.items() if k != "sweep_points"}
            for r in val_results
        ],
        "val_budget_paced_full": val_results,
        "val_nonstationary": nonstat_results,
    }

    out_path = RESULTS_DIR / "hparam_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nResults written to %s", out_path)

    best_path = RESULTS_DIR / "best_hparams.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "selection_method": "epsilon_constraint",
                "epsilon": EPSILON,
                "best_per_variant_val": per_variant_best,
                "auc_only_best": auc_only_best,
                "test_per_variant": test_results,
            },
            f,
            indent=2,
        )
    logger.info("Best hparams written to %s", best_path)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (epsilon-constraint, epsilon=%.2f)", EPSILON)
    logger.info("=" * 70)
    for variant in VARIANTS:
        b = per_variant_best[variant]
        t = test_results[variant]
        auc_b = auc_only_best[variant]
        if variant == "sw_ucb":
            logger.info(
                "  %-12s  SELECTED: alpha=%.3f  W=%4d  "
                "val_BP_AUC=%.6f  val_P2_regret=%.1f  "
                "test_BP_AUC=%.6f (Δ=%+.3f%%)",
                variant, b["alpha"], b["window_size"],
                b["val_pareto_auc"], b["val_phase2_regret"],
                t["test_pareto_auc"], t["test_delta_pct"],
            )
            logger.info(
                "  %-12s  AUC-only: alpha=%.3f  W=%4d  "
                "val_BP_AUC=%.6f",
                "", auc_b["alpha"], auc_b["window_size"],
                auc_b["val_pareto_auc"],
            )
        else:
            logger.info(
                "  %-12s  SELECTED: alpha=%.3f  n_eff=%7.0f  γ=%.4f  "
                "val_BP_AUC=%.6f  val_P2_regret=%.1f  "
                "test_BP_AUC=%.6f (Δ=%+.3f%%)",
                variant, b["alpha"], b["n_eff"], b["gamma"],
                b["val_pareto_auc"], b["val_phase2_regret"],
                t["test_pareto_auc"], t["test_delta_pct"],
            )
            logger.info(
                "  %-12s  AUC-only: alpha=%.3f  n_eff=%7.0f  γ=%.4f  "
                "val_BP_AUC=%.6f",
                "", auc_b["alpha"], auc_b["n_eff"], auc_b["gamma"],
                auc_b["val_pareto_auc"],
            )
    logger.info("=" * 70)
    logger.info("Wall time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
