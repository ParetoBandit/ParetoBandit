#!/usr/bin/env python3
"""Figure 1: Cost-Quality Pareto Frontier for BanditGPT (K=2).

Runs the *actual* BanditRouter end-to-end on real prompt data from
the canonical test split, sweeping ``cost_penalty`` to trace the
Pareto frontier.  The static baseline (linear interpolation between
the weak- and strong-model endpoints) is computed from the same data.

Statistical treatment
---------------------
- Each seed runs an independent train→test simulation (different
  shuffle order → different online learning trajectory).
- **Pareto AUC** is computed per-seed first, then averaged — avoiding
  the "phantom frontier" artifact of averaging coordinates before
  building the hull.
- **CostSave@Q** 95 % confidence intervals use a *prompt-level
  paired bootstrap* (n = test-set size, typically ~1 800 prompts).
  Per-prompt bandit outcomes are seed-averaged, then the same
  resampled indices are applied to both the bandit and static
  baselines, preserving covariance.  Both Pareto hulls and the
  strong-arm reference are recomputed each iteration.
  **Caveat:** bandit test-phase outcomes have mild temporal
  dependence from online learning; after >8 K training prompts the
  policy is near-convergent and the dependence is weak, making the
  i.i.d. bootstrap a reasonable approximation (standard practice in
  bandit / RL evaluation).
- All figures report ±1 SE bands (``n_seeds`` trajectories) on the
  frontier and 95 % prompt-bootstrap CIs on CostSave metrics.

Outputs
-------
``results/figure1_pareto_k2.pdf``
    Publication-quality single-panel Pareto figure (Panel A).  Panel B
    (K=3 post-onboarding) will be generated after Experiment 3 runs.

``results/figure1_data.json``
    Machine-readable results for downstream table generation.

Usage
-----
    python experiments/01_figure/plot_pareto_frontier.py
    python experiments/01_figure/plot_pareto_frontier.py --n-seeds 5
    python experiments/01_figure/plot_pareto_frontier.py --fast
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    BEST_K2_HPARAMS,
    HOLDOUT_DATA_PATH,
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.pareto import (
    interpolate_pareto_cost,
    pareto_auc,
    pareto_hull,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

for _noisy in (
    "bandit_gpt.router",
    "bandit_gpt.router_v2",
    "bandit_gpt.feature_service",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"
ARM_ORDER = K2_ARM_ORDER
ARM_LABELS = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
}
COST_PENALTY_SWEEP = [
    0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2,
    0.22, 0.25, 0.28,
    0.3, 0.35, 0.4, 0.45,
    0.5, 0.6, 0.7, 0.8,
    1.0, 2.0, 5.0, 10.0,
]
COSTSAVE_THRESHOLDS = [0.90, 0.95, 0.99]


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SplitData:
    """Pre-processed split with embeddings ready for bandit simulation."""

    prompts: List[str]
    rewards: Dict[str, np.ndarray]
    costs: Dict[str, np.ndarray]
    embeddings: np.ndarray

    @property
    def n(self) -> int:
        return len(self.prompts)


def load_split(path: Path, fs: FeatureService) -> SplitData:
    """Load a JSONL split and encode prompts into feature vectors.

    Args:
        path: Path to a JSONL file where each line contains ``prompt``
            and ``arms`` with per-model ``reward`` and ``cost``.
        fs: Feature service for encoding prompts.

    Returns:
        Fully loaded and embedded split data.
    """
    prompts: List[str] = []
    per_arm_rewards: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_costs: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}

    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["prompt"])
            for arm_id in ARM_ORDER:
                info = r["arms"][arm_id]
                per_arm_rewards[arm_id].append(info["reward"])
                per_arm_costs[arm_id].append(info["cost"])

    rewards = {a: np.array(v) for a, v in per_arm_rewards.items()}
    costs = {a: np.array(v) for a, v in per_arm_costs.items()}

    logger.info("  Encoding %d prompts from %s ...", len(prompts), path.name)
    embeddings = fs.extract_features_batch(prompts)

    return SplitData(
        prompts=prompts, rewards=rewards, costs=costs, embeddings=embeddings,
    )


def build_model_registry() -> Dict[str, Any]:
    """Build model registry filtered to the K=2 arm set."""
    config_path = PROJECT_ROOT / "data_collection" / "config" / "models_k3.json"
    with open(config_path) as f:
        data = json.load(f)
    registry: Dict[str, Any] = {}
    for m in data["models"]:
        if m["model_id"] in ARM_ORDER:
            registry[m["model_id"]] = {
                "model_id": m["model_id"],
                "display_name": m.get("display", m["model_id"]),
                "input_cost_per_m": m["input_cost_per_m"],
                "output_cost_per_m": m["output_cost_per_m"],
            }
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# Bandit Simulation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SimResult:
    """Per-prompt results from a single bandit simulation run.

    Arrays are in *evaluation order* (the shuffled sequence the bandit
    processed).  ``eval_idx`` maps evaluation position ``j`` to the
    original prompt index ``i``, enabling un-shuffling back to the
    canonical prompt order for paired bootstrap resampling.
    """

    rewards: np.ndarray
    costs: np.ndarray
    choices: np.ndarray
    eval_idx: np.ndarray


def simulate_bandit(
    train: SplitData,
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    cost_penalty: float,
    hparams: Dict[str, Any],
    warmup_path: str,
    seed: int,
) -> SimResult:
    """Run a full train→test bandit simulation with the actual router.

    The bandit first processes the training split (updating parameters
    without recording metrics), then processes the test split where
    per-prompt rewards, costs, and arm choices are recorded.  The
    bandit continues learning during the test phase, faithfully
    reflecting online deployment conditions.

    Args:
        train: Training split data.
        test: Test split data.
        registry: Model registry (filtered to K=2 arms).
        feature_dim: Dimensionality of feature vectors.
        cost_penalty: Cost penalty weight for routing.
        hparams: Non-cost hyperparameters (alpha, policy, etc.).
        warmup_path: Path to warmup priors joblib file.
        seed: Random seed controlling data shuffle order.

    Returns:
        Per-prompt reward, cost, and arm-choice arrays for the test
        split.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    is_tabula_rasa = hparams.get("policy") == "tabula_rasa"
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="none" if is_tabula_rasa else "warmup",
        warmup_path=None if is_tabula_rasa else warmup_path,
        prior_n_effective=hparams["prior_n_effective"],
        alpha=hparams["alpha"],
        use_corralling=hparams["use_corralling"],
        cost_penalty=cost_penalty,
        forgetting_factor=hparams["forgetting_factor"],
        policy="disjoint" if is_tabula_rasa else hparams["policy"],
    )

    arm_to_idx = {arm: i for i, arm in enumerate(ARM_ORDER)}

    train_idx = rng.permutation(train.n)
    for i in train_idx:
        emb = train.embeddings[i]
        model, log = router.route(emb)
        reward = float(train.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)

    eval_rewards = np.zeros(test.n)
    eval_costs = np.zeros(test.n)
    eval_choices = np.zeros(test.n, dtype=np.int32)
    eval_idx = rng.permutation(test.n)
    for j, i in enumerate(eval_idx):
        emb = test.embeddings[i]
        model, log = router.route(emb)
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)
        eval_rewards[j] = reward
        eval_costs[j] = cost
        eval_choices[j] = arm_to_idx[model]

    return SimResult(
        rewards=eval_rewards, costs=eval_costs, choices=eval_choices,
        eval_idx=eval_idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Sweep + Pareto Computation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SweepPoint:
    """Aggregated statistics for a single cost_penalty value."""

    cost_penalty: float
    mean_reward: float
    std_reward: float
    mean_cost: float
    std_cost: float
    pct_weak: float
    pct_strong: float
    per_seed_rewards: List[float] = field(default_factory=list)
    per_seed_costs: List[float] = field(default_factory=list)


@dataclass
class FrontierResult:
    """Complete frontier evaluation results.

    Per-prompt arrays (``per_prompt_bandit_*``, ``baseline_per_prompt_*``)
    are stored in the *original* prompt order (matching the JSONL row
    index) so the bootstrap can pair bandit and baseline outcomes for
    the same prompt.
    """

    sweep_points: List[SweepPoint]
    pareto_auc_mean: float
    pareto_auc_std: float
    pareto_auc_per_seed: List[float]
    hull_costs: List[float]
    hull_rewards: List[float]
    baselines: Dict[str, Dict[str, float]]
    static_auc: float
    hparams: Dict[str, Any]
    n_seeds: int
    oracle_reward: float

    per_prompt_bandit_rewards: Optional[Dict[float, np.ndarray]] = None
    """``{cost_penalty: (n_prompts,)}`` seed-averaged rewards, original order."""

    per_prompt_bandit_costs: Optional[Dict[float, np.ndarray]] = None
    """``{cost_penalty: (n_prompts,)}`` seed-averaged costs, original order."""

    baseline_per_prompt_rewards: Optional[Dict[str, np.ndarray]] = None
    """``{arm_id: (n_prompts,)}`` ground-truth per-prompt rewards."""

    baseline_per_prompt_costs: Optional[Dict[str, np.ndarray]] = None
    """``{arm_id: (n_prompts,)}`` ground-truth per-prompt costs."""


def run_frontier_sweep(
    train: SplitData,
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    hparams: Dict[str, Any],
    warmup_path: str,
    cost_penalties: List[float],
    n_seeds: int,
    seed_offset: int = 1000,
) -> FrontierResult:
    """Sweep cost_penalty and build the bandit Pareto frontier.

    For each seed independently:
      1. Run simulation for every cost_penalty value.
      2. Build the Pareto hull from sweep points + fixed-model endpoints.
      3. Compute that seed's Pareto AUC.

    Per-prompt results are un-shuffled to canonical prompt order and
    averaged across seeds, enabling prompt-level paired bootstrap in
    :func:`compute_costsave_with_bootstrap`.

    Args:
        train: Training data (bandit learns, not evaluated).
        test: Test data (bandit evaluated, continues learning).
        registry: K=2 model registry.
        feature_dim: Feature vector dimensionality.
        hparams: Best hyperparameters (non-cost).
        warmup_path: Path to warmup priors.
        cost_penalties: Grid of cost_penalty values to sweep.
        n_seeds: Number of independent random seeds.
        seed_offset: Base offset added to seed index.

    Returns:
        Comprehensive frontier results with per-seed statistics and
        per-prompt arrays for bootstrap CIs.
    """
    n_arms = len(ARM_ORDER)
    fixed_costs = [float(test.costs[a].mean()) for a in ARM_ORDER]
    fixed_rewards = [float(test.rewards[a].mean()) for a in ARM_ORDER]
    cost_lo = min(fixed_costs)
    cost_hi = max(fixed_costs)

    oracle_per_prompt = np.maximum.reduce(
        [test.rewards[a] for a in ARM_ORDER]
    )
    oracle_reward = float(oracle_per_prompt.mean())

    baselines = {}
    for arm_id in ARM_ORDER:
        label = ARM_LABELS[arm_id]
        baselines[label] = {
            "mean_reward": float(test.rewards[arm_id].mean()),
            "mean_cost": float(test.costs[arm_id].mean()),
        }

    static_auc = pareto_auc(fixed_costs, fixed_rewards, cost_lo, cost_hi)

    cp_rewards: Dict[float, List[float]] = {cp: [] for cp in cost_penalties}
    cp_costs: Dict[float, List[float]] = {cp: [] for cp in cost_penalties}
    cp_choices: Dict[float, List[np.ndarray]] = {cp: [] for cp in cost_penalties}

    pp_rewards_accum: Dict[float, List[np.ndarray]] = {cp: [] for cp in cost_penalties}
    pp_costs_accum: Dict[float, List[np.ndarray]] = {cp: [] for cp in cost_penalties}

    per_seed_auc: List[float] = []

    for s in range(n_seeds):
        seed = seed_offset + s
        seed_costs_list: List[float] = []
        seed_rewards_list: List[float] = []

        logger.info("  Seed %d/%d (seed=%d)", s + 1, n_seeds, seed)
        for cp in cost_penalties:
            result = simulate_bandit(
                train, test, registry, feature_dim,
                cost_penalty=cp,
                hparams=hparams,
                warmup_path=warmup_path,
                seed=seed,
            )
            mr = float(result.rewards.mean())
            mc = float(result.costs.mean())
            seed_costs_list.append(mc)
            seed_rewards_list.append(mr)

            cp_rewards[cp].append(mr)
            cp_costs[cp].append(mc)
            arm_counts = np.bincount(result.choices, minlength=n_arms)
            cp_choices[cp].append(arm_counts)

            prompt_order_r = np.empty(test.n)
            prompt_order_c = np.empty(test.n)
            prompt_order_r[result.eval_idx] = result.rewards
            prompt_order_c[result.eval_idx] = result.costs
            pp_rewards_accum[cp].append(prompt_order_r)
            pp_costs_accum[cp].append(prompt_order_c)

        all_c = seed_costs_list + fixed_costs
        all_r = seed_rewards_list + fixed_rewards
        seed_auc = pareto_auc(all_c, all_r, cost_lo, cost_hi)
        per_seed_auc.append(seed_auc)
        logger.info("    Seed %d AUC: %.6f (static: %.6f, Δ=%+.3f%%)",
                     s + 1, seed_auc, static_auc,
                     (seed_auc - static_auc) / static_auc * 100)

    sweep_points: List[SweepPoint] = []
    for cp in cost_penalties:
        rewards_arr = np.array(cp_rewards[cp])
        costs_arr = np.array(cp_costs[cp])
        choices_arr = np.array([c.astype(float) for c in cp_choices[cp]])
        mean_choices = choices_arr.mean(axis=0)
        total = mean_choices.sum()
        sweep_points.append(SweepPoint(
            cost_penalty=cp,
            mean_reward=float(rewards_arr.mean()),
            std_reward=float(rewards_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
            mean_cost=float(costs_arr.mean()),
            std_cost=float(costs_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
            pct_weak=float(mean_choices[0] / total) if total > 0 else 0.0,
            pct_strong=float(mean_choices[-1] / total) if total > 0 else 0.0,
            per_seed_rewards=cp_rewards[cp],
            per_seed_costs=cp_costs[cp],
        ))

    avg_costs = [sp.mean_cost for sp in sweep_points] + fixed_costs
    avg_rewards = [sp.mean_reward for sp in sweep_points] + fixed_rewards
    hull_c, hull_r = pareto_hull(avg_costs, avg_rewards)

    auc_mean = float(np.mean(per_seed_auc))
    auc_std = float(np.std(per_seed_auc, ddof=1)) if n_seeds > 1 else 0.0

    pp_bandit_rewards = {
        cp: np.mean(np.stack(arrs), axis=0)
        for cp, arrs in pp_rewards_accum.items()
    }
    pp_bandit_costs = {
        cp: np.mean(np.stack(arrs), axis=0)
        for cp, arrs in pp_costs_accum.items()
    }

    return FrontierResult(
        sweep_points=sweep_points,
        pareto_auc_mean=auc_mean,
        pareto_auc_std=auc_std,
        pareto_auc_per_seed=per_seed_auc,
        hull_costs=hull_c,
        hull_rewards=hull_r,
        baselines=baselines,
        static_auc=static_auc,
        hparams=hparams,
        n_seeds=n_seeds,
        oracle_reward=oracle_reward,
        per_prompt_bandit_rewards=pp_bandit_rewards,
        per_prompt_bandit_costs=pp_bandit_costs,
        baseline_per_prompt_rewards={a: test.rewards[a] for a in ARM_ORDER},
        baseline_per_prompt_costs={a: test.costs[a] for a in ARM_ORDER},
    )


# ═══════════════════════════════════════════════════════════════════════════
# CostSave@Q with Bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CostSaveResult:
    """CostSave at a single quality threshold with uncertainty."""

    threshold: float
    target_reward: float
    bandit_cost: Optional[float]
    bandit_saving_pct: Optional[float]
    baseline_cost: Optional[float]
    baseline_saving_pct: Optional[float]
    advantage_pp: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]


def _costsave_at_threshold(
    hull_c: List[float],
    hull_r: List[float],
    strong_cost: float,
    strong_reward: float,
    q: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute cost and saving percentage at quality threshold q.

    Returns:
        ``(cost_at_q, saving_pct)`` or ``(None, None)`` if the target
        reward falls outside the hull's range.
    """
    target_r = q * strong_reward
    cost_at_q = interpolate_pareto_cost(hull_c, hull_r, target_r)
    if cost_at_q is not None:
        saving = (1.0 - cost_at_q / strong_cost) * 100
        return cost_at_q, saving
    return None, None


def compute_costsave_with_bootstrap(
    frontier: FrontierResult,
    thresholds: List[float],
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
) -> List[CostSaveResult]:
    """Compute CostSave@Q with prompt-level paired bootstrap CIs.

    For each bootstrap iteration the same resampled prompt indices
    are used for *both* the bandit sweep points and the fixed-model
    baselines (paired bootstrap), then the Pareto hulls and CostSave
    are recomputed.  This captures test-set sampling uncertainty —
    variability attributable to the finite draw of evaluation prompts
    from the task distribution.

    The per-prompt bandit arrays are seed-averaged (computed in
    :func:`run_frontier_sweep`), so each element represents the
    expected bandit outcome for that prompt across trajectories.
    Bandit outcomes have mild temporal dependence from online
    learning; after >8 K training prompts the policy is
    near-convergent and the dependence is weak.

    Args:
        frontier: Results from ``run_frontier_sweep`` (must contain
            per-prompt arrays).
        thresholds: Quality fractions (e.g. [0.90, 0.95, 0.99]).
        n_bootstrap: Number of bootstrap resamples.
        bootstrap_seed: RNG seed for reproducibility.

    Returns:
        One ``CostSaveResult`` per threshold.

    Raises:
        ValueError: If ``frontier`` lacks per-prompt data.
    """
    if frontier.per_prompt_bandit_rewards is None:
        raise ValueError(
            "FrontierResult missing per-prompt data — "
            "re-run run_frontier_sweep with the latest code."
        )

    rng = np.random.default_rng(bootstrap_seed)

    pp_rewards = frontier.per_prompt_bandit_rewards
    pp_costs = frontier.per_prompt_bandit_costs
    bl_pp_rewards = frontier.baseline_per_prompt_rewards
    bl_pp_costs = frontier.baseline_per_prompt_costs
    n_prompts = len(next(iter(pp_rewards.values())))

    strong_arm = max(
        ARM_ORDER,
        key=lambda a: frontier.baselines[ARM_LABELS[a]]["mean_reward"],
    )
    strong_label = ARM_LABELS[strong_arm]
    strong_cost = frontier.baselines[strong_label]["mean_cost"]
    strong_reward = frontier.baselines[strong_label]["mean_reward"]

    fixed_costs = [
        frontier.baselines[ARM_LABELS[a]]["mean_cost"] for a in ARM_ORDER
    ]
    fixed_rewards = [
        frontier.baselines[ARM_LABELS[a]]["mean_reward"] for a in ARM_ORDER
    ]

    bandit_hull_c, bandit_hull_r = pareto_hull(
        [sp.mean_cost for sp in frontier.sweep_points] + fixed_costs,
        [sp.mean_reward for sp in frontier.sweep_points] + fixed_rewards,
    )
    baseline_hull_c, baseline_hull_r = pareto_hull(fixed_costs, fixed_rewards)

    observed: Dict[float, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for q in thresholds:
        _, b_sav = _costsave_at_threshold(
            bandit_hull_c, bandit_hull_r, strong_cost, strong_reward, q,
        )
        _, bl_sav = _costsave_at_threshold(
            baseline_hull_c, baseline_hull_r, strong_cost, strong_reward, q,
        )
        adv = None
        if b_sav is not None and bl_sav is not None:
            adv = b_sav - bl_sav
        observed[q] = (b_sav, bl_sav, adv)

    boot_advantages: Dict[float, List[float]] = {q: [] for q in thresholds}
    cost_penalties = [sp.cost_penalty for sp in frontier.sweep_points]

    for _ in range(n_bootstrap):
        idx = rng.choice(n_prompts, size=n_prompts, replace=True)

        boot_sweep_costs: List[float] = []
        boot_sweep_rewards: List[float] = []
        for cp in cost_penalties:
            boot_sweep_rewards.append(float(pp_rewards[cp][idx].mean()))
            boot_sweep_costs.append(float(pp_costs[cp][idx].mean()))

        boot_fixed_costs = [
            float(bl_pp_costs[a][idx].mean()) for a in ARM_ORDER
        ]
        boot_fixed_rewards = [
            float(bl_pp_rewards[a][idx].mean()) for a in ARM_ORDER
        ]

        boot_strong_idx = int(np.argmax(boot_fixed_rewards))
        boot_strong_cost = boot_fixed_costs[boot_strong_idx]
        boot_strong_reward = boot_fixed_rewards[boot_strong_idx]

        boot_bandit_hull_c, boot_bandit_hull_r = pareto_hull(
            boot_sweep_costs + boot_fixed_costs,
            boot_sweep_rewards + boot_fixed_rewards,
        )
        boot_baseline_hull_c, boot_baseline_hull_r = pareto_hull(
            boot_fixed_costs, boot_fixed_rewards,
        )

        for q in thresholds:
            _, b_sav = _costsave_at_threshold(
                boot_bandit_hull_c, boot_bandit_hull_r,
                boot_strong_cost, boot_strong_reward, q,
            )
            _, bl_sav = _costsave_at_threshold(
                boot_baseline_hull_c, boot_baseline_hull_r,
                boot_strong_cost, boot_strong_reward, q,
            )
            if b_sav is not None and bl_sav is not None:
                boot_advantages[q].append(b_sav - bl_sav)

    results: List[CostSaveResult] = []
    for q in thresholds:
        b_sav, bl_sav, adv = observed[q]
        target_r = q * strong_reward

        b_cost, _ = _costsave_at_threshold(
            bandit_hull_c, bandit_hull_r, strong_cost, strong_reward, q,
        )
        bl_cost, _ = _costsave_at_threshold(
            baseline_hull_c, baseline_hull_r, strong_cost, strong_reward, q,
        )

        ci_lo: Optional[float] = None
        ci_hi: Optional[float] = None
        boot_arr = boot_advantages[q]
        if len(boot_arr) >= 20:
            ci_lo = float(np.percentile(boot_arr, 2.5))
            ci_hi = float(np.percentile(boot_arr, 97.5))

        results.append(CostSaveResult(
            threshold=q,
            target_reward=round(target_r, 4),
            bandit_cost=b_cost,
            bandit_saving_pct=round(b_sav, 2) if b_sav is not None else None,
            baseline_cost=bl_cost,
            baseline_saving_pct=round(bl_sav, 2) if bl_sav is not None else None,
            advantage_pp=round(adv, 2) if adv is not None else None,
            ci_lower=round(ci_lo, 2) if ci_lo is not None else None,
            ci_upper=round(ci_hi, 2) if ci_hi is not None else None,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Gap@Oracle
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GapAtOracleResult:
    """Gap between a router's quality and the per-instance oracle.

    Gap@Oracle := (R_oracle - R_router) / (R_oracle - R_weak) × 100,
    yielding a normalised percentage in [0, 100] where 0 = oracle-optimal
    and 100 = always-weak.  This metric is comparable across datasets
    because it is anchored to the portfolio's ceiling and floor.
    """

    cost_penalty: float
    router_reward: float
    oracle_reward: float
    weak_reward: float
    gap_pct: float


def compute_gap_at_oracle(
    frontier: FrontierResult,
) -> List[GapAtOracleResult]:
    """Compute normalised Gap@Oracle for every sweep operating point.

    Args:
        frontier: Results from ``run_frontier_sweep``.

    Returns:
        One ``GapAtOracleResult`` per sweep point, sorted by ascending
        cost penalty.
    """
    oracle_r = frontier.oracle_reward
    weak_r = min(
        frontier.baselines[ARM_LABELS[a]]["mean_reward"] for a in ARM_ORDER
    )
    quality_range = oracle_r - weak_r

    results: List[GapAtOracleResult] = []
    for sp in frontier.sweep_points:
        gap_pct = (
            (oracle_r - sp.mean_reward) / quality_range * 100
            if quality_range > 0 else 0.0
        )
        results.append(GapAtOracleResult(
            cost_penalty=sp.cost_penalty,
            router_reward=sp.mean_reward,
            oracle_reward=oracle_r,
            weak_reward=weak_r,
            gap_pct=round(gap_pct, 2),
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GRAY = "#999999"
CB_RED = "#D55E00"
CB_GREEN = "#009E73"

MARKER_STRONG = "D"
MARKER_WEAK = "o"


def _dollar_formatter(x: float, _pos: Any) -> str:
    """Format tick labels as dollar amounts."""
    if x < 0.001:
        return f"${x:.1e}"
    return f"${x:.4f}"


def plot_pareto_panel_a(
    frontier: FrontierResult,
    costsave_results: List[CostSaveResult],
    out_dir: Path,
    gap_results: Optional[List[GapAtOracleResult]] = None,
) -> Path:
    """Generate the K=2 Pareto frontier figure (Panel A).

    Design decisions for KDD publication:
    - Colorblind-safe palette (Tol / Wong).
    - Minimal chartjunk: no background shading, no 3D effects.
    - Confidence bands use ±1 SE (not ±1 SD) to represent
      uncertainty in the *mean* frontier, not individual runs.
    - CostSave annotations use arrows from the static baseline
      to the bandit frontier at matched quality levels, with
      exact numerical annotations.
    - X-axis is linear (not log) so the visual area between
      curves is proportional to the actual cost saving.

    Args:
        frontier: Sweep results from ``run_frontier_sweep``.
        costsave_results: CostSave@Q metrics with CIs.
        out_dir: Directory for output files.

    Returns:
        Path to the saved figure.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)

    # ── Static baseline: linear interpolation between endpoints ───────
    weak_label = ARM_LABELS[ARM_ORDER[0]]
    strong_label = ARM_LABELS[ARM_ORDER[-1]]
    weak_c = frontier.baselines[weak_label]["mean_cost"]
    weak_r = frontier.baselines[weak_label]["mean_reward"]
    strong_c = frontier.baselines[strong_label]["mean_cost"]
    strong_r = frontier.baselines[strong_label]["mean_reward"]

    ax.plot(
        [weak_c, strong_c], [weak_r, strong_r],
        "--", color=CB_GRAY, lw=2.0, zorder=3,
        label="Static random mix",
    )

    # ── Fixed-model endpoints ─────────────────────────────────────────
    ax.scatter(
        [weak_c], [weak_r],
        marker=MARKER_WEAK, c="white", edgecolors=CB_GRAY, s=80,
        linewidths=1.5, zorder=7,
    )
    ax.annotate(
        weak_label, (weak_c, weak_r),
        textcoords="offset points", xytext=(8, -2),
        fontsize=8, color=CB_GRAY, fontstyle="italic",
    )

    ax.scatter(
        [strong_c], [strong_r],
        marker=MARKER_STRONG, c="white", edgecolors=CB_GRAY, s=80,
        linewidths=1.5, zorder=7,
    )
    ax.annotate(
        strong_label, (strong_c, strong_r),
        textcoords="offset points", xytext=(0, 10),
        fontsize=8, color=CB_GRAY, fontstyle="italic",
        ha="center",
    )

    # ── Bandit frontier ───────────────────────────────────────────────
    hull_c = np.array(frontier.hull_costs)
    hull_r = np.array(frontier.hull_rewards)

    interior_mask = []
    for c, r in zip(hull_c, hull_r):
        is_fixed = any(
            abs(c - frontier.baselines[ARM_LABELS[a]]["mean_cost"]) < 1e-12
            and abs(r - frontier.baselines[ARM_LABELS[a]]["mean_reward"]) < 1e-12
            for a in ARM_ORDER
        )
        interior_mask.append(not is_fixed)
    interior_mask = np.array(interior_mask)

    ax.plot(
        hull_c, hull_r, "-",
        color=CB_BLUE, lw=2.2, zorder=5,
        label="BanditGPT frontier",
    )
    if np.any(interior_mask):
        ax.scatter(
            hull_c[interior_mask], hull_r[interior_mask],
            marker="o", c=CB_BLUE, s=30, zorder=6,
            edgecolors="white", linewidths=0.5,
        )

    # ── ±1 SE band from per-seed variance ─────────────────────────────
    n_seeds = frontier.n_seeds
    if n_seeds > 1:
        band_c: List[float] = []
        band_upper: List[float] = []
        band_lower: List[float] = []
        for sp in sorted(frontier.sweep_points, key=lambda s: s.mean_cost):
            se = sp.std_reward / np.sqrt(n_seeds)
            band_c.append(sp.mean_cost)
            band_upper.append(sp.mean_reward + se)
            band_lower.append(sp.mean_reward - se)

        ax.fill_between(
            band_c, band_lower, band_upper,
            color=CB_BLUE, alpha=0.12, zorder=2,
            label=f"±1 SE ({n_seeds} seeds)",
        )

    # ── Shaded contextual routing gain ────────────────────────────────
    overlap_lo = max(weak_c, hull_c[0])
    overlap_hi = min(strong_c, hull_c[-1])
    if overlap_lo < overlap_hi:
        fill_x = np.linspace(overlap_lo, overlap_hi, 300)
        fill_bandit = np.interp(fill_x, hull_c, hull_r)
        fill_static = np.interp(fill_x, [weak_c, strong_c], [weak_r, strong_r])
        ax.fill_between(
            fill_x, fill_static, fill_bandit,
            where=fill_bandit > fill_static,
            color=CB_BLUE, alpha=0.06, zorder=1,
        )

    # ── CostSave@Q annotations ────────────────────────────────────────
    q_colors = {0.90: CB_GREEN, 0.95: CB_ORANGE, 0.99: CB_RED}
    annotation_offsets = {0.90: (40, -18), 0.95: (-160, -12), 0.99: (-130, 18)}

    for cs in costsave_results:
        q = cs.threshold
        if cs.bandit_cost is None or cs.baseline_cost is None:
            continue

        target_r = cs.target_reward
        color = q_colors.get(q, CB_GRAY)

        ax.plot(
            [cs.baseline_cost, cs.bandit_cost], [target_r, target_r],
            "-", color=color, lw=1.5, alpha=0.7, zorder=4,
        )
        ax.scatter(
            [cs.bandit_cost], [target_r],
            marker="*", c=color, s=120, zorder=8, edgecolors="white",
            linewidths=0.3,
        )

        offset = annotation_offsets.get(q, (15, 0))
        ci_str = ""
        if cs.ci_lower is not None and cs.ci_upper is not None:
            ci_str = f"\n95% CI [{cs.ci_lower:+.1f}, {cs.ci_upper:+.1f}] pp"
        ax.annotate(
            f"CostSave@{q:.0%}: {cs.bandit_saving_pct:.1f}%\n"
            f"(Δ={cs.advantage_pp:+.1f} pp vs static"
            f" {cs.baseline_saving_pct:.1f}%){ci_str}",
            xy=(cs.bandit_cost, target_r),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            color=color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                edgecolor=color, alpha=0.85, linewidth=0.5,
            ),
            arrowprops=dict(
                arrowstyle="-", color=color, lw=0.8, ls=":",
            ),
            zorder=9,
        )

    # ── Oracle reference line ─────────────────────────────────────────
    oracle_r = frontier.oracle_reward
    ax.axhline(
        oracle_r, color=CB_RED, lw=1.0, ls=":", alpha=0.5, zorder=2,
    )
    ax.annotate(
        f"Oracle ({oracle_r:.3f})",
        xy=(strong_c * 0.40, oracle_r),
        xytext=(0, 6), textcoords="offset points",
        fontsize=7, color=CB_RED, fontstyle="italic", alpha=0.7,
    )

    # ── Gap@Oracle annotation at λ=0 ──────────────────────────────────
    if gap_results:
        gap_at_zero = next(
            (g for g in gap_results if g.cost_penalty == 0.0), None,
        )
        if gap_at_zero is not None:
            gap_text = (
                f"Gap@Oracle (λ=0): {gap_at_zero.gap_pct:.1f}%"
            )
        else:
            gap_text = ""
    else:
        gap_text = ""

    # ── Pareto AUC annotation (small, non-prominent) ──────────────────
    auc_text = (
        f"Pareto AUC: {frontier.pareto_auc_mean:.4f} "
        f"(static: {frontier.static_auc:.4f}, "
        f"Δ={((frontier.pareto_auc_mean - frontier.static_auc) / frontier.static_auc * 100):+.2f}%)"
    )
    combined_text = auc_text
    if gap_text:
        combined_text = f"{auc_text}    |    {gap_text}"
    ax.text(
        0.02, 0.02, combined_text,
        transform=ax.transAxes, fontsize=6.5, color="#666666",
        verticalalignment="bottom",
    )

    # ── Axes and legend ───────────────────────────────────────────────
    ax.set_xlabel("Average Cost per Request ($)")
    ax.set_ylabel("Average Reward (Quality)")
    ax.set_title(
        "Cost\u2013Quality Pareto Frontier: BanditGPT (K=2, Online, No Routing Labels)",
        fontsize=11, fontweight="bold",
    )

    ax.xaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
    ax.grid(True, alpha=0.15, ls="--")
    ax.legend(loc="lower right", framealpha=0.92)

    y_lo = weak_r - 0.02
    y_hi = strong_r + 0.035
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-strong_c * 0.02, strong_c * 1.14)

    out_path = out_dir / "figure1_pareto_k2.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        out_path.with_suffix(".png"), dpi=300,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# JSON Export
# ═══════════════════════════════════════════════════════════════════════════


def export_results(
    frontier: FrontierResult,
    costsave_results: List[CostSaveResult],
    elapsed_s: float,
    out_dir: Path,
    gap_results: Optional[List[GapAtOracleResult]] = None,
) -> Path:
    """Write machine-readable results for downstream consumption.

    Args:
        frontier: Frontier sweep results.
        costsave_results: CostSave metrics with CIs.
        elapsed_s: Total wall-clock time in seconds.
        out_dir: Output directory.
        gap_results: Gap@Oracle metrics per sweep point.

    Returns:
        Path to the saved JSON file.
    """
    data = {
        "experiment": "Figure 1 — Cost-Quality Pareto Frontier (K=2)",
        "hparams": frontier.hparams,
        "n_seeds": frontier.n_seeds,
        "oracle_reward": round(frontier.oracle_reward, 6),
        "pareto_auc": {
            "bandit_mean": round(frontier.pareto_auc_mean, 6),
            "bandit_std": round(frontier.pareto_auc_std, 6),
            "bandit_per_seed": [round(a, 6) for a in frontier.pareto_auc_per_seed],
            "static": round(frontier.static_auc, 6),
            "delta_pct": round(
                (frontier.pareto_auc_mean - frontier.static_auc)
                / frontier.static_auc * 100, 3,
            ),
        },
        "costsave": [
            {
                "threshold": f"{cs.threshold:.0%}",
                "target_reward": cs.target_reward,
                "bandit_saving_pct": cs.bandit_saving_pct,
                "baseline_saving_pct": cs.baseline_saving_pct,
                "advantage_pp": cs.advantage_pp,
                "ci_95_lower_pp": cs.ci_lower,
                "ci_95_upper_pp": cs.ci_upper,
            }
            for cs in costsave_results
        ],
        "gap_at_oracle": [
            {
                "cost_penalty": g.cost_penalty,
                "router_reward": round(g.router_reward, 6),
                "gap_pct": g.gap_pct,
            }
            for g in (gap_results or [])
        ],
        "sweep_points": [
            {
                "cost_penalty": sp.cost_penalty,
                "mean_reward": round(sp.mean_reward, 6),
                "std_reward": round(sp.std_reward, 6),
                "mean_cost": round(sp.mean_cost, 8),
                "std_cost": round(sp.std_cost, 8),
                "pct_weak": round(sp.pct_weak * 100, 1),
                "pct_strong": round(sp.pct_strong * 100, 1),
            }
            for sp in frontier.sweep_points
        ],
        "pareto_hull": {
            "costs": [round(c, 8) for c in frontier.hull_costs],
            "rewards": [round(r, 6) for r in frontier.hull_rewards],
        },
        "baselines": frontier.baselines,
        "arm_order": ARM_ORDER,
        "elapsed_s": round(elapsed_s, 1),
    }

    out_path = out_dir / "figure1_data.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Console Summary
# ═══════════════════════════════════════════════════════════════════════════


def print_summary(
    frontier: FrontierResult,
    costsave_results: List[CostSaveResult],
    gap_results: Optional[List[GapAtOracleResult]] = None,
) -> None:
    """Print a concise, reviewer-friendly summary to stdout."""
    print("\n" + "=" * 72)
    print("FIGURE 1 — COST-QUALITY PARETO FRONTIER (K=2)")
    print("=" * 72)

    print(f"\nConfig: {json.dumps(frontier.hparams)}")
    print(f"Seeds:  {frontier.n_seeds} (offset=1000)")
    print(f"Arms:   {', '.join(ARM_LABELS[a] for a in ARM_ORDER)}")

    print(f"\n{'Model':<20s}  {'Reward':>8s}  {'Cost':>12s}")
    print("-" * 44)
    for arm_id in ARM_ORDER:
        label = ARM_LABELS[arm_id]
        b = frontier.baselines[label]
        print(f"{label:<20s}  {b['mean_reward']:8.4f}  ${b['mean_cost']:11.8f}")
    print(f"{'Oracle (per-prompt)':<20s}  {frontier.oracle_reward:8.4f}")

    print(f"\nPareto AUC")
    print(f"  Bandit:  {frontier.pareto_auc_mean:.6f} ± {frontier.pareto_auc_std:.6f}")
    print(f"  Static:  {frontier.static_auc:.6f}")
    delta = (frontier.pareto_auc_mean - frontier.static_auc) / frontier.static_auc * 100
    print(f"  Delta:   {delta:+.3f}%")

    if gap_results:
        print(f"\nGap@Oracle (normalised: 0% = oracle, 100% = always-weak)")
        print(f"  {'λ':<6s}  {'Reward':>8s}  {'Gap':>8s}")
        print("  " + "-" * 26)
        for g in gap_results:
            if g.cost_penalty in (0.0, 0.05, 0.1, 0.2, 0.5):
                print(f"  {g.cost_penalty:<6.2f}  {g.router_reward:8.4f}  {g.gap_pct:7.1f}%")

    n_prompts_str = ""
    if frontier.per_prompt_bandit_rewards is not None:
        n_pp = len(next(iter(frontier.per_prompt_bandit_rewards.values())))
        n_prompts_str = f" (prompt-level, n={n_pp})"
    print(f"\nCostSave{n_prompts_str}")
    print(f"{'Threshold':<12s}  {'Bandit':>8s}  {'Static':>8s}  {'Advantage':>10s}  {'95% CI':>18s}")
    print("-" * 62)
    for cs in costsave_results:
        b_str = f"{cs.bandit_saving_pct:.1f}%" if cs.bandit_saving_pct is not None else "N/A"
        bl_str = f"{cs.baseline_saving_pct:.1f}%" if cs.baseline_saving_pct is not None else "N/A"
        adv_str = f"{cs.advantage_pp:+.1f} pp" if cs.advantage_pp is not None else "N/A"
        if cs.ci_lower is not None and cs.ci_upper is not None:
            ci_str = f"[{cs.ci_lower:+.1f}, {cs.ci_upper:+.1f}] pp"
        else:
            ci_str = "N/A"
        print(f"  @{cs.threshold:.0%}       {b_str:>8s}  {bl_str:>8s}  {adv_str:>10s}  {ci_str:>18s}")

    print(f"\nSweep Detail")
    print(f"  {'cp':<6s}  {'reward':>8s}  {'cost':>12s}  {'%weak':>6s}  {'%strong':>8s}")
    print("  " + "-" * 46)
    for sp in frontier.sweep_points:
        print(
            f"  {sp.cost_penalty:<6.2f}  {sp.mean_reward:8.4f}  "
            f"${sp.mean_cost:11.8f}  {sp.pct_weak*100:5.1f}%  {sp.pct_strong*100:7.1f}%"
        )

    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of independent random seeds (default: 5)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=2000,
        help="Number of bootstrap resamples for CostSave CIs (default: 2000)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Reduced sweep for quick testing (3 seeds, 5 cost penalties)",
    )
    args = parser.parse_args()

    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_seeds = 3 if args.fast else args.n_seeds
    cost_penalties = (
        [0.0, 0.1, 0.5, 2.0, 10.0] if args.fast else COST_PENALTY_SWEEP
    )
    n_bootstrap = 200 if args.fast else args.n_bootstrap

    hparams = dict(BEST_K2_HPARAMS)
    warmup_path = str(K2_WARMUP_PRIORS_PATH)

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading data and encoding prompts ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    logger.info("  Feature dim: %d", feature_dim)

    train = load_split(TRAIN_DATA_PATH, fs)
    test = load_split(HOLDOUT_DATA_PATH, fs)
    logger.info("  Train: %d prompts, Test: %d prompts", train.n, test.n)

    registry = build_model_registry()
    logger.info("  Registry: %s", list(registry.keys()))

    # ── Run frontier sweep ────────────────────────────────────────────
    logger.info(
        "\nRunning frontier sweep: %d cost_penalties × %d seeds = %d simulations",
        len(cost_penalties), n_seeds, len(cost_penalties) * n_seeds,
    )
    frontier = run_frontier_sweep(
        train, test, registry, feature_dim,
        hparams=hparams,
        warmup_path=warmup_path,
        cost_penalties=cost_penalties,
        n_seeds=n_seeds,
        seed_offset=1000,
    )

    # ── CostSave@Q with bootstrap CIs ────────────────────────────────
    logger.info("\nComputing CostSave@Q with %d bootstrap resamples ...", n_bootstrap)
    costsave_results = compute_costsave_with_bootstrap(
        frontier, COSTSAVE_THRESHOLDS,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=42,
    )

    # ── Gap@Oracle ────────────────────────────────────────────────────
    logger.info("\nComputing Gap@Oracle ...")
    gap_results = compute_gap_at_oracle(frontier)

    # ── Generate outputs ──────────────────────────────────────────────
    elapsed = time.time() - t0

    fig_path = plot_pareto_panel_a(
        frontier, costsave_results, RESULTS_DIR, gap_results=gap_results,
    )
    logger.info("Figure saved to %s", fig_path)

    json_path = export_results(
        frontier, costsave_results, elapsed, RESULTS_DIR,
        gap_results=gap_results,
    )
    logger.info("Data saved to %s", json_path)

    print_summary(frontier, costsave_results, gap_results=gap_results)

    logger.info("\nTotal wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
