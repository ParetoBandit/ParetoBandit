#!/usr/bin/env python3
"""Experiment 03: Budget Pacing Under Cost Drift.

Demonstrates that the BudgetPacer automatically exploits a mid-stream
model pricing change to improve routing quality while maintaining budget
compliance --- the most production-relevant scenario for cost-constrained
LLM routing.

Experimental setup
------------------
The router is deployed on a two-phase data stream constructed from
real benchmark data (K=3: Llama-8B, Mistral-Large, Gemini-Pro):

  **Phase 1** (steps 1--893): Normal pricing.  Gemini-Pro is expensive
  (normalized cost 0.67); the BudgetPacer enforces the dollar budget
  target by raising lambda_t, which suppresses Gemini selection.

  **Phase 2** (steps 894--1785): **Gemini price drop** — pricing falls
  to $0.10/$0.10 per million tokens (normalized cost ~0.0).  The router
  registry is updated at the boundary.  The cost EMA should decline,
  driving lambda_t downward and allowing Gemini routing.

Three budget targets span the constraint regime:

  - **Tight**    ($2.3 × 10⁻⁴ $/req): lambda high, Llama-heavy Phase 1
  - **Moderate** ($6.6 × 10⁻⁴ $/req): mixed routing Phase 1
  - **Loose**    ($1.9 × 10⁻³ $/req): light constraint Phase 1

Per target, three conditions are compared, representing increasing
levels of routing sophistication:

  1. **Fixed Policy (offline)** — Warmup priors with a matched static
     cost penalty but no online learning.  The dominant production
     pattern: train offline, deploy, never update.  Helpless under drift.
  2. **Naive Bandit** — LinUCB with warmup priors, infinite memory
     (γ=1.0), and a matched static cost penalty.  The obvious first
     attempt at online routing — adapts, but Phase 1 inertia dilutes
     Phase 2 signal, and has no principled budget mechanism.
  3. **BanditGPT** — Warmup priors + geometric forgetting (γ=0.997) +
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
from typing import Any, Dict, List, Optional, Tuple

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
from utils.simulation import SplitData, build_model_registry, compute_normalized_costs

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
ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

GEMINI_ID: str = "google/gemini-2.5-pro"
GEMINI_NEW_INPUT_COST: float = 0.10
GEMINI_NEW_OUTPUT_COST: float = 0.10

N_SEEDS: int = 20
SEED_OFFSET: int = 7000
RESULTS_DIR = Path(__file__).parent / "results"

PHASE1_N: int = 893
PHASE2_N: int = 892
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
    oracle_utility: float
    chosen_utility: float
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
        regrets = [s.oracle_utility - s.chosen_utility for s in phase_steps]
        arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
        for s in phase_steps:
            arm_counts[s.model] += 1
        n = len(phase_steps)
        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_cost": float(np.mean(costs)),
            "cumulative_regret": float(np.sum(regrets)),
            "arm_fractions": {a: cnt / n for a, cnt in arm_counts.items()},
            "mean_lambda": float(np.mean([s.lambda_t for s in phase_steps])),
            "mean_cost_ema": float(np.mean([s.cost_ema for s in phase_steps])),
            "n_steps": n,
        }

    def total_regret(self) -> float:
        return sum(s.oracle_utility - s.chosen_utility for s in self.steps)


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


def _build_phase2_registry(
    registry: Dict[str, Any],
    gemini_id: str,
    new_input: float,
    new_output: float,
) -> Dict[str, Any]:
    """Return a deep copy of the registry with Gemini's pricing updated."""
    new_reg = copy.deepcopy(registry)
    new_reg[gemini_id]["input_cost_per_m"] = new_input
    new_reg[gemini_id]["output_cost_per_m"] = new_output
    return new_reg


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
        drift_threshold=0.0,
        budget_pacer=budget_pacer,
    )


# ======================================================================
# Two-Phase Learning Curve
# ======================================================================


def _run_two_phase_trial(
    *,
    condition_label: str,
    phase1: SplitData,
    phase2: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    normalized_costs_p1: Dict[str, float],
    normalized_costs_p2: Dict[str, float],
    cost_penalty: float,
    warmup: bool = True,
    forgetting_factor: float = 1.0,
    online_learn: bool = True,
    budget_pacer: Optional[BudgetPacer] = None,
    seed: int,
) -> SeedResult:
    """Run one seed through the two-phase cost-drift scenario.

    Phase 1 uses original pricing; at the boundary the registry is
    updated to reflect the Gemini price drop, and Phase 2 continues
    with the new costs.  The BudgetPacer (if present) receives actual
    per-request costs and adapts lambda_t continuously.

    Parameters
    ----------
    condition_label : str
        Human-readable condition name.
    phase1, phase2 : SplitData
        Online learning data for each phase.
    registry : dict
        Original model registry (Phase 1 pricing).
    feature_dim : int
        Context vector dimensionality.
    normalized_costs_p1, normalized_costs_p2 : dict
        Per-model normalized costs for oracle regret computation.
    cost_penalty : float
        Static cost penalty weight (0.0 for pacer conditions).
    warmup : bool
        Whether to load warmup priors.
    forgetting_factor : float
        Fixed forgetting factor.
    online_learn : bool
        If False, the policy is frozen at deployment — ``process_feedback``
        is never called.
    budget_pacer : BudgetPacer or None
        Budget pacer instance (reset before each seed).
    seed : int
        Random seed for prompt ordering.

    Returns
    -------
    SeedResult
        Per-step metrics for this seed.
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
            router.registry[GEMINI_ID]["input_cost_per_m"] = GEMINI_NEW_INPUT_COST
            router.registry[GEMINI_ID]["output_cost_per_m"] = GEMINI_NEW_OUTPUT_COST
            registry_updated = True

        nc = normalized_costs_p1 if t < n_p1 else normalized_costs_p2
        phase = 1 if t < n_p1 else 2

        emb = all_emb[t]
        model, log = router.route(emb)
        reward = float(all_rewards[model][t])
        cost = float(all_costs[model][t])

        log.cost_usd = cost
        if online_learn:
            router.process_feedback(log.request_id, reward=reward)

        oracle_utility = max(float(all_rewards[a][t]) for a in ARM_ORDER)
        chosen_utility = reward

        lam = budget_pacer.lambda_t if budget_pacer is not None else 0.0
        ema = budget_pacer.cost_ema if budget_pacer is not None else 0.0
        gamma = router.bandit.gamma

        result.steps.append(StepRecord(
            step=t + 1,
            phase=phase,
            model=model,
            reward=reward,
            cost=cost,
            oracle_utility=oracle_utility,
            chosen_utility=chosen_utility,
            lambda_t=lam,
            cost_ema=ema,
            gamma=gamma,
        ))

    return result


# ======================================================================
# Condition definitions
# ======================================================================


def _build_conditions(
    budget_target: float,
    budget_label: str,
    matched_cp: float,
) -> List[Dict[str, Any]]:
    """Build three conditions for a given budget target.

    The conditions represent increasing routing sophistication:
    Fixed Policy → Naive Bandit → BanditGPT.

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
            "label": f"BanditGPT ({budget_label})",
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
        Aggregated metrics including checkpoint curves, phase summaries,
        and per-seed regrets for statistical tests.
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
        lambdas, cost_emas, cum_regrets, gammas = [], [], [], []
        arm_frac_lists: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
        rewards_agg, costs_agg = [], []

        avg_costs: List[float] = []

        for sr in seed_results:
            steps_so_far = sr.steps[:cp_step]
            regret = sum(s.oracle_utility - s.chosen_utility for s in steps_so_far)
            cum_regrets.append(regret)
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
            "mean_cumulative_regret": float(np.mean(cum_regrets)),
            "std_cumulative_regret": float(np.std(cum_regrets)),
            "se_cumulative_regret": float(np.std(cum_regrets) / np.sqrt(n_seeds)),
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
    per_seed_regret = [sr.total_regret() for sr in seed_results]

    return {
        "label": seed_results[0].condition,
        "curves": curves,
        "phase1_summary": {
            "mean_reward": float(np.mean([m["mean_reward"] for m in phase1_metrics])),
            "mean_cost": float(np.mean([m["mean_cost"] for m in phase1_metrics])),
            "mean_regret": float(np.mean([m["cumulative_regret"] for m in phase1_metrics])),
            "mean_lambda": float(np.mean([m["mean_lambda"] for m in phase1_metrics])),
            "arm_fractions": {
                ARM_SHORT[a]: float(np.mean([m["arm_fractions"][a] for m in phase1_metrics]))
                for a in ARM_ORDER
            },
        },
        "phase2_summary": {
            "mean_reward": float(np.mean([m["mean_reward"] for m in phase2_metrics])),
            "mean_cost": float(np.mean([m["mean_cost"] for m in phase2_metrics])),
            "mean_regret": float(np.mean([m["cumulative_regret"] for m in phase2_metrics])),
            "mean_lambda": float(np.mean([m["mean_lambda"] for m in phase2_metrics])),
            "arm_fractions": {
                ARM_SHORT[a]: float(np.mean([m["arm_fractions"][a] for m in phase2_metrics]))
                for a in ARM_ORDER
            },
        },
        "total_regret": {
            "mean": float(np.mean(per_seed_regret)),
            "std": float(np.std(per_seed_regret)),
            "se": float(np.std(per_seed_regret) / np.sqrt(n_seeds)),
        },
        "per_seed_regret": per_seed_regret,
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
    fs = FeatureService()
    feature_dim = fs.dimension

    train_all = _load_all(VAL_DATA_PATH, fs, ARM_ORDER)
    test_all = _load_all(HOLDOUT_DATA_PATH, fs, ARM_ORDER)

    logger.info("  Online (val): %d prompts", train_all.n)
    logger.info("  Holdout (test): %d prompts", test_all.n)

    rng_global = np.random.default_rng(42)
    all_indices = rng_global.permutation(train_all.n)
    p1_indices = all_indices[:PHASE1_N]
    p2_indices = all_indices[PHASE1_N : PHASE1_N + PHASE2_N]

    phase1 = SplitData(
        prompts=[train_all.prompts[i] for i in p1_indices],
        rewards={a: train_all.rewards[a][p1_indices] for a in ARM_ORDER},
        costs={a: train_all.costs[a][p1_indices] for a in ARM_ORDER},
        embeddings=train_all.embeddings[p1_indices],
    )

    phase2_raw = SplitData(
        prompts=[train_all.prompts[i] for i in p2_indices],
        rewards={a: train_all.rewards[a][p2_indices] for a in ARM_ORDER},
        costs={a: train_all.costs[a][p2_indices] for a in ARM_ORDER},
        embeddings=train_all.embeddings[p2_indices],
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

    normalized_costs_p1 = compute_normalized_costs(registry, ARM_ORDER)
    registry_p2 = _build_phase2_registry(
        registry, GEMINI_ID,
        GEMINI_NEW_INPUT_COST, GEMINI_NEW_OUTPUT_COST,
    )
    normalized_costs_p2 = compute_normalized_costs(registry_p2, ARM_ORDER)

    logger.info("  Phase 1 norm costs: %s", {
        ARM_SHORT[a]: f"{v:.4f}" for a, v in normalized_costs_p1.items()
    })
    logger.info("  Phase 2 norm costs: %s", {
        ARM_SHORT[a]: f"{v:.4f}" for a, v in normalized_costs_p2.items()
    })

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
                sr = _run_two_phase_trial(
                    condition_label=label,
                    phase1=phase1,
                    phase2=phase2,
                    registry=registry,
                    feature_dim=feature_dim,
                    normalized_costs_p1=normalized_costs_p1,
                    normalized_costs_p2=normalized_costs_p2,
                    cost_penalty=cond["cost_penalty"],
                    warmup=cond["warmup"],
                    forgetting_factor=cond["forgetting_factor"],
                    online_learn=cond.get("online_learn", True),
                    budget_pacer=pacer,
                    seed=seed,
                )
                seed_results.append(sr)

            agg = _aggregate_seeds(seed_results, PHASE1_N)
            all_condition_results[label] = agg

            logger.info(
                "  Total regret: %.1f ± %.1f (SE %.1f)",
                agg["total_regret"]["mean"],
                agg["total_regret"]["std"],
                agg["total_regret"]["se"],
            )
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
            phase1=phase1,
            phase2=phase2,
            registry=registry,
            feature_dim=feature_dim,
            normalized_costs_p1=normalized_costs_p1,
            normalized_costs_p2=normalized_costs_p2,
            cost_penalty=0.0,
            warmup=True,
            forgetting_factor=1.0,
            budget_pacer=None,
            seed=seed,
        )
        unconstrained_seeds.append(sr)

    unconstrained_agg = _aggregate_seeds(unconstrained_seeds, PHASE1_N)
    all_condition_results["Unconstrained"] = unconstrained_agg
    logger.info(
        "  Total regret: %.1f ± %.1f",
        unconstrained_agg["total_regret"]["mean"],
        unconstrained_agg["total_regret"]["std"],
    )

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
        "pacer_lr": PACER_LR,
        "pacer_lambda_max": PACER_LAMBDA_MAX,
        "pacer_ema_alpha": PACER_EMA_ALPHA,
        "prior_n_effective": PRIOR_N_EFFECTIVE,
        "alpha": ALPHA,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "normalized_costs_p1": {
            ARM_SHORT[a]: v for a, v in normalized_costs_p1.items()
        },
        "normalized_costs_p2": {
            ARM_SHORT[a]: v for a, v in normalized_costs_p2.items()
        },
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
        "  %-35s  %8s  %8s  %8s  %8s  %8s",
        "Condition", "P1 Rwd", "P2 Rwd", "P1 λ", "P2 λ", "Regret",
    )
    logger.info("  " + "-" * 90)
    for label, agg in all_condition_results.items():
        p1 = agg["phase1_summary"]
        p2 = agg["phase2_summary"]
        logger.info(
            "  %-35s  %8.4f  %8.4f  %8.3f  %8.3f  %8.1f",
            label,
            p1["mean_reward"], p2["mean_reward"],
            p1["mean_lambda"], p2["mean_lambda"],
            agg["total_regret"]["mean"],
        )
    logger.info("=" * 100)
    logger.info("Wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
