#!/usr/bin/env python3
"""Appendix: Cold-Start vs Warmup Prior Regret.

Compares ParetoBandit with warmup priors against a tabula-rasa cold start
on the K=3 portfolio under stationary conditions.  Demonstrates that
warmup priors substantially reduce early regret and improve sample
efficiency — the router begins with informed beliefs rather than
blindly exploring all arms.

Three conditions share the same prompt stream (val split, n=1,785),
seeds, and hyperparameters; only the prior initialization differs:

  1. **ParetoBandit (warmup)** — offline priors from training set
  2. **Tabula Rasa** — cold start (A=λI, b=0)
  3. **Random** — uniform random arm selection (floor baseline)

Usage:
    python experiments/appendix/warmup_ablation/run_warmup_ablation.py
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
    BEST_K3_TABULA_RASA_HPARAMS,
    DEFAULT_PACER_LAMBDA_MAX,
    DEFAULT_PACER_LR,
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_BUDGET_LABELS,
    K3_BUDGET_TARGETS,
    K3_WARMUP_PRIORS_PATH,
    N_SEEDS,
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
ARM_SHORT: Dict[str, str] = K3_ARM_SHORT

SEED_OFFSET: int = 9000
RESULTS_DIR = Path(__file__).parent / "results"

CHECKPOINT_INTERVAL: int = 25

WARMUP_ALPHA: float = BEST_K3_HPARAMS["alpha"]
WARMUP_N_EFF: float = BEST_K3_HPARAMS["prior_n_effective"]
WARMUP_GAMMA: float = BEST_K3_HPARAMS["forgetting_factor"]

TABULA_ALPHA: float = BEST_K3_TABULA_RASA_HPARAMS["alpha"]
TABULA_N_EFF: float = BEST_K3_TABULA_RASA_HPARAMS["prior_n_effective"]
TABULA_GAMMA: float = BEST_K3_TABULA_RASA_HPARAMS["forgetting_factor"]

EARLY_STEP: int = 200
"""Step at which to report Regret@200 for the early-learning comparison."""


def _snapshot_trace_A_inv(
    router: Optional["BanditRouter"],
    arms: List[str],
) -> Dict[str, float]:
    """Return tr(A_inv) for each arm — scalar summary of uncertainty.

    Returns zeros if the router is None (random baseline).
    """
    if router is None:
        return {a: 0.0 for a in arms}
    return {
        a: float(np.trace(router.bandit.A_inv[a]))
        for a in arms
        if a in router.bandit.A_inv
    }


# ======================================================================
# Data types
# ======================================================================


@dataclass
class StepRecord:
    """Per-step metrics recorded during the trial."""

    step: int
    model: str
    reward: float
    oracle_reward: float

    @property
    def regret(self) -> float:
        return self.oracle_reward - self.reward


@dataclass
class UncertaintyCheckpoint:
    """Snapshot of per-arm uncertainty at a given step."""

    step: int
    trace_A_inv: Dict[str, float]


@dataclass
class SeedResult:
    """Aggregate metrics for one (condition, seed) trial."""

    condition: str
    seed: int
    steps: List[StepRecord] = field(default_factory=list)
    uncertainty_checkpoints: List[UncertaintyCheckpoint] = field(
        default_factory=list,
    )

    @property
    def n(self) -> int:
        return len(self.steps)

    def total_regret(self) -> float:
        return sum(s.regret for s in self.steps)

    def regret_at(self, step: int) -> float:
        """Cumulative regret through the first *step* steps."""
        return sum(s.regret for s in self.steps[:step])

    def mean_reward(self) -> float:
        return float(np.mean([s.reward for s in self.steps]))

    def oracle_agreement(self, window: int = 50) -> float:
        """Fraction of last *window* steps where the chosen arm was oracle-best."""
        tail = self.steps[-min(window, len(self.steps)) :]
        return float(
            np.mean(
                [
                    1.0 if abs(s.reward - s.oracle_reward) < 1e-9 else 0.0
                    for s in tail
                ]
            )
        )


# ======================================================================
# Router Factory
# ======================================================================


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup: bool = True,
    alpha: float = WARMUP_ALPHA,
    prior_n_effective: float = WARMUP_N_EFF,
    forgetting_factor: float = WARMUP_GAMMA,
    budget_pacer: Optional[BudgetPacer] = None,
) -> BanditRouter:
    """Build a K=3 router with optional warmup priors and budget pacer.

    Parameters
    ----------
    registry : dict
        Model registry from ``build_model_registry``.
    feature_dim : int
        Context vector dimensionality.
    warmup : bool
        If True, load warmup priors from ``K3_WARMUP_PRIORS_PATH``.
        If False, cold start (``A=λI, b=0``).
    alpha : float
        Exploration coefficient.
    prior_n_effective : float
        Number of pseudo-observations for warmup priors.
    forgetting_factor : float
        Geometric decay factor.
    budget_pacer : BudgetPacer or None
        If provided, enables primal-dual budget pacing.
    """
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    return BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if warmup else "none",
        warmup_path=str(K3_WARMUP_PRIORS_PATH) if warmup else None,
        prior_n_effective=prior_n_effective,
        alpha=alpha,
        use_corralling=False,
        cost_penalty=0.0,
        forgetting_factor=forgetting_factor,
        budget_pacer=budget_pacer,
    )


# ======================================================================
# Trial Runner
# ======================================================================


def _run_trial(
    *,
    condition_label: str,
    data: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    seed: int,
    warmup: bool = True,
    is_random: bool = False,
    alpha: float = WARMUP_ALPHA,
    prior_n_effective: float = WARMUP_N_EFF,
    forgetting_factor: float = WARMUP_GAMMA,
    budget_pacer: Optional[BudgetPacer] = None,
) -> SeedResult:
    """Run one seed for one condition (cumulative-regret protocol).

    The router learns and is evaluated simultaneously on the same split.
    Cumulative regret is measured from step 1, capturing the full cost
    of learning (including the cold-start transient).  This is the
    standard contextual-bandit evaluation protocol.

    Parameters
    ----------
    condition_label : str
        Human-readable condition name.
    data : SplitData
        Evaluation split (test.jsonl, held out from warmup-prior
        fitting and hyperparameter selection).
    registry : dict
        Model registry.
    feature_dim : int
        Context vector dimensionality.
    seed : int
        Random seed for prompt ordering.
    warmup : bool
        Load warmup priors (True) or cold start (False).
    is_random : bool
        If True, select arms uniformly at random (no router).
    alpha : float
        Exploration coefficient.
    prior_n_effective : float
        Number of pseudo-observations for warmup priors.
    forgetting_factor : float
        Geometric decay factor.
    budget_pacer : BudgetPacer or None
        If provided, enables primal-dual budget pacing.

    Returns
    -------
    SeedResult
        Per-step metrics for this seed.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(data.n)

    if budget_pacer is not None:
        budget_pacer.reset()

    router: Optional[BanditRouter] = None
    if not is_random:
        router = _create_router(
            registry, feature_dim,
            warmup=warmup,
            alpha=alpha,
            prior_n_effective=prior_n_effective,
            forgetting_factor=forgetting_factor,
            budget_pacer=budget_pacer,
        )

    result = SeedResult(condition=condition_label, seed=seed)

    # Snapshot initial uncertainty (step 0, before any online learning)
    if not is_random:
        result.uncertainty_checkpoints.append(
            UncertaintyCheckpoint(
                step=0,
                trace_A_inv=_snapshot_trace_A_inv(router, ARM_ORDER),
            )
        )

    for t_idx in range(data.n):
        orig_idx = order[t_idx]

        if is_random:
            model = rng.choice(ARM_ORDER)
        else:
            emb = data.embeddings[orig_idx]
            model, log = router.route(emb)
            reward = float(data.rewards[model][orig_idx])
            log.cost_usd = float(data.costs[model][orig_idx])
            router.process_feedback(log.request_id, reward=reward)

        gt_reward = float(data.rewards[model][orig_idx])
        oracle_reward = max(
            float(data.rewards[a][orig_idx]) for a in ARM_ORDER
        )

        result.steps.append(
            StepRecord(
                step=t_idx + 1,
                model=model,
                reward=gt_reward,
                oracle_reward=oracle_reward,
            )
        )

        step = t_idx + 1
        if not is_random and (
            step % CHECKPOINT_INTERVAL == 0 or step == data.n
        ):
            result.uncertainty_checkpoints.append(
                UncertaintyCheckpoint(
                    step=step,
                    trace_A_inv=_snapshot_trace_A_inv(router, ARM_ORDER),
                )
            )

    return result


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_seeds(
    seed_results: List[SeedResult],
) -> Dict[str, Any]:
    """Aggregate per-seed results into checkpoint curves and summary stats."""
    n_seeds = len(seed_results)
    n_total = seed_results[0].n

    checkpoints = sorted(
        set(
            [1]
            + list(range(CHECKPOINT_INTERVAL, n_total + 1, CHECKPOINT_INTERVAL))
            + [n_total]
        )
    )

    curves: List[Dict[str, Any]] = []
    for cp_step in checkpoints:
        cum_regrets = [
            sum(s.regret for s in sr.steps[:cp_step]) for sr in seed_results
        ]
        window_size = min(50, cp_step)
        oracle_agreements = []
        for sr in seed_results:
            window = sr.steps[cp_step - window_size : cp_step]
            agree = float(
                np.mean(
                    [
                        1.0 if abs(s.reward - s.oracle_reward) < 1e-9 else 0.0
                        for s in window
                    ]
                )
            )
            oracle_agreements.append(agree)

        arm_frac_lists: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
        for sr in seed_results:
            window = sr.steps[max(0, cp_step - 50) : cp_step]
            arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
            for s in window:
                arm_counts[s.model] += 1
            wn = len(window)
            for a in ARM_ORDER:
                arm_frac_lists[a].append(arm_counts[a] / wn)

        arm_fracs = {
            ARM_SHORT[a]: float(np.mean(arm_frac_lists[a])) for a in ARM_ORDER
        }

        curves.append(
            {
                "step": cp_step,
                "mean_cumulative_regret": float(np.mean(cum_regrets)),
                "std_cumulative_regret": float(np.std(cum_regrets)),
                "se_cumulative_regret": float(
                    np.std(cum_regrets) / np.sqrt(n_seeds)
                ),
                "per_seed_cumulative_regret": [float(r) for r in cum_regrets],
                "mean_oracle_agreement": float(np.mean(oracle_agreements)),
                "se_oracle_agreement": float(
                    np.std(oracle_agreements) / np.sqrt(n_seeds)
                ),
                "arm_fractions": arm_fracs,
                "n_seeds": n_seeds,
            }
        )

    # Aggregate uncertainty checkpoints (only for non-random conditions)
    uncertainty_curves: List[Dict[str, Any]] = []
    has_uncertainty = any(
        len(sr.uncertainty_checkpoints) > 0 for sr in seed_results
    )
    if has_uncertainty:
        ref_cps = seed_results[0].uncertainty_checkpoints
        for cp_idx, ref_cp in enumerate(ref_cps):
            per_arm: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
            for sr in seed_results:
                if cp_idx < len(sr.uncertainty_checkpoints):
                    cp = sr.uncertainty_checkpoints[cp_idx]
                    for a in ARM_ORDER:
                        per_arm[a].append(cp.trace_A_inv.get(a, 0.0))
            uncertainty_curves.append({
                "step": ref_cp.step,
                "trace_A_inv_mean": {
                    ARM_SHORT[a]: float(np.mean(per_arm[a]))
                    for a in ARM_ORDER
                },
                "trace_A_inv_se": {
                    ARM_SHORT[a]: (
                        float(np.std(per_arm[a], ddof=1) / np.sqrt(n_seeds))
                        if n_seeds > 1 else 0.0
                    )
                    for a in ARM_ORDER
                },
                "per_seed_trace_A_inv": {
                    ARM_SHORT[a]: [float(v) for v in per_arm[a]]
                    for a in ARM_ORDER
                },
            })

    per_seed_regret = [sr.total_regret() for sr in seed_results]
    per_seed_reward = [sr.mean_reward() for sr in seed_results]
    per_seed_agree = [sr.oracle_agreement(window=50) for sr in seed_results]
    per_seed_regret_early = [sr.regret_at(EARLY_STEP) for sr in seed_results]

    return {
        "label": seed_results[0].condition,
        "curves": curves,
        "total_regret": {
            "mean": float(np.mean(per_seed_regret)),
            "std": float(np.std(per_seed_regret)),
            "se": float(np.std(per_seed_regret) / np.sqrt(n_seeds)),
        },
        "mean_reward": {
            "mean": float(np.mean(per_seed_reward)),
            "std": float(np.std(per_seed_reward)),
            "se": float(np.std(per_seed_reward) / np.sqrt(n_seeds)),
        },
        "oracle_agreement": {
            "mean": float(np.mean(per_seed_agree)),
            "std": float(np.std(per_seed_agree)),
            "se": float(np.std(per_seed_agree) / np.sqrt(n_seeds)),
        },
        f"regret_at_{EARLY_STEP}": {
            "mean": float(np.mean(per_seed_regret_early)),
            "std": float(np.std(per_seed_regret_early)),
            "se": float(np.std(per_seed_regret_early) / np.sqrt(n_seeds)),
        },
        "per_seed_regret": per_seed_regret,
        "per_seed_reward": per_seed_reward,
        "per_seed_oracle_agreement": per_seed_agree,
        f"per_seed_regret_at_{EARLY_STEP}": per_seed_regret_early,
        "uncertainty_curves": uncertainty_curves,
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

    # Cumulative-regret protocol: learn and evaluate simultaneously on
    # the held-out test split (disjoint from warmup-prior training on
    # train.jsonl and hyperparameter selection on val.jsonl).
    test_data = load_split(HOLDOUT_DATA_PATH, fs, ARM_ORDER)
    logger.info("  Test: %d prompts, K=%d arms", test_data.n, len(ARM_ORDER))

    registry = build_model_registry(ARM_ORDER)

    # -- Unconstrained conditions --
    conditions: List[Dict[str, Any]] = [
        {
            "label": "ParetoBandit (warmup)",
            "warmup": True,
            "is_random": False,
            "alpha": WARMUP_ALPHA,
            "prior_n_effective": WARMUP_N_EFF,
            "forgetting_factor": WARMUP_GAMMA,
            "budget_target": None,
        },
        {
            "label": "Tabula Rasa",
            "warmup": False,
            "is_random": False,
            "alpha": TABULA_ALPHA,
            "prior_n_effective": TABULA_N_EFF,
            "forgetting_factor": TABULA_GAMMA,
            "budget_target": None,
        },
        {
            "label": "Random",
            "warmup": False,
            "is_random": True,
            "alpha": TABULA_ALPHA,
            "prior_n_effective": TABULA_N_EFF,
            "forgetting_factor": TABULA_GAMMA,
            "budget_target": None,
        },
    ]

    # -- Budget-constrained conditions (tight + moderate) --
    for target, blabel in zip(K3_BUDGET_TARGETS[:2], K3_BUDGET_LABELS[:2]):
        conditions.append({
            "label": f"Warmup ({blabel} budget)",
            "warmup": True,
            "is_random": False,
            "alpha": WARMUP_ALPHA,
            "prior_n_effective": WARMUP_N_EFF,
            "forgetting_factor": WARMUP_GAMMA,
            "budget_target": target,
        })
        conditions.append({
            "label": f"Tabula Rasa ({blabel} budget)",
            "warmup": False,
            "is_random": False,
            "alpha": TABULA_ALPHA,
            "prior_n_effective": TABULA_N_EFF,
            "forgetting_factor": TABULA_GAMMA,
            "budget_target": target,
        })

    all_results: Dict[str, Dict[str, Any]] = {}

    for cond in conditions:
        label = cond["label"]
        logger.info("=== %s ===", label)
        seed_results: List[SeedResult] = []
        budget_target: Optional[float] = cond["budget_target"]

        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            pacer: Optional[BudgetPacer] = None
            if budget_target is not None:
                pacer = BudgetPacer(
                    target_avg_spend_usd=budget_target,
                    mode=PacingMode.ADAPTIVE,
                    lr=DEFAULT_PACER_LR,
                    lambda_max=DEFAULT_PACER_LAMBDA_MAX,
                )
            sr = _run_trial(
                condition_label=label,
                data=test_data,
                registry=registry,
                feature_dim=feature_dim,
                seed=seed,
                warmup=cond["warmup"],
                is_random=cond["is_random"],
                alpha=cond["alpha"],
                prior_n_effective=cond["prior_n_effective"],
                forgetting_factor=cond["forgetting_factor"],
                budget_pacer=pacer,
            )
            seed_results.append(sr)
            if (s + 1) % 5 == 0:
                logger.info(
                    "  seed %d/%d  regret=%.1f  regret@%d=%.1f  reward=%.4f",
                    s + 1,
                    N_SEEDS,
                    sr.total_regret(),
                    EARLY_STEP,
                    sr.regret_at(EARLY_STEP),
                    sr.mean_reward(),
                )

        agg = _aggregate_seeds(seed_results)
        all_results[label] = agg
        logger.info(
            "  FINAL: regret=%.1f±%.1f  regret@%d=%.1f±%.1f  agree=%.3f",
            agg["total_regret"]["mean"],
            agg["total_regret"]["se"],
            EARLY_STEP,
            agg[f"regret_at_{EARLY_STEP}"]["mean"],
            agg[f"regret_at_{EARLY_STEP}"]["se"],
            agg["oracle_agreement"]["mean"],
        )

    output = {
        "experiment": "appendix_warmup_ablation",
        "n_seeds": N_SEEDS,
        "n_prompts": test_data.n,
        "split": "test",
        "arms": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "early_step": EARLY_STEP,
        "hparams": {
            "warmup": {
                "alpha": WARMUP_ALPHA,
                "prior_n_effective": WARMUP_N_EFF,
                "forgetting_factor": WARMUP_GAMMA,
            },
            "tabula_rasa": {
                "alpha": TABULA_ALPHA,
                "prior_n_effective": TABULA_N_EFF,
                "forgetting_factor": TABULA_GAMMA,
            },
            "policy": "disjoint",
            "pacer_lr": DEFAULT_PACER_LR,
            "pacer_lambda_max": DEFAULT_PACER_LAMBDA_MAX,
        },
        "budget_targets": {
            label: target
            for label, target in zip(K3_BUDGET_LABELS[:2], K3_BUDGET_TARGETS[:2])
        },
        "conditions": all_results,
    }

    out_path = RESULTS_DIR / "warmup_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", out_path)

    elapsed = time.time() - t0
    logger.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    main()
