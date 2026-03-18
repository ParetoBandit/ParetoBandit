#!/usr/bin/env python3
"""Appendix: Validation Burn-In Ablation.

Quantifies how much of the reported test-set performance depends on the
online-learning burn-in that occurs on the validation split before the
test trajectory begins.

The paper's standard protocol is: warmup priors (train) → online
learning on val (1,785 prompts, no reported metrics) → evaluation on
test (1,824 prompts, reported metrics).  A reviewer may reasonably ask
how much of the test-time quality originates from the val burn-in versus
the warmup priors themselves.

This experiment answers that question through two complementary views:

**View 1 — Burn-in fraction sweep.**
Six conditions share the same warmup priors and test split; only the
amount of val burn-in varies (0%, 25%, 50%, 75%, 100%).  A Tabula Rasa
baseline (no priors, no burn-in) anchors the floor.  Cumulative regret
on test is the primary metric.

**View 2 — Combined trajectory.**
The val and test splits are concatenated into a single 3,609-step
stream.  Cumulative regret is reported from step 1, giving the "no free
lunch" perspective where all learning costs are visible.  A vertical
line marks the val→test boundary.

Usage:
    python experiments/appendix/val_burnin_ablation/run_val_burnin_ablation.py
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
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.config import (
    BEST_K3_HPARAMS,
    BEST_K3_TABULA_RASA_HPARAMS,
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_WARMUP_PRIORS_PATH,
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

N_SEEDS: int = 20
SEED_OFFSET: int = 7000
RESULTS_DIR = Path(__file__).parent / "results"

CHECKPOINT_INTERVAL: int = 25

BURNIN_FRACTIONS: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0]
"""Fraction of the val split used for online learning before test."""

WARMUP_ALPHA: float = BEST_K3_HPARAMS["alpha"]
WARMUP_N_EFF: float = BEST_K3_HPARAMS["prior_n_effective"]
WARMUP_GAMMA: float = BEST_K3_HPARAMS["forgetting_factor"]

TABULA_ALPHA: float = BEST_K3_TABULA_RASA_HPARAMS["alpha"]
TABULA_N_EFF: float = BEST_K3_TABULA_RASA_HPARAMS["prior_n_effective"]
TABULA_GAMMA: float = BEST_K3_TABULA_RASA_HPARAMS["forgetting_factor"]

EARLY_STEP: int = 200

# With γ=0.995, effective memory ≈ 1/(1−γ) = 200 steps.  After a
# full 1,785-step burn-in the warmup priors are decayed by γ^1785
# ≈ 1.3e−5 — effectively erased.  This interaction between forgetting
# and burn-in length is a key part of the analysis: at 100% burn-in
# the bandit operates on recent online evidence, not the original
# priors.  The 0% condition is the only one where priors are fully
# intact at the start of test.
GAMMA_DECAY_AT_FULL_BURNIN: float = WARMUP_GAMMA ** 1785


# ======================================================================
# Data types
# ======================================================================


@dataclass
class StepRecord:
    """Per-step metrics recorded during a trial."""

    step: int
    model: str
    reward: float
    oracle_reward: float
    phase: str

    @property
    def regret(self) -> float:
        return self.oracle_reward - self.reward


@dataclass
class SeedResult:
    """Aggregate metrics for one (condition, seed) trial."""

    condition: str
    seed: int
    steps: List[StepRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.steps)

    def total_regret(self) -> float:
        return sum(s.regret for s in self.steps)

    def phase_regret(self, phase: str) -> float:
        return sum(s.regret for s in self.steps if s.phase == phase)

    def regret_at(self, step: int) -> float:
        """Cumulative regret through the first *step* steps."""
        return sum(s.regret for s in self.steps[:step])

    def mean_reward(self, phase: Optional[str] = None) -> float:
        relevant = [s for s in self.steps if phase is None or s.phase == phase]
        return float(np.mean([s.reward for s in relevant])) if relevant else 0.0

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

    def test_steps(self) -> List[StepRecord]:
        return [s for s in self.steps if s.phase == "test"]

    def test_regret_curve(self) -> List[float]:
        """Cumulative regret over test-phase steps only."""
        cum = 0.0
        curve: List[float] = []
        for s in self.test_steps():
            cum += s.regret
            curve.append(cum)
        return curve


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
) -> BanditRouter:
    """Build a K=3 router with optional warmup priors.

    Parameters
    ----------
    registry : dict
        Model registry from ``build_model_registry``.
    feature_dim : int
        Context vector dimensionality.
    warmup : bool
        If True, load warmup priors from ``K3_WARMUP_PRIORS_PATH``.
    alpha : float
        Exploration coefficient.
    prior_n_effective : float
        Number of pseudo-observations for warmup priors.
    forgetting_factor : float
        Geometric decay factor.
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
        policy="disjoint",
        adaptive_gamma=False,
    )


# ======================================================================
# Trial Runners
# ======================================================================


def _run_burnin_trial(
    *,
    condition_label: str,
    val_data: SplitData,
    test_data: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    seed: int,
    burnin_fraction: float,
    warmup: bool = True,
    alpha: float = WARMUP_ALPHA,
    prior_n_effective: float = WARMUP_N_EFF,
    forgetting_factor: float = WARMUP_GAMMA,
) -> SeedResult:
    """Run one seed with a specified fraction of val as burn-in.

    Parameters
    ----------
    condition_label : str
        Human-readable condition name.
    val_data : SplitData
        Validation split used for burn-in.
    test_data : SplitData
        Held-out test split for evaluation.
    registry : dict
        Model registry.
    feature_dim : int
        Context vector dimensionality.
    seed : int
        Random seed for prompt ordering.
    burnin_fraction : float
        Fraction of val_data to use for burn-in (0.0 to 1.0).
    warmup : bool
        Load warmup priors (True) or cold start (False).
    alpha : float
        Exploration coefficient.
    prior_n_effective : float
        Prior strength.
    forgetting_factor : float
        Geometric decay factor.

    Returns
    -------
    SeedResult
        Per-step metrics including both val and test phases.
    """
    rng = np.random.default_rng(seed)
    val_order = rng.permutation(val_data.n)
    test_order = rng.permutation(test_data.n)

    n_burnin = int(round(burnin_fraction * val_data.n))

    router = _create_router(
        registry,
        feature_dim,
        warmup=warmup,
        alpha=alpha,
        prior_n_effective=prior_n_effective,
        forgetting_factor=forgetting_factor,
    )

    result = SeedResult(condition=condition_label, seed=seed)
    global_step = 0

    # -- Val burn-in phase (metrics recorded but labelled "val") --
    for t_idx in range(n_burnin):
        orig_idx = val_order[t_idx]
        emb = val_data.embeddings[orig_idx]
        model, log = router.route(emb)
        reward = float(val_data.rewards[model][orig_idx])
        log.cost_usd = float(val_data.costs[model][orig_idx])
        router.process_feedback(log.request_id, reward=reward)

        oracle_reward = max(
            float(val_data.rewards[a][orig_idx]) for a in ARM_ORDER
        )
        global_step += 1
        result.steps.append(
            StepRecord(
                step=global_step,
                model=model,
                reward=reward,
                oracle_reward=oracle_reward,
                phase="val",
            )
        )

    # -- Test phase (metrics labelled "test") --
    for t_idx in range(test_data.n):
        orig_idx = test_order[t_idx]
        emb = test_data.embeddings[orig_idx]
        model, log = router.route(emb)
        reward = float(test_data.rewards[model][orig_idx])
        log.cost_usd = float(test_data.costs[model][orig_idx])
        router.process_feedback(log.request_id, reward=reward)

        oracle_reward = max(
            float(test_data.rewards[a][orig_idx]) for a in ARM_ORDER
        )
        global_step += 1
        result.steps.append(
            StepRecord(
                step=global_step,
                model=model,
                reward=reward,
                oracle_reward=oracle_reward,
                phase="test",
            )
        )

    return result


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_test_metrics(
    seed_results: List[SeedResult],
    n_test: int,
) -> Dict[str, Any]:
    """Aggregate test-phase metrics across seeds.

    Produces checkpoint curves for cumulative regret on the test split,
    plus summary statistics.
    """
    n_seeds = len(seed_results)

    checkpoints = sorted(
        set(
            [1]
            + list(range(CHECKPOINT_INTERVAL, n_test + 1, CHECKPOINT_INTERVAL))
            + [n_test]
        )
    )

    test_regret_curves = [sr.test_regret_curve() for sr in seed_results]

    curves: List[Dict[str, Any]] = []
    for cp_step in checkpoints:
        cum_regrets = [c[cp_step - 1] for c in test_regret_curves]

        test_steps_all = [sr.test_steps() for sr in seed_results]
        window_size = min(50, cp_step)
        oracle_agreements: List[float] = []
        for ts in test_steps_all:
            window = ts[cp_step - window_size : cp_step]
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
        for ts in test_steps_all:
            window = ts[max(0, cp_step - 50) : cp_step]
            arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
            for s in window:
                arm_counts[s.model] += 1
            wn = len(window)
            for a in ARM_ORDER:
                arm_frac_lists[a].append(arm_counts[a] / wn if wn > 0 else 0.0)

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

    per_seed_test_regret = [sr.phase_regret("test") for sr in seed_results]
    per_seed_test_reward = [sr.mean_reward("test") for sr in seed_results]
    per_seed_test_agree = [sr.oracle_agreement(window=50) for sr in seed_results]
    per_seed_test_regret_early = [
        c[min(EARLY_STEP, len(c)) - 1] if c else 0.0 for c in test_regret_curves
    ]

    return {
        "curves": curves,
        "test_regret": {
            "mean": float(np.mean(per_seed_test_regret)),
            "std": float(np.std(per_seed_test_regret)),
            "se": float(np.std(per_seed_test_regret) / np.sqrt(n_seeds)),
        },
        "test_reward": {
            "mean": float(np.mean(per_seed_test_reward)),
            "std": float(np.std(per_seed_test_reward)),
            "se": float(np.std(per_seed_test_reward) / np.sqrt(n_seeds)),
        },
        "oracle_agreement": {
            "mean": float(np.mean(per_seed_test_agree)),
            "std": float(np.std(per_seed_test_agree)),
            "se": float(np.std(per_seed_test_agree) / np.sqrt(n_seeds)),
        },
        f"test_regret_at_{EARLY_STEP}": {
            "mean": float(np.mean(per_seed_test_regret_early)),
            "std": float(np.std(per_seed_test_regret_early)),
            "se": float(np.std(per_seed_test_regret_early) / np.sqrt(n_seeds)),
        },
        "per_seed_test_regret": per_seed_test_regret,
        "per_seed_test_reward": per_seed_test_reward,
        "per_seed_oracle_agreement": per_seed_test_agree,
        f"per_seed_test_regret_at_{EARLY_STEP}": per_seed_test_regret_early,
    }


def _aggregate_combined_trajectory(
    seed_results: List[SeedResult],
    n_val_burnin: int,
    n_test: int,
) -> Dict[str, Any]:
    """Aggregate combined val+test trajectory across seeds.

    Produces checkpoint curves over the full stream (val burn-in + test)
    with cumulative regret counted from step 1.
    """
    n_seeds = len(seed_results)
    n_total = n_val_burnin + n_test

    checkpoints = sorted(
        set(
            [1]
            + list(range(CHECKPOINT_INTERVAL, n_total + 1, CHECKPOINT_INTERVAL))
            + [n_val_burnin]
            + [n_total]
        )
    )

    curves: List[Dict[str, Any]] = []
    for cp_step in checkpoints:
        cum_regrets = [sr.regret_at(cp_step) for sr in seed_results]
        curves.append(
            {
                "step": cp_step,
                "mean_cumulative_regret": float(np.mean(cum_regrets)),
                "std_cumulative_regret": float(np.std(cum_regrets)),
                "se_cumulative_regret": float(
                    np.std(cum_regrets) / np.sqrt(n_seeds)
                ),
                "per_seed_cumulative_regret": [float(r) for r in cum_regrets],
                "n_seeds": n_seeds,
            }
        )

    return {
        "n_val_burnin": n_val_burnin,
        "n_test": n_test,
        "n_total": n_total,
        "curves": curves,
    }


# ======================================================================
# Statistical tests
# ======================================================================


def _paired_test(
    a_regrets: List[float],
    b_regrets: List[float],
) -> Dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on paired per-seed regret."""
    diff = np.array(a_regrets) - np.array(b_regrets)
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = wilcoxon(a_regrets, b_regrets, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p)}


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading K=3 data ...")
    fs = FeatureService()
    feature_dim = fs.dimension

    val_data = load_split(VAL_DATA_PATH, fs, ARM_ORDER)
    test_data = load_split(HOLDOUT_DATA_PATH, fs, ARM_ORDER)
    registry = build_model_registry(ARM_ORDER)
    logger.info(
        "  Val: %d prompts  Test: %d prompts  dim=%d",
        val_data.n, test_data.n, feature_dim,
    )

    # ==================================================================
    # View 1: Burn-in fraction sweep (warmup conditions)
    # ==================================================================
    conditions: Dict[str, Dict[str, Any]] = {}

    for frac in BURNIN_FRACTIONS:
        n_burnin = int(round(frac * val_data.n))
        label = f"Warmup ({int(frac * 100)}% burn-in)"
        logger.info("=== %s (n_burnin=%d) ===", label, n_burnin)
        seed_results: List[SeedResult] = []

        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            sr = _run_burnin_trial(
                condition_label=label,
                val_data=val_data,
                test_data=test_data,
                registry=registry,
                feature_dim=feature_dim,
                seed=seed,
                burnin_fraction=frac,
                warmup=True,
                alpha=WARMUP_ALPHA,
                prior_n_effective=WARMUP_N_EFF,
                forgetting_factor=WARMUP_GAMMA,
            )
            seed_results.append(sr)
            if (s + 1) % 5 == 0:
                logger.info(
                    "  seed %d/%d  test_regret=%.1f  test_reward=%.4f",
                    s + 1, N_SEEDS,
                    sr.phase_regret("test"), sr.mean_reward("test"),
                )

        test_agg = _aggregate_test_metrics(seed_results, test_data.n)

        combined_agg = _aggregate_combined_trajectory(
            seed_results, n_burnin, test_data.n,
        )

        conditions[label] = {
            "label": label,
            "burnin_fraction": frac,
            "n_burnin": n_burnin,
            "warmup": True,
            "hparams": {
                "alpha": WARMUP_ALPHA,
                "prior_n_effective": WARMUP_N_EFF,
                "forgetting_factor": WARMUP_GAMMA,
            },
            "test_metrics": test_agg,
            "combined_trajectory": combined_agg,
        }

        logger.info(
            "  FINAL: test_regret=%.1f±%.1f  test_reward=%.4f  "
            "test_regret@%d=%.1f",
            test_agg["test_regret"]["mean"],
            test_agg["test_regret"]["se"],
            test_agg["test_reward"]["mean"],
            EARLY_STEP,
            test_agg[f"test_regret_at_{EARLY_STEP}"]["mean"],
        )

    # ==================================================================
    # Tabula Rasa baselines (complete the 2×2 factorial)
    # ==================================================================
    tabula_conditions: List[Dict[str, Any]] = [
        {
            "label": "Tabula Rasa (no burn-in)",
            "burnin_fraction": 0.0,
        },
        {
            "label": "Tabula Rasa (100% burn-in)",
            "burnin_fraction": 1.0,
        },
    ]

    for tc in tabula_conditions:
        label = tc["label"]
        frac = tc["burnin_fraction"]
        n_burnin = int(round(frac * val_data.n))
        logger.info("=== %s (n_burnin=%d) ===", label, n_burnin)
        seed_results = []
        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            sr = _run_burnin_trial(
                condition_label=label,
                val_data=val_data,
                test_data=test_data,
                registry=registry,
                feature_dim=feature_dim,
                seed=seed,
                burnin_fraction=frac,
                warmup=False,
                alpha=TABULA_ALPHA,
                prior_n_effective=TABULA_N_EFF,
                forgetting_factor=TABULA_GAMMA,
            )
            seed_results.append(sr)

        test_agg = _aggregate_test_metrics(seed_results, test_data.n)
        combined_agg = _aggregate_combined_trajectory(
            seed_results, n_burnin, test_data.n,
        )

        conditions[label] = {
            "label": label,
            "burnin_fraction": frac,
            "n_burnin": n_burnin,
            "warmup": False,
            "hparams": {
                "alpha": TABULA_ALPHA,
                "prior_n_effective": TABULA_N_EFF,
                "forgetting_factor": TABULA_GAMMA,
            },
            "test_metrics": test_agg,
            "combined_trajectory": combined_agg,
        }
        logger.info(
            "  FINAL: test_regret=%.1f±%.1f  test_reward=%.4f",
            test_agg["test_regret"]["mean"],
            test_agg["test_regret"]["se"],
            test_agg["test_reward"]["mean"],
        )

    # ==================================================================
    # Pairwise tests: each burn-in level vs the full (100%) condition
    # ==================================================================
    full_label = "Warmup (100% burn-in)"
    full_regrets = conditions[full_label]["test_metrics"]["per_seed_test_regret"]

    statistical_tests: Dict[str, Dict[str, Any]] = {}
    for cond_label, cond_data in conditions.items():
        if cond_label == full_label:
            continue
        other_regrets = cond_data["test_metrics"]["per_seed_test_regret"]
        test_result = _paired_test(other_regrets, full_regrets)
        diff_regret = float(
            np.mean(other_regrets) - np.mean(full_regrets)
        )
        statistical_tests[cond_label] = {
            "vs": full_label,
            "delta_regret": diff_regret,
            "delta_pct": (
                100.0 * diff_regret / np.mean(full_regrets)
                if np.mean(full_regrets) > 0
                else 0.0
            ),
            **test_result,
        }
        logger.info(
            "  %s vs %s: Δregret=%+.1f (%+.1f%%)  p=%.4f",
            cond_label, full_label,
            diff_regret,
            statistical_tests[cond_label]["delta_pct"],
            test_result["p_value"],
        )

    # ==================================================================
    # Save
    # ==================================================================
    effective_memory = 1.0 / (1.0 - WARMUP_GAMMA)
    prior_decay_by_fraction = {
        frac: float(WARMUP_GAMMA ** int(round(frac * val_data.n)))
        for frac in BURNIN_FRACTIONS
    }

    output = {
        "experiment": "appendix_val_burnin_ablation",
        "n_seeds": N_SEEDS,
        "n_val": val_data.n,
        "n_test": test_data.n,
        "arms": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "early_step": EARLY_STEP,
        "burnin_fractions": BURNIN_FRACTIONS,
        "forgetting_analysis": {
            "gamma": WARMUP_GAMMA,
            "effective_memory_steps": effective_memory,
            "prior_decay_by_burnin_fraction": prior_decay_by_fraction,
            "note": (
                "With gamma=0.995, effective memory is ~200 steps. "
                "After 100% burn-in (1785 steps), warmup priors are "
                f"decayed by gamma^1785 = {GAMMA_DECAY_AT_FULL_BURNIN:.2e}, "
                "effectively erased. The 0% condition is the only one "
                "where priors are fully intact at the start of test."
            ),
        },
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
        },
        "conditions": conditions,
        "statistical_tests": statistical_tests,
    }

    out_path = RESULTS_DIR / "val_burnin_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", out_path)

    # ==================================================================
    # Summary table
    # ==================================================================
    hdr_w = 100
    print("\n" + "=" * hdr_w)
    print("VAL BURN-IN ABLATION — Summary")
    print("=" * hdr_w)
    print(
        f"  {'Condition':<30s}  {'Burn-in':>7s}  "
        f"{'Test Regret':>12s}  {'R@200':>10s}  "
        f"{'Reward':>8s}  {'Agree':>6s}  {'Δ%':>7s}  {'p':>8s}"
    )
    print(f"  {'-' * 30}  {'-' * 7}  {'-' * 12}  {'-' * 10}  "
          f"{'-' * 8}  {'-' * 6}  {'-' * 7}  {'-' * 8}")

    for cond_label, cond_data in conditions.items():
        tm = cond_data["test_metrics"]
        n_bi = cond_data["n_burnin"]
        delta_str = "—"
        p_str = "—"
        if cond_label in statistical_tests:
            st = statistical_tests[cond_label]
            delta_str = f"{st['delta_pct']:+.1f}%"
            p_str = f"{st['p_value']:.2e}" if st["p_value"] < 0.01 else f"{st['p_value']:.3f}"
        print(
            f"  {cond_label:<30s}  {n_bi:>7d}  "
            f"{tm['test_regret']['mean']:>8.1f}±{tm['test_regret']['se']:<4.1f}"
            f"{tm[f'test_regret_at_{EARLY_STEP}']['mean']:>7.1f}±{tm[f'test_regret_at_{EARLY_STEP}']['se']:<4.1f}"
            f"{tm['test_reward']['mean']:>8.4f}  "
            f"{tm['oracle_agreement']['mean']:>6.3f}  "
            f"{delta_str:>7s}  {p_str:>8s}"
        )
    print("=" * hdr_w)

    elapsed = time.time() - t0
    logger.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    main()
