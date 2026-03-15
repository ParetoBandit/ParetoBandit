#!/usr/bin/env python3
"""Appendix: Cold-Start vs Warmup Prior Regret.

Compares BanditGPT with warmup priors against a tabula-rasa cold start
on the K=3 portfolio under stationary conditions.  Demonstrates that
warmup priors substantially reduce early regret and improve sample
efficiency — the router begins with informed beliefs rather than
blindly exploring all arms.

Three conditions share the same prompt stream (val split, n=1,785),
seeds, and hyperparameters; only the prior initialization differs:

  1. **BanditGPT (warmup)** — offline priors from training set
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

from bandit_gpt.config import (
    K3_ARM_ORDER,
    K3_WARMUP_PRIORS_PATH,
    VAL_DATA_PATH,
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
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service", "bandit_gpt.policy"):
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

N_SEEDS: int = 20
SEED_OFFSET: int = 9000
RESULTS_DIR = Path(__file__).parent / "results"

CHECKPOINT_INTERVAL: int = 25
PRIOR_N_EFFECTIVE: float = 5000.0
ALPHA: float = 1.0
FORGETTING_FACTOR: float = 1.0

EARLY_STEP: int = 200
"""Step at which to report Regret@200 for the early-learning comparison."""


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
        If False, cold start (``A=λI, b=0``).
    """
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
        use_corralling=False,
        cost_penalty=0.0,
        forgetting_factor=FORGETTING_FACTOR,
        drift_threshold=0.0,
        policy="disjoint",
        adaptive_gamma=False,
        budget_pacer=None,
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
) -> SeedResult:
    """Run one seed for one condition.

    Parameters
    ----------
    condition_label : str
        Human-readable condition name.
    data : SplitData
        Online learning data (val split).
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

    Returns
    -------
    SeedResult
        Per-step metrics for this seed.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(data.n)

    router: Optional[BanditRouter] = None
    if not is_random:
        router = _create_router(registry, feature_dim, warmup=warmup)

    result = SeedResult(condition=condition_label, seed=seed)

    for t_idx in range(data.n):
        orig_idx = order[t_idx]

        if is_random:
            model = rng.choice(ARM_ORDER)
        else:
            emb = data.embeddings[orig_idx]
            model, log = router.route(emb)
            reward = float(data.rewards[model][orig_idx])
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
                "mean_oracle_agreement": float(np.mean(oracle_agreements)),
                "se_oracle_agreement": float(
                    np.std(oracle_agreements) / np.sqrt(n_seeds)
                ),
                "arm_fractions": arm_fracs,
                "n_seeds": n_seeds,
            }
        )

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

    val_data = load_split(VAL_DATA_PATH, fs, ARM_ORDER)
    logger.info("  Val: %d prompts, K=%d arms", val_data.n, len(ARM_ORDER))

    registry = build_model_registry(ARM_ORDER)

    conditions = [
        {"label": "BanditGPT (warmup)", "warmup": True, "is_random": False},
        {"label": "Tabula Rasa", "warmup": False, "is_random": False},
        {"label": "Random", "warmup": False, "is_random": True},
    ]

    all_results: Dict[str, Dict[str, Any]] = {}

    for cond in conditions:
        label = cond["label"]
        logger.info("=== %s ===", label)
        seed_results: List[SeedResult] = []

        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            sr = _run_trial(
                condition_label=label,
                data=val_data,
                registry=registry,
                feature_dim=feature_dim,
                seed=seed,
                warmup=cond["warmup"],
                is_random=cond["is_random"],
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
        "n_prompts": val_data.n,
        "arms": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "early_step": EARLY_STEP,
        "hparams": {
            "alpha": ALPHA,
            "prior_n_effective": PRIOR_N_EFFECTIVE,
            "forgetting_factor": FORGETTING_FACTOR,
            "policy": "disjoint",
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
