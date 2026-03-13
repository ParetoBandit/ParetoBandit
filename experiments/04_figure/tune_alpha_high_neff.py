#!/usr/bin/env python3
"""Re-tune alpha at high prior_n_effective for the distribution shift experiment.

Motivation
----------
Figures 1/3 use ``n_eff=50`` (tuned via Pareto AUC on val).  Figure 4
uses ``n_eff=5000`` to simulate a production deployment with strong
prior trust.  Alpha was originally tuned at ``n_eff=50``; at ``n_eff=5000``
the priors dominate for hundreds of steps, so the optimal exploration
parameter may differ.

This script sweeps ``alpha × n_eff`` on the val split using Pareto AUC
(identical protocol to ``experiments/benchmark/tune_router.py``) and
reports the best alpha for each n_eff level.  The comparison with
Figure 1's AUC (alpha=1.0, n_eff=50) quantifies any Pareto AUC
degradation from strengthening the priors.

Protocol
--------
1. Train on train split (8,374 prompts) with disjoint LinUCB.
2. Evaluate on val split (1,785 prompts) — bandit continues learning.
3. Sweep cost_penalty to trace the (cost, reward) frontier.
4. Compute Pareto AUC per seed, average across seeds.

Usage::

    python experiments/04_figure/tune_alpha_high_neff.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    BEST_K2_HPARAMS,
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.pareto import pareto_auc
from utils.simulation import SplitData, build_model_registry, load_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)
for _noisy in (
    "bandit_gpt.router",
    "bandit_gpt.router_v2",
    "bandit_gpt.feature_service",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ============================================================================
# Grid
# ============================================================================

ARM_ORDER: List[str] = K2_ARM_ORDER
ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

ALPHA_GRID: List[float] = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
N_EFF_GRID: List[float] = [50.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
COST_PENALTY_SWEEP: List[float] = [
    0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
]

N_SEEDS: int = 3
SEED_OFFSET: int = 0

# Reference from Figure 1 / benchmark sweep (alpha=1.0, n_eff=50)
FIGURE1_VAL_AUC: float = 0.8688
FIGURE1_TEST_AUC: float = 0.8699


# ============================================================================
# Simulation helpers
# ============================================================================


def _simulate_sweep(
    train: SplitData,
    val: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    alpha: float,
    prior_n_effective: float,
    warmup_path: str,
    seed: int,
) -> Tuple[List[float], List[float]]:
    """Run train→val for each cost_penalty, return (costs, rewards) lists."""
    sweep_costs: List[float] = []
    sweep_rewards: List[float] = []

    for cp in COST_PENALTY_SWEEP:
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        fs = FeatureService.for_precomputed(feature_dim)
        store = EphemeralContextStore()
        router = BanditRouter.create(
            model_registry=registry,
            feature_service=fs,
            context_store=store,
            priors="warmup",
            warmup_path=warmup_path,
            prior_n_effective=prior_n_effective,
            alpha=alpha,
            use_corralling=False,
            cost_penalty=cp,
            forgetting_factor=1.0,
            policy="disjoint",
        )

        train_idx = rng.permutation(train.n)
        for i in train_idx:
            model, log = router.route(train.embeddings[i])
            reward = float(train.rewards[model][i])
            router.process_feedback(log.request_id, reward=reward)

        val_rewards_list: List[float] = []
        val_costs_list: List[float] = []
        val_idx = rng.permutation(val.n)
        for i in val_idx:
            model, log = router.route(val.embeddings[i])
            reward = float(val.rewards[model][i])
            router.process_feedback(log.request_id, reward=reward)
            val_rewards_list.append(reward)
            val_costs_list.append(float(val.costs[model][i]))

        sweep_costs.append(float(np.mean(val_costs_list)))
        sweep_rewards.append(float(np.mean(val_rewards_list)))

    return sweep_costs, sweep_rewards


def compute_pareto_auc_for_config(
    train: SplitData,
    val: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    alpha: float,
    prior_n_effective: float,
    warmup_path: str,
) -> Tuple[float, float]:
    """Return (mean_auc, std_auc) across seeds."""
    fixed_costs = [float(val.costs[a].mean()) for a in ARM_ORDER]
    fixed_rewards = [float(val.rewards[a].mean()) for a in ARM_ORDER]
    cost_lo = min(fixed_costs)
    cost_hi = max(fixed_costs)

    per_seed_auc: List[float] = []
    for s in range(N_SEEDS):
        seed = SEED_OFFSET + s
        sc, sr = _simulate_sweep(
            train, val, registry, feature_dim,
            alpha=alpha,
            prior_n_effective=prior_n_effective,
            warmup_path=warmup_path,
            seed=seed,
        )
        all_c = sc + fixed_costs
        all_r = sr + fixed_rewards
        per_seed_auc.append(pareto_auc(all_c, all_r, cost_lo, cost_hi))

    mean_auc = float(np.mean(per_seed_auc))
    std_auc = (
        float(np.std(per_seed_auc, ddof=1)) if N_SEEDS > 1 else 0.0
    )
    return mean_auc, std_auc


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("ALPHA RE-TUNING AT HIGH n_eff (K=2, Pareto AUC)")
    logger.info("  Alpha grid: %s", ALPHA_GRID)
    logger.info("  n_eff grid: %s", N_EFF_GRID)
    logger.info("  %d seeds, %d cost_penalty values",
                N_SEEDS, len(COST_PENALTY_SWEEP))
    total_sims = len(ALPHA_GRID) * len(N_EFF_GRID) * len(COST_PENALTY_SWEEP) * N_SEEDS
    logger.info("  Total simulations: %d", total_sims)
    logger.info("=" * 70)

    logger.info("\nLoading data ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    train = load_split(TRAIN_DATA_PATH, fs, ARM_ORDER)
    val = load_split(VAL_DATA_PATH, fs, ARM_ORDER)
    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K2_WARMUP_PRIORS_PATH)
    logger.info("  Train=%d  Val=%d  dim=%d", train.n, val.n, feature_dim)

    fixed_auc = pareto_auc(
        [float(val.costs[a].mean()) for a in ARM_ORDER],
        [float(val.rewards[a].mean()) for a in ARM_ORDER],
        min(float(val.costs[a].mean()) for a in ARM_ORDER),
        max(float(val.costs[a].mean()) for a in ARM_ORDER),
    )
    logger.info("  Fixed-model val AUC: %.6f", fixed_auc)
    logger.info("  Figure 1 reference (α=1.0, n_eff=50): val=%.4f, test=%.4f",
                FIGURE1_VAL_AUC, FIGURE1_TEST_AUC)

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    best_per_neff: Dict[float, Dict[str, Any]] = {}

    n_total = len(ALPHA_GRID) * len(N_EFF_GRID)
    idx = 0

    for n_eff in N_EFF_GRID:
        for alpha in ALPHA_GRID:
            idx += 1
            t_start = time.time()
            mean_auc, std_auc = compute_pareto_auc_for_config(
                train, val, registry, feature_dim,
                alpha=alpha,
                prior_n_effective=n_eff,
                warmup_path=warmup_path,
            )
            elapsed = time.time() - t_start
            delta_pct = (mean_auc - fixed_auc) / fixed_auc * 100
            delta_fig1 = (mean_auc - FIGURE1_VAL_AUC) / FIGURE1_VAL_AUC * 100

            entry = {
                "alpha": alpha,
                "n_eff": n_eff,
                "val_auc": round(mean_auc, 6),
                "val_auc_std": round(std_auc, 6),
                "delta_vs_fixed_pct": round(delta_pct, 3),
                "delta_vs_figure1_pct": round(delta_fig1, 3),
            }
            results.append(entry)

            is_best = (
                n_eff not in best_per_neff
                or mean_auc > best_per_neff[n_eff]["val_auc"]
            )
            marker = " ***" if is_best else ""
            if is_best:
                best_per_neff[n_eff] = entry

            logger.info(
                "  [%3d/%d] α=%.1f n_eff=%5.0f  AUC=%.6f±%.6f "
                "(Δfixed=%+.3f%%, Δfig1=%+.3f%%)  %.1fs%s",
                idx, n_total, alpha, n_eff,
                mean_auc, std_auc, delta_pct, delta_fig1, elapsed, marker,
            )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: Best alpha per n_eff")
    logger.info("=" * 70)
    logger.info("  %-8s  %-6s  %-12s  %-14s  %-14s",
                "n_eff", "alpha", "val_AUC", "Δ vs fixed", "Δ vs Fig.1")
    logger.info("  %s", "-" * 62)

    for n_eff in N_EFF_GRID:
        if n_eff in best_per_neff:
            b = best_per_neff[n_eff]
            logger.info(
                "  %-8.0f  %-6.1f  %.6f     %+.3f%%         %+.3f%%",
                n_eff, b["alpha"], b["val_auc"],
                b["delta_vs_fixed_pct"], b["delta_vs_figure1_pct"],
            )

    # Recommendation
    best_5k = best_per_neff.get(5000.0)
    if best_5k:
        logger.info(
            "\n  RECOMMENDATION for Figure 4 (n_eff=5000):"
            "\n    alpha=%.1f, Pareto AUC=%.6f (Δfig1=%+.3f%%)",
            best_5k["alpha"], best_5k["val_auc"],
            best_5k["delta_vs_figure1_pct"],
        )

    elapsed_total = time.time() - t0
    logger.info("\n  Total wall time: %.1fs (%.1f min)", elapsed_total,
                elapsed_total / 60)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "alpha_tuning_high_neff.json"

    output = {
        "description": "Alpha re-tuning at high n_eff for Figure 4",
        "protocol": "Pareto AUC on val (train→val, disjoint, K=2)",
        "n_seeds": N_SEEDS,
        "alpha_grid": ALPHA_GRID,
        "n_eff_grid": N_EFF_GRID,
        "cost_penalty_sweep": COST_PENALTY_SWEEP,
        "fixed_model_val_auc": round(fixed_auc, 6),
        "figure1_reference": {
            "alpha": 1.0,
            "n_eff": 50.0,
            "val_auc": FIGURE1_VAL_AUC,
            "test_auc": FIGURE1_TEST_AUC,
        },
        "best_per_neff": {
            str(int(k)): v for k, v in best_per_neff.items()
        },
        "all_results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results → %s", out_path)


if __name__ == "__main__":
    main()
