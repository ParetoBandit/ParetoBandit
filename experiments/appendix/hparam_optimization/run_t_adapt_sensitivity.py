#!/usr/bin/env python3
"""T_adapt sensitivity analysis for the Pareto knee-point selection.

Verifies that the selected (alpha, gamma) configuration is stable
across different adaptation horizon values.  For each T_adapt in
{250, 500, 1000}, the full Pareto knee-point selection procedure is
repeated: n_eff is re-derived from gamma, budget-paced AUC and
catastrophic-failure Phase-2 reward are evaluated on the validation
split, the Pareto frontier is built, and the knee point is identified.

If the selected gamma is consistent (or varies by at most one grid
step) across T_adapt values, the method is robust to the T_adapt
anchor.  If the selected config changes materially, the result is
fragile and the T_adapt choice requires stronger justification.

Uses the same grid, data splits, seeds, and evaluation protocol as
:mod:`run_hparam_sweep` — only T_adapt (and therefore the derived
n_eff for each gamma) varies.

Usage::

    python experiments/appendix/hparam_optimization/run_t_adapt_sensitivity.py
"""

from __future__ import annotations

import itertools
import json
import logging
import time
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from pareto_bandit.feature_service import FeatureService

from run_hparam_sweep import (
    ALPHA_VALUES,
    GAMMA_VALUES,
    PCA_DIM,
    BUDGET_TARGET_COUNT,
    SEED_OFFSET_VAL,
    SEED_OFFSET_FAILURE_VAL,
    _derive_n_eff,
    _load_jsonl,
    _parse_and_embed,
    _split_data,
    compute_budget_paced_pareto_auc,
    compute_failure_resilience,
)
from utils.pareto import pareto_auc
from utils.simulation import build_model_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("pareto_bandit.router", "pareto_bandit.feature_service", "pareto_bandit.policy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Configuration
# ======================================================================

T_ADAPT_VALUES: List[int] = [250, 500, 1000]
"""Adaptation horizons to test.  500 is the canonical value used in
the main sweep; 250 and 1000 bracket it to test sensitivity."""

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = K3_ARM_SHORT
RESULTS_DIR = Path(__file__).parent / "results"

FAILURE_ARMS: List[str] = [K3_FAILURE_ARM]
FAILURE_REWARD: float = K3_FAILURE_REWARD
FAILURE_BUDGET_TARGETS: List[float] = K3_BUDGET_TARGETS
PACER_LR: float = DEFAULT_PACER_LR
PACER_LAMBDA_MAX: float = DEFAULT_PACER_LAMBDA_MAX


# ======================================================================
# Pareto frontier and knee-point (duplicated to avoid import of nested
# functions from run_hparam_sweep.main)
# ======================================================================


def _find_pareto_frontier(
    aucs: np.ndarray,
    p2s: np.ndarray,
) -> List[int]:
    """Return indices of Pareto-optimal configs (both objectives maximized)."""
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


def _find_knee_point(
    aucs: np.ndarray,
    p2s: np.ndarray,
    pareto_indices: List[int],
) -> int:
    """Find the knee point on the Pareto frontier.

    See :func:`run_hparam_sweep._find_knee_point` for full documentation.
    """
    import math

    if len(pareto_indices) <= 1:
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


def _cfg_matches(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True when two result dicts share the same hyperparameter config."""
    return (
        a["alpha"] == b["alpha"]
        and a["gamma"] == b["gamma"]
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data (same protocol as run_hparam_sweep)
    # ------------------------------------------------------------------
    logger.info("Loading data records ...")
    train_records = _load_jsonl(TRAIN_DATA_PATH)
    val_records = _load_jsonl(VAL_DATA_PATH)
    logger.info("  train=%d  val=%d", len(train_records), len(val_records))

    logger.info("Initializing FeatureService (PCA-%d) ...", PCA_DIM)
    fs = FeatureService(pca_components=PCA_DIM)
    feature_dim = fs.dimension

    logger.info("Encoding and embedding prompts (val) ...")
    val_data = _parse_and_embed(val_records, fs)
    val_burnin, val_eval = _split_data(val_data, ARM_ORDER)
    logger.info(
        "  val split → val_burnin=%d  val_eval=%d",
        val_burnin["n"], val_eval["n"],
    )

    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K3_WARMUP_PRIORS_PATH)

    # Fixed-model baselines (for AUC normalization)
    val_fixed_costs = [float(val_eval["costs"][a].mean()) for a in ARM_ORDER]
    val_fixed_rewards = [float(val_eval["rewards"][a].mean()) for a in ARM_ORDER]
    val_fixed_auc = pareto_auc(
        val_fixed_costs, val_fixed_rewards,
        min(val_fixed_costs), max(val_fixed_costs),
    )

    per_model_means = {
        a: float(np.mean([r["arms"][a]["cost"] for r in train_records]))
        for a in ARM_ORDER
    }
    budget_targets = list(np.geomspace(
        min(per_model_means.values()),
        max(per_model_means.values()),
        num=BUDGET_TARGET_COUNT,
    ))

    # ------------------------------------------------------------------
    # 2. Run Pareto knee-point selection for each T_adapt
    # ------------------------------------------------------------------
    per_t_adapt: Dict[str, Dict[str, Any]] = {}

    for t_adapt in T_ADAPT_VALUES:
        logger.info("\n" + "=" * 70)
        logger.info("T_ADAPT = %d", t_adapt)
        logger.info("=" * 70)

        gamma_neff = {
            g: round(_derive_n_eff(g, t_adapt), 1) for g in GAMMA_VALUES
        }
        logger.info("  gamma → n_eff:")
        for g, ne in sorted(gamma_neff.items()):
            logger.info("    γ=%.4f → n_eff=%.1f", g, ne)

        # Build configs for paretobandit only (tabula_rasa n_eff is always 1.0)
        configs: List[Dict[str, Any]] = []
        for alpha, gamma in itertools.product(ALPHA_VALUES, GAMMA_VALUES):
            configs.append({
                "alpha": alpha,
                "n_eff": gamma_neff[gamma],
                "gamma": gamma,
            })

        # --- Budget-paced Pareto AUC ---
        logger.info("\n  Budget-paced AUC (%d configs) ...", len(configs))
        auc_results: List[Dict[str, Any]] = []
        for idx, cfg in enumerate(configs):
            auc, auc_std, sweep = compute_budget_paced_pareto_auc(
                val_burnin, val_eval, registry, feature_dim,
                budget_targets,
                warmup_path=warmup_path,
                alpha=cfg["alpha"],
                n_eff=cfg["n_eff"],
                gamma=cfg["gamma"],
                n_seeds=N_SEEDS,
                seed_offset=SEED_OFFSET_VAL,
            )
            auc_results.append({
                "alpha": cfg["alpha"],
                "n_eff": cfg["n_eff"],
                "gamma": cfg["gamma"],
                "val_pareto_auc": round(auc, 6),
                "val_pareto_auc_std": round(auc_std, 6),
            })
            if (idx + 1) % 10 == 0:
                logger.info("    [%d/%d] ...", idx + 1, len(configs))

        # --- Failure resilience ---
        logger.info("  Failure resilience (%d configs) ...", len(configs))
        failure_results: List[Dict[str, Any]] = []
        for idx, cfg in enumerate(configs):
            p2_rwd, p2_std, p1_rwd = compute_failure_resilience(
                val_burnin, val_eval, registry, feature_dim,
                warmup_path=warmup_path,
                alpha=cfg["alpha"],
                n_eff=cfg["n_eff"],
                gamma=cfg["gamma"],
                n_seeds=N_SEEDS,
                seed_offset=SEED_OFFSET_FAILURE_VAL,
                failure_arms=FAILURE_ARMS,
                failure_reward=FAILURE_REWARD,
                budget_targets=FAILURE_BUDGET_TARGETS,
            )
            failure_results.append({
                "alpha": cfg["alpha"],
                "n_eff": cfg["n_eff"],
                "gamma": cfg["gamma"],
                "phase2_reward": round(p2_rwd, 4),
                "phase2_reward_std": round(p2_std, 4),
            })
            if (idx + 1) % 10 == 0:
                logger.info("    [%d/%d] ...", idx + 1, len(configs))

        # --- Pareto knee-point selection ---
        aucs = np.array([r["val_pareto_auc"] for r in auc_results])
        p2s = np.array([
            next(
                f["phase2_reward"]
                for f in failure_results if _cfg_matches(f, r)
            )
            for r in auc_results
        ])

        pareto_idx = _find_pareto_frontier(aucs, p2s)
        knee_idx = _find_knee_point(aucs, p2s, pareto_idx)

        knee_cfg = auc_results[knee_idx]
        knee_p2 = float(p2s[knee_idx])

        frontier = [
            {
                "alpha": auc_results[i]["alpha"],
                "n_eff": auc_results[i]["n_eff"],
                "gamma": auc_results[i]["gamma"],
                "val_pareto_auc": round(float(aucs[i]), 6),
                "phase2_reward": round(float(p2s[i]), 4),
                "is_knee": i == knee_idx,
            }
            for i in pareto_idx
        ]

        per_t_adapt[str(t_adapt)] = {
            "t_adapt": t_adapt,
            "alpha": knee_cfg["alpha"],
            "n_eff": knee_cfg["n_eff"],
            "gamma": knee_cfg["gamma"],
            "val_pareto_auc": knee_cfg["val_pareto_auc"],
            "val_phase2_reward": knee_p2,
            "pareto_frontier": frontier,
        }

        logger.info(
            "\n  KNEE POINT (T_adapt=%d): alpha=%.3f, γ=%.4f, n_eff=%.0f  "
            "AUC=%.6f  P2=%.4f",
            t_adapt, knee_cfg["alpha"], knee_cfg["gamma"],
            knee_cfg["n_eff"], knee_cfg["val_pareto_auc"], knee_p2,
        )

    # ------------------------------------------------------------------
    # 3. Stability assessment
    # ------------------------------------------------------------------
    selected_gammas = [
        per_t_adapt[str(t)]["gamma"] for t in T_ADAPT_VALUES
    ]
    gamma_unique = sorted(set(selected_gammas))
    gamma_range = max(selected_gammas) - min(selected_gammas)
    gamma_step = GAMMA_VALUES[1] - GAMMA_VALUES[0] if len(GAMMA_VALUES) > 1 else 0.001

    stable = gamma_range <= gamma_step
    logger.info("\n" + "=" * 70)
    logger.info("T_ADAPT SENSITIVITY SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "  Selected gammas: %s",
        {str(t): per_t_adapt[str(t)]["gamma"] for t in T_ADAPT_VALUES},
    )
    logger.info(
        "  Gamma range: %.4f (grid step: %.4f)",
        gamma_range, gamma_step,
    )
    logger.info(
        "  Stable (within one grid step): %s", "YES" if stable else "NO",
    )

    for t in T_ADAPT_VALUES:
        r = per_t_adapt[str(t)]
        logger.info(
            "  T_adapt=%4d  →  α=%.3f  γ=%.4f  n_eff=%7.0f  "
            "AUC=%.6f  P2=%.4f",
            t, r["alpha"], r["gamma"], r["n_eff"],
            r["val_pareto_auc"], r["val_phase2_reward"],
        )
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "experiment": "t_adapt_sensitivity",
        "description": (
            "Pareto knee-point selection repeated for multiple T_adapt values "
            "to verify stability.  Uses the same grid, data splits, seeds, "
            "and evaluation protocol as run_hparam_sweep.py."
        ),
        "t_adapt_values": T_ADAPT_VALUES,
        "grid": {
            "alpha_values": ALPHA_VALUES,
            "gamma_values": GAMMA_VALUES,
            "pca_dim": PCA_DIM,
            "n_seeds": N_SEEDS,
        },
        "val_fixed_auc": round(val_fixed_auc, 6),
        "per_t_adapt": per_t_adapt,
        "stability": {
            "selected_gammas": {str(t): per_t_adapt[str(t)]["gamma"] for t in T_ADAPT_VALUES},
            "gamma_range": round(gamma_range, 4),
            "gamma_grid_step": round(gamma_step, 4),
            "within_one_grid_step": stable,
        },
        "stable": stable,
    }

    out_path = RESULTS_DIR / "t_adapt_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nResults written to %s", out_path)

    elapsed = time.time() - t0
    logger.info("Wall time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
