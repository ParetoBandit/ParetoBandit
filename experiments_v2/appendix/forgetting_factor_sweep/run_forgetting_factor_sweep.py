#!/usr/bin/env python3
"""Appendix: Forgetting Factor (γ) Sweep.

Evaluates a finer grid of fixed forgetting-factor values on the
Experiment 02a reward-shift scenario to justify γ=0.999 as the
default and reveal the regret-vs-γ curve shape.

The main text tests only three values {0.995, 0.999, 1.0}. This
appendix sweeps {0.99, 0.995, 0.997, 0.999, 0.9995, 1.0} plus the
adaptive-γ condition as a reference, showing that moderate forgetting
(γ ≈ 0.999) sits at the bottom of a shallow U-shaped regret curve.

Protocol
--------
Same two-phase reward-swap setup as Experiment 02a:
  - Phase 1 (893 steps): normal reward landscape
  - Phase 2 (892 steps): Llama ↔ Mistral column swap
  - 20 seeds, K=3 portfolio, warmup priors

All hyperparameters (alpha, n_eff, cost_penalty) match Experiment 02.

Usage::

    python experiments_v2/appendix/forgetting_factor_sweep/run_forgetting_factor_sweep.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from utils.simulation import SplitData, build_model_registry, compute_normalized_costs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Constants — match Experiment 02a
# ======================================================================

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

SWAP_ARMS: Tuple[str, str] = (
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
)

PHASE1_N: int = 893
PHASE2_N: int = 892
COST_PENALTY: float = 0.20
PRIOR_N_EFFECTIVE: float = 50.0
ALPHA: float = 0.5

N_SEEDS: int = 20
SEED_OFFSET: int = 9000
RESULTS_DIR = Path(__file__).parent / "results"

# ======================================================================
# Sweep grid
# ======================================================================

GAMMA_VALUES: List[float] = [0.99, 0.995, 0.997, 0.999, 0.9995, 1.0]

CONDITIONS: List[Dict[str, Any]] = [
    {
        "label": f"γ={g}",
        "forgetting_factor": g,
        "adaptive_gamma": False,
    }
    for g in GAMMA_VALUES
] + [
    {
        "label": "Adaptive γ",
        "forgetting_factor": 1.0,
        "adaptive_gamma": True,
    },
]


# ======================================================================
# Data helpers (identical to run_reward_shift.py)
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

    embeddings = fs.extract_features_batch(prompts)
    return SplitData(
        prompts=prompts,
        rewards={a: np.array(v) for a, v in rewards.items()},
        costs={a: np.array(v) for a, v in costs.items()},
        embeddings=embeddings,
    )


def _apply_reward_swap(
    split: SplitData,
    arm_a: str,
    arm_b: str,
) -> SplitData:
    """Return a new ``SplitData`` with rewards and costs swapped between two arms."""
    new_rewards = dict(split.rewards)
    new_costs = dict(split.costs)
    new_rewards[arm_a], new_rewards[arm_b] = (
        split.rewards[arm_b].copy(),
        split.rewards[arm_a].copy(),
    )
    new_costs[arm_a], new_costs[arm_b] = (
        split.costs[arm_b].copy(),
        split.costs[arm_a].copy(),
    )
    return SplitData(
        prompts=split.prompts,
        rewards=new_rewards,
        costs=new_costs,
        embeddings=split.embeddings,
    )


# ======================================================================
# Router factory
# ======================================================================


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    forgetting_factor: float = 1.0,
    adaptive_gamma: bool = False,
) -> BanditRouter:
    """Build a K=3 router with warmup priors.

    Parameters
    ----------
    registry : dict
        Model registry from ``build_model_registry``.
    feature_dim : int
        Context vector dimensionality.
    forgetting_factor : float
        Fixed forgetting factor (ignored when adaptive_gamma=True).
    adaptive_gamma : bool
        Enable the dual-EMA adaptive forgetting mechanism.

    Returns
    -------
    BanditRouter
    """
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    return BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup",
        warmup_path=str(K3_WARMUP_PRIORS_PATH),
        prior_n_effective=PRIOR_N_EFFECTIVE,
        alpha=ALPHA,
        use_corralling=False,
        cost_penalty=COST_PENALTY,
        forgetting_factor=forgetting_factor,
        drift_threshold=0.0,
        policy="disjoint",
        adaptive_gamma=adaptive_gamma,
    )


# ======================================================================
# Single-condition evaluation
# ======================================================================


def _evaluate_condition(
    condition: Dict[str, Any],
    phase1: SplitData,
    phase2_online: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    normalized_costs: Dict[str, float],
) -> Dict[str, Any]:
    """Run one forgetting-factor condition across all seeds.

    Parameters
    ----------
    condition : dict
        Must contain ``label``, ``forgetting_factor``, ``adaptive_gamma``.
    phase1, phase2_online : SplitData
        Phase 1 (normal) and Phase 2 (swapped) online data.
    registry : dict
        Model registry.
    feature_dim : int
        Context dimensionality.
    normalized_costs : dict
        Per-model normalized costs.

    Returns
    -------
    dict
        Aggregated results with per-seed regret and summary statistics.
    """
    label = condition["label"]
    n_p1 = phase1.n
    n_p2 = phase2_online.n

    per_seed_total: List[float] = []
    per_seed_phase1: List[float] = []
    per_seed_phase2: List[float] = []

    for s in range(N_SEEDS):
        seed = SEED_OFFSET + s
        rng = np.random.default_rng(seed)

        router = _create_router(
            registry,
            feature_dim,
            forgetting_factor=condition["forgetting_factor"],
            adaptive_gamma=condition.get("adaptive_gamma", False),
        )

        p1_order = rng.permutation(n_p1)
        p2_order = rng.permutation(n_p2)

        all_emb = np.concatenate([
            phase1.embeddings[p1_order],
            phase2_online.embeddings[p2_order],
        ], axis=0)

        all_rewards: Dict[str, np.ndarray] = {}
        all_costs: Dict[str, np.ndarray] = {}
        for arm in ARM_ORDER:
            all_rewards[arm] = np.concatenate([
                phase1.rewards[arm][p1_order],
                phase2_online.rewards[arm][p2_order],
            ])
            all_costs[arm] = np.concatenate([
                phase1.costs[arm][p1_order],
                phase2_online.costs[arm][p2_order],
            ])

        regret_phase1 = 0.0
        regret_phase2 = 0.0

        for t in range(n_p1 + n_p2):
            emb = all_emb[t]
            model, log = router.route(emb)
            reward = float(all_rewards[model][t])
            cost = float(all_costs[model][t])

            log.cost_usd = cost
            router.process_feedback(log.request_id, reward=reward)

            oracle_utility = max(
                float(all_rewards[a][t]) - COST_PENALTY * normalized_costs[a]
                for a in ARM_ORDER
            )
            chosen_utility = reward - COST_PENALTY * normalized_costs[model]
            step_regret = oracle_utility - chosen_utility

            if t < n_p1:
                regret_phase1 += step_regret
            else:
                regret_phase2 += step_regret

        per_seed_total.append(regret_phase1 + regret_phase2)
        per_seed_phase1.append(regret_phase1)
        per_seed_phase2.append(regret_phase2)

    total_arr = np.array(per_seed_total)
    p1_arr = np.array(per_seed_phase1)
    p2_arr = np.array(per_seed_phase2)

    half_life = (
        np.log(2) / (1 - condition["forgetting_factor"])
        if condition["forgetting_factor"] < 1.0
        else float("inf")
    )

    return {
        "label": label,
        "forgetting_factor": condition["forgetting_factor"],
        "adaptive_gamma": condition.get("adaptive_gamma", False),
        "effective_half_life": half_life,
        "mean_regret": float(total_arr.mean()),
        "se_regret": float(total_arr.std(ddof=1) / np.sqrt(N_SEEDS)),
        "std_regret": float(total_arr.std(ddof=1)),
        "mean_phase1_regret": float(p1_arr.mean()),
        "se_phase1_regret": float(p1_arr.std(ddof=1) / np.sqrt(N_SEEDS)),
        "mean_phase2_regret": float(p2_arr.mean()),
        "se_phase2_regret": float(p2_arr.std(ddof=1) / np.sqrt(N_SEEDS)),
        "per_seed_regret": [float(r) for r in per_seed_total],
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
    logger.info("  Online (val): %d prompts", train_all.n)

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
    phase2_online = _apply_reward_swap(phase2_raw, *SWAP_ARMS)

    registry = build_model_registry(ARM_ORDER)
    normalized_costs = compute_normalized_costs(registry, ARM_ORDER)
    logger.info("  Normalized costs: %s", {
        ARM_SHORT[a]: f"{v:.4f}" for a, v in normalized_costs.items()
    })

    logger.info("Evaluating %d conditions x %d seeds ...", len(CONDITIONS), N_SEEDS)

    results: List[Dict[str, Any]] = []
    for i, cond in enumerate(CONDITIONS):
        label = cond["label"]
        logger.info("  [%d/%d] %s", i + 1, len(CONDITIONS), label)
        result = _evaluate_condition(
            cond, phase1, phase2_online, registry, feature_dim, normalized_costs,
        )
        results.append(result)
        logger.info(
            "    regret=%.1f±%.1f  (P1=%.1f  P2=%.1f)  half-life=%.0f",
            result["mean_regret"], result["se_regret"],
            result["mean_phase1_regret"], result["mean_phase2_regret"],
            result["effective_half_life"],
        )

    output: Dict[str, Any] = {
        "experiment": "forgetting_factor_sweep",
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "phase1_n": PHASE1_N,
        "phase2_n": PHASE2_N,
        "cost_penalty": COST_PENALTY,
        "alpha": ALPHA,
        "prior_n_effective": PRIOR_N_EFFECTIVE,
        "gamma_values": GAMMA_VALUES,
        "results": results,
    }

    out_path = RESULTS_DIR / "forgetting_factor_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nSaved results to %s", out_path)

    logger.info("\n" + "=" * 80)
    logger.info("FORGETTING FACTOR SWEEP — Summary")
    logger.info("=" * 80)
    logger.info(
        "  %-18s  %10s  %8s  %8s  %8s",
        "Condition", "Half-life", "Total", "Phase1", "Phase2",
    )
    logger.info("  " + "-" * 62)
    for r in results:
        hl = f"{r['effective_half_life']:.0f}" if np.isfinite(r["effective_half_life"]) else "∞"
        logger.info(
            "  %-18s  %10s  %7.1f±%.1f  %7.1f  %7.1f",
            r["label"], hl,
            r["mean_regret"], r["se_regret"],
            r["mean_phase1_regret"], r["mean_phase2_regret"],
        )
    logger.info("=" * 80)
    logger.info("Wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
