#!/usr/bin/env python3
"""Appendix: Multi-Seed Adaptive Gamma Trajectories.

Records per-seed, per-checkpoint forgetting factor trajectories for the
adaptive-gamma condition under both sub-experiments (reward shift and
cost shift).  The main text figure (Figure gamma_trajectory) shows
mean ± std; this appendix provides quartile bands and representative
individual seed traces to demonstrate cross-seed consistency.

Only the adaptive-gamma condition is evaluated (the figure's focus).
All other hyperparameters match Experiment 02a/02b exactly.

Protocol
--------
For each sub-experiment (reward shift, cost shift):
  - Phase 1 (893 steps): normal environment
  - Phase 2 (892 steps): shifted environment
  - 40 seeds (matching Experiment 02), checkpoint every 25 steps
  - Record gamma at each checkpoint for every seed

Usage::

    python experiments/appendix/gamma_trajectory_multiseed/run_gamma_trajectories.py
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
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service", "bandit_gpt.policy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Constants — match Experiment 02
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

GEMINI_ID: str = "google/gemini-2.5-pro"
GEMINI_NEW_INPUT_COST: float = 0.10
GEMINI_NEW_OUTPUT_COST: float = 0.10

PHASE1_N: int = 893
PHASE2_N: int = 892
COST_PENALTY: float = 0.20
PRIOR_N_EFFECTIVE: float = 50.0
ALPHA: float = 0.5
CHECKPOINT_INTERVAL: int = 25

N_SEEDS: int = 40
SEED_OFFSET: int = 4000
RESULTS_DIR = Path(__file__).parent / "results"


# ======================================================================
# Data helpers
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


def _apply_gemini_cost_reduction(split: SplitData) -> SplitData:
    """Scale Gemini costs to reflect the price drop.

    Parameters
    ----------
    split : SplitData
        Data with original Gemini pricing.

    Returns
    -------
    SplitData
        New split with scaled Gemini costs.
    """
    old_avg = (1.25 + 10.0) / 2.0
    new_avg = (GEMINI_NEW_INPUT_COST + GEMINI_NEW_OUTPUT_COST) / 2.0
    scale = new_avg / old_avg

    new_costs = dict(split.costs)
    new_costs[GEMINI_ID] = split.costs[GEMINI_ID] * scale
    return SplitData(
        prompts=split.prompts,
        rewards=split.rewards,
        costs=new_costs,
        embeddings=split.embeddings,
    )


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
) -> BanditRouter:
    """Build a K=3 router with warmup priors and adaptive gamma."""
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
        forgetting_factor=1.0,
        drift_threshold=0.0,
        policy="disjoint",
        adaptive_gamma=True,
    )


# ======================================================================
# Trajectory recording
# ======================================================================


def _record_gamma_trajectories(
    phase1: SplitData,
    phase2_online: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    normalized_costs: Dict[str, float],
    *,
    apply_registry_update: bool = False,
) -> Dict[str, Any]:
    """Run adaptive-gamma and record per-seed gamma at each checkpoint.

    Parameters
    ----------
    phase1, phase2_online : SplitData
        Phase 1 (normal) and Phase 2 (shifted) online data.
    registry : dict
        Model registry.
    feature_dim : int
        Context dimensionality.
    normalized_costs : dict
        Per-model normalized costs for Phase 1 (and Phase 2 if no
        registry update).
    apply_registry_update : bool
        If True, update the registry at the phase boundary to reflect
        Gemini price drop (cost-shift scenario).

    Returns
    -------
    dict
        ``checkpoints`` (list of step indices) and ``per_seed_gamma``
        (list of lists, one per seed).
    """
    n_p1 = phase1.n
    n_p2 = phase2_online.n
    n_total = n_p1 + n_p2

    checkpoints = sorted(set(
        list(range(CHECKPOINT_INTERVAL, n_total, CHECKPOINT_INTERVAL))
        + [n_total]
    ))
    checkpoint_set = set(checkpoints)

    per_seed_gamma: List[List[float]] = []

    for s in range(N_SEEDS):
        seed = SEED_OFFSET + s
        rng = np.random.default_rng(seed)

        router = _create_router(registry, feature_dim)
        registry_updated = False

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

        seed_gammas: List[float] = []

        for t in range(n_total):
            if apply_registry_update and t == n_p1 and not registry_updated:
                router.registry[GEMINI_ID]["input_cost_per_m"] = GEMINI_NEW_INPUT_COST
                router.registry[GEMINI_ID]["output_cost_per_m"] = GEMINI_NEW_OUTPUT_COST
                registry_updated = True

            emb = all_emb[t]
            model, log = router.route(emb)
            reward = float(all_rewards[model][t])
            cost = float(all_costs[model][t])

            log.cost_usd = cost
            router.process_feedback(log.request_id, reward=reward)

            step = t + 1
            if step in checkpoint_set:
                seed_gammas.append(float(router.bandit.gamma))

        per_seed_gamma.append(seed_gammas)

    return {
        "checkpoints": checkpoints,
        "per_seed_gamma": per_seed_gamma,
        "phase_boundary": n_p1,
        "n_seeds": N_SEEDS,
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

    registry = build_model_registry(ARM_ORDER)
    normalized_costs = compute_normalized_costs(registry, ARM_ORDER)

    # ---- Reward shift ----
    logger.info("\n=== Reward Shift ===")
    phase2_reward_swap = _apply_reward_swap(phase2_raw, *SWAP_ARMS)
    reward_shift_data = _record_gamma_trajectories(
        phase1, phase2_reward_swap, registry, feature_dim, normalized_costs,
    )
    logger.info(
        "  Recorded %d checkpoints x %d seeds",
        len(reward_shift_data["checkpoints"]),
        reward_shift_data["n_seeds"],
    )

    # ---- Cost shift ----
    logger.info("\n=== Cost Shift ===")
    phase2_cost_drop = _apply_gemini_cost_reduction(phase2_raw)
    cost_shift_data = _record_gamma_trajectories(
        phase1, phase2_cost_drop, registry, feature_dim, normalized_costs,
        apply_registry_update=True,
    )
    logger.info(
        "  Recorded %d checkpoints x %d seeds",
        len(cost_shift_data["checkpoints"]),
        cost_shift_data["n_seeds"],
    )

    output: Dict[str, Any] = {
        "experiment": "gamma_trajectory_multiseed",
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "reward_shift": reward_shift_data,
        "cost_shift": cost_shift_data,
    }

    out_path = RESULTS_DIR / "gamma_trajectory_multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nSaved results to %s", out_path)
    logger.info("Wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
