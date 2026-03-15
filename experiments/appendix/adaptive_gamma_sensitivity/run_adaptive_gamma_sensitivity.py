#!/usr/bin/env python3
"""Appendix: Adaptive Gamma Hyperparameter Sensitivity.

Sweeps the three user-configurable parameters of the adaptive forgetting
mechanism — EMA time constants (alpha_s, alpha_l), burn-in length, and
noise-margin multiplier — on the reward-shift scenario from Experiment 02a
to show that cumulative regret is robust across a wide range of settings.

Protocol
--------
We reuse the Experiment 02a two-phase reward-swap setup (Llama <-> Mistral,
K=3 portfolio) and evaluate the adaptive-gamma condition under different
parameter configurations.  All other hyperparameters (alpha, n_eff,
cost_penalty, etc.) are fixed at their Experiment 02 values.

Three one-at-a-time sweeps, each holding the others at their defaults:

1. **EMA grid** — alpha_s x alpha_l (4x4 = 16 configs)
2. **Burn-in sweep** — aw_burn_in_steps in {25, 50, 100}
3. **Noise-margin sweep** — aw_noise_margin_k in {1.0, 2.0, 3.0}

Default parameter vector (the production config):
    alpha_s = 0.1, alpha_l = 0.01, burn_in = 50, noise_margin_k = 2.0

Usage::

    python experiments/appendix/adaptive_gamma_sensitivity/run_adaptive_gamma_sensitivity.py
"""

from __future__ import annotations

import itertools
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
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_WARMUP_PRIORS_PATH,
    VAL_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.simulation import (
    SplitData,
    apply_reward_swap,
    build_model_registry,
    compute_normalized_costs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service", "bandit_gpt.policy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Constants — match Experiment 02a exactly
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
SEED_OFFSET: int = 8000
RESULTS_DIR = Path(__file__).parent / "results"

# ======================================================================
# Sweep grids
# ======================================================================

DEFAULT_ALPHA_S: float = 0.1
DEFAULT_ALPHA_L: float = 0.01
DEFAULT_BURN_IN: int = 50
DEFAULT_NOISE_K: float = 2.0

ALPHA_S_VALUES: List[float] = [0.05, 0.1, 0.2, 0.3]
ALPHA_L_VALUES: List[float] = [0.005, 0.01, 0.03, 0.05]
BURN_IN_VALUES: List[int] = [25, 50, 100]
NOISE_K_VALUES: List[float] = [1.0, 2.0, 3.0]


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


# ======================================================================
# Router factory
# ======================================================================


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    aw_alpha_short: float = DEFAULT_ALPHA_S,
    aw_alpha_long: float = DEFAULT_ALPHA_L,
    aw_burn_in_steps: int = DEFAULT_BURN_IN,
    aw_noise_margin_k: float = DEFAULT_NOISE_K,
) -> BanditRouter:
    """Build a K=3 router with warmup priors and adaptive gamma.

    Parameters
    ----------
    registry : dict
        Model registry from ``build_model_registry``.
    feature_dim : int
        Context vector dimensionality.
    aw_alpha_short : float
        Short-horizon EMA decay for the adaptive gamma mechanism.
    aw_alpha_long : float
        Long-horizon EMA decay for the adaptive gamma mechanism.
    aw_burn_in_steps : int
        Number of burn-in observations before activating adaptation.
    aw_noise_margin_k : float
        Multiplier for the dead-zone noise margin.

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
        forgetting_factor=1.0,
        drift_threshold=0.0,
        policy="disjoint",
        adaptive_gamma=True,
        aw_alpha_short=aw_alpha_short,
        aw_alpha_long=aw_alpha_long,
        aw_burn_in_steps=aw_burn_in_steps,
        aw_noise_margin_k=aw_noise_margin_k,
    )


# ======================================================================
# Single-config evaluation
# ======================================================================


def _evaluate_config(
    config: Dict[str, Any],
    phase1: SplitData,
    phase2_online: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    normalized_costs: Dict[str, float],
) -> Dict[str, Any]:
    """Run one adaptive-gamma configuration across all seeds.

    Returns a summary dict with per-seed terminal regret, mean/se,
    and phase-decomposed regret.

    Parameters
    ----------
    config : dict
        Must contain ``aw_alpha_short``, ``aw_alpha_long``,
        ``aw_burn_in_steps``, ``aw_noise_margin_k``.
    phase1, phase2_online : SplitData
        Phase 1 (normal) and Phase 2 (swapped) online data.
    registry : dict
        Model registry.
    feature_dim : int
        Context dimensionality.
    normalized_costs : dict
        Per-model normalized costs for cost-adjusted scoring.

    Returns
    -------
    dict
        Aggregated results including per-seed and summary statistics.
    """
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
            aw_alpha_short=config["aw_alpha_short"],
            aw_alpha_long=config["aw_alpha_long"],
            aw_burn_in_steps=config["aw_burn_in_steps"],
            aw_noise_margin_k=config["aw_noise_margin_k"],
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

    return {
        **config,
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
# Config generation
# ======================================================================


def _build_configs() -> List[Dict[str, Any]]:
    """Generate all parameter configurations to evaluate.

    Returns three groups:
    1. EMA grid: alpha_s x alpha_l (all combinations)
    2. Burn-in sweep (one-at-a-time, other params at defaults)
    3. Noise-margin sweep (one-at-a-time, other params at defaults)

    Duplicates (configs matching the default vector) are deduplicated.

    Returns
    -------
    list[dict]
        Each dict has ``aw_alpha_short``, ``aw_alpha_long``,
        ``aw_burn_in_steps``, ``aw_noise_margin_k``, and ``sweep_group``.
    """
    seen: set = set()
    configs: List[Dict[str, Any]] = []

    def _key(c: Dict[str, Any]) -> Tuple:
        return (c["aw_alpha_short"], c["aw_alpha_long"],
                c["aw_burn_in_steps"], c["aw_noise_margin_k"])

    # 1. EMA grid
    for alpha_s, alpha_l in itertools.product(ALPHA_S_VALUES, ALPHA_L_VALUES):
        c = {
            "aw_alpha_short": alpha_s,
            "aw_alpha_long": alpha_l,
            "aw_burn_in_steps": DEFAULT_BURN_IN,
            "aw_noise_margin_k": DEFAULT_NOISE_K,
            "sweep_group": "ema_grid",
        }
        k = _key(c)
        if k not in seen:
            seen.add(k)
            configs.append(c)

    # 2. Burn-in sweep
    for burn_in in BURN_IN_VALUES:
        c = {
            "aw_alpha_short": DEFAULT_ALPHA_S,
            "aw_alpha_long": DEFAULT_ALPHA_L,
            "aw_burn_in_steps": burn_in,
            "aw_noise_margin_k": DEFAULT_NOISE_K,
            "sweep_group": "burn_in",
        }
        k = _key(c)
        if k not in seen:
            seen.add(k)
            configs.append(c)

    # 3. Noise-margin sweep
    for noise_k in NOISE_K_VALUES:
        c = {
            "aw_alpha_short": DEFAULT_ALPHA_S,
            "aw_alpha_long": DEFAULT_ALPHA_L,
            "aw_burn_in_steps": DEFAULT_BURN_IN,
            "aw_noise_margin_k": noise_k,
            "sweep_group": "noise_margin",
        }
        k = _key(c)
        if k not in seen:
            seen.add(k)
            configs.append(c)

    return configs


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
    phase2_online = apply_reward_swap(phase2_raw, *SWAP_ARMS)

    registry = build_model_registry(ARM_ORDER)
    normalized_costs = compute_normalized_costs(registry, ARM_ORDER)
    logger.info("  Normalized costs: %s", {
        ARM_SHORT[a]: f"{v:.4f}" for a, v in normalized_costs.items()
    })

    configs = _build_configs()
    logger.info("Evaluating %d configurations x %d seeds ...", len(configs), N_SEEDS)

    results: List[Dict[str, Any]] = []
    for i, cfg in enumerate(configs):
        logger.info(
            "  [%d/%d] group=%s  α_s=%.3f  α_l=%.3f  burn_in=%d  noise_k=%.1f",
            i + 1, len(configs),
            cfg["sweep_group"],
            cfg["aw_alpha_short"],
            cfg["aw_alpha_long"],
            cfg["aw_burn_in_steps"],
            cfg["aw_noise_margin_k"],
        )
        result = _evaluate_config(
            cfg, phase1, phase2_online, registry, feature_dim, normalized_costs,
        )
        results.append(result)
        logger.info(
            "    regret=%.1f±%.1f  (P1=%.1f  P2=%.1f)",
            result["mean_regret"], result["se_regret"],
            result["mean_phase1_regret"], result["mean_phase2_regret"],
        )

    output: Dict[str, Any] = {
        "experiment": "adaptive_gamma_sensitivity",
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "phase1_n": PHASE1_N,
        "phase2_n": PHASE2_N,
        "cost_penalty": COST_PENALTY,
        "alpha": ALPHA,
        "prior_n_effective": PRIOR_N_EFFECTIVE,
        "defaults": {
            "aw_alpha_short": DEFAULT_ALPHA_S,
            "aw_alpha_long": DEFAULT_ALPHA_L,
            "aw_burn_in_steps": DEFAULT_BURN_IN,
            "aw_noise_margin_k": DEFAULT_NOISE_K,
        },
        "sweep_grids": {
            "alpha_s_values": ALPHA_S_VALUES,
            "alpha_l_values": ALPHA_L_VALUES,
            "burn_in_values": BURN_IN_VALUES,
            "noise_k_values": NOISE_K_VALUES,
        },
        "results": results,
    }

    out_path = RESULTS_DIR / "adaptive_gamma_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nSaved results to %s", out_path)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY — Adaptive Gamma Sensitivity")
    logger.info("=" * 80)
    logger.info("  %-8s  %-6s  %-6s  %-8s  %-8s  %8s  %8s  %8s",
                "Group", "α_s", "α_l", "Burn-in", "Noise-k",
                "Total", "Phase1", "Phase2")
    logger.info("  " + "-" * 76)
    for r in results:
        logger.info(
            "  %-8s  %-6.3f  %-6.3f  %-8d  %-8.1f  %7.1f±%.1f  %7.1f  %7.1f",
            r["sweep_group"][:8],
            r["aw_alpha_short"], r["aw_alpha_long"],
            r["aw_burn_in_steps"], r["aw_noise_margin_k"],
            r["mean_regret"], r["se_regret"],
            r["mean_phase1_regret"], r["mean_phase2_regret"],
        )
    logger.info("=" * 80)
    logger.info("Wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
