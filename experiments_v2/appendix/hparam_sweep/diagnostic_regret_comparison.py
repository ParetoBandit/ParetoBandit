#!/usr/bin/env python3
"""Diagnostic: compare regret distributions at alpha=0.5, n_eff=50.

Runs budget-paced and non-stationary simulations on the **same val split**
with a consistent pure-reward oracle to determine how the two regret
signals compare and what standardization is needed for a combined metric.

Usage::

    python experiments_v2/appendix/hparam_sweep/diagnostic_regret_comparison.py
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

from bandit_gpt.budget_pacer import BudgetPacer, PacingMode
from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    K3_ARM_ORDER,
    K3_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.simulation import build_model_registry, compute_normalized_costs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

ALPHA = 0.5
N_EFF = 50.0
PCA_DIM = 25
N_SEEDS = 10
SEED_OFFSET = 0
BUDGET_TARGET_COUNT = 7
NONSTAT_COST_PENALTY = 0.2
NONSTAT_SWAP_PAIRS: List[Tuple[str, str]] = [
    ("meta-llama/llama-3.1-8b-instruct", "mistralai/mistral-large-2512"),
    ("meta-llama/llama-3.1-8b-instruct", "google/gemini-2.5-pro"),
    ("mistralai/mistral-large-2512", "google/gemini-2.5-pro"),
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _parse_and_embed(
    records: List[Dict[str, Any]], fs: FeatureService
) -> Dict[str, Any]:
    prompts = [r["prompt"] for r in records]
    rewards: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    costs: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    for r in records:
        for arm_id in ARM_ORDER:
            info = r["arms"][arm_id]
            rewards[arm_id].append(info["reward"])
            costs[arm_id].append(info["cost"])
    embeddings = fs.extract_features_batch(prompts)
    return {
        "rewards": {a: np.array(v) for a, v in rewards.items()},
        "costs": {a: np.array(v) for a, v in costs.items()},
        "embeddings": embeddings,
        "n": len(prompts),
    }


def _pure_reward_oracle(data: Dict[str, Any], idx: int) -> float:
    """Max reward across all arms for prompt idx (no cost in oracle)."""
    return max(float(data["rewards"][a][idx]) for a in ARM_ORDER)


def _simulate_budget_paced(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup_path: str,
    budget_target: float,
    seed: int,
) -> Dict[str, Any]:
    """Budget-paced simulation returning per-step regret on val.

    Oracle is pure reward (max arm reward), consistent across both
    diagnostic conditions.
    """
    rng = np.random.default_rng(seed)
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    pacer = BudgetPacer(
        target_avg_spend_usd=budget_target,
        mode=PacingMode.ADAPTIVE,
        lr=0.05,
        lambda_max=5.0,
    )

    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup",
        warmup_path=warmup_path,
        prior_n_effective=N_EFF,
        alpha=ALPHA,
        use_corralling=False,
        cost_penalty=0.0,
        forgetting_factor=1.0,
        drift_threshold=0.0,
        policy="disjoint",
        adaptive_gamma=False,
        budget_pacer=pacer,
    )

    train_order = rng.permutation(train_data["n"])
    for i in train_order:
        model, log = router.route(train_data["embeddings"][i])
        reward = float(train_data["rewards"][model][i])
        log.cost_usd = float(train_data["costs"][model][i])
        router.process_feedback(log.request_id, reward=reward)

    val_order = rng.permutation(val_data["n"])
    cum_regret = 0.0
    per_step_regret: List[float] = []
    rewards_collected: List[float] = []
    costs_collected: List[float] = []

    for idx in val_order:
        model, log = router.route(val_data["embeddings"][idx])
        reward = float(val_data["rewards"][model][idx])
        cost = float(val_data["costs"][model][idx])
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

        oracle = _pure_reward_oracle(val_data, idx)
        cum_regret += oracle - reward
        per_step_regret.append(cum_regret)
        rewards_collected.append(reward)
        costs_collected.append(cost)

    return {
        "cumulative_regret": cum_regret,
        "per_step_regret": per_step_regret,
        "mean_reward": float(np.mean(rewards_collected)),
        "mean_cost": float(np.mean(costs_collected)),
        "n_steps": len(val_order),
    }


def _simulate_nonstationary(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    warmup_path: str,
    swap_arms: Tuple[str, str],
    seed: int,
) -> Dict[str, Any]:
    """Two-phase non-stationary simulation with adaptive gamma.

    Oracle is pure reward (max arm reward) using the phase-appropriate
    reward arrays, consistent with the budget-paced diagnostic.
    """
    rng = np.random.default_rng(seed)
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup",
        warmup_path=warmup_path,
        prior_n_effective=N_EFF,
        alpha=ALPHA,
        use_corralling=False,
        cost_penalty=NONSTAT_COST_PENALTY,
        forgetting_factor=1.0,
        drift_threshold=0.0,
        policy="disjoint",
        adaptive_gamma=True,
        budget_pacer=None,
    )

    train_order = rng.permutation(train_data["n"])
    for i in train_order:
        model, log = router.route(train_data["embeddings"][i])
        reward = float(train_data["rewards"][model][i])
        router.process_feedback(log.request_id, reward=reward)

    a1, a2 = swap_arms
    swapped_rewards = dict(val_data["rewards"])
    swapped_rewards[a1] = val_data["rewards"][a2]
    swapped_rewards[a2] = val_data["rewards"][a1]

    n_val = val_data["n"]
    val_order = rng.permutation(n_val)
    mid = n_val // 2

    p1_cum_regret = 0.0
    p1_per_step: List[float] = []
    for idx in val_order[:mid]:
        model, log = router.route(val_data["embeddings"][idx])
        reward = float(val_data["rewards"][model][idx])
        router.process_feedback(log.request_id, reward=reward)
        oracle = _pure_reward_oracle(val_data, idx)
        p1_cum_regret += oracle - reward
        p1_per_step.append(p1_cum_regret)

    p2_cum_regret = 0.0
    p2_per_step: List[float] = []
    swapped_data = dict(val_data)
    swapped_data["rewards"] = swapped_rewards
    for idx in val_order[mid:]:
        model, log = router.route(val_data["embeddings"][idx])
        reward = float(swapped_rewards[model][idx])
        router.process_feedback(log.request_id, reward=reward)
        oracle = _pure_reward_oracle(swapped_data, idx)
        p2_cum_regret += oracle - reward
        p2_per_step.append(p2_cum_regret)

    return {
        "phase1_regret": p1_cum_regret,
        "phase2_regret": p2_cum_regret,
        "total_regret": p1_cum_regret + p2_cum_regret,
        "phase1_per_step": p1_per_step,
        "phase2_per_step": p2_per_step,
        "phase1_n": mid,
        "phase2_n": n_val - mid,
    }


def main() -> None:
    t0 = time.time()

    logger.info("Loading data ...")
    fs = FeatureService(pca_path=str(DEFAULT_PCA_PATH), pca_components=PCA_DIM)
    feature_dim = fs.dimension

    train_records = _load_jsonl(TRAIN_DATA_PATH)
    val_records = _load_jsonl(VAL_DATA_PATH)
    train_data = _parse_and_embed(train_records, fs)
    val_data = _parse_and_embed(val_records, fs)
    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K3_WARMUP_PRIORS_PATH)

    logger.info("  train=%d  val=%d  dim=%d", train_data["n"], val_data["n"], feature_dim)

    per_model_means = {
        a: float(np.mean(train_data["costs"][a])) for a in ARM_ORDER
    }
    lo = min(per_model_means.values())
    hi = max(per_model_means.values())
    budget_targets = list(np.geomspace(lo, hi, num=BUDGET_TARGET_COUNT))

    logger.info("Per-model mean costs (train):")
    for a in ARM_ORDER:
        logger.info("  %s: $%.8f", ARM_SHORT[a], per_model_means[a])
    logger.info(
        "Budget targets: %s",
        [f"${t:.6f}" for t in budget_targets],
    )

    # ==================================================================
    # Diagnostic A: Budget-Paced Regret
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC A: Budget-Paced Regret (alpha=%.2f, n_eff=%.0f)", ALPHA, N_EFF)
    logger.info("=" * 70)

    bp_results: Dict[float, List[Dict[str, Any]]] = {}
    for target in budget_targets:
        bp_results[target] = []
        for s in range(N_SEEDS):
            res = _simulate_budget_paced(
                train_data, val_data, registry, feature_dim,
                warmup_path=warmup_path,
                budget_target=target,
                seed=SEED_OFFSET + s,
            )
            bp_results[target].append(res)

        regrets = [r["cumulative_regret"] for r in bp_results[target]]
        n_steps = bp_results[target][0]["n_steps"]
        mean_r = float(np.mean(regrets))
        std_r = float(np.std(regrets, ddof=1))
        per_step = mean_r / n_steps
        logger.info(
            "  target=$%.6f  regret=%.1f ± %.1f  (%.4f/step)  "
            "reward=%.4f  cost=$%.8f",
            target, mean_r, std_r, per_step,
            float(np.mean([r["mean_reward"] for r in bp_results[target]])),
            float(np.mean([r["mean_cost"] for r in bp_results[target]])),
        )

    all_bp_regrets = [
        r["cumulative_regret"]
        for target_results in bp_results.values()
        for r in target_results
    ]
    n_steps_bp = bp_results[budget_targets[0]][0]["n_steps"]

    logger.info("\n  SUMMARY (across all targets and seeds):")
    logger.info("    n_steps: %d", n_steps_bp)
    logger.info("    Total regret: mean=%.1f  std=%.1f  min=%.1f  max=%.1f",
                np.mean(all_bp_regrets), np.std(all_bp_regrets),
                np.min(all_bp_regrets), np.max(all_bp_regrets))
    logger.info("    Per-step:     mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                np.mean(all_bp_regrets) / n_steps_bp,
                np.std(all_bp_regrets) / n_steps_bp,
                np.min(all_bp_regrets) / n_steps_bp,
                np.max(all_bp_regrets) / n_steps_bp)

    per_target_means = [
        float(np.mean([r["cumulative_regret"] for r in bp_results[t]]))
        for t in budget_targets
    ]
    logger.info("    Per-target means: %s",
                [f"{v:.1f}" for v in per_target_means])
    logger.info("    Average of per-target means: %.1f (%.4f/step)",
                np.mean(per_target_means),
                np.mean(per_target_means) / n_steps_bp)

    # ==================================================================
    # Diagnostic B: Non-Stationary Regret
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC B: Non-Stationary Regret (alpha=%.2f, n_eff=%.0f)", ALPHA, N_EFF)
    logger.info("=" * 70)

    ns_results: Dict[str, List[Dict[str, Any]]] = {}
    for a1, a2 in NONSTAT_SWAP_PAIRS:
        pair_key = f"{ARM_SHORT[a1]}↔{ARM_SHORT[a2]}"
        ns_results[pair_key] = []
        for s in range(N_SEEDS):
            res = _simulate_nonstationary(
                train_data, val_data, registry, feature_dim,
                warmup_path=warmup_path,
                swap_arms=(a1, a2),
                seed=SEED_OFFSET + s,
            )
            ns_results[pair_key].append(res)

        p2_regrets = [r["phase2_regret"] for r in ns_results[pair_key]]
        total_regrets = [r["total_regret"] for r in ns_results[pair_key]]
        p1_n = ns_results[pair_key][0]["phase1_n"]
        p2_n = ns_results[pair_key][0]["phase2_n"]
        logger.info(
            "  %s:  P2_regret=%.1f ± %.1f (%.4f/step)  "
            "total=%.1f ± %.1f  P1=%.1f",
            pair_key,
            np.mean(p2_regrets), np.std(p2_regrets, ddof=1),
            float(np.mean(p2_regrets)) / p2_n,
            np.mean(total_regrets), np.std(total_regrets, ddof=1),
            float(np.mean([r["phase1_regret"] for r in ns_results[pair_key]])),
        )

    all_p2_regrets = [
        r["phase2_regret"]
        for pair_results in ns_results.values()
        for r in pair_results
    ]
    all_total_regrets = [
        r["total_regret"]
        for pair_results in ns_results.values()
        for r in pair_results
    ]
    p2_n_steps = ns_results[list(ns_results.keys())[0]][0]["phase2_n"]
    total_n_steps = p2_n_steps + ns_results[list(ns_results.keys())[0]][0]["phase1_n"]

    logger.info("\n  SUMMARY (across all swap pairs and seeds):")
    logger.info("    phase1_n: %d  phase2_n: %d  total: %d",
                ns_results[list(ns_results.keys())[0]][0]["phase1_n"],
                p2_n_steps, total_n_steps)
    logger.info("    Phase 2 regret: mean=%.1f  std=%.1f  min=%.1f  max=%.1f",
                np.mean(all_p2_regrets), np.std(all_p2_regrets),
                np.min(all_p2_regrets), np.max(all_p2_regrets))
    logger.info("    Phase 2 /step:  mean=%.4f  std=%.4f",
                np.mean(all_p2_regrets) / p2_n_steps,
                np.std(all_p2_regrets) / p2_n_steps)
    logger.info("    Total regret:   mean=%.1f  std=%.1f  min=%.1f  max=%.1f",
                np.mean(all_total_regrets), np.std(all_total_regrets),
                np.min(all_total_regrets), np.max(all_total_regrets))
    logger.info("    Total /step:    mean=%.4f",
                np.mean(all_total_regrets) / total_n_steps)

    per_pair_p2_means = {
        k: float(np.mean([r["phase2_regret"] for r in v]))
        for k, v in ns_results.items()
    }
    logger.info("    Per-pair P2 means: %s",
                {k: f"{v:.1f}" for k, v in per_pair_p2_means.items()})

    # ==================================================================
    # Comparison
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)

    avg_bp_regret = float(np.mean(per_target_means))
    avg_ns_p2_regret = float(np.mean(all_p2_regrets))
    avg_ns_total_regret = float(np.mean(all_total_regrets))

    logger.info("  Budget-paced avg regret:      %.1f over %d steps (%.4f/step)",
                avg_bp_regret, n_steps_bp, avg_bp_regret / n_steps_bp)
    logger.info("  Non-stat Phase-2 avg regret:  %.1f over %d steps (%.4f/step)",
                avg_ns_p2_regret, p2_n_steps, avg_ns_p2_regret / p2_n_steps)
    logger.info("  Non-stat total avg regret:    %.1f over %d steps (%.4f/step)",
                avg_ns_total_regret, total_n_steps, avg_ns_total_regret / total_n_steps)

    logger.info("\n  Raw |Δ|:")
    logger.info("    |BP - NS_total|:  %.1f", abs(avg_bp_regret - avg_ns_total_regret))
    logger.info("    |BP - NS_P2|:     %.1f", abs(avg_bp_regret - avg_ns_p2_regret))

    logger.info("\n  Per-step |Δ|:")
    bp_rate = avg_bp_regret / n_steps_bp
    ns_total_rate = avg_ns_total_regret / total_n_steps
    ns_p2_rate = avg_ns_p2_regret / p2_n_steps
    logger.info("    |BP_rate - NS_total_rate|: %.4f", abs(bp_rate - ns_total_rate))
    logger.info("    |BP_rate - NS_P2_rate|:    %.4f", abs(bp_rate - ns_p2_rate))

    logger.info("\n  Ratio:")
    logger.info("    BP / NS_total: %.2f", avg_bp_regret / avg_ns_total_regret if avg_ns_total_regret > 0 else float("inf"))
    logger.info("    BP / NS_P2:    %.2f", avg_bp_regret / avg_ns_p2_regret if avg_ns_p2_regret > 0 else float("inf"))

    logger.info("\n  Step-count ratio: BP(%d) / NS_total(%d) = %.2f",
                n_steps_bp, total_n_steps, n_steps_bp / total_n_steps)

    logger.info("\nDone in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
