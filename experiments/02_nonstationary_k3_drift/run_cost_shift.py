#!/usr/bin/env python3
"""Experiment 02b: Non-stationary K=3 Adaptation via Cost Shift.

Demonstrates that BanditGPT adapts when a model's pricing changes ---
the most common non-stationarity in production LLM routing.

Experimental setup
------------------
The router is deployed on a two-phase data stream constructed from
real benchmark data:

  **Phase 1** (steps 1--1000): Normal pricing.  Gemini-Pro is the
  most expensive model (normalized cost 0.67); Mistral-Large is
  utility-best under the cost penalty.  Gemini is rarely selected
  because the cost penalty outweighs its quality advantage.

  **Phase 2** (steps 1001--2000): **Gemini price drop** --- Gemini's
  pricing is reduced to Llama-level ($0.10/M tokens input+output).
  The router's model registry is updated at the boundary, so
  ``route()`` immediately computes the new normalized cost (~0.0).
  Post-drop, Gemini is both highest-quality *and* cheapest by
  cost-adjusted utility (0.933 vs Mistral's 0.840).

  The adaptation challenge: despite the immediate pricing signal,
  the bandit's theta estimates carry Phase-1 inertia.  Gemini was
  under-explored (rarely selected when expensive), so theta_gemini
  has high variance.  Mistral was heavily exploited, so
  theta_mistral is overly confident.  The forgetting factor decays
  this confidence, enabling faster convergence to Gemini.

Four conditions are compared at a fixed cost penalty (lambda=0.2):

  - **BanditGPT** (gamma=0.999): Warmup priors + forgetting.
  - **No forgetting** (gamma=1.0): Warmup, no adaptation.
  - **Fast forgetting** (gamma=0.995): Warmup, aggressive decay.
  - **Tabula Rasa**: No priors, learns from scratch.

Outputs (``results/``)
    cost_shift_results.json

Usage:
    python -m experiments.02_nonstationary_k3_drift.run_cost_shift
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

N_SEEDS: int = 40
SEED_OFFSET: int = 5000
RESULTS_DIR = Path(__file__).parent / "results"

PHASE1_N: int = 893
PHASE2_N: int = 892
COST_PENALTY: float = 0.20
CHECKPOINT_INTERVAL: int = 50

PRIOR_N_EFFECTIVE: float = 10.0
ALPHA_WARMUP: float = 0.1
ALPHA_TABULA_RASA: float = 0.01

CONDITIONS: List[Dict[str, Any]] = [
    {
        "label": "BanditGPT (γ=0.997)",
        "warmup": True,
        "forgetting_factor": 0.997,
        "alpha": ALPHA_WARMUP,
    },
    {
        "label": "No forgetting (γ=1.0)",
        "warmup": True,
        "forgetting_factor": 1.0,
        "alpha": ALPHA_WARMUP,
    },
    {
        "label": "Fast forgetting (γ=0.995)",
        "warmup": True,
        "forgetting_factor": 0.995,
        "alpha": ALPHA_WARMUP,
    },
    {
        "label": "Tabula Rasa",
        "warmup": False,
        "forgetting_factor": 0.999,
        "alpha": ALPHA_TABULA_RASA,
    },
]


# ======================================================================
# Data Loading
# ======================================================================


def _load_all(
    path: Path,
    fs: FeatureService,
    arm_order: List[str],
) -> SplitData:
    """Load all prompts from a JSONL file into a single ``SplitData``.

    Parameters
    ----------
    path : Path
        JSONL file with ``prompt``, ``arms`` fields.
    fs : FeatureService
        For encoding prompts into feature vectors.
    arm_order : list[str]
        Model identifiers.

    Returns
    -------
    SplitData
    """
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
    """Return a new ``SplitData`` with Gemini's costs scaled to new pricing.

    The per-prompt costs are scaled by the ratio of new to old average
    cost-per-million-tokens, preserving the per-prompt cost variance.

    Parameters
    ----------
    split : SplitData
        Original data.
    gemini_id : str
        Model ID for Gemini.
    old_input, old_output : float
        Original pricing ($/M tokens).
    new_input, new_output : float
        New pricing ($/M tokens).

    Returns
    -------
    SplitData
        New object with scaled Gemini costs.
    """
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
    """Return a deep copy of the registry with Gemini's pricing updated.

    Parameters
    ----------
    registry : dict
        Original model registry.
    gemini_id : str
        Model ID for Gemini.
    new_input, new_output : float
        New pricing ($/M tokens).

    Returns
    -------
    dict
        New registry with updated Gemini pricing.
    """
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
    alpha: float = ALPHA_WARMUP,
) -> BanditRouter:
    """Build a K=3 router with optional warmup priors.

    Parameters
    ----------
    registry : dict
        Model registry from ``build_model_registry``.
    feature_dim : int
        Context vector dimensionality.
    warmup : bool
        Whether to load warmup priors.
    forgetting_factor : float
        Exponential discount on prior observations.
    alpha : float
        LinUCB exploration coefficient.
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
        alpha=alpha,
        cost_penalty=COST_PENALTY,
        forgetting_factor=forgetting_factor,
        drift_threshold=0.0,
    )


# ======================================================================
# Frozen Holdout Evaluation
# ======================================================================


def _frozen_holdout_eval(
    router: BanditRouter,
    holdout: SplitData,
    arm_order: List[str],
    normalized_costs: Dict[str, float],
    cost_penalty: float,
) -> Dict[str, Any]:
    """Evaluate the router's current policy on holdout data.

    Uses ``bandit.select_arm()`` directly --- a pure read of the
    A_inv/b matrices with no state mutation.

    Parameters
    ----------
    router : BanditRouter
        Router with current learned policy.
    holdout : SplitData
        Held-out evaluation data.
    arm_order : list[str]
        Model identifiers.
    normalized_costs : dict[str, float]
        Per-model normalized costs for cost-adjusted scoring.
    cost_penalty : float
        Cost penalty weight.

    Returns
    -------
    dict
        Evaluation metrics: reward, cost, arm_counts.
    """
    cp: Optional[Dict[str, float]] = None
    if cost_penalty > 0:
        cp = {m: cost_penalty * normalized_costs[m] for m in arm_order}

    rewards: List[float] = []
    costs: List[float] = []
    arm_counts: Dict[str, int] = {a: 0 for a in arm_order}

    for j in range(holdout.n):
        model, _score = router.bandit.select_arm(
            holdout.embeddings[j], cost_penalties=cp,
        )
        rewards.append(float(holdout.rewards[model][j]))
        costs.append(float(holdout.costs[model][j]))
        arm_counts[model] += 1

    return {
        "reward": float(np.mean(rewards)),
        "cost": float(np.mean(costs)),
        "arm_counts": arm_counts,
    }


# ======================================================================
# Learning Curve Runner
# ======================================================================


def _run_learning_curve(
    condition: Dict[str, Any],
    phase1: SplitData,
    phase2_online: SplitData,
    phase2_holdout: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    normalized_costs_p1: Dict[str, float],
    normalized_costs_p2: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Run a two-phase learning curve with periodic holdout evaluation.

    Phase 1 streams prompts at original pricing; Phase 2 streams
    prompts at reduced Gemini pricing.  At the phase boundary, the
    router's registry is updated to reflect the new Gemini pricing.
    Frozen holdout evaluation uses Phase 2 normalized costs throughout
    (measuring policy quality under post-price-drop conditions).

    Parameters
    ----------
    condition : dict
        Condition definition with ``label``, ``warmup``,
        ``forgetting_factor``.
    phase1 : SplitData
        Prompts with original costs.
    phase2_online : SplitData
        Prompts with reduced Gemini costs for online learning.
    phase2_holdout : SplitData
        Prompts with reduced Gemini costs for frozen evaluation.
    registry : dict
        Model registry (original pricing).
    feature_dim : int
        Context dimensionality.
    normalized_costs_p1 : dict[str, float]
        Per-model normalized costs under original pricing.
    normalized_costs_p2 : dict[str, float]
        Per-model normalized costs under reduced Gemini pricing.

    Returns
    -------
    list[dict]
        Checkpoint data aggregated across seeds.
    """
    label = condition["label"]
    n_p1 = phase1.n
    n_p2 = phase2_online.n
    n_train = n_p1 + n_p2

    checkpoints = sorted(set(
        [0]
        + list(range(CHECKPOINT_INTERVAL, n_train, CHECKPOINT_INTERVAL))
        + [n_train]
    ))

    per_seed_curves: List[Dict[int, Dict[str, Any]]] = []

    for s in range(N_SEEDS):
        seed = SEED_OFFSET + s
        rng = np.random.default_rng(seed)

        router = _create_router(
            registry,
            feature_dim,
            warmup=condition["warmup"],
            forgetting_factor=condition["forgetting_factor"],
            alpha=condition.get("alpha", ALPHA_WARMUP),
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

        curve: Dict[int, Dict[str, Any]] = {}
        checkpoint_set = set(checkpoints)
        cumulative_regret: float = 0.0
        registry_updated = False

        def _eval() -> Dict[str, Any]:
            snapshot = _frozen_holdout_eval(
                router, phase2_holdout, ARM_ORDER,
                normalized_costs_p2, COST_PENALTY,
            )
            snapshot["cumulative_regret"] = cumulative_regret
            snapshot["forgetting_factor"] = router.bandit.gamma
            return snapshot

        if 0 in checkpoint_set:
            curve[0] = _eval()

        for t in range(n_train):
            if t == n_p1 and not registry_updated:
                router.registry[GEMINI_ID]["input_cost_per_m"] = GEMINI_NEW_INPUT_COST
                router.registry[GEMINI_ID]["output_cost_per_m"] = GEMINI_NEW_OUTPUT_COST
                registry_updated = True

            # Select normalized costs for regret based on current phase
            nc = normalized_costs_p1 if t < n_p1 else normalized_costs_p2

            emb = all_emb[t]
            model, log = router.route(emb)
            reward = float(all_rewards[model][t])
            cost = float(all_costs[model][t])

            log.cost_usd = cost
            router.process_feedback(log.request_id, reward=reward)

            oracle_utility = max(
                float(all_rewards[a][t]) - COST_PENALTY * nc[a]
                for a in ARM_ORDER
            )
            chosen_utility = reward - COST_PENALTY * nc[model]
            cumulative_regret += oracle_utility - chosen_utility

            step = t + 1
            if step in checkpoint_set:
                curve[step] = _eval()

        per_seed_curves.append(curve)

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    result: List[Dict[str, Any]] = []
    for step in checkpoints:
        seed_data = [c[step] for c in per_seed_curves if step in c]
        if not seed_data:
            continue

        rewards_agg = [d["reward"] for d in seed_data]
        costs_agg = [d["cost"] for d in seed_data]
        regrets = [d["cumulative_regret"] for d in seed_data]
        gammas = [d["forgetting_factor"] for d in seed_data]

        arm_frac: Dict[str, float] = {}
        arm_frac_std: Dict[str, float] = {}
        n_eval = phase2_holdout.n
        for arm in ARM_ORDER:
            fracs = [d["arm_counts"][arm] / n_eval for d in seed_data]
            arm_frac[ARM_SHORT[arm]] = float(np.mean(fracs))
            arm_frac_std[ARM_SHORT[arm]] = float(np.std(fracs))

        entry: Dict[str, Any] = {
            "step": step,
            "phase": "normal" if step <= n_p1 else "price-drop",
            "phase_boundary": n_p1,
            "mean_reward": float(np.mean(rewards_agg)),
            "std_reward": float(np.std(rewards_agg)),
            "se_reward": float(np.std(rewards_agg) / np.sqrt(len(rewards_agg))),
            "mean_cost": float(np.mean(costs_agg)),
            "std_cost": float(np.std(costs_agg)),
            "mean_cumulative_regret": float(np.mean(regrets)),
            "std_cumulative_regret": float(np.std(regrets)),
            "mean_forgetting_factor": float(np.mean(gammas)),
            "std_forgetting_factor": float(np.std(gammas)),
            "arm_fractions": arm_frac,
            "arm_fractions_std": arm_frac_std,
            "n_seeds": len(seed_data),
            "label": label,
        }

        if step == checkpoints[-1]:
            entry["per_seed_regret"] = [float(r) for r in regrets]

        result.append(entry)

    return result


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading K=3 data ...")
    fs = FeatureService()
    feature_dim = fs.dimension

    # Online stream uses val (unseen by warmup priors, which were
    # trained exclusively on train.jsonl).  Holdout uses the test split.
    train_all = _load_all(VAL_DATA_PATH, fs, ARM_ORDER)
    test_all = _load_all(HOLDOUT_DATA_PATH, fs, ARM_ORDER)

    logger.info("  Online (val): %d prompts", train_all.n)
    logger.info("  Holdout (test): %d prompts", test_all.n)

    # ---- Subsample Phase 1 and Phase 2 ----
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

    # ---- Compute original pricing info for cost scaling ----
    registry = build_model_registry(ARM_ORDER)
    gemini_meta = registry[GEMINI_ID]
    old_input = gemini_meta["input_cost_per_m"]
    old_output = gemini_meta["output_cost_per_m"]

    phase2_online = _apply_gemini_cost_reduction(
        phase2_raw, GEMINI_ID,
        old_input, old_output,
        GEMINI_NEW_INPUT_COST, GEMINI_NEW_OUTPUT_COST,
    )
    phase2_holdout = _apply_gemini_cost_reduction(
        test_all, GEMINI_ID,
        old_input, old_output,
        GEMINI_NEW_INPUT_COST, GEMINI_NEW_OUTPUT_COST,
    )

    logger.info(
        "  Phase 1: %d prompts (normal pricing)", phase1.n,
    )
    logger.info(
        "  Phase 2 online: %d prompts (Gemini price drop)", phase2_online.n,
    )
    logger.info(
        "  Phase 2 holdout: %d prompts (Gemini price drop)", phase2_holdout.n,
    )

    # ---- Compute normalized costs for both phases ----
    normalized_costs_p1 = compute_normalized_costs(registry, ARM_ORDER)
    registry_p2 = _build_phase2_registry(
        registry, GEMINI_ID,
        GEMINI_NEW_INPUT_COST, GEMINI_NEW_OUTPUT_COST,
    )
    normalized_costs_p2 = compute_normalized_costs(registry_p2, ARM_ORDER)

    logger.info("  Phase 1 normalized costs: %s", {
        ARM_SHORT[a]: f"{v:.4f}" for a, v in normalized_costs_p1.items()
    })
    logger.info("  Phase 2 normalized costs: %s", {
        ARM_SHORT[a]: f"{v:.4f}" for a, v in normalized_costs_p2.items()
    })

    # ---- Summarize utility landscape shift ----
    for label, split, nc in [
        ("Phase 1 (normal pricing)", phase1, normalized_costs_p1),
        ("Phase 2 (Gemini price drop)", phase2_online, normalized_costs_p2),
    ]:
        mean_rewards: Dict[str, float] = {}
        mean_utilities: Dict[str, float] = {}
        for a in ARM_ORDER:
            mean_rewards[a] = float(np.mean(split.rewards[a]))
            mean_utilities[a] = mean_rewards[a] - COST_PENALTY * nc[a]
        utility_best = max(ARM_ORDER, key=lambda a: mean_utilities[a])
        logger.info(
            "  %s  rewards: %s  utilities: %s  best: %s",
            label,
            {ARM_SHORT[a]: f"{v:.3f}" for a, v in mean_rewards.items()},
            {ARM_SHORT[a]: f"{v:.3f}" for a, v in mean_utilities.items()},
            ARM_SHORT[utility_best],
        )

    # ---- Run conditions ----
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for cond in CONDITIONS:
        label = cond["label"]
        logger.info("\n=== %s ===", label)
        curve = _run_learning_curve(
            cond, phase1, phase2_online, phase2_holdout,
            registry, feature_dim,
            normalized_costs_p1, normalized_costs_p2,
        )
        all_results[label] = curve

        final = curve[-1]
        logger.info(
            "  Final: reward=%.4f+/-%.4f  cost=$%.6f  regret=%.1f  arm=%s",
            final["mean_reward"], final["se_reward"],
            final["mean_cost"],
            final["mean_cumulative_regret"],
            {k: f"{v:.0%}" for k, v in final["arm_fractions"].items()},
        )

    # ---- Save results ----
    output: Dict[str, Any] = {
        "experiment": "02b_cost_shift",
        "arm_order": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "cost_shift_model": GEMINI_ID,
        "gemini_new_input_cost": GEMINI_NEW_INPUT_COST,
        "gemini_new_output_cost": GEMINI_NEW_OUTPUT_COST,
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "phase1_n": phase1.n,
        "phase2_online_n": phase2_online.n,
        "phase2_holdout_n": phase2_holdout.n,
        "cost_penalty": COST_PENALTY,
        "prior_n_effective": PRIOR_N_EFFECTIVE,
        "alpha_warmup": ALPHA_WARMUP,
        "alpha_tabula_rasa": ALPHA_TABULA_RASA,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "normalized_costs_p1": {
            ARM_SHORT[a]: v for a, v in normalized_costs_p1.items()
        },
        "normalized_costs_p2": {
            ARM_SHORT[a]: v for a, v in normalized_costs_p2.items()
        },
        "conditions": all_results,
    }

    out_path = RESULTS_DIR / "cost_shift_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nSaved results to %s", out_path)

    # ---- Summary table ----
    logger.info("\n" + "=" * 80)
    logger.info("COST SHIFT — Final Checkpoint Summary")
    logger.info("=" * 80)
    logger.info(
        "  %-22s  %8s  %10s  %8s  %s",
        "Condition", "Reward", "Cost", "Regret", "Arm Fractions",
    )
    logger.info("  " + "-" * 76)
    for label, curve in all_results.items():
        final = curve[-1]
        arm_str = "  ".join(
            f"{k}={v:.0%}" for k, v in final["arm_fractions"].items()
        )
        logger.info(
            "  %-22s  %8.4f  $%9.6f  %8.1f  %s",
            label,
            final["mean_reward"],
            final["mean_cost"],
            final["mean_cumulative_regret"],
            arm_str,
        )
    logger.info("=" * 80)
    logger.info("Wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
