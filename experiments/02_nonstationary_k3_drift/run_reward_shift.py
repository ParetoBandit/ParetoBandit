#!/usr/bin/env python3
"""Experiment 02: Non-stationary K=3 Adaptation via Reward Swap.

Demonstrates that ParetoBandit's combination of warmup priors and
geometric forgetting enables automatic adaptation when a model's
quality changes — a common production event as LLM providers iterate
on their APIs.

Experimental setup
------------------
The router is deployed on a two-phase data stream constructed from
real benchmark data:

  **Phase 1** (steps 1--893): Normal reward landscape.  The warmup
  priors (trained on the full K=3 training set) are well-calibrated.
  Mistral-Large and Gemini-Pro dominate; Llama-8B is weakest but
  cheapest.

  **Phase 2** (steps 894--1785): **Reward swap** — Llama-8B and
  Mistral-Large exchange their per-prompt reward columns, simulating
  provider-side quality changes: Llama receives a major quality
  upgrade while Mistral regresses.  API pricing (costs) is unchanged,
  so the router must navigate a new quality--cost tradeoff where
  Llama-8B (previously cheapest-but-worst) now delivers the highest
  quality at its original low price, while Mistral-Large (previously
  utility-best) drops to worst quality at its original high price.
  Cost-level shifts are tested separately in Experiment 03.

  Gemini-Pro is unchanged (anchor), preserving a realistic three-way
  competition.

Four conditions are compared at a fixed cost penalty (λ=0.2),
representing increasing levels of routing sophistication:

  - **Fixed Policy (offline)**: Warmup priors deployed frozen — the
    industry standard pattern of training offline and deploying without
    online adaptation.  Under drift it is helpless.
  - **Naive Bandit (γ=1.0)**: LinUCB with infinite memory and warmup
    priors.  The obvious first attempt at online routing — adapts, but
    Phase 1 inertia dilutes Phase 2 signal.
  - **SW-UCB (W=200)**: Sliding-Window LinUCB without priors.  Retains
    only the last W observations with equal weighting (Garivier &
    Moulines 2011).  A structurally different non-stationary baseline.
  - **ParetoBandit (γ=0.995)**: Warmup priors with jointly-tuned
    geometric forgetting.  Effective memory ~200 steps.

Outputs (``results/``)
    reward_shift_results.json

Usage:
    python -m experiments.02_nonstationary_k3_drift.run_reward_shift
"""

from __future__ import annotations

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

from pareto_bandit.config import (
    BEST_K3_HPARAMS,
    BEST_K3_SW_UCB_HPARAMS,
    DEFAULT_NONSTAT_COST_PENALTY,
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_DEFAULT_SWAP_ARMS,
    K3_WARMUP_PRIORS_PATH,
    N_SEEDS,
    VAL_DATA_PATH,
)
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.policy import SlidingWindowLinUCBPolicy
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
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
for _noisy in ("pareto_bandit.router", "pareto_bandit.feature_service", "pareto_bandit.policy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Constants
# ======================================================================

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = K3_ARM_SHORT

SWAP_ARMS = K3_DEFAULT_SWAP_ARMS

SEED_OFFSET: int = 4000
RESULTS_DIR = Path(__file__).parent / "results"

PHASE1_N: int = 893
PHASE2_N: int = 892
COST_PENALTY: float = DEFAULT_NONSTAT_COST_PENALTY
CHECKPOINT_INTERVAL: int = 50

PRIOR_N_EFFECTIVE: float = BEST_K3_HPARAMS["prior_n_effective"]
ALPHA_WARMUP: float = BEST_K3_HPARAMS["alpha"]
ALPHA_SW_UCB: float = BEST_K3_SW_UCB_HPARAMS["alpha"]
SW_UCB_WINDOW: int = BEST_K3_SW_UCB_HPARAMS["window_size"]

CONDITIONS: List[Dict[str, Any]] = [
    {
        "label": "Fixed Policy (offline)",
        "warmup": True,
        "forgetting_factor": 1.0,
        "alpha": ALPHA_WARMUP,
        "online_learn": False,
    },
    {
        "label": "Naive Bandit (γ=1.0)",
        "warmup": True,
        "forgetting_factor": 1.0,
        "alpha": ALPHA_WARMUP,
        "online_learn": True,
    },
    {
        "label": f"SW-UCB (W={SW_UCB_WINDOW})",
        "warmup": False,
        "forgetting_factor": 1.0,
        "alpha": ALPHA_SW_UCB,
        "online_learn": True,
        "window_size": SW_UCB_WINDOW,
    },
    {
        "label": "ParetoBandit (γ=0.995)",
        "warmup": True,
        "forgetting_factor": BEST_K3_HPARAMS["forgetting_factor"],
        "alpha": ALPHA_WARMUP,
        "online_learn": True,
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
    window_size: int = 0,
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
    window_size : int
        If > 0, replace the default policy with a
        :class:`SlidingWindowLinUCBPolicy` retaining the last
        *window_size* observations (SW-UCB baseline).
    """
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if warmup else "none",
        warmup_path=str(K3_WARMUP_PRIORS_PATH) if warmup else None,
        prior_n_effective=PRIOR_N_EFFECTIVE,
        alpha=alpha,
        cost_penalty=COST_PENALTY,
        forgetting_factor=forgetting_factor,
    )
    if window_size > 0:
        router.bandit = SlidingWindowLinUCBPolicy(
            model_names=ARM_ORDER,
            dim=feature_dim,
            alpha=alpha,
            window_size=window_size,
        )
    return router


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

    Uses ``bandit.select_arm()`` directly — a pure read of the
    A_inv/b matrices with no state mutation.

    Parameters
    ----------
    router : BanditRouter
        Router with current learned policy.
    holdout : SplitData
        Held-out evaluation data with Phase 2 reward landscape
        (reward-swapped, costs unchanged).
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
    normalized_costs: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Run a two-phase learning curve with periodic holdout evaluation.

    Phase 1 streams normal-reward prompts; Phase 2 streams reward-swapped
    prompts.  Within each phase, prompts are shuffled per seed.  Frozen
    holdout evaluation (on Phase 2 holdout with swapped rewards) is
    performed at every checkpoint.

    Parameters
    ----------
    condition : dict
        Condition definition with ``label``, ``warmup``,
        ``forgetting_factor``.
    phase1 : SplitData
        In-distribution prompts (normal rewards).
    phase2_online : SplitData
        Reward-swapped prompts for online learning.
    phase2_holdout : SplitData
        Reward-swapped prompts for frozen evaluation.
    registry : dict
        Model registry.
    feature_dim : int
        Context dimensionality.
    normalized_costs : dict[str, float]
        Per-model normalized costs.

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
        + [n_p1]  # exact phase boundary for precise phase-split regret
        + [n_train]
    ))

    online_learn = condition.get("online_learn", True)
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
            window_size=condition.get("window_size", 0),
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

        def _eval() -> Dict[str, Any]:
            snapshot = _frozen_holdout_eval(
                router, phase2_holdout, ARM_ORDER,
                normalized_costs, COST_PENALTY,
            )
            snapshot["cumulative_regret"] = cumulative_regret
            snapshot["forgetting_factor"] = router.bandit.gamma
            return snapshot

        if 0 in checkpoint_set:
            curve[0] = _eval()

        for t in range(n_train):
            emb = all_emb[t]
            model, log = router.route(emb)
            reward = float(all_rewards[model][t])
            cost = float(all_costs[model][t])

            log.cost_usd = cost
            if online_learn:
                router.process_feedback(log.request_id, reward=reward)

            oracle_utility = max(
                float(all_rewards[a][t]) - COST_PENALTY * normalized_costs[a]
                for a in ARM_ORDER
            )
            chosen_utility = reward - COST_PENALTY * normalized_costs[model]
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
        per_seed_arm_fracs: Dict[str, List[float]] = {}
        n_eval = phase2_holdout.n
        for arm in ARM_ORDER:
            fracs = [d["arm_counts"][arm] / n_eval for d in seed_data]
            short = ARM_SHORT[arm]
            arm_frac[short] = float(np.mean(fracs))
            arm_frac_std[short] = float(np.std(fracs))
            per_seed_arm_fracs[short] = [float(f) for f in fracs]

        entry: Dict[str, Any] = {
            "step": step,
            "phase": "normal" if step <= n_p1 else "swapped",
            "phase_boundary": n_p1,
            "mean_reward": float(np.mean(rewards_agg)),
            "std_reward": float(np.std(rewards_agg)),
            "se_reward": float(np.std(rewards_agg) / np.sqrt(len(rewards_agg))),
            "mean_cost": float(np.mean(costs_agg)),
            "std_cost": float(np.std(costs_agg)),
            "mean_cumulative_regret": float(np.mean(regrets)),
            "std_cumulative_regret": float(np.std(regrets)),
            "per_seed_cumulative_regret": [float(r) for r in regrets],
            "mean_forgetting_factor": float(np.mean(gammas)),
            "std_forgetting_factor": float(np.std(gammas)),
            "arm_fractions": arm_frac,
            "arm_fractions_std": arm_frac_std,
            "per_seed_arm_fractions": per_seed_arm_fracs,
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
    # PCA projection is pre-fitted on ~46K disjoint LMSYS prompts and frozen;
    # only .transform() is called during evaluation (no leakage).
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

    phase2_online = apply_reward_swap(phase2_raw, *SWAP_ARMS)
    phase2_holdout = apply_reward_swap(test_all, *SWAP_ARMS)

    logger.info(
        "  Phase 1: %d prompts (normal rewards)",
        phase1.n,
    )
    logger.info(
        "  Phase 2 online: %d prompts (Llama <-> Mistral rewards swapped)",
        phase2_online.n,
    )
    logger.info(
        "  Phase 2 holdout: %d prompts (rewards swapped, costs unchanged)",
        phase2_holdout.n,
    )

    # ---- Compute normalized costs ----
    registry = build_model_registry(ARM_ORDER)
    normalized_costs = compute_normalized_costs(registry, ARM_ORDER)
    logger.info("  Normalized costs: %s", {
        ARM_SHORT[a]: f"{v:.4f}" for a, v in normalized_costs.items()
    })

    # ---- Summarize reward landscape shift ----
    for label, split in [
        ("Phase 1 (normal)", phase1),
        ("Phase 2 (swapped)", phase2_online),
    ]:
        best_arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
        mean_rewards: Dict[str, float] = {}
        for a in ARM_ORDER:
            mean_rewards[a] = float(np.mean(split.rewards[a]))
        for i in range(split.n):
            best = max(ARM_ORDER, key=lambda a: split.rewards[a][i])
            best_arm_counts[best] += 1
        logger.info("  %s best-arm: %s  means: %s", label,
            {ARM_SHORT[a]: f"{c / split.n:.0%}" for a, c in best_arm_counts.items()},
            {ARM_SHORT[a]: f"{v:.3f}" for a, v in mean_rewards.items()},
        )

    # ---- Run conditions ----
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for cond in CONDITIONS:
        label = cond["label"]
        logger.info("\n=== %s ===", label)
        curve = _run_learning_curve(
            cond, phase1, phase2_online, phase2_holdout,
            registry, feature_dim, normalized_costs,
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
        "experiment": "02_reward_shift",
        "arm_order": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "swap_arms": list(SWAP_ARMS),
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "phase1_n": phase1.n,
        "phase2_online_n": phase2_online.n,
        "phase2_holdout_n": phase2_holdout.n,
        "cost_penalty": COST_PENALTY,
        "prior_n_effective": PRIOR_N_EFFECTIVE,
        "alpha_warmup": ALPHA_WARMUP,
        "alpha_sw_ucb": ALPHA_SW_UCB,
        "sw_ucb_window": SW_UCB_WINDOW,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "normalized_costs": {
            ARM_SHORT[a]: v for a, v in normalized_costs.items()
        },
        "conditions": all_results,
    }

    out_path = RESULTS_DIR / "reward_shift_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nSaved results to %s", out_path)

    # ---- Summary table ----
    logger.info("\n" + "=" * 80)
    logger.info("REWARD SHIFT — Final Checkpoint Summary")
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
