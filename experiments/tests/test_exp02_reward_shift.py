"""Regression tests for Experiment 02: Non-stationary Reward Shift.

Fast tests verify config integration.  The ``@pytest.mark.slow`` smoke
test runs one seed of the ParetoBandit condition through the two-phase
learning curve and compares final-checkpoint metrics against a pinned
reference.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.config import BEST_K3_HPARAMS, K3_ARM_ORDER
from pareto_bandit.feature_service import FeatureService
from utils.simulation import (
    SplitData,
    apply_reward_swap,
    build_model_registry,
    compute_normalized_costs,
)

from helpers import assert_metrics_match, load_reference, save_reference

REFERENCE_NAME = "exp02_seed4000_paretobandit"
SEED = 4000


def _import_exp02():
    """Import the experiment module via importlib to avoid collisions."""
    mod_name = "_exp_run_reward_shift"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    exp_dir = PROJECT_ROOT / "experiments" / "02_nonstationary_k3_drift"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        exp_dir / "run_reward_shift.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# Config integration (fast)
# ======================================================================


@pytest.mark.experiment
class TestExp02Config:
    """Verify Exp 02 hyperparameters come from the central config."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        self.mod = _import_exp02()

    def test_alpha_warmup_from_config(self):
        assert self.mod.ALPHA_WARMUP == BEST_K3_HPARAMS["alpha"]

    def test_prior_n_effective_from_config(self):
        assert self.mod.PRIOR_N_EFFECTIVE == BEST_K3_HPARAMS["prior_n_effective"]

    def test_paretobandit_forgetting_factor(self):
        paretobandit_cond = [
            c for c in self.mod.CONDITIONS if "ParetoBandit" in c["label"]
        ][0]
        assert paretobandit_cond["forgetting_factor"] == BEST_K3_HPARAMS["forgetting_factor"]

    def test_fixed_policy_gamma_is_one(self):
        fixed = [c for c in self.mod.CONDITIONS if "Fixed" in c["label"]][0]
        assert fixed["forgetting_factor"] == 1.0

    def test_naive_bandit_gamma_is_one(self):
        naive = [c for c in self.mod.CONDITIONS if "Naive" in c["label"]][0]
        assert naive["forgetting_factor"] == 1.0

    def test_arm_order_matches_config(self):
        assert self.mod.ARM_ORDER == K3_ARM_ORDER


# ======================================================================
# Single-seed regression (slow)
# ======================================================================


@pytest.mark.experiment
@pytest.mark.slow
def test_exp02_single_seed_regression(
    val_split,
    test_split,
    model_registry,
    feature_dim,
):
    """Run one ParetoBandit seed through Phase 1 + Phase 2 and compare.

    Monkeypatches ``N_SEEDS=1`` inside ``_run_learning_curve`` by calling
    the per-seed inner loop directly rather than the aggregating wrapper.
    """
    mod = _import_exp02()

    arm_order = mod.ARM_ORDER
    phase1_n = mod.PHASE1_N
    phase2_n = mod.PHASE2_N
    swap_arms = mod.SWAP_ARMS

    rng_global = np.random.default_rng(42)
    all_indices = rng_global.permutation(val_split.n)
    p1_idx = all_indices[:phase1_n]
    p2_idx = all_indices[phase1_n : phase1_n + phase2_n]

    phase1 = SplitData(
        prompts=[val_split.prompts[i] for i in p1_idx],
        rewards={a: val_split.rewards[a][p1_idx] for a in arm_order},
        costs={a: val_split.costs[a][p1_idx] for a in arm_order},
        embeddings=val_split.embeddings[p1_idx],
    )
    phase2_raw = SplitData(
        prompts=[val_split.prompts[i] for i in p2_idx],
        rewards={a: val_split.rewards[a][p2_idx] for a in arm_order},
        costs={a: val_split.costs[a][p2_idx] for a in arm_order},
        embeddings=val_split.embeddings[p2_idx],
    )
    phase2_online = apply_reward_swap(phase2_raw, *swap_arms)
    phase2_holdout = apply_reward_swap(test_split, *swap_arms)

    normalized_costs = compute_normalized_costs(model_registry, arm_order)

    paretobandit_cond = [c for c in mod.CONDITIONS if "ParetoBandit" in c["label"]][0]

    # Run a single seed directly (the inner loop of _run_learning_curve).
    rng = np.random.default_rng(SEED)
    router = mod._create_router(
        model_registry,
        feature_dim,
        warmup=paretobandit_cond["warmup"],
        forgetting_factor=paretobandit_cond["forgetting_factor"],
        alpha=paretobandit_cond.get("alpha", mod.ALPHA_WARMUP),
    )

    p1_order = rng.permutation(phase1.n)
    p2_order = rng.permutation(phase2_online.n)

    all_emb = np.concatenate([
        phase1.embeddings[p1_order],
        phase2_online.embeddings[p2_order],
    ], axis=0)

    all_rewards: Dict[str, np.ndarray] = {}
    all_costs: Dict[str, np.ndarray] = {}
    for arm in arm_order:
        all_rewards[arm] = np.concatenate([
            phase1.rewards[arm][p1_order],
            phase2_online.rewards[arm][p2_order],
        ])
        all_costs[arm] = np.concatenate([
            phase1.costs[arm][p1_order],
            phase2_online.costs[arm][p2_order],
        ])

    cumulative_regret: float = 0.0
    model_counts: Dict[str, int] = {a: 0 for a in arm_order}
    rewards_collected: List[float] = []
    costs_collected: List[float] = []
    n_train = phase1.n + phase2_online.n

    for t in range(n_train):
        model, log = router.route(all_emb[t])
        reward = float(all_rewards[model][t])
        cost = float(all_costs[model][t])
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

        oracle_utility = max(
            float(all_rewards[a][t]) - mod.COST_PENALTY * normalized_costs[a]
            for a in arm_order
        )
        chosen_utility = reward - mod.COST_PENALTY * normalized_costs[model]
        cumulative_regret += oracle_utility - chosen_utility

        rewards_collected.append(reward)
        costs_collected.append(cost)
        model_counts[model] += 1

    # Frozen holdout eval at end.
    holdout_eval = mod._frozen_holdout_eval(
        router, phase2_holdout, arm_order, normalized_costs, mod.COST_PENALTY,
    )

    actual: Dict[str, Any] = {
        "cumulative_regret": cumulative_regret,
        "mean_online_reward": float(np.mean(rewards_collected)),
        "mean_online_cost": float(np.mean(costs_collected)),
        "holdout_reward": holdout_eval["reward"],
        "holdout_cost": holdout_eval["cost"],
        "model_fractions": {
            a: model_counts[a] / n_train for a in arm_order
        },
    }

    try:
        reference = load_reference(REFERENCE_NAME)
    except FileNotFoundError:
        path = save_reference(REFERENCE_NAME, actual)
        pytest.fail(
            f"Reference generated at {path}.  Review the values and "
            "re-run to validate.",
        )

    assert_metrics_match(actual, reference)
