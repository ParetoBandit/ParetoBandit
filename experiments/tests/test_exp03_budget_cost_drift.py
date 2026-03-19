"""Regression tests for Experiment 03: Budget Pacing under Cost Drift.

Fast tests verify config integration.  The ``@pytest.mark.slow`` smoke
test runs one ParetoBandit (moderate budget) seed through the three-phase
cost-drift scenario and compares metrics against a pinned reference.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
from pareto_bandit.config import BEST_K3_HPARAMS, K3_ARM_ORDER
from utils.simulation import SplitData, build_model_registry

from helpers import assert_metrics_match, load_reference, save_reference

REFERENCE_NAME = "exp03_seed8000_paretobandit_moderate_3phase"
SEED = 8000


def _import_exp03():
    """Import the experiment module via importlib to avoid collisions."""
    mod_name = "_exp_run_budget_cost_drift"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    exp_dir = PROJECT_ROOT / "experiments" / "02_budget_plus_drift"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        exp_dir / "run_budget_cost_drift.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# Config integration (fast)
# ======================================================================


@pytest.mark.experiment
class TestExp03Config:
    """Verify Exp 03 hyperparameters come from the central config."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        self.mod = _import_exp03()

    def test_alpha_from_config(self):
        assert self.mod.ALPHA == BEST_K3_HPARAMS["alpha"]

    def test_prior_n_effective_from_config(self):
        assert self.mod.PRIOR_N_EFFECTIVE == BEST_K3_HPARAMS["prior_n_effective"]

    def test_paretobandit_forgetting_factor(self):
        conditions = self.mod._build_conditions(
            budget_target=6.62e-4,
            budget_label="moderate",
            matched_cp=0.30,
            recalibrated_phase2_cp=0.0,
            recalibrated_phase3_cp=0.30,
        )
        paretobandit = [c for c in conditions if "ParetoBandit" in c["label"]][0]
        assert paretobandit["forgetting_factor"] == BEST_K3_HPARAMS["forgetting_factor"]

    def test_fixed_policy_gamma_is_one(self):
        conditions = self.mod._build_conditions(
            budget_target=6.62e-4,
            budget_label="moderate",
            matched_cp=0.30,
            recalibrated_phase2_cp=0.0,
            recalibrated_phase3_cp=0.30,
        )
        fixed = [c for c in conditions if "Fixed" in c["label"]][0]
        assert fixed["forgetting_factor"] == 1.0

    def test_arm_order_matches_config(self):
        assert self.mod.ARM_ORDER == K3_ARM_ORDER

    def test_three_phase_structure(self):
        assert self.mod.N_PHASES == 3
        assert self.mod.PHASE_N == 608


# ======================================================================
# Single-seed regression (slow)
# ======================================================================


@pytest.mark.experiment
@pytest.mark.slow
def test_exp03_single_seed_regression(
    val_split,
    test_split,
    model_registry,
    feature_dim,
):
    """Run one ParetoBandit (moderate) seed through the three-phase scenario.

    Sets up the train-then-evaluate cost drift exactly as ``main()``
    does, but for a single seed only.
    """
    mod = _import_exp03()

    arm_order = mod.ARM_ORDER
    phase_n = mod.PHASE_N
    gemini_id = mod.GEMINI_ID

    rng_global = np.random.default_rng(42)
    all_indices = rng_global.permutation(test_split.n)
    p1_idx = all_indices[:phase_n]
    p2_idx = all_indices[phase_n:2 * phase_n]

    phase1 = SplitData(
        prompts=[test_split.prompts[i] for i in p1_idx],
        rewards={a: test_split.rewards[a][p1_idx] for a in arm_order},
        costs={a: test_split.costs[a][p1_idx] for a in arm_order},
        embeddings=test_split.embeddings[p1_idx],
    )

    gemini_meta = model_registry[gemini_id]
    old_input = gemini_meta["input_cost_per_m"]
    old_output = gemini_meta["output_cost_per_m"]

    phase2_raw = SplitData(
        prompts=[test_split.prompts[i] for i in p2_idx],
        rewards={a: test_split.rewards[a][p2_idx] for a in arm_order},
        costs={a: test_split.costs[a][p2_idx] for a in arm_order},
        embeddings=test_split.embeddings[p2_idx],
    )
    phase2 = mod._apply_gemini_cost_reduction(
        phase2_raw, gemini_id,
        old_input, old_output,
        mod.GEMINI_NEW_INPUT_COST, mod.GEMINI_NEW_OUTPUT_COST,
    )

    phase3 = phase1

    budget_target = mod.BUDGET_TARGETS[1]  # moderate
    pacer = BudgetPacer(
        target_avg_spend_usd=budget_target,
        mode=PacingMode.ADAPTIVE,
        lr=mod.PACER_LR,
        lambda_max=mod.PACER_LAMBDA_MAX,
        ema_alpha=mod.PACER_EMA_ALPHA,
    )

    seed_result = mod._run_three_phase_trial(
        condition_label="ParetoBandit (moderate)",
        train_data=val_split,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
        registry=copy.deepcopy(model_registry),
        original_gemini_input=old_input,
        original_gemini_output=old_output,
        feature_dim=feature_dim,
        cost_penalty=0.0,
        warmup=True,
        forgetting_factor=BEST_K3_HPARAMS["forgetting_factor"],
        online_learn=True,
        budget_pacer=pacer,
        seed=SEED,
    )

    p1_metrics = seed_result.phase_metrics(1)
    p2_metrics = seed_result.phase_metrics(2)
    p3_metrics = seed_result.phase_metrics(3)

    actual: Dict[str, Any] = {
        "p1_mean_reward": p1_metrics["mean_reward"],
        "p1_mean_cost": p1_metrics["mean_cost"],
        "p2_mean_reward": p2_metrics["mean_reward"],
        "p2_mean_cost": p2_metrics["mean_cost"],
        "p2_mean_lambda": p2_metrics["mean_lambda"],
        "p3_mean_reward": p3_metrics["mean_reward"],
        "p3_mean_cost": p3_metrics["mean_cost"],
        "p3_mean_lambda": p3_metrics["mean_lambda"],
        "p1_arm_fractions": p1_metrics["arm_fractions"],
        "p2_arm_fractions": p2_metrics["arm_fractions"],
        "p3_arm_fractions": p3_metrics["arm_fractions"],
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
