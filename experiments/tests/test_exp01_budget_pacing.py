"""Regression tests for Experiment 01: Stationary Budget Pacing.

Fast tests (``test_config_*``) verify that hyperparameters are correctly
sourced from the centralised config.  The ``@pytest.mark.slow`` smoke test
runs a single-seed trial and compares key metrics against a pinned
reference to detect unintended behavioural changes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.config import BEST_K3_HPARAMS, K3_ARM_ORDER

from helpers import assert_metrics_match, load_reference, save_reference

REFERENCE_NAME = "exp01_seed3000_cp020"
SEED = 3000


def _import_exp01():
    """Import the experiment module via importlib to avoid collisions."""
    mod_name = "_exp_run_budget_pacing"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    exp_dir = PROJECT_ROOT / "experiments" / "01_stationary_budget_pacing"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        exp_dir / "run_budget_pacing.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# Config integration (fast)
# ======================================================================


@pytest.mark.experiment
class TestExp01Config:
    """Verify Exp 01 hyperparameters come from the central config."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        self.mod = _import_exp01()

    def test_warmup_hparams_is_best_k3(self):
        assert self.mod.WARMUP_HPARAMS is BEST_K3_HPARAMS

    def test_alpha_value(self):
        assert self.mod.WARMUP_HPARAMS["alpha"] == BEST_K3_HPARAMS["alpha"]

    def test_prior_n_effective_value(self):
        assert (
            self.mod.WARMUP_HPARAMS["prior_n_effective"]
            == BEST_K3_HPARAMS["prior_n_effective"]
        )

    def test_forgetting_factor_value(self):
        assert (
            self.mod.WARMUP_HPARAMS["forgetting_factor"]
            == BEST_K3_HPARAMS["forgetting_factor"]
        )

    def test_arm_order_matches_config(self):
        assert self.mod.ARM_ORDER == K3_ARM_ORDER


# ======================================================================
# Single-seed regression (slow)
# ======================================================================


@pytest.mark.experiment
@pytest.mark.slow
def test_exp01_single_seed_regression(
    val_split,
    test_split,
    model_registry,
    feature_dim,
):
    """Run one seed with cost_penalty=0.20 and compare to pinned reference.

    On first run the reference file is generated and the test fails with
    a message to review it.  Subsequent runs compare deterministically.
    """
    mod = _import_exp01()

    trial = mod._run_trial(
        val_split,
        test_split,
        model_registry,
        feature_dim,
        condition="static_cp0.20",
        cost_penalty=0.20,
        seed=SEED,
    )

    actual: Dict[str, Any] = {
        "mean_reward": trial.mean_reward,
        "mean_cost": trial.mean_cost,
        "cumulative_quality_gap": trial.cumulative_quality_gap,
        "model_fractions": trial.model_fractions,
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
