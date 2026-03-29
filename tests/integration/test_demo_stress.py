"""Integration stress test for the ``paretobandit[demo]`` extra.

Verifies that the demo module is importable, its public API is
accessible, data loading works, trials produce sane metrics, each
scenario function runs end-to-end, and the CLI entry point responds
to ``--help``.

Requires ``pip install paretobandit[demo]``  (embeddings + matplotlib).

Run via Docker (recommended):
    ./scripts/run_integration_test.sh --demo

Skip marker:
    These tests are automatically skipped when sentence-transformers or
    matplotlib is not installed, so lighter Docker targets can safely
    collect this file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

_demo_available = True
_skip_reason = ""

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _demo_available = False
    _skip_reason = "sentence-transformers not installed"

try:
    import matplotlib  # noqa: F401
except ImportError:
    _demo_available = False
    _skip_reason = "matplotlib not installed"

requires_demo = pytest.mark.skipif(
    not _demo_available,
    reason=_skip_reason or "demo dependencies not installed (pip install paretobandit[demo])",
)


def _package_pip_installed() -> bool:
    """True when ``pareto_bandit`` is on the interpreter's default path."""
    probe = subprocess.run(
        [sys.executable, "-c", "import pareto_bandit"],
        capture_output=True,
        timeout=10,
    )
    return probe.returncode == 0


requires_pip_install = pytest.mark.skipif(
    not _package_pip_installed(),
    reason="CLI tests require a real pip install (run via Docker target)",
)


# ---------------------------------------------------------------------------
# 1. Public API importability
# ---------------------------------------------------------------------------

@requires_demo
class TestDemoImports:
    """All documented public symbols must be importable."""

    def test_module_importable(self) -> None:
        import pareto_bandit.demo
        assert hasattr(pareto_bandit.demo, "main")

    def test_public_symbols(self) -> None:
        from pareto_bandit.demo import (
            ARM_ORDER,
            ARM_SHORT,
            _create_router,
            load_demo_splits,
            run_scenario_1,
            run_scenario_2,
            run_scenario_3,
            run_scenario_4,
            run_trial,
        )
        assert len(ARM_ORDER) == 3
        assert len(ARM_SHORT) == 3
        assert callable(load_demo_splits)
        assert callable(run_trial)
        assert callable(run_scenario_1)
        assert callable(run_scenario_2)
        assert callable(run_scenario_3)
        assert callable(run_scenario_4)
        assert callable(_create_router)

    def test_demo_config_defaults(self) -> None:
        from pareto_bandit.demo import DemoConfig

        cfg = DemoConfig()
        assert cfg.seed == 42
        assert cfg.n_seeds == 10
        assert isinstance(cfg.alpha, float)
        assert isinstance(cfg.forgetting_factor, float)
        assert cfg.scenario is None


# ---------------------------------------------------------------------------
# 2. Data loading from shipped holdout
# ---------------------------------------------------------------------------

@requires_demo
class TestDataLoading:
    """The shipped holdout file must load and split cleanly."""

    @pytest.fixture(scope="class")
    def splits(self):
        from pareto_bandit.demo import ARM_ORDER, DemoConfig, load_demo_splits
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        cfg = DemoConfig()
        train, holdout = load_demo_splits(
            val_file=cfg.val_file,
            holdout_file=cfg.holdout_file,
            feature_service=fs,
        )
        return train, holdout, ARM_ORDER

    def test_split_sizes(self, splits) -> None:
        train, holdout, _ = splits
        assert train.n > 0
        assert holdout.n > 0

    def test_embeddings_shape(self, splits) -> None:
        train, holdout, _ = splits
        assert train.embeddings.ndim == 2
        assert train.embeddings.shape[1] == holdout.embeddings.shape[1]
        assert train.embeddings.shape[1] >= 2  # at least 1 feature + bias

    def test_bias_column_is_one(self, splits) -> None:
        train, _, _ = splits
        np.testing.assert_allclose(train.embeddings[:, -1], 1.0)

    def test_rewards_and_costs_present(self, splits) -> None:
        train, _, arm_order = splits
        for arm in arm_order:
            assert arm in train.rewards
            assert arm in train.costs
            assert train.rewards[arm].shape == (train.n,)
            assert train.costs[arm].shape == (train.n,)

    def test_rewards_bounded(self, splits) -> None:
        train, _, arm_order = splits
        for arm in arm_order:
            assert np.all(train.rewards[arm] >= 0.0)
            assert np.all(train.rewards[arm] <= 1.0)

    def test_costs_non_negative(self, splits) -> None:
        train, _, arm_order = splits
        for arm in arm_order:
            assert np.all(train.costs[arm] >= 0.0)


# ---------------------------------------------------------------------------
# 3. Trial execution produces sane metrics
# ---------------------------------------------------------------------------

@requires_demo
class TestRunTrial:
    """A single trial must produce finite, bounded metrics."""

    @pytest.fixture(scope="class")
    def trial_result(self):
        from pareto_bandit.demo import DemoConfig, load_demo_splits, run_trial
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        cfg = DemoConfig()
        train, holdout = load_demo_splits(
            val_file=cfg.val_file,
            holdout_file=cfg.holdout_file,
            feature_service=fs,
        )
        return run_trial(
            train, holdout,
            alpha=0.01,
            cost_penalty=0.3,
            seed=7,
            record_steps=True,
        )

    def test_mean_reward_bounded(self, trial_result) -> None:
        assert 0.0 <= trial_result.mean_reward <= 1.0

    def test_mean_cost_positive(self, trial_result) -> None:
        assert trial_result.mean_cost > 0.0

    def test_model_fractions_sum_to_one(self, trial_result) -> None:
        total = sum(trial_result.model_fractions.values())
        assert abs(total - 1.0) < 1e-6

    def test_per_step_lists_populated(self, trial_result) -> None:
        assert len(trial_result.per_step_models) > 0
        assert len(trial_result.per_step_rewards) == len(trial_result.per_step_models)
        assert len(trial_result.per_step_costs) == len(trial_result.per_step_models)

    def test_all_selected_models_are_known_arms(self, trial_result) -> None:
        from pareto_bandit.demo import ARM_ORDER

        arm_set = set(ARM_ORDER)
        for model in trial_result.per_step_models:
            assert model in arm_set


# ---------------------------------------------------------------------------
# 4. Router factory
# ---------------------------------------------------------------------------

@requires_demo
class TestRouterFactory:
    """_create_router must produce a working BanditRouter."""

    def test_create_router_cold_start(self) -> None:
        from pareto_bandit.demo import _create_router

        router = _create_router(26, warmup=False, seed=0)
        assert len(router.registry) == 3

    def test_create_router_warmup(self) -> None:
        from pareto_bandit.demo import _create_router

        router = _create_router(26, warmup=True, seed=0)
        for mid in router.bandit.models:
            assert np.all(np.isfinite(router.bandit.theta[mid]))

    def test_router_routes_precomputed_vector(self) -> None:
        from pareto_bandit.demo import _create_router

        dim = 26
        router = _create_router(dim, warmup=True, seed=42)
        x = np.random.default_rng(0).standard_normal(dim)
        x[-1] = 1.0
        model, log = router.route(x)
        assert model in router.registry
        assert log.selected_model == model


# ---------------------------------------------------------------------------
# 5. Scenario end-to-end (small data, fast)
# ---------------------------------------------------------------------------

@requires_demo
class TestScenarioEndToEnd:
    """Each scenario function must run without error and produce a plot."""

    @pytest.fixture(scope="class")
    def demo_env(self, tmp_path_factory):
        from pareto_bandit.demo import DemoConfig, load_demo_splits
        from pareto_bandit.feature_service import FeatureService

        out_dir = tmp_path_factory.mktemp("demo_plots")
        fs = FeatureService()
        cfg = DemoConfig(
            seed=42,
            n_seeds=2,
            n_budget_targets=3,
            output_dir=str(out_dir),
        )
        train, holdout = load_demo_splits(
            val_file=cfg.val_file,
            holdout_file=cfg.holdout_file,
            feature_service=fs,
        )
        return cfg, train, holdout, out_dir

    def test_scenario_1(self, demo_env) -> None:
        from pareto_bandit.demo import run_scenario_1

        cfg, train, holdout, out_dir = demo_env
        path = run_scenario_1(cfg, train, holdout)
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_scenario_2(self, demo_env) -> None:
        from pareto_bandit.demo import run_scenario_2

        cfg, train, holdout, out_dir = demo_env
        path = run_scenario_2(cfg, train, holdout)
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_scenario_3(self, demo_env) -> None:
        from pareto_bandit.demo import run_scenario_3

        cfg, train, holdout, out_dir = demo_env
        path = run_scenario_3(cfg, train, holdout)
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_scenario_4(self, demo_env) -> None:
        from pareto_bandit.demo import run_scenario_4

        cfg, train, holdout, out_dir = demo_env
        path = run_scenario_4(cfg, train, holdout)
        assert path.exists()
        assert path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# 6. CLI entry point
# ---------------------------------------------------------------------------

@requires_demo
@requires_pip_install
class TestDemoCLI:
    """The ``paretobandit-demo`` console script must respond to --help."""

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pareto_bandit.demo", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "scenario" in result.stdout.lower()
        assert "n-seeds" in result.stdout.lower()

    def test_cli_scenario_flag_accepted(self) -> None:
        """--scenario 1 with minimal data must not crash."""
        result = subprocess.run(
            [
                sys.executable, "-m", "pareto_bandit.demo",
                "--scenario", "1",
                "--n-seeds", "1",
                "--n-budget-targets", "2",
                "--output-dir", "/tmp/paretobandit_demo_test",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Demo CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert Path("/tmp/paretobandit_demo_test/scenario1_budget_pacing.png").exists()
