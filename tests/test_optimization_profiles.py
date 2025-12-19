"""
Unit tests for OptimizationProfile and ExplorationRate.

Tests:
  - OptimizationProfile: named presets, aliases, error handling
  - ExplorationRate: named presets, float parsing, error handling
  - Integration: using profiles in route() and rank_prompt()
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Test OptimizationProfile
# ---------------------------------------------------------------------------


class TestOptimizationProfile:
    """Tests for OptimizationProfile class."""

    def test_import(self):
        """OptimizationProfile can be imported."""
        from banditgpt.core import OptimizationProfile

        assert OptimizationProfile is not None

    def test_predefined_profiles(self):
        """Predefined profiles have expected values."""
        from banditgpt.core.bandit_router import OptimizationProfile

        # Check all four profiles exist
        assert OptimizationProfile.QUALITY_FIRST == {"lambda_cost": 0.1, "lambda_latency": 0.05}
        assert OptimizationProfile.BALANCED == {"lambda_cost": 10.0, "lambda_latency": 0.10}
        assert OptimizationProfile.COST_SAVER == {"lambda_cost": 50.0, "lambda_latency": 0.20}
        assert OptimizationProfile.LOW_LATENCY == {"lambda_cost": 1.0, "lambda_latency": 0.50}

    def test_get_by_name(self):
        """get() returns correct profile by name."""
        from banditgpt.core.bandit_router import OptimizationProfile

        assert OptimizationProfile.get("quality_first") == OptimizationProfile.QUALITY_FIRST
        assert OptimizationProfile.get("balanced") == OptimizationProfile.BALANCED
        assert OptimizationProfile.get("cost_saver") == OptimizationProfile.COST_SAVER
        assert OptimizationProfile.get("low_latency") == OptimizationProfile.LOW_LATENCY

    def test_get_case_insensitive(self):
        """get() is case-insensitive."""
        from banditgpt.core.bandit_router import OptimizationProfile

        assert OptimizationProfile.get("QUALITY_FIRST") == OptimizationProfile.QUALITY_FIRST
        assert OptimizationProfile.get("Quality_First") == OptimizationProfile.QUALITY_FIRST
        assert OptimizationProfile.get("BALANCED") == OptimizationProfile.BALANCED

    def test_get_with_hyphens(self):
        """get() accepts hyphens instead of underscores."""
        from banditgpt.core.bandit_router import OptimizationProfile

        assert OptimizationProfile.get("quality-first") == OptimizationProfile.QUALITY_FIRST
        assert OptimizationProfile.get("cost-saver") == OptimizationProfile.COST_SAVER
        assert OptimizationProfile.get("low-latency") == OptimizationProfile.LOW_LATENCY

    def test_get_aliases(self):
        """get() supports aliases."""
        from banditgpt.core.bandit_router import OptimizationProfile

        # penny_pincher -> COST_SAVER
        assert OptimizationProfile.get("penny_pincher") == OptimizationProfile.COST_SAVER
        # realtime -> LOW_LATENCY
        assert OptimizationProfile.get("realtime") == OptimizationProfile.LOW_LATENCY

    def test_get_unknown_profile_raises(self):
        """get() raises ValueError for unknown profile."""
        from banditgpt.core.bandit_router import OptimizationProfile

        with pytest.raises(ValueError, match="Unknown profile"):
            OptimizationProfile.get("nonexistent")

    def test_list_profiles(self):
        """list_profiles() returns all primary profile names."""
        from banditgpt.core.bandit_router import OptimizationProfile

        profiles = OptimizationProfile.list_profiles()
        assert "quality_first" in profiles
        assert "balanced" in profiles
        assert "cost_saver" in profiles
        assert "low_latency" in profiles
        assert len(profiles) == 4


# ---------------------------------------------------------------------------
# Test ExplorationRate
# ---------------------------------------------------------------------------


class TestExplorationRate:
    """Tests for ExplorationRate class."""

    def test_import(self):
        """ExplorationRate can be imported."""
        from banditgpt.core import ExplorationRate

        assert ExplorationRate is not None

    def test_predefined_rates(self):
        """Predefined rates have expected values."""
        from banditgpt.core.bandit_router import ExplorationRate

        assert ExplorationRate.STATIC == 0.0
        assert ExplorationRate.SAFE == 0.1
        assert ExplorationRate.BALANCED == 0.5
        assert ExplorationRate.AGGRESSIVE == 2.0

    def test_get_by_name(self):
        """get() returns correct rate by name."""
        from banditgpt.core.bandit_router import ExplorationRate

        assert ExplorationRate.get("static") == 0.0
        assert ExplorationRate.get("safe") == 0.1
        assert ExplorationRate.get("balanced") == 0.5
        assert ExplorationRate.get("aggressive") == 2.0

    def test_get_case_insensitive(self):
        """get() is case-insensitive."""
        from banditgpt.core.bandit_router import ExplorationRate

        assert ExplorationRate.get("STATIC") == 0.0
        assert ExplorationRate.get("Safe") == 0.1
        assert ExplorationRate.get("AGGRESSIVE") == 2.0

    def test_get_aliases(self):
        """get() supports aliases."""
        from banditgpt.core.bandit_router import ExplorationRate

        # none/zero/off -> STATIC
        assert ExplorationRate.get("none") == 0.0
        assert ExplorationRate.get("zero") == 0.0
        assert ExplorationRate.get("off") == 0.0

        # production/default/low -> SAFE
        assert ExplorationRate.get("production") == 0.1
        assert ExplorationRate.get("default") == 0.1
        assert ExplorationRate.get("low") == 0.1

        # medium/normal -> BALANCED
        assert ExplorationRate.get("medium") == 0.5
        assert ExplorationRate.get("normal") == 0.5

        # high/calibration/shadow/day1 -> AGGRESSIVE
        assert ExplorationRate.get("high") == 2.0
        assert ExplorationRate.get("calibration") == 2.0
        assert ExplorationRate.get("shadow") == 2.0
        assert ExplorationRate.get("day1") == 2.0

    def test_get_float_string(self):
        """get() parses float strings."""
        from banditgpt.core.bandit_router import ExplorationRate

        assert ExplorationRate.get("0.75") == 0.75
        assert ExplorationRate.get("1.5") == 1.5
        assert ExplorationRate.get("0") == 0.0
        assert ExplorationRate.get("3.0") == 3.0

    def test_get_unknown_rate_raises(self):
        """get() raises ValueError for unknown rate."""
        from banditgpt.core.bandit_router import ExplorationRate

        with pytest.raises(ValueError, match="Unknown exploration"):
            ExplorationRate.get("nonexistent")

    def test_list_rates(self):
        """list_rates() returns all primary rate names."""
        from banditgpt.core.bandit_router import ExplorationRate

        rates = ExplorationRate.list_rates()
        assert "static" in rates
        assert "safe" in rates
        assert "balanced" in rates
        assert "aggressive" in rates
        assert len(rates) == 4


# ---------------------------------------------------------------------------
# Test Integration with BanditRouter
# ---------------------------------------------------------------------------


class TestBanditRouterWithProfiles:
    """Test BanditRouter integration with profiles and exploration rates."""

    @pytest.fixture
    def sample_registry(self):
        """Minimal model registry for testing."""
        return {
            "model-a": {"display_name": "Model A", "cost_per_1k_input": 0.001},
            "model-b": {"display_name": "Model B", "cost_per_1k_input": 0.002},
        }

    def test_create_with_exploration_safe(self, sample_registry):
        """BanditRouter.create() accepts exploration='safe'."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(
            model_registry=sample_registry,
            exploration="safe",
            priors="none",
        )

        assert router._default_exploration == 0.1

    def test_create_with_exploration_aggressive(self, sample_registry):
        """BanditRouter.create() accepts exploration='aggressive'."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(
            model_registry=sample_registry,
            exploration="aggressive",
            priors="none",
        )

        assert router._default_exploration == 2.0

    def test_create_with_exploration_float(self, sample_registry):
        """BanditRouter.create() accepts exploration as float string."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(
            model_registry=sample_registry,
            exploration="0.75",
            priors="none",
        )

        assert router._default_exploration == 0.75

    def test_create_alpha_overrides_exploration(self, sample_registry):
        """alpha parameter takes precedence over exploration."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(
            model_registry=sample_registry,
            exploration="aggressive",  # would be 2.0
            alpha=0.3,  # explicit override
            priors="none",
        )

        assert router._default_exploration == 0.3

    def test_init_with_exploration(self, sample_registry):
        """BanditRouter.__init__() accepts exploration parameter."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter(
            model_registry=sample_registry,
            exploration="balanced",
        )

        assert router._default_exploration == 0.5

    def test_route_with_profile(self, sample_registry):
        """route() accepts profile parameter."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter(
            model_registry=sample_registry,
            exploration="safe",
        )

        # Should not raise
        model, log = router.route(
            "Test prompt",
            profile="balanced",
            epsilon=0.0,  # Disable random exploration for determinism
        )

        assert model in sample_registry

    def test_route_with_exploration_override(self, sample_registry):
        """route() exploration parameter overrides default."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter(
            model_registry=sample_registry,
            exploration="safe",  # default 0.1
        )

        # Override with aggressive
        model, log = router.route(
            "Test prompt",
            exploration="aggressive",
            epsilon=0.0,
        )

        assert model in sample_registry

    def test_rank_prompt_with_profile(self, sample_registry):
        """rank_prompt() accepts profile parameter."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter(
            model_registry=sample_registry,
            exploration="safe",
        )

        rows = router.rank_prompt(
            "Test prompt",
            profile="cost_saver",
            top_k=2,
        )

        assert len(rows) == 2
        assert all("model_id" in r for r in rows)

    def test_rank_prompt_with_exploration(self, sample_registry):
        """rank_prompt() accepts exploration parameter."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter(
            model_registry=sample_registry,
            exploration="safe",
        )

        rows = router.rank_prompt(
            "Test prompt",
            exploration="static",
            top_k=2,
        )

        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Test CLI Arguments
# ---------------------------------------------------------------------------


class TestCLIArguments:
    """Test that CLI properly parses profile and exploration arguments."""

    def test_add_recommend_args_has_profile(self):
        """add_recommend_args includes --profile argument."""
        import argparse
        from banditgpt.core.cli import add_recommend_args

        parser = argparse.ArgumentParser()
        add_recommend_args(parser)

        # Parse with --profile
        args = parser.parse_args(["--prompt", "test", "--profile", "balanced"])
        assert args.profile == "balanced"

    def test_add_recommend_args_has_exploration(self):
        """add_recommend_args includes --exploration argument."""
        import argparse
        from banditgpt.core.cli import add_recommend_args

        parser = argparse.ArgumentParser()
        add_recommend_args(parser)

        # Parse with --exploration
        args = parser.parse_args(["--prompt", "test", "--exploration", "aggressive"])
        assert args.exploration == "aggressive"

    def test_profile_choices(self):
        """--profile only accepts valid choices."""
        import argparse
        from banditgpt.core.cli import add_recommend_args

        parser = argparse.ArgumentParser()
        add_recommend_args(parser)

        # Valid choices
        for profile in ["quality_first", "balanced", "cost_saver", "low_latency"]:
            args = parser.parse_args(["--prompt", "test", "--profile", profile])
            assert args.profile == profile

        # Invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--prompt", "test", "--profile", "invalid"])


# ---------------------------------------------------------------------------
# Test Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_exploration_static_means_no_exploration_bonus(self):
        """With exploration='static', UCB should equal mean (no uncertainty bonus)."""
        from banditgpt.core.bandit_router import BanditRouter

        registry = {
            "model-a": {"display_name": "A"},
            "model-b": {"display_name": "B"},
        }

        router = BanditRouter(model_registry=registry, exploration="static")

        # With static (alpha=0), the quality_hat should be just mean + prior
        # Since we use UCB, with alpha=0, ucb = mean + 0*std = mean
        rows = router.rank_prompt("test", top_k=2)

        # All models should have quality_hat = mean (no exploration bonus)
        # We can't easily verify this without mocking, but at least ensure no crash
        assert len(rows) == 2

    def test_exploration_and_profile_together(self):
        """Both exploration and profile can be specified together."""
        from banditgpt.core.bandit_router import BanditRouter

        registry = {"model-a": {"display_name": "A"}}

        router = BanditRouter(model_registry=registry, exploration="safe")

        # Use both profile (for cost/latency) and exploration (for UCB alpha)
        model, log = router.route(
            "Test",
            profile="cost_saver",
            exploration="aggressive",
            epsilon=0.0,
        )

        assert model == "model-a"

    def test_none_exploration_uses_default(self):
        """If exploration=None in route(), use router's default."""
        from banditgpt.core.bandit_router import BanditRouter

        registry = {"model-a": {"display_name": "A"}}

        router = BanditRouter(model_registry=registry, exploration="balanced")
        assert router._default_exploration == 0.5

        # Calling route() without exploration should use default
        model, log = router.route("Test", epsilon=0.0)
        assert model == "model-a"
