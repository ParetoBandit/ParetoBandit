import unittest
import numpy as np
from pathlib import Path
from bandit_gpt.router import BanditRouter, OptimizationProfile, ExplorationRate


class TestRouterConfigurationUpdates(unittest.TestCase):
    """
    Unit tests for recent router configuration updates:
    1. ARBITRAGE profile weights updated to w_q=0.80, w_c=0.20
    2. Default alpha changed to 0.1 (via exploration="safe")
    3. Default prior_n_effective changed to 20.0
    """

    def setUp(self):
        """Set up minimal test registry."""
        self.model_registry = {
            "test-model-1": {
                "openrouter_id": "test-model-1",
                "hle": 0.15,
                "input_cost_per_m": 1.0,
                "output_cost_per_m": 1.0,
                "time_to_first_token_seconds": 0.5
            },
            "test-model-2": {
                "openrouter_id": "test-model-2",
                "hle": 0.25,
                "input_cost_per_m": 5.0,
                "output_cost_per_m": 5.0,
                "time_to_first_token_seconds": 1.0
            }
        }

    def test_arbitrage_profile_weights(self):
        """Test that ARBITRAGE profile has updated weights."""
        arbitrage = OptimizationProfile.ARBITRAGE
        
        self.assertEqual(arbitrage["w_q"], 0.80, 
                        "ARBITRAGE quality weight should be 0.80")
        self.assertEqual(arbitrage["w_c"], 0.20, 
                        "ARBITRAGE cost weight should be 0.20")
        self.assertEqual(arbitrage["w_l"], 0.00, 
                        "ARBITRAGE latency weight should be 0.00")

    def test_arbitrage_profile_ratio(self):
        """Test that ARBITRAGE w_q/w_c ratio is 4.0."""
        arbitrage = OptimizationProfile.ARBITRAGE
        ratio = arbitrage["w_q"] / arbitrage["w_c"]
        
        self.assertAlmostEqual(ratio, 4.0, places=1,
                              msg="ARBITRAGE should have w_q/w_c ratio of 4.0")

    def test_arbitrage_profile_retrieval(self):
        """Test that 'arbitrage' string maps to correct weights."""
        weights = OptimizationProfile.get("arbitrage")
        
        self.assertEqual(weights["w_q"], 0.80)
        self.assertEqual(weights["w_c"], 0.20)
        self.assertEqual(weights["w_l"], 0.00)

    def test_default_alpha_via_exploration_safe(self):
        """Test that exploration='safe' maps to alpha=0.1."""
        router = BanditRouter.create(
            model_registry=self.model_registry,
            exploration="safe"
        )
        
        self.assertEqual(router.bandit.alpha, 0.1,
                        "exploration='safe' should result in alpha=0.1")

    def test_default_alpha_via_create_no_args(self):
        """Test that BanditRouter.create() defaults to alpha=0.1."""
        router = BanditRouter.create(model_registry=self.model_registry)
        
        # Default exploration is "safe" which should map to 0.1
        self.assertEqual(router.bandit.alpha, 0.1,
                        "BanditRouter.create() should default to alpha=0.1")

    def test_exploration_rate_safe_constant(self):
        """Test that ExplorationRate.SAFE is 0.1."""
        self.assertEqual(ExplorationRate.SAFE, 0.1,
                        "ExplorationRate.SAFE should be 0.1")

    def test_prior_n_effective_default(self):
        """
        Test that prior_n_effective defaults to 20.0.
        
        We can't directly inspect prior_n_effective after router creation,
        but we can verify it's used by checking bias term magnitudes.
        """
        router = BanditRouter.create(
            model_registry=self.model_registry,
            priors="hle"  # Use HLE priors to ensure bias terms are set
        )
        
        # Check that bias terms have been scaled by ~20.0 * hle
        # For test-model-1 with hle=0.15, bias should be ~20.0 * 0.15 = 3.0
        bias_value = router.bandit.b["test-model-1"][-1]
        
        # The bias should be approximately N_eff * hle
        # With N_eff=20 and hle=0.15, we expect ~3.0
        expected_bias = 20.0 * 0.15
        
        self.assertAlmostEqual(bias_value, expected_bias, places=1,
                              msg=f"Bias term should reflect N_eff=20.0 * hle=0.15")

    def test_explicit_prior_n_effective_override(self):
        """Test that prior_n_effective can still be overridden."""
        custom_n_eff = 50.0
        router = BanditRouter.create(
            model_registry=self.model_registry,
            priors="hle",
            prior_n_effective=custom_n_eff
        )
        
        # Check that bias term reflects custom N_eff
        bias_value = router.bandit.b["test-model-1"][-1]
        expected_bias = custom_n_eff * 0.15  # hle=0.15
        
        self.assertAlmostEqual(bias_value, expected_bias, places=1,
                              msg=f"Bias term should reflect custom N_eff={custom_n_eff}")

    def test_default_profile_is_arbitrage(self):
        """Test that route() defaults to 'arbitrage' profile."""
        router = BanditRouter.create(model_registry=self.model_registry)
        
        # Capture the profile used by mocking _resolve_utility_weights
        captured_profile = None
        original_resolve = router._resolve_utility_weights
        
        def mock_resolve(profile, max_cost, max_latency):
            nonlocal captured_profile
            captured_profile = profile
            return original_resolve(profile, max_cost, max_latency)
        
        router._resolve_utility_weights = mock_resolve
        
        try:
            router.route("test prompt")
        except:
            pass  # We don't care if routing fails, just want to see the profile
        
        self.assertEqual(captured_profile, "arbitrage",
                        "route() should default to 'arbitrage' profile")

    def test_all_other_profiles_unchanged(self):
        """Ensure other profiles remain unchanged."""
        max_quality = OptimizationProfile.MAX_QUALITY
        self.assertEqual(max_quality["w_q"], 0.99)
        self.assertEqual(max_quality["w_c"], 0.01)
        
        best_value = OptimizationProfile.BEST_VALUE
        self.assertEqual(best_value["w_q"], 0.70)
        self.assertEqual(best_value["w_c"], 0.30)
        
        cost_saver = OptimizationProfile.COST_SAVER
        self.assertEqual(cost_saver["w_q"], 0.40)
        self.assertEqual(cost_saver["w_c"], 0.60)


if __name__ == "__main__":
    unittest.main()
