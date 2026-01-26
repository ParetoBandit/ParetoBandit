import py_compile
import unittest
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from bandit_gpt.router import BanditRouter, OptimizationProfile

class TestCustomProfiles(unittest.TestCase):
    def setUp(self):
        # Mock registry with extreme models
        self.model_registry = {
            "expensive-high-quality": {
                "openrouter_id": "expensive-high-quality",
                "hle": 0.40,  # Max quality
                "input_cost_per_m": 10.0,
                "output_cost_per_m": 10.0,
                "time_to_first_token_seconds": 2.0
            },
            "cheap-low-quality": {
                "openrouter_id": "cheap-low-quality",
                "hle": 0.05,  # Min quality
                "input_cost_per_m": 0.001,
                "output_cost_per_m": 0.001,
                "time_to_first_token_seconds": 0.1
            }
        }
        self.router = BanditRouter(self.model_registry, use_corralling=True)

    def test_custom_weight_dict(self):
        # Test 1: Extreme Quality focus
        # Even with high cost, it should pick the high quality model
        custom_q = {"w_q": 1.0, "w_c": 0.0, "w_l": 0.0}
        model_q, _ = self.router.route("Hello", profile=custom_q)
        self.assertEqual(model_q, "expensive-high-quality")

        # Test 2: Extreme Cost focus
        # It should pick the cheap model
        custom_c = {"w_q": 0.0, "w_c": 1.0, "w_l": 0.0}
        model_c, _ = self.router.route("Hello", profile=custom_c)
        self.assertEqual(model_c, "cheap-low-quality")

    def test_normalization(self):
        # NOTE: We NO LONGER normalize weights to sum to 1.0.
        # This allows "Unbounded Weights" where users can set high priorities
        # for multiple metrics simultaneously.
        custom_unnormalized = {"w_q": 10.0, "w_c": 10.0} 
        weights = OptimizationProfile.get(custom_unnormalized)
        
        self.assertEqual(weights["w_q"], 10.0)
        self.assertEqual(weights["w_c"], 10.0)
        self.assertEqual(weights["w_l"], 0.0)

    def test_auto_profile(self):
        # Test that "auto" profile returns a marker for Pareto routing
        weights_auto = OptimizationProfile.get("auto")
        self.assertIn("_pareto_mode", weights_auto)
        self.assertTrue(weights_auto["_pareto_mode"])
    
    def test_unknown_profile_raises_error(self):
        # Test that unknown profile names raise errors
        with self.assertRaises(ValueError) as ctx:
            OptimizationProfile.get("unknown_profile")
        self.assertIn("Unknown profile", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
