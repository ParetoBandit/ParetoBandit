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
        self.router = BanditRouter(self.model_registry)

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

    def test_new_profiles(self):
        # Test that cost_saver and low_latency aliases work
        weights_cs = OptimizationProfile.get("cost_saver")
        # COST_SAVER = {"w_q": 3.6, "w_c": 1.0, "w_l": 0.0}
        self.assertEqual(weights_cs["w_q"], 3.6)
        self.assertEqual(weights_cs["w_c"], 1.0)
        
        weights_ll = OptimizationProfile.get("low_latency")
        # LOW_LATENCY = {"w_q": 0.20, "w_c": 0.10, "w_l": 0.70}
        self.assertEqual(weights_ll["w_l"], 0.70)

if __name__ == "__main__":
    unittest.main()
