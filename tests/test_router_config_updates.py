import unittest
import numpy as np
from pathlib import Path
from pareto_bandit.router import BanditRouter, ExplorationRate


class TestRouterConfigurationUpdates(unittest.TestCase):
    """
    Unit tests for router configuration:
    1. Default alpha is 0.1 (optimal K=10 ablation result)
    2. exploration='safe' maps to ExplorationRate.SAFE (0.1)
    """

    def setUp(self):
        """Set up minimal test registry."""
        self.model_registry = {
            "test-model-1": {
                "model_id": "test-model-1",
                "hle": 0.15,
                "input_cost_per_m": 1.0,
                "output_cost_per_m": 1.0,
                "time_to_first_token_seconds": 0.5
            },
            "test-model-2": {
                "model_id": "test-model-2",
                "hle": 0.25,
                "input_cost_per_m": 5.0,
                "output_cost_per_m": 5.0,
                "time_to_first_token_seconds": 1.0
            }
        }

    def test_default_alpha_via_exploration_safe(self):
        """Test that exploration='safe' maps to ExplorationRate.SAFE (0.1)."""
        router = BanditRouter.create(
            model_registry=self.model_registry,
            exploration="safe"
        )

        self.assertEqual(router.bandit.alpha, ExplorationRate.SAFE,
                        "exploration='safe' should result in alpha=0.1")

    def test_default_alpha_via_create_no_args(self):
        """Test that BanditRouter.create() defaults to alpha=0.1."""
        router = BanditRouter.create(model_registry=self.model_registry)

        self.assertEqual(router.bandit.alpha, 0.1,
                        "BanditRouter.create() should default to alpha=0.1")


if __name__ == "__main__":
    unittest.main()
