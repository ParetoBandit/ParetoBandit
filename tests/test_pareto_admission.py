import unittest
from unittest.mock import MagicMock
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bandit_gpt.router import BanditRouter, RouterConfig

class TestParetoAdmission(unittest.TestCase):
    def setUp(self):
        # Reset RouterConfig to defaults before each test
        self.old_max_probation = RouterConfig.max_probation_models
        self.old_probation_reqs = RouterConfig.probation_requests
        
    def tearDown(self):
        RouterConfig.max_probation_models = self.old_max_probation
        RouterConfig.probation_requests = self.old_probation_reqs

    def test_rejection_of_expensive_model(self):
        """Verify that a ridiculously expensive model is rejected even with optimistic quality."""
        # Create a config with specific parameters if needed, or just rely on defaults
        router = BanditRouter.create()
        
        # Ridiculously expensive model (1000x market standard)
        # We need it to be dominated by ALL profiles.
        # Arbitrage/CostSaver/LowLatency: dominated by cheap/fast models.
        # MaxQuality: dominated by something like GPT-4o if it's cheaper.
        expensive_model = {
            "openrouter_id": "scam/expensive-model",
            "cost_per_1m_tokens": 100000.0, # Mega expensive
            "median_latency_s": 100.0,      # Mega slow
            "initial_quality": 0.1
        }
        
        # At quality=0.95 (optimistic), if it's $100/1k and 100s latency,
        # it should still be dominated by an existing model like GPT-4o
        # if GPT-4o has quality > 0.90 but cost $0.01/1k.
        result = router.admit_new_model(expensive_model)
        self.assertFalse(result, "Expensive model should have been Pareto rejected.")

    def test_probation_spam_guard(self):
        """Verify that the probation spam guard triggers correctly."""
        config = RouterConfig()
        config.max_probation_models = 2
        config.probation_requests = 100
        
        router = BanditRouter.create(config=config)
        # Ensure probation is empty
        router.probation_models = {}
        
        # Models that are decent enough to pass Pareto check
        model1 = {
            "openrouter_id": "test/model-1",
            "cost_per_1m_tokens": 0.1,
            "median_latency_s": 0.5,
            "initial_quality": 0.8
        }
        model2 = {
            "openrouter_id": "test/model-2",
            "cost_per_1m_tokens": 0.2,
            "median_latency_s": 0.5,
            "initial_quality": 0.8
        }
        model3 = {
            "openrouter_id": "test/model-3",
            "cost_per_1m_tokens": 0.3,
            "median_latency_s": 0.5,
            "initial_quality": 0.8
        }
        
        # First two should be admitted
        self.assertTrue(router.admit_new_model(model1))
        self.assertTrue(router.admit_new_model(model2))
        
        # Third one should be rejected due to probation limit
        self.assertFalse(router.admit_new_model(model3), "Third model should be rejected due to probation limit.")
        
    def test_probation_graduation(self):
        """Verify that models graduate from probation and free up slots."""
        config = RouterConfig()
        config.max_probation_models = 1
        config.probation_requests = 100
        
        router = BanditRouter.create(config=config)
        router.probation_models = {}
        
        model1 = {
            "openrouter_id": "test/model-1",
            "cost_per_1m_tokens": 0.1,
            "median_latency_s": 0.5,
            "initial_quality": 0.8
        }
        
        # Admit first model
        self.assertTrue(router.admit_new_model(model1))
        
        # Try to admit another - should fail
        model2 = {
            "openrouter_id": "test/model-2",
            "cost_per_1m_tokens": 0.2,
            "median_latency_s": 0.5,
            "initial_quality": 0.8
        }
        self.assertFalse(router.admit_new_model(model2))
        
        # "Graduate" model1 by advancing bandit time
        router.bandit.t = 200 # > 100 (immune_until was t_start + 100)
        
        # Now model2 should be admitted
        self.assertTrue(router.admit_new_model(model2), "Model 2 should be admitted after Model 1 graduates.")

if __name__ == "__main__":
    unittest.main()
