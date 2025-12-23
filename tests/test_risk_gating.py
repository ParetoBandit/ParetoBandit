#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

# Add final_release to path
sys.path.append(str(Path(__file__).parent.parent / "final_release"))

from bandit import BanditRouter, OptimizationProfile

class TestRiskGating(unittest.TestCase):
    def setUp(self):
        # Mock Registry with Known Risk Profiles
        self.mock_registry = {
            "safe_model": {
                "openrouter_id": "safe_model",
                "hallucination_composite": 1.5,
                "input_cost_per_m": 5.0,
                "time_to_first_token_seconds": 0.5
            },
            "risky_model": {
                "openrouter_id": "risky_model",
                "hallucination_composite": 15.0, # High Risk
                "input_cost_per_m": 0.1, # Cheap
                "time_to_first_token_seconds": 0.1 # Fast
            },
            "borderline_model": {
                "openrouter_id": "borderline_model",
                "hallucination_composite": 4.9, # Just safe
                "input_cost_per_m": 1.0,
                "time_to_first_token_seconds": 0.3
            }
        }
        self.router = BanditRouter(self.mock_registry)
        
    def test_classifier(self):
        """Verify the heuristics classify correctly."""
        high_risk = [
            "What is the dose for Ibuprofen?",
            "Write a python function to sort list",
            "Solve this math equation: 2x + 4 = 10",
            "Who is the president of France?",
            "Provide legal advice for a contract"
        ]
        low_risk = [
            "Write a poem about the sun",
            "Tell me a joke",
            "Summarize this short story",
            "Generate a creative idea for a logo"
        ]
        
        for p in high_risk:
            self.assertEqual(self.router._classify_sensitivity(p), "HIGH", f"Failed on: {p}")
            
        for p in low_risk:
            self.assertEqual(self.router._classify_sensitivity(p), "LOW", f"Failed on: {p}")

    def test_gating_high_sensitivity(self):
        """High sensitivity prompt should BLOCK risky_model."""
        prompt = "Medical advice: dosage for ibuprofen"
        
        # We can inspect the logs or monkeypatch, but router returns the selected model.
        # Let's force a scenario where risky_model would win on utility IF included.
        # Risky is significantly cheaper/faster, so it has high utility in "cost_saver" profile.
        
        # With Gating: Risky should be excluded. Safe or Borderline should win.
        model, log = self.router.route(prompt, profile="cost_saver")
        
        print(f"HIGH Sensitivity Selected: {model}")
        self.assertNotEqual(model, "risky_model", "Risky model was selected despite High Sensitivity!")
        self.assertTrue(model in ["safe_model", "borderline_model"])

    def test_gating_low_sensitivity(self):
        """Low sensitivity prompt should ALLOW risky_model if it's best."""
        prompt = "Write a creative poem about clouds."
        
        # In 'cost_saver' mode, risky_model is way cheaper ($0.1 vs $5.0) and faster.
        # It should win easily if allowed.
        model, log = self.router.route(prompt, profile="cost_saver")
        
        print(f"LOW Sensitivity Selected: {model}")
        self.assertEqual(model, "risky_model", "Risky model (efficient) should be selected for creative task.")

if __name__ == "__main__":
    unittest.main()
