
import json
import numpy as np
from pathlib import Path
from final_release.bandit import BanditRouter

def test_risk_penalty():
    print("=== Testing Risk-Aware Utility (Hallucination Penalty) ===")
    
    # Setup dummy models
    # Model A: High Quality, Low Cost, HIGH Hallucination (The "Liar")
    # Model B: Slightly Lower Quality, Higher Cost, LOW Hallucination (The "Truthful")
    registry = {
        "provider/liar-specialist": {
            "openrouter_id": "provider/liar-specialist",
            "hle": 0.25, # High capability
            "input_cost_per_m": 0.1, # Cheap
            "output_cost_per_m": 0.1,
            "hallucination_rate": 12.0, # 12% Hallucination (Bad)
            "description": "A very smart but dishonest model."
        },
        "provider/truthful-anchor": {
            "openrouter_id": "provider/truthful-anchor",
            "hle": 0.22, # Slightly lower capability
            "input_cost_per_m": 2.0, # More expensive
            "output_cost_per_m": 2.0,
            "hallucination_rate": 2.0, # 2% Hallucination (Very Reliable)
            "description": "An honest, reliable model."
        }
    }
    
    # 1. Test with LOW risk tolerance (COST_SAVER)
    print("\nScenario 1: Cost Saver Profile (Risk Lambda = 2.0)")
    router = BanditRouter.create(model_registry=registry, exploration="static")
    selected, _ = router.route("Solve a complex math problem.", profile="cost_saver")
    print(f"Selected: {selected}")
    
    # 2. Test with HIGH risk tolerance (QUALITY_FIRST)
    print("\nScenario 2: Quality First Profile (Risk Lambda = 10.0)")
    selected, _ = router.route("Solve a complex math problem.", profile="quality_first")
    print(f"Selected: {selected}")
    
    # 3. Test with BALANCED (Risk Lambda = 5.0)
    print("\nScenario 3: Balanced Profile (Risk Lambda = 5.0)")
    selected, _ = router.route("Solve a complex math problem.", profile="best_value")
    print(f"Selected: {selected}")

if __name__ == "__main__":
    test_risk_penalty()
