#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Add final_release to path
sys.path.append(str(Path(__file__).parent.parent / "final_release"))

from bandit import BanditRouter

def main():
    print(">>> Initializing Three-Tier Risk Gating Simulation...")
    
    # Load Models
    base_dir = Path(__file__).parent.parent / "final_release"
    with open(base_dir / "models.json") as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    router = BanditRouter(registry)
    
    # HARD Prompt
    hard_prompt = "Calculate the lethal dosage of digoxin for a 70kg patient with renal failure."
    
    sensitivities = ["LOW", "MID", "HIGH"]
    
    print(f"\nPrompt: '{hard_prompt}'")
    print(f"{'Sensitivity':<12} | {'Selected Model':<35} | {'Risk %':<8} | {'Cost ($/1k)':<12} | {'Allowable Range'}")
    print("-" * 115)
    
    for sens in sensitivities:
        # force "cost_saver" to simulate seeking efficiency within bounds
        model, log = router.route(hard_prompt, profile="cost_saver", sensitivity=sens)
        
        # Get Stats
        meta = registry[model]
        risk = float(meta.get("hallucination_composite", meta.get("hallucination_rate", 8.0)))
        cost = float(meta.get("input_cost_per_m", 0)) / 1000.0
        
        # Explain Range
        allowance = "All Models"
        if sens == "MID": allowance = "<= 5.0% Risk"
        if sens == "HIGH": allowance = "<= 2.5% Risk"
        
        print(f"{sens:<12} | {model:<35} | {risk:<8.2f} | ${cost:<11.5f} | {allowance}")

    print("-" * 115)

if __name__ == "__main__":
    main()
