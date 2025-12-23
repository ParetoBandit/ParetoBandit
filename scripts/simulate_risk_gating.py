#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from collections import Counter

# Add final_release to path
sys.path.append(str(Path(__file__).parent.parent / "final_release"))

from bandit import BanditRouter, OptimizationProfile

def main():
    print(">>> Initializing Two-Stage Risk Gating Router...")
    
    # Load Models
    base_dir = Path(__file__).parent.parent / "final_release"
    with open(base_dir / "models.json") as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    router = BanditRouter(registry)
    
    # Define Scenarios
    scenarios = {
        "CREATIVE (Low Sensitivity)": [
            "Write a sci-fi story about robots",
            "Generate a creative name for a startup",
            "Summarize this fun movie plot",
            "Tell me a joke about programming",
            "Write a poem about the ocean"
        ],
        "SENSITIVE (High Sensitivity)": [
            "What is the recommended dosage for Ibuprofen?",
            "Write a python function to merge sort",
            "Solve this calculus equation",
            "Who is the current Prime Minister of UK?",
            "Explain the legal definition of negligence"
        ]
    }
    
    print(f"{'Scenario':<30} | {'Selected Model':<35} | {'Risk %':<8} | {'Cost ($/1k)':<12} | {'Why?'}")
    print("-" * 115)
    
    for category, prompts in scenarios.items():
        counts = Counter()
        
        for p in prompts:
            # force "cost_saver" to see if it allows cheap/risky models when safe
            model, log = router.route(p, profile="cost_saver")
            counts[model] += 1
            
            # Get Stats
            meta = registry[model]
            risk = float(meta.get("hallucination_composite", meta.get("hallucination_rate", 8.0)))
            cost = float(meta.get("input_cost_per_m", 0)) / 1000.0
            
            # Truncate prompt for display
            display_p = (p[:25] + "..") if len(p) > 25 else p
            
            print(f"{category:<30} | {model:<35} | {risk:<8.2f} | ${cost:<11.5f} | {display_p}")

    print("-" * 115)
    print("\n>>> SUMMARY: PORTFOLIO SHIFT")
    # ... logic to print summary if needed, but the table above is good.

if __name__ == "__main__":
    main()
