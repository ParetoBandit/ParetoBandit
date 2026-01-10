#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    results_path = Path(__file__).parent / "results" / "arbitrage_results.json"
    
    if not results_path.exists():
        print("Results file not found.")
        return

    with open(results_path) as f:
        data = json.load(f)

    print("\n### ⚖️ Stable Model Frontier (Static Baseline)")
    print("| Model Name | Cost ($/1M) | Quality (Hard %) |")
    print("| :--- | :--- | :--- |")
    
    # Sort frontier by cost
    frontier = sorted(data["pareto_frontier"], key=lambda x: x["cost"])
    for m in frontier:
        if m["cost"] > 0: # Filter out weird zero-cost artifacts if any
            name = m["display_name"]
            cost = f"${m['cost']:.2f}"
            qual = f"{m['quality']*100:.1f}%"
            print(f"| {name} | {cost} | {qual} |")

    print("\n### 🚀 BanditGPT (Arbitrage Curve)")
    print("| Profile | Cost ($/1M) | Quality (Hard %) | Efficiency Gain |")
    print("| :--- | :--- | :--- | :--- |")
    
    # Sort bandit curve by cost
    curve = sorted(data["bandit_curve"], key=lambda x: x["cost_mean"])
    
    for b in curve:
        name = b["profile"]
        cost_val = b["cost_mean"]
        qual_val = b["quality_mean"]
        
        cost = f"${cost_val:.2f}"
        qual = f"{qual_val*100:.1f}%"
        
        # Simple gain heuristic: "How much cheaper than a model with similar quality?"
        # Find closest frontier model with quality >= bandit quality
        comparable_model = next((m for m in frontier if m["quality"] >= qual_val), None)
        
        if comparable_model:
            ref_cost = comparable_model["cost"]
            if ref_cost > cost_val:
                reduction = (ref_cost - cost_val) / ref_cost * 100
                gain = f"⬇ {reduction:.0f}% Cost"
            else:
                gain = "-"
        else:
             gain = "Top Quality"

        print(f"| {name} | {cost} | {qual} | {gain} |")

if __name__ == "__main__":
    main()
