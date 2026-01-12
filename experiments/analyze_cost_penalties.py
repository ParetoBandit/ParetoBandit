#!/usr/bin/env python3
"""
Analyze cost penalty distribution to verify if clustering is an issue.

This script loads actual model costs from models.json and calculates
the normalized cost penalties to see if they cluster in a narrow range.
"""

import json
import math
import numpy as np
from pathlib import Path

# Current normalization parameters (from router.py - UPDATED)
MARKET_COST_FLOOR = 0.0001  # $/1k tokens
MARKET_COST_CEILING = 0.04  # $/1k tokens



def calculate_cost_per_1k(model_data):
    """Calculate blended cost per 1k tokens."""
    input_cost = model_data.get("input_cost_per_m") or 0.0
    output_cost = model_data.get("output_cost_per_m") or 0.0
    cost_per_1m = 0.5 * input_cost + 0.5 * output_cost
    return cost_per_1m / 1000.0


def calculate_penalty(cost_per_1k, floor=MARKET_COST_FLOOR, ceiling=MARKET_COST_CEILING):
    """Calculate normalized cost penalty using current formula."""
    safe_cost = max(cost_per_1k, floor)
    log_cost = math.log(safe_cost)
    log_floor = math.log(floor)
    log_ceiling = math.log(ceiling)
    log_range = log_ceiling - log_floor
    
    penalty = (log_cost - log_floor) / log_range
    return max(0.0, min(1.0, penalty))


def main():
    #Load models
    project_root = Path(__file__).parent.parent
    models_path = project_root / "src" / "bandit_gpt" / "config" / "models.json"
    
    with open(models_path) as f:
        data = json.load(f)
    models_list = data.get("models", [])
    
    print("=" * 70)
    print("COST PENALTY DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 Current Normalization Parameters:")
    print(f"  Floor: ${MARKET_COST_FLOOR:.6f}/1k")
    print(f"  Ceiling: ${MARKET_COST_CEILING:.6f}/1k")
    log_range = math.log(MARKET_COST_CEILING) - math.log(MARKET_COST_FLOOR)
    print(f"  Log range: {log_range:.2f}")
    
    # Calculate costs and penalties
    results = []
    for model_data in models_list:
        model_id = model_data.get("openrouter_id")
        cost_per_1k = calculate_cost_per_1k(model_data)
        penalty = calculate_penalty(cost_per_1k)
        
        results.append({
            "id": model_id,
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "cost": cost_per_1k,
            "penalty": penalty
        })
    
    # Sort by cost
    results.sort(key=lambda x: x["cost"])
    
    # Statistics
    costs = [r["cost"] for r in results]
    penalties = [r["penalty"] for r in results]
    
    print(f"\n📈 Portfolio Statistics:")
    print(f"  Models: {len(results)}")
    print(f"  Cost range: ${min(costs):.6f} - ${max(costs):.6f}/1k")
    print(f"  Cost mean: ${np.mean(costs):.6f} ± ${np.std(costs):.6f}")
    print(f"  Penalty range: [{min(penalties):.3f}, {max(penalties):.3f}]")
    print(f"  Penalty mean: {np.mean(penalties):.3f} ± {np.std(penalties):.3f}")
    
    # Check for clustering
    penalty_std = np.std(penalties)
    if penalty_std < 0.15:
        print(f"\n⚠️ WARNING: Low penalty spread (std={penalty_std:.3f})")
        print("   Most models cluster in narrow range - weak cost differentiation!")
    else:
        print(f"\n✅ Good penalty spread (std={penalty_std:.3f})")
    
    # Show distribution by decile
    print(f"\n📊 Penalty Distribution:")
    penalty_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, bin_edges = np.histogram(penalties, bins=penalty_bins)
    
    for i, count in enumerate(hist):
        pct = (count / len(penalties)) * 100
        bar = "█" * int(pct / 2)
        print(f"  [{penalty_bins[i]:.1f}-{penalty_bins[i+1]:.1f}]: {count:2d} models ({pct:5.1f}%) {bar}")
    
    # Show examples
    print(f"\n📋 Example Models (sorted by cost):")
    print(f"{'Model':<40} {'Cost ($/1k)':<12} {'Penalty':<8}")
    print("-" * 70)
    
    # Show first 5, middle 5, last 5
    examples = results[:5] + results[len(results)//2-2:len(results)//2+3] + results[-5:]
    shown = set()
    for r in examples:
        if r["id"] not in shown:
            print(f"{r['name'][:40]:<40} ${r['cost']:<11.6f} {r['penalty']:.3f}")
            shown.add(r["id"])
    
    # Calculate optimal anchors
    print(f"\n🔧 Suggested Optimal Anchors:")
    p5 = np.percentile(costs, 5)
    p95 = np.percentile(costs, 95)
    print(f"  5th percentile: ${p5:.6f}/1k")
    print(f"  95th percentile: ${p95:.6f}/1k")
    print(f"  Suggested floor: ${p5:.6f}")
    print(f"  Suggested ceiling: ${p95:.6f}")
    
    # Show what penalties would be with new anchors
    print(f"\n📊 Penalties with Suggested Anchors:")
    new_penalties = [calculate_penalty(r["cost"], floor=p5, ceiling=p95) for r in results]
    print(f"  New penalty range: [{min(new_penalties):.3f}, {max(new_penalties):.3f}]")
    print(f"  New penalty mean: {np.mean(new_penalties):.3f} ± {np.std(new_penalties):.3f}")
    print(f"  Spread improvement: {np.std(new_penalties) / penalty_std:.2f}x")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
