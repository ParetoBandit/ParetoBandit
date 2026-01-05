#!/usr/bin/env python3
"""
Investigate cost normalization impact on model selection.
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load registry
models_path = Path(__file__).parent.parent.parent / "models.json"
with open(models_path) as f:
    data = json.load(f)
registry = {m["openrouter_id"]: m for m in data["models"]}

# Get model costs
def get_cost(model, in_tok=100, out_tok=200):
    if "price_1m_input" not in model or "price_1m_output" not in model:
        return None
    return (in_tok * model["price_1m_input"] + out_tok * model["price_1m_output"]) / 1_000_000

model_costs = []
for model_id, model in registry.items():
    cost = get_cost(model)
    if cost:
        model_costs.append({
            "id": model_id,
            "name": model.get("display_name", model_id)[:30],
            "cost": cost,
            "quality": model.get("overall_success_rate", 0)
        })

model_costs.sort(key=lambda x: x["cost"])

print("=" * 90)
print("COST DISTRIBUTION ANALYSIS")
print("=" * 90)

costs = [m["cost"] for m in model_costs]
log_costs = [np.log(c) for c in costs]

print(f"\nCost range (linear): ${min(costs)*1000:.4f} to ${max(costs)*1000:.4f} per 1k")
print(f"Cost range (log): {min(log_costs):.2f} to {max(log_costs):.2f}")
print(f"Log range: {max(log_costs) - min(log_costs):.2f}")
print(f"GLOBAL_COST_RANGE in code: 5.3")

GLOBAL_COST_RANGE = 5.3
actual_range = max(log_costs) - min(log_costs)
print(f"\nUsing max(actual_range, GLOBAL_COST_RANGE) = max({actual_range:.2f}, 5.3) = {max(actual_range, GLOBAL_COST_RANGE):.2f}")

# Show cost bands
print("\n" + "=" * 90)
print("COST BANDS (with normalized cost using GLOBAL_COST_RANGE)")
print("=" * 90)

min_log = min(log_costs)
max_log = max(log_costs)
range_c = max(max_log - min_log, GLOBAL_COST_RANGE)

bands = [
    ("Ultra Cheap", 0.00, 0.10),
    ("Very Cheap", 0.10, 0.25),
    ("Cheap", 0.25, 0.40),
    ("Mid-Range", 0.40, 0.60),
    ("Expensive", 0.60, 0.80),
    ("Premium", 0.80, 1.00)
]

for band_name, norm_min, norm_max in bands:
    in_band = [m for m in model_costs 
               if norm_min <= (np.log(m["cost"]) - min_log) / range_c < norm_max]
    if in_band:
        print(f"\n{band_name} (norm {norm_min:.2f}-{norm_max:.2f}):")
        for m in in_band[:3]:
            norm_cost = (np.log(m["cost"]) - min_log) / range_c
            print(f"  {m['name']:\u003c30} ${m['cost']*1000:6.4f}/1k  norm={norm_cost:.3f}  qual={m['quality']:.3f}")
        if len(in_band) > 3:
            print(f"  ... and {len(in_band)-3} more")

# Simulate ARBITRAGE penalty
print("\n" + "=" * 90)
print("ARBITRAGE PENALTY SIMULATION (lambda_cost=0.55)")
print("=" * 90)

lambda_cost = 0.55
assumed_quality = 0.85  # Assume all models have 0.85 quality for comparison

print(f"\nAssuming constant quality UCB = {assumed_quality:.2f} for all models")
print("Utility = Quality - (lambda_cost * norm_cost)")
print()

for m in model_costs[:5]:
    log_cost = np.log(m["cost"])
    norm_cost = (log_cost - min_log) / range_c
    penalty = lambda_cost * norm_cost
    utility = assumed_quality - penalty
    print(f"{m['name']:\u003c30} ${m['cost']*1000:6.4f}/1k  norm={norm_cost:.3f}  penalty={penalty:.3f}  utility={utility:.3f}")

print("\n..." + " " * 78 + "...")

for m in model_costs[-5:]:
    log_cost = np.log(m["cost"])
    norm_cost = (log_cost - min_log) / range_c
    penalty = lambda_cost * norm_cost
    utility = assumed_quality - penalty
    print(f"{m['name']:\u003c30} ${m['cost']*1000:6.4f}/1k  norm={norm_cost:.3f}  penalty={penalty:.3f}  utility={utility:.3f}")

# Find the crossover point
print("\n" + "=" * 90)
print("ANALYSIS: When does a premium model beat a cheap model?")
print("=" * 90)

cheap = model_costs[0]
cheap_log = np.log(cheap["cost"])
cheap_norm = (cheap_log - min_log) / range_c

print(f"\nCheapest model: {cheap['name']}")
print(f"  Cost: ${cheap['cost']*1000:.4f}/1k (norm={cheap_norm:.4f})")
print(f"  Penalty: {lambda_cost * cheap_norm:.4f}")
print(f"  Required quality: {lambda_cost * cheap_norm:.4f} (to break even with utility=0)")

print(f"\nFor a premium model to beat this, it needs:")
for pct in [0.5, 0.7, 0.9]:
    prem_norm = pct
    prem_penalty = lambda_cost * prem_norm
    quality_needed = (lambda_cost * cheap_norm) + prem_penalty
    print(f"  At norm_cost={prem_norm:.2f}: quality must be {quality_needed:.3f} higher (Δ = {prem_penalty:.3f})")
