#!/usr/bin/env python3
"""
Detailed UCB and penalty analysis for ARBITRAGE profile.
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from banditgpt.bandit import BanditRouter
from sentence_transformers import SentenceTransformer

# Load data
data_dir = Path(__file__).parent.parent.parent / "data"
models_path = Path(__file__).parent.parent.parent / "models.json"

with open(models_path) as f:
    data = json.load(f)
registry = {m["openrouter_id"]: m for m in data["models"]}

print("=" * 90)
print("DETAILED UCB + PENALTY ANALYSIS")
print("=" * 90)

# Initialize router
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER
encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
router = BanditRouter.create(
    registry,
    exploration="static",  # Pure exploitation (alpha=0)
    priors="csr",
    prior_n_effective=50.0,
    prior_structure_n_effective=50.0,
    context_encoder=encoder
)

# Test prompt
test_prompt = "Explain the difference between supervised and unsupervised learning in machine learning."

# Get context vector
x = router._get_context_vector(test_prompt)

# Get UCB values for all models
print(f"\nTest prompt: '{test_prompt[:60]}...'")
print(f"\nAnalyzing model UCBs and penalties with ARBITRAGE profile (lambda_cost=0.55):\n")

model_analysis = []
for model_id, model in registry.items():
    if model_id not in router.bandit.models:
        continue
        
    # Get UCB
    theta = router.bandit.A_inv[model_id] @ router.bandit.b[model_id]
    ucb = float(theta.dot(x))
    
    # Get cost
    cost = router._estimate_cost(model_id, 100, 200)
    if cost == float('inf'):
        continue
    cost_per_1k = cost * 1000
    
    # Get absolute penalty
    penalty = router._calculate_absolute_penalty(cost_per_1k)
    
    # Calculate utility with ARBITRAGE lambda
    lambda_cost = 0.55
    utility = ucb - (lambda_cost * penalty)
    
    model_analysis.append({
        "id": model_id,
        "name": model.get("display_name", model_id)[:30],
        "ucb": ucb,
        "cost": cost_per_1k,
        "penalty": penalty,
        "utility": utility,
        "quality": model.get("overall_success_rate", 0)
    })

# Sort by utility
model_analysis.sort(key=lambda x: x["utility"], reverse=True)

print("TOP 15 Models by Utility (UCB - 0.55*Penalty):")
print("-" * 90)
print(f"{'Model':\u003c30} {'UCB':^7} {'Cost/1k':^8} {'Penalty':^8} {'Utility':^8}")
print("-" * 90)
for m in model_analysis[:15]:
    print(f"{m['name']:\u003c30} {m['ucb']:7.4f} ${m['cost']:7.4f} {m['penalty']:8.4f} {m['utility']:8.4f}")

print("\n\nCOMPARISON: Top 5 vs Bottom 5")
print("=" * 90)
print("\nTOP 5 (Winning the Arbitrage):")
for m in model_analysis[:5]:
    print(f"  {m['name']:\u003c30} UCB={m['ucb']:.4f} - 0.55*{m['penalty']:.3f} = {m['utility']:.4f}")

print("\nBOTTOM 5 (Losing the Arbitrage):")
for m in model_analysis[-5:]:
    print(f"  {m['name']:\u003c30} UCB={m['ucb']:.4f} - 0.55*{m['penalty']:.3f} = {m['utility']:.4f}")

# Show penalty distribution
print("\n\n" + "=" * 90)
print("PENALTY DISTRIBUTION")
print("=" * 90)

penalty_bands = [
    ("Ultra Low", 0.0, 0.2),
    ("Low", 0.2, 0.4),
    ("Medium", 0.4, 0.6),
    ("High", 0.6, 0.8),
    ("Ultra High", 0.8, 1.0)
]

for band_name, p_min, p_max in penalty_bands:
    in_band = [m for m in model_analysis if p_min <= m["penalty"] < p_max]
    if in_band:
        avg_ucb = np.mean([m["ucb"] for m in in_band])
        avg_cost = np.mean([m["cost"] for m in in_band])
        print(f"\n{band_name} Penalty ({p_min:.1f}-{p_max:.1f}):")
        print(f"  {len(in_band)} models, avg UCB={avg_ucb:.3f}, avg cost=${avg_cost:.4f}/1k")
        if len(in_band) <= 3:
            for m in in_band:
                print(f"    {m['name']}: UCB={m['ucb']:.3f}, penalty={m['penalty']:.3f}")

# Check if premium models are competitive
print("\n\n" + "=" * 90)
print("PREMIUM MODEL COMPETITIVENESS")
print("=" * 90)

premium_threshold = 1.0  # $1/1k
premium_models = [m for m in model_analysis if m["cost"] >= premium_threshold]
cheap_models = [m for m in model_analysis if m["cost"] < 0.05]

if premium_models:
    best_premium = max(premium_models, key=lambda x: x["utility"])
    print(f"\nBest premium model (>${premium_threshold}/1k):")
    print(f"  {best_premium['name']}")
    print(f"  UCB: {best_premium['ucb']:.4f}")
    print(f"  Cost: ${best_premium['cost']:.4f}/1k")
    print(f"  Penalty: {best_premium['penalty']:.4f}")
    print(f"  Utility: {best_premium['utility']:.4f}")
    
if cheap_models:
    best_cheap = max(cheap_models, key=lambda x: x["utility"])
    print(f"\nBest cheap model (\u003c$0.05/1k):")
    print(f"  {best_cheap['name']}")
    print(f"  UCB: {best_cheap['ucb']:.4f}")
    print(f"  Cost: ${best_cheap['cost']:.4f}/1k")
    print(f"  Penalty: {best_cheap['penalty']:.4f}")
    print(f"  Utility: {best_cheap['utility']:.4f}")
    
    if premium_models:
        gap = best_cheap["utility"] - best_premium["utility"]
        print(f"\n  Gap (cheap - premium): {gap:+.4f}")
        if gap > 0:
            print(f"  ❌ Cheap model wins by {gap:.4f}")
        else:
            print(f"  ✓ Premium model wins by {-gap:.4f}")
