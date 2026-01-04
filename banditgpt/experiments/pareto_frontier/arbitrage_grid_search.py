#!/usr/bin/env python3
"""
2D Grid Search: Optimize (N_prior, N_struct) with Max Quality Profile

Search Space:
- N_prior (prior_n_effective): Controls b vector (mean beliefs)
- N_struct (prior_structure_n_effective): Controls A matrix (covariance structure)

Test Profile: lambda_cost=0 (Max Quality - ignore cost completely)
Goal: Find if ANY combination allows priors to select premium models when cost is ignored.
"""
import sys
from pathlib import Path
import json
import numpy as np
import random
from collections import defaultdict
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from banditgpt.bandit import BanditRouter
from sentence_transformers import SentenceTransformer

# Load data
data_dir = Path(__file__).parent.parent.parent / "data"
test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
models_path = Path(__file__).parent.parent.parent / "models.json"

with open(models_path) as f:
    data = json.load(f)
registry = {m["openrouter_id"]: m for m in data["models"]}

# Build z-score lookup
zscore_lookup = {}
for model_id, model in registry.items():
    if "cluster_success_rates" in model:
        for cluster_id_str, cluster_data in model["cluster_success_rates"].items():
            if isinstance(cluster_data, dict) and "z_score" in cluster_data:
                zscore_lookup[(model_id, int(cluster_id_str))] = cluster_data["z_score"]

# Load test data
test_data = defaultdict(lambda: {"cluster_id": None, "rewards": {}, "zscores": {}})
with open(test_rewards_path) as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("ok"):
            prompt = entry["prompt"]
            model_id = entry["model_id"]
            cluster_id = entry.get("cluster_id", 0)
            test_data[prompt]["cluster_id"] = cluster_id
            test_data[prompt]["rewards"][model_id] = entry["raw_score"]
            zscore = zscore_lookup.get((model_id, cluster_id), 0.0)
            test_data[prompt]["zscores"][model_id] = zscore

print("=" * 90)
print("2D GRID SEARCH: MAX QUALITY PROFILE (lambda_cost=0)")
print("=" * 90)

# Define search space
n_prior_values = [1, 2, 5, 10, 20, 50]
n_struct_values = [5, 10, 20, 50, 100]

print(f"\nSearch space:")
print(f"  N_prior (prior_n_effective): {n_prior_values}")
print(f"  N_struct (prior_structure_n_effective): {n_struct_values}")
print(f"  Total combinations: {len(n_prior_values) * len(n_struct_values)}")

# Initialize encoder once
print(f"\nInitializing encoder...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sample test prompts (use subset for speed)
test_prompts = list(test_data.keys())
random.seed(42)
random.shuffle(test_prompts)
n_test = 100  # Use 100 prompts for evaluation
test_subset = test_prompts[:n_test]

print(f"Using {n_test} test prompts for evaluation\n")

# Run grid search
results = []
total_combinations = len(n_prior_values) * len(n_struct_values)
current = 0

for n_prior, n_struct in product(n_prior_values, n_struct_values):
    current += 1
    print(f"[{current}/{total_combinations}] Testing N_prior={n_prior}, N_struct={n_struct}...", end=" ")
    
    try:
        # Create router with these parameters
        router = BanditRouter.create(
            registry,
            exploration="static",  # Pure exploitation
            priors="csr",
            prior_n_effective=float(n_prior),
            prior_structure_n_effective=float(n_struct),
            context_encoder=encoder
        )
        
        # Evaluate on test subset
        total_zscore = 0.0
        total_cost = 0.0
        count = 0
        selections = defaultdict(int)
        
        # Use lambda_cost=0 (Max Quality - ignore cost)
        max_quality_profile = {"lambda_cost": 0.0, "lambda_latency": 0.0}
        
        for prompt in test_subset:
            data = test_data[prompt]
            selected, log = router.route(prompt, profile=max_quality_profile, input_tokens=100)
            
            if selected in data["zscores"]:
                zscore = data["zscores"][selected]
                model = registry[selected]
                cost = (100 * model["price_1m_input"] + 200 * model["price_1m_output"]) / 1_000_000
                
                if cost < float('inf'):
                    total_cost += cost * 1000  # Convert to $/1k
                    total_zscore += zscore
                    count += 1
                    selections[selected] += 1
        
        if count > 0:
            avg_zscore = total_zscore / count
            avg_cost = total_cost / count
            n_unique = len(selections)
            
            # Check if premium models are selected
            premium_count = sum(1 for m, cnt in selections.items() 
                              if registry[m]["price_1m_input"] * 100 / 1_000_000 > 1.0)
            has_premium = premium_count > 0
            
            results.append({
                "n_prior": n_prior,
                "n_struct": n_struct,
                "avg_zscore": avg_zscore,
                "avg_cost": avg_cost,
                "n_unique": n_unique,
                "has_premium": has_premium,
                "selections": dict(selections)
            })
            
            premium_marker = "✓" if has_premium else "✗"
            print(f"Z={avg_zscore:+.3f}σ, Cost=${avg_cost:.4f}, Models={n_unique}, Premium={premium_marker}")
        else:
            print("FAILED (no valid selections)")
            
    except Exception as e:
        print(f"ERROR: {e}")

# Analyze results
print("\n" + "=" * 90)
print("GRID SEARCH RESULTS")
print("=" * 90)

if results:
    # Sort by Z-score
    results.sort(key=lambda x: x["avg_zscore"], reverse=True)
    
    print(f"\nTOP 10 Configurations (by Z-score):")
    print("-" * 90)
    print(f"{'Rank':<6} {'N_prior':<10} {'N_struct':<10} {'Z-score':<12} {'Cost/1k':<12} {'Models':<8} {'Premium'}")
    print("-" * 90)
    
    for i, r in enumerate(results[:10], 1):
        premium = "Yes" if r["has_premium"] else "No"
        print(f"{i:<6} {r['n_prior']:<10} {r['n_struct']:<10} {r['avg_zscore']:+.4f}σ     "
              f"${r['avg_cost']:<11.4f} {r['n_unique']:<8} {premium}")
    
    # Find best with premium models
    with_premium = [r for r in results if r["has_premium"]]
    if with_premium:
        best_with_premium = with_premium[0]
        print(f"\n✓ Best configuration WITH premium models:")
        print(f"  N_prior = {best_with_premium['n_prior']}")
        print(f"  N_struct = {best_with_premium['n_struct']}")
        print(f"  Z-score = {best_with_premium['avg_zscore']:+.3f}σ")
        print(f"  Cost = ${best_with_premium['avg_cost']:.4f}/1k")
        print(f"  Unique models = {best_with_premium['n_unique']}")
        
        # Show model breakdown
        print(f"\n  Top 5 selected models:")
        sorted_sel = sorted(best_with_premium["selections"].items(), key=lambda x: -x[1])
        for model_id, cnt in sorted_sel[:5]:
            model = registry[model_id]
            name = model.get("display_name", model_id)[:30]
            cost = (100 * model["price_1m_input"] + 200 * model["price_1m_output"]) / 1_000_000 * 1000
            print(f"    {name:<30} {cnt:3}x  ${cost:.4f}/1k")
    
    # Save results
    output_file = Path(__file__).parent / "grid_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved to: {output_file}")
    
else:
    print("\n❌ No successful results!")

print("\n" + "=" * 90)
print("GRID SEARCH COMPLETE")
print("=" * 90)
