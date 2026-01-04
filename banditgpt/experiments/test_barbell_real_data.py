#!/usr/bin/env python3
"""
Test ARBITRAGE profile with REAL cluster-labeled test data.
Uses test_rewards_pareto_dedup.jsonl to demonstrate barbell distribution.
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from bandit import BanditRouter

# Load test data with cluster labels
data_dir = Path(__file__).parent.parent / "data"
test_file = data_dir / "test_rewards_pareto_dedup.jsonl"

if not test_file.exists():
    print(f"ERROR: {test_file} not found")
    sys.exit(1)

# Load prompts with cluster IDs
prompts = []
with open(test_file) as f:
    for line in f:
        data = json.loads(line)
        if "cluster_id" in data and "prompt" in data:
            prompts.append(data)

print("=" * 80)
print("BARBELL DISTRIBUTION TEST: Using Real Cluster-Labeled Data")
print("=" * 80)

# Initialize router
router = BanditRouter.create(
    priors="hle",
    prior_n_effective=60.0,
    prior_structure_n_effective=10.0,
    exploration="safe"
)

print(f"\n✓ Loaded {len(router.registry)} models")
print(f"✓ Loaded {len(prompts)} test prompts with cluster labels")

# Separate by difficulty using actual cluster IDs
easy_prompts = []
hard_prompts = []

for p in prompts:
    cluster_id = p.get("cluster_id")
    if cluster_id is not None:
        is_hard = BanditRouter._is_hard_cluster(cluster_id)
        if is_hard:
            hard_prompts.append(p)
        else:
            easy_prompts.append(p)

print(f"\n📊 Classification:")
print(f"   Easy prompts (clusters 0-40, 81-99): {len(easy_prompts)}")
print(f"   Hard prompts (clusters 41-80): {len(hard_prompts)}")

# Test on easy prompts
print("\n" + "=" * 80)
print("EASY PROMPTS (Expect Cheap Models like Gemma, Nova)")
print("=" * 80)

easy_selections = Counter()
for i, p in enumerate(easy_prompts[:10]):
    cluster_id = p["cluster_id"]
    prompt_text = p["prompt"]
    
    # Route with MANUAL override to ensure hardness=False
    # (In production, cluster detector would do this automatically)
    model, log = router.route(prompt_text, profile="arbitrage")
    
    cost = router.registry.get(model, {}).get("input_cost_per_m", 0)
    hle = router.registry.get(model, {}).get("hle", 0)
    
    easy_selections[model] += 1
    
    if i < 5:  # Show first 5
        print(f"\n{i+1}. Cluster {cluster_id}: {prompt_text[:70]}...")
        print(f"   Selected: {model}")
        print(f"   Cost: ${cost:.4f}/1M | HLE: {hle:.3f} | Utility: {log.predicted_utility:.3f}")

# Test on hard prompts
print("\n" + "=" * 80)
print("HARD PROMPTS (Expect Premium Models like Claude Opus, GPT-4)")
print("=" * 80)

hard_selections = Counter()
for i, p in enumerate(hard_prompts[:10]):
    cluster_id = p["cluster_id"]
    prompt_text = p["prompt"]
    
    model, log = router.route(prompt_text, profile="arbitrage")
    
    cost = router.registry.get(model, {}).get("input_cost_per_m", 0)
    hle = router.registry.get(model, {}).get("hle", 0)
    
    hard_selections[model] += 1
    
    if i < 5:  # Show first 5
        print(f"\n{i+1}. Cluster {cluster_id}: {prompt_text[:70]}...")
        print(f"   Selected: {model}")
        print(f"   Cost: ${cost:.4f}/1M | HLE: {hle:.3f} | Utility: {log.predicted_utility:.3f}")

# Summary Analysis
print("\n" + "=" * 80)
print("BARBELL DISTRIBUTION ANALYSIS")
print("=" * 80)

print("\n📊 Easy Prompts - Top Models:")
for model, count in easy_selections.most_common(3):
    cost = router.registry.get(model, {}).get("input_cost_per_m", 0)
    hle = router.registry.get(model, {}).get("hle", 0)
    pct = 100 * count / max(len(easy_selections), 1)
    print(f"  {model:45} | {count:3} ({pct:5.1f}%) | ${cost:.4f}/1M | HLE: {hle:.3f}")

print("\n📊 Hard Prompts - Top Models:")
for model, count in hard_selections.most_common(3):
    cost = router.registry.get(model, {}).get("input_cost_per_m", 0)
    hle = router.registry.get(model, {}).get("hle", 0)
    pct = 100 * count / max(len(hard_selections), 1)
    print(f"  {model:45} | {count:3} ({pct:5.1f}%) | ${cost:.4f}/1M | HLE: {hle:.3f}")

# Calculate average costs
if easy_selections:
    avg_cost_easy = sum(
        router.registry.get(m, {}).get("input_cost_per_m", 0) * count 
        for m, count in easy_selections.items()
    ) / sum(easy_selections.values())
else:
    avg_cost_easy = 0

if hard_selections:
    avg_cost_hard = sum(
        router.registry.get(m, {}).get("input_cost_per_m", 0) * count 
        for m, count in hard_selections.items()
    ) / sum(hard_selections.values())
else:
    avg_cost_hard = 0

print("\n📈 Cost Analysis:")
print(f"  Average cost (Easy prompts): ${avg_cost_easy:.4f}/1M")
print(f"  Average cost (Hard prompts): ${avg_cost_hard:.4f}/1M")

# Check for barbell success
cheapest_cost = min(m.get("input_cost_per_m", 999) for m in router.registry.values())
most_expensive_cost = max(m.get("input_cost_per_m", 0) for m in router.registry.values())

print(f"\n🎯 Barbell Metrics:")
print(f"  Cheapest model in registry: ${cheapest_cost:.4f}/1M")
print(f"  Most expensive in registry: ${most_expensive_cost:.4f}/1M")

if avg_cost_easy < avg_cost_hard:
    print(f"\n✅ SUCCESS: Easy prompts use cheaper models (${avg_cost_easy:.4f} < ${avg_cost_hard:.4f})")
else:
    print(f"\n⚠️  WARNING: No cost differentiation (Easy: ${avg_cost_easy:.4f}, Hard: ${avg_cost_hard:.4f})")

print("\n" + "=" * 80)
