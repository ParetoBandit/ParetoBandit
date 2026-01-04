#!/usr/bin/env python3
"""
Benchmark Bandit Profiles with Real Test Data.
Compares model selections across all optimization profiles (Arbitrage, Quality, Value, Cost).
"""

import sys
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict

# Adjust path to find banditgpt
sys.path.insert(0, str(Path(__file__).parent.parent))

from bandit import BanditRouter, OptimizationProfile

# Configure logging to avoid noise
logging.getLogger("banditgpt").setLevel(logging.ERROR)

# Load test data with cluster labels
data_dir = Path(__file__).parent.parent / "data"
test_file = data_dir / "test_rewards_pareto_dedup.jsonl"

if not test_file.exists():
    print(f"ERROR: {test_file} not found")
    sys.exit(1)

# Load prompts
prompts = []
with open(test_file) as f:
    for line in f:
        data = json.loads(line)
        if "cluster_id" in data and "prompt" in data:
            prompts.append(data)

print("=" * 100)
print(f"PROFILE BENCHMARK: {len(prompts)} Test Prompts")
print("=" * 100)

# Initialize router
router = BanditRouter.create(
    priors="hle",
    prior_n_effective=60.0,
    prior_structure_n_effective=10.0,
    exploration="safe"
)

# Identify profiles to test
profiles = [
    "quality_first",
    "best_value",     # Balanced
    "cost_saver",
    "value_efficient",
    "arbitrage"       # Barbell
]

# Limit sample size for speed
SAMPLE_SIZE = 500
easy_prompts = [p for p in prompts if not BanditRouter._is_hard_cluster(p.get("cluster_id"))][:SAMPLE_SIZE]
hard_prompts = [p for p in prompts if BanditRouter._is_hard_cluster(p.get("cluster_id"))][:SAMPLE_SIZE]

print(f"Testing on {len(easy_prompts)} Easy and {len(hard_prompts)} Hard prompts per profile...")

results = {}

for profile in profiles:
    print(f"\nRunning Profile: {profile.upper()}...")
    
    profile_stats = {
        "easy": Counter(),
        "hard": Counter(),
        "easy_cost": 0.0,
        "hard_cost": 0.0
    }
    
    # Run Easy Prompts
    for p in easy_prompts:
        model, _ = router.route(p["prompt"], profile=profile)
        profile_stats["easy"][model] += 1
        profile_stats["easy_cost"] += router.registry[model]["input_cost_per_m"]
        
    # Run Hard Prompts
    for p in hard_prompts:
        model, _ = router.route(p["prompt"], profile=profile)
        profile_stats["hard"][model] += 1
        profile_stats["hard_cost"] += router.registry[model]["input_cost_per_m"]
        
    results[profile] = profile_stats

# --- REPORTING ---

print("\n" + "=" * 100)
print("RESULTS SUMMARY")
print("=" * 100)

# Print Header
print(f"{'PROFILE':<18} | {'TYPE':<5} | {'TOP MODEL SELECTION':<40} | {'AVG COST':<10}")
print("-" * 100)

for profile in profiles:
    stats = results[profile]
    
    # Easy Stats
    top_easy, count_easy = stats["easy"].most_common(1)[0]
    pct_easy = count_easy / len(easy_prompts) * 100
    avg_cost_easy = stats["easy_cost"] / len(easy_prompts)
    
    # Hard Stats
    top_hard, count_hard = stats["hard"].most_common(1)[0]
    pct_hard = count_hard / len(hard_prompts) * 100
    avg_cost_hard = stats["hard_cost"] / len(hard_prompts)
    
    print(f"{profile:<18} | {'Easy':<5} | {top_easy:<30} ({pct_easy:.0f}%) | ${avg_cost_easy:.4f}")
    print(f"{'':<18} | {'Hard':<5} | {top_hard:<30} ({pct_hard:.0f}%) | ${avg_cost_hard:.4f}")
    print("-" * 100)

print("\n" + "=" * 100)
print("DETAILED COST EFFICIENCY ANALYSIS (Hard/Easy Ratio)")
print("=" * 100)

for profile in profiles:
    stats = results[profile]
    avg_cost_easy = stats["easy_cost"] / len(easy_prompts)
    avg_cost_hard = stats["hard_cost"] / len(hard_prompts)
    
    ratio = avg_cost_hard / avg_cost_easy if avg_cost_easy > 0 else 0
    
    print(f"{profile.upper():<16}: {ratio:5.1f}x  (Easy: ${avg_cost_easy:.4f} -> Hard: ${avg_cost_hard:.4f})")
