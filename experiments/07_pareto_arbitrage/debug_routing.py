"""
Debug script to analyze BanditGPT routing decisions.
Shows which models are selected for each hard prompt.
"""

import json
from collections import defaultdict, Counter
from pathlib import Path

# Load results
results_path = Path(__file__).parent / "results" / "arbitrage_results.json"
with open(results_path) as f:
    results = json.load(f)

# Load splits to get hard prompts
data_dir = Path(__file__).parent.parent.parent / "data"
with open(data_dir / "processed" / "splits.json") as f:
    splits = json.load(f)

# Load oracle rewards
with open(data_dir / "processed" / "oracle_rewards.json") as f:
    all_rewards = json.load(f)

# Load registry
with open(data_dir / "processed" / "registry.json") as f:
    registry_data = json.load(f)
registry = {m["openrouter_id"]: m for m in registry_data}

# Filter for hard prompts
test_prompts = splits["test"]
hard_prompts = []

for prompt in test_prompts:
    oracle_scores = [all_rewards.get(prompt, {}).get(mid) 
                   for mid in registry.keys() 
                   if all_rewards.get(prompt, {}).get(mid) is not None]
    if oracle_scores:
        failures = sum(1 for r in oracle_scores if r == 0.0)
        is_solvable = max(oracle_scores) == 1.0
        if failures >= 3 and is_solvable:
            hard_prompts.append(prompt)

print(f"\n{'='*70}")
print(f"BANDIT ROUTING DECISIONS ANALYSIS")
print(f"{'='*70}")
print(f"\n📊 Dataset: {len(hard_prompts)} hard prompts (failures >= 3, max == 1.0)")
print(f"   Test set: {len(test_prompts)} total prompts")
print(f"   Hard ratio: {len(hard_prompts)/len(test_prompts)*100:.1f}%")

# For debugging, let's simulate what the CURRENT router would choose
# We can check if it's loading warmup priors or not
print(f"\n{'='*70}")
print(f"CHECKING CURRENT EXPERIMENT RESULTS")
print(f"{'='*70}")

# Show BanditGPT results
bandit_results = results.get("bandit", [])
for profile_result in bandit_results:
    name = profile_result["name"]
    cost = profile_result["cost"]
    quality = profile_result["quality"] * 100
    print(f"\n{name}:")
    print(f"  Cost: ${cost:.2f}/1M")
    print(f"  Quality: {quality:.1f}%")
    
    # Check if we have routing logs (we'd need to add this to run_arbitrage.py)
    if "selections" in profile_result:
        selections = Counter(profile_result["selections"])
        print(f"  Model selections:")
        for model_id, count in selections.most_common():
            pct = count / len(hard_prompts) * 100
            model_name = registry.get(model_id, {}).get("display_name", model_id)
            print(f"    {model_name}: {count}/{len(hard_prompts)} ({pct:.1f}%)")

# Show individual model baselines for comparison
print(f"\n{'='*70}")
print(f"INDIVIDUAL MODEL BASELINES (for comparison)")
print(f"{'='*70}")

individual_results = results.get("individual", [])
for model_result in sorted(individual_results, key=lambda x: x["cost"]):
    name = model_result["name"]
    cost = model_result["cost"]
    quality = model_result["quality"] * 100
    print(f"\n{name}:")
    print(f"  Cost: ${cost:.2f}/1M")
    print(f"  Quality: {quality:.1f}%")

print(f"\n{'='*70}")
print(f"KEY INSIGHT")
print(f"{'='*70}")
print(f"\ngpt-oss-120B: 68.8% at $0.11/1M")
print(f"BanditGPT should BEAT this by:")
print(f"  1. Using gpt-oss for 33/49 easy-ish hard prompts (67%)")
print(f"  2. Escalating to flagships for 16/49 trap prompts (33%)")
print(f"  3. Target: ~72-75% at ~$0.50-1.00/1M")
