#!/usr/bin/env python3
"""
Quick diagnostic: Analyze ARBITRAGE profile model selections.
"""
import sys
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from banditgpt.bandit import BanditRouter

# Load data
data_dir = Path(__file__).parent.parent.parent / "data"
test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
models_path = Path(__file__).parent.parent.parent / "models.json"

# Load registry
with open(models_path) as f:
    data = json.load(f)
registry = {m["openrouter_id"]: m for m in data["models"]}

# Load test data
test_data = defaultdict(lambda: {"cluster_id": None, "rewards": {}, "zscores": {}})
zscore_lookup = {}
for model_id, model in registry.items():
    if "cluster_success_rates" in model:
        for cluster_id_str, cluster_data in model["cluster_success_rates"].items():
            if isinstance(cluster_data, dict) and "z_score" in cluster_data:
                zscore_lookup[(model_id, int(cluster_id_str))] = cluster_data["z_score"]

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

print("=" * 80)
print("ARBITRAGE PROFILE DIAGNOSTIC")
print("=" * 80)

# Test ARBITRAGE profile
from sentence_transformers import SentenceTransformer
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER
encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

router = BanditRouter.create(
    registry,
    exploration="static",  # Pure exploitation
    priors="csr",
    prior_n_effective=50.0,
    prior_structure_n_effective=50.0,
    context_encoder=encoder
)

# Test on 20 prompts with ARBITRAGE profile
test_prompts = list(test_data.keys())[:20]
selections = defaultdict(int)
details = []

for prompt in test_prompts:
    data = test_data[prompt]
    
    # Route with ARBITRAGE
    selected, log = router.route(prompt, profile="arbitrage", input_tokens=100)
    
    if selected in data["zscores"]:
        model = registry[selected]
        cost = (100 * model["price_1m_input"] + 200 * model["price_1m_output"]) / 1_000_000
        
        details.append({
            "model": model.get("display_name", selected)[:30],
            "cost": cost * 1000,
            "zscore": data["zscores"][selected],
            "latency": log.latency_s
        })
        selections[selected] += 1

print(f"\nTested on {len(test_prompts)} prompts")
print(f"Unique models selected: {len(selections)}")

print("\n" + "=" * 80)
print("MODEL SELECTIONS (by frequency)")
print("=" * 80)
sorted_sel = sorted(selections.items(), key=lambda x: -x[1])
for model_id, count in sorted_sel[:10]:
    model = registry[model_id]
    name = model.get("display_name", model_id)[:35]
    cost = (100 * model["price_1m_input"] + 200 * model["price_1m_output"]) / 1_000_000
    avg_z = np.mean([d["zscore"] for d in details if d["model"].startswith(name[:20])])
    print(f"  {name:\u003c35} {count:2}x  ${cost*1000:.4f}/1k  Z={avg_z:+.3f}σ")

print("\n" + "=" * 80)
print("SAMPLE ROUTING DECISIONS")
print("=" * 80)
for i, d in enumerate(details[:10]):
    print(f"{i+1:2d}. {d['model']:\u003c30} ${d['cost']:.4f}/1k  Z={d['zscore']:+.3f}σ  {d['latency']:.2f}s")

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
avg_cost = np.mean([d["cost"] for d in details])
avg_zscore = np.mean([d["zscore"] for d in details])
print(f"Average cost: ${avg_cost:.4f} per 1k tokens")
print(f"Average Z-score: {avg_zscore:+.3f}σ")
print(f"Cost std: ${np.std([d['cost'] for d in details]):.4f}")
print(f"Z-score std: {np.std([d['zscore'] for d in details]):.3f}σ")
