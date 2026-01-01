import json
from collections import defaultdict
from pathlib import Path

def update_success_rates():
    base = Path(__file__).parent
    rewards_file = base / "data/test_rewards_pareto.jsonl"
    models_file = base / "models.json"
    
    print(f"Loading rewards from {rewards_file}...")
    
    # Storage: model_id -> cluster_id -> [rewards]
    stats = defaultdict(lambda: defaultdict(list))
    
    with open(rewards_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                mid = data["model_id"]
                cid = str(data["cluster_id"]) # JSON keys are strings
                
                # Check for raw_score (binary 0.0/1.0)
                # Or reward_logit (need to reverse transform?)
                # User set 'raw_score' in rejudge_cot.py as 0.0 or 1.0
                score = data.get("raw_score")
                
                # Handle NaN or missing
                if score is None: continue
                import math
                if isinstance(score, float) and math.isnan(score): continue
                
                stats[mid][cid].append(float(score))
                
            except Exception as e:
                continue

    print(f"Aggregated stats for {len(stats)} models.")
    
    # Calculate Rates
    new_rates = {} # model_id -> {cluster_id: rate}
    for mid, clusters in stats.items():
        new_rates[mid] = {}
        for cid, scores in clusters.items():
            if not scores: continue
            rate = sum(scores) / len(scores)
            new_rates[mid][cid] = rate
            
    # Update models.json
    print(f"Updating {models_file}...")
    with open(models_file, 'r') as f:
        registry = json.load(f)
        
    updated_count = 0
    for model in registry["models"]:
        mid = model["openrouter_id"]
        if mid in new_rates:
            model["cluster_success_rates"] = new_rates[mid]
            updated_count += 1
            
    with open(models_file, 'w') as f:
        json.dump(registry, f, indent=4)
        
    print(f"Updated success rates for {updated_count} models.")

if __name__ == "__main__":
    update_success_rates()
