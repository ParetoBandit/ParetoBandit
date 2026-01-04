import json
import numpy as np
from collections import defaultdict
from pathlib import Path

def update_success_rates():
    base = Path(__file__).parent
    rewards_file = base / "data/test_rewards_pareto_dedup.jsonl"
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
                score = data.get("raw_score")
                
                # Handle NaN or missing
                if score is None: continue
                import math
                if isinstance(score, float) and math.isnan(score): continue
                
                stats[mid][cid].append(float(score))
                
            except Exception as e:
                continue

    print(f"Aggregated stats for {len(stats)} models.")
    
    # First pass: Calculate raw rates for all models/clusters
    raw_rates = {} # model_id -> {cluster_id: raw_rate}
    for mid, clusters in stats.items():
        raw_rates[mid] = {}
        for cid, scores in clusters.items():
            if not scores: continue
            rate = sum(scores) / len(scores)
            raw_rates[mid][cid] = rate
    
    # Second pass: Compute per-cluster statistics
    # For each cluster, gather all model rates to compute mean/std
    cluster_stats = {}
    all_cluster_ids = set()
    for rates_dict in raw_rates.values():
        all_cluster_ids.update(rates_dict.keys())
    
    print(f"Computing z-scores across {len(all_cluster_ids)} clusters...")
    
    for cid in all_cluster_ids:
        # Gather all model rates for this cluster
        rates_for_cluster = []
        for mid in raw_rates:
            if cid in raw_rates[mid]:
                rates_for_cluster.append(raw_rates[mid][cid])
        
        if len(rates_for_cluster) < 2:
            # Not enough data for z-score
            cluster_stats[cid] = {'mean': 0.5, 'std': 0.0}
        else:
            cluster_stats[cid] = {
                'mean': float(np.mean(rates_for_cluster)),
                'std': float(np.std(rates_for_cluster))
            }
    
    # Third pass: Compute z-scores for each model/cluster
    new_rates = {} # model_id -> {cluster_id: {raw, z_score}}
    for mid in raw_rates:
        new_rates[mid] = {}
        for cid, raw_rate in raw_rates[mid].items():
            mean = cluster_stats[cid]['mean']
            std = cluster_stats[cid]['std']
            
            # Compute z-score with protection against zero std
            if std < 0.01:
                z_score = 0.0  # No variation in cluster, all models equal
            else:
                z_score = (raw_rate - mean) / std
            
            new_rates[mid][cid] = {
                'raw': float(raw_rate),
                'z_score': float(z_score)
            }
    
    # Log some statistics
    print("\nCluster Statistics (sample of first 5):")
    for i, cid in enumerate(sorted(all_cluster_ids)[:5]):
        stats_cid = cluster_stats[cid]
        print(f"  Cluster {cid}: mean={stats_cid['mean']:.3f}, std={stats_cid['std']:.3f}")
    
    # Update models.json
    print(f"\nUpdating {models_file}...")
    with open(models_file, 'r') as f:
        registry = json.load(f)
        
    updated_count = 0
    for model in registry["models"]:
        mid = model["openrouter_id"]
        if mid in new_rates:
            model["cluster_success_rates"] = new_rates[mid]
            updated_count += 1
    
    # Log z-score distribution for verification
    all_z_scores = []
    for mid in new_rates:
        for cid in new_rates[mid]:
            all_z_scores.append(new_rates[mid][cid]['z_score'])
    
    if all_z_scores:
        print(f"\nZ-score distribution across all model/cluster pairs:")
        print(f"  Mean: {np.mean(all_z_scores):.3f}")
        print(f"  Std: {np.std(all_z_scores):.3f}")
        print(f"  Min: {np.min(all_z_scores):.3f}")
        print(f"  Max: {np.max(all_z_scores):.3f}")
        
    with open(models_file, 'w') as f:
        json.dump(registry, f, indent=4)
        
    print(f"\n✓ Updated success rates for {updated_count} models.")
    print(f"✓ Added z-scores for fair per-cluster normalization")

if __name__ == "__main__":
    update_success_rates()
