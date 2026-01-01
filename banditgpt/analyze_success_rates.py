import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_distribution():
    base = Path(__file__).parent
    models_file = base / "models.json"
    
    with open(models_file) as f:
        data = json.load(f)
        
    all_rates = []
    model_averages = {}
    cluster_rates = defaultdict(list)
    
    for m in data["models"]:
        mid = m["openrouter_id"]
        rates = m.get("cluster_success_rates", {})
        if not rates: continue
        
        vals = list(rates.values())
        all_rates.extend(vals)
        model_averages[mid] = np.mean(vals)
        
        for cid, val in rates.items():
            cluster_rates[cid].append(val)
            
    if not all_rates:
        print("No success rates found.")
        return

    print("=== SUCCESS RATE DISTRIBUTION ANALYSIS ===")
    print(f"Total Datapoints (Model x Cluster): {len(all_rates)}")
    print(f"Global Mean Success Rate: {np.mean(all_rates):.4f}")
    print(f"Global Median: {np.median(all_rates):.4f}")
    print(f"Global Std Dev: {np.std(all_rates):.4f}")
    
    print("\n--- Distribution Buckets ---")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.01]
    hist, _ = np.histogram(all_rates, bins=bins)
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-0.95", "0.95-1.0"]
    for label, count in zip(labels, hist):
        pct = count / len(all_rates) * 100
        print(f"{label}: {count} ({pct:.1f}%)")
        
    print("\n--- Top 5 Models (by Mean Cluster Success) ---")
    sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
    for mid, avg in sorted_models[:5]:
        print(f"{mid}: {avg:.4f}")
        
    print("\n--- Bottom 5 Models ---")
    for mid, avg in sorted_models[-5:]:
        print(f"{mid}: {avg:.4f}")
        
    # Analyze Cluster Difficulty
    cluster_avgs = {cid: np.mean(vals) for cid, vals in cluster_rates.items()}
    sorted_clusters = sorted(cluster_avgs.items(), key=lambda x: x[1])
    
    print("\n--- Hardest 5 Clusters (Lowest Avg Success) ---")
    for cid, avg in sorted_clusters[:5]:
        print(f"Cluster {cid}: {avg:.4f}")
        
    print("\n--- Easiest 5 Clusters (Highest Avg Success) ---")
    for cid, avg in sorted_clusters[-5:]:
        print(f"Cluster {cid}: {avg:.4f}")

if __name__ == "__main__":
    analyze_distribution()
