#!/usr/bin/env python3
"""
Convert cluster success rates to z-scores for cluster boost.

Z-score indicates how well a model performs in a cluster relative to:
- Its overall average performance
-The standard deviation of its performance across clusters

Positive z-score = model excels in this cluster
Negative z-score = model weak in this cluster
"""

import json
import numpy as np
from pathlib import Path

def compute_z_scores(models_path: Path):
    """Add cluster_z_scores to models based on cluster_success_rates."""
    with open(models_path) as f:
        models_data = json.load(f)
    
    updated_count = 0
    
    for model in models_data['models']:
        if 'cluster_success_rates' not in model:
            continue
            
        rates = np.array(model['cluster_success_rates'])
        
        # Only use clusters with data (non-zero rates)
        non_zero_rates = rates[rates > 0]
        
        if len(non_zero_rates) < 2:
            # Not enough data for z-scores
            model['cluster_z_scores'] = [0.0] * len(rates)
            continue
        
        # Compute z-scores: (rate - mean) / std
        mean_rate = np.mean(non_zero_rates)
        std_rate = np.std(non_zero_rates)
        
        if std_rate < 0.01:
            # Model performance is too uniform
            z_scores = [0.0] * len(rates)
        else:
            z_scores = []
            for rate in rates:
                if rate > 0:
                    # Compute z-score for clusters with data
                    z = (rate - mean_rate) / std_rate
                else:
                    # Neutral for clusters without data
                    z = 0.0
                z_scores.append(float(z))
        
        model['cluster_z_scores'] = z_scores
        updated_count += 1
        
        # Print sample
        if updated_count <= 3:
            print(f"\nModel: {model['openrouter_id']}")
            print(f"  Mean rate: {mean_rate:.3f}")
            print(f"  Std rate: {std_rate:.3f}")
            max_z_idx = np.argmax(z_scores)
            min_z_idx = np.argmin(z_scores)
            print(f"  Best cluster: {max_z_idx} (z={z_scores[max_z_idx]:.2f}, rate={rates[max_z_idx]:.3f})")
            print(f"  Worst cluster: {min_z_idx} (z={z_scores[min_z_idx]:.2f}, rate={rates[min_z_idx]:.3f})")
    
    # Backup and save
    backup_path = models_path.with_suffix('.json.backup_pre_z')
    with open(backup_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    print(f"\n✓ Backup: {backup_path}")
    
    with open(models_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    
    print(f"✓ Added z-scores to {updated_count} models")

if __name__ == '__main__':
    base_dir = Path(__file__).parent
    models_path = base_dir.parent / 'models.json'
    compute_z_scores(models_path)
    print("\n✓ Complete!")
