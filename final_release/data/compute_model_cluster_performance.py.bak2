#!/usr/bin/env python3
"""
Compute per-cluster performance metrics for each LLM model.

This script:
1. Loads training and test reward data
2. Calculates success rate for each model across all 500 prompt clusters
3. Identifies the best-performing cluster for each model
4. Updates models.json with cluster performance data
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import numpy as np

def load_rewards(filepath: Path) -> List[dict]:
    """Load reward data from JSONL file."""
    rewards = []
    with open(filepath) as f:
        for line in f:
            rewards.append(json.loads(line))
    return rewards

def compute_cluster_performance(rewards: List[dict]) -> Dict[str, Dict[int, List[float]]]:
    """
    Compute performance per cluster for each model.
    
    Returns:
        Dict mapping model_id -> cluster_id -> list of raw scores
    """
    model_cluster_scores = defaultdict(lambda: defaultdict(list))
    skipped_count = 0
    
    for reward in rewards:
        # Skip failed evaluations (ok=False, no raw_score)
        if not reward.get('ok', True) or 'raw_score' not in reward:
            skipped_count += 1
            continue
            
        model_id = reward['model_id']
        cluster_id = reward['cluster_id']
        raw_score = reward['raw_score']
        
        model_cluster_scores[model_id][cluster_id].append(raw_score)
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} failed evaluations")
    
    return model_cluster_scores

def compute_cluster_success_rates(model_cluster_scores: Dict[str, Dict[int, List[float]]], 
                                   num_clusters: int = 500) -> Dict[str, dict]:
    """
    Compute success rate vector and best cluster for each model.
    
    Returns:
        Dict mapping model_id -> {
            'cluster_success_rates': List[float],  # Success rate per cluster (0-500)
            'best_cluster_id': int,                # Cluster with highest success rate
            'best_cluster_success_rate': float     # Success rate of best cluster
        }
    """
    model_metrics = {}
    
    for model_id, cluster_scores in model_cluster_scores.items():
        # Initialize success rates for all clusters (0 if no data)
        success_rates = []
        
        for cluster_id in range(num_clusters):
            if cluster_id in cluster_scores:
                scores = cluster_scores[cluster_id]
                # Success rate = mean of raw scores
                success_rate = np.mean(scores)
            else:
                # No data for this cluster
                success_rate = 0.0
            success_rates.append(float(success_rate))
        
        # Find best performing cluster (ignore clusters with no data)
        cluster_rates = [(i, rate) for i, rate in enumerate(success_rates) if rate > 0]
        if cluster_rates:
            best_cluster_id, best_rate = max(cluster_rates, key=lambda x: x[1])
        else:
            best_cluster_id = 0
            best_rate = 0.0
        
        model_metrics[model_id] = {
            'cluster_success_rates': success_rates,
            'best_cluster_id': int(best_cluster_id),
            'best_cluster_success_rate': float(best_rate),
            'clusters_evaluated': len(cluster_rates),  # How many clusters have data
            'overall_success_rate': float(np.mean([r for r in success_rates if r > 0])) if cluster_rates else 0.0
        }
    
    return model_metrics

def update_models_json(models_path: Path, model_metrics: Dict[str, dict]) -> None:
    """Update models.json with cluster performance data."""
    # Load models.json
    with open(models_path) as f:
        models_data = json.load(f)
    
    # Map openrouter_id to model_id used in rewards
    # The rewards use openrouter_id directly as model_id
    updated_count = 0
    
    for model in models_data['models']:
        model_id = model.get('openrouter_id')
        
        if model_id and model_id in model_metrics:
            metrics = model_metrics[model_id]
            
            # Add cluster performance fields
            model['cluster_success_rates'] = metrics['cluster_success_rates']
            model['best_cluster_id'] = metrics['best_cluster_id']
            model['best_cluster_success_rate'] = metrics['best_cluster_success_rate']
            model['clusters_evaluated'] = metrics['clusters_evaluated']
            model['overall_success_rate'] = metrics['overall_success_rate']
            
            updated_count += 1
    
    # Save updated models.json
    backup_path = models_path.with_suffix('.json.backup')
    models_path.rename(backup_path)
    print(f"Created backup: {backup_path}")
    
    with open(models_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    
    print(f"Updated {updated_count} models in {models_path}")

def main():
    """Main execution."""
    base_dir = Path(__file__).parent
    
    # Load reward data
    print("Loading reward data...")
    train_rewards = load_rewards(base_dir / 'train_rewards.jsonl')
    test_rewards = load_rewards(base_dir / 'test_rewards.jsonl')
    
    print(f"Loaded {len(train_rewards)} training rewards")
    print(f"Loaded {len(test_rewards)} test rewards")
    
    # Combine training and test data
    all_rewards = train_rewards + test_rewards
    print(f"Total rewards: {len(all_rewards)}")
    
    # Compute cluster performance
    print("\nComputing cluster performance per model...")
    model_cluster_scores = compute_cluster_performance(all_rewards)
    print(f"Found {len(model_cluster_scores)} unique models")
    
    # Compute success rates
    print("\nComputing success rate vectors...")
    model_metrics = compute_cluster_success_rates(model_cluster_scores, num_clusters=500)
    
    # Print sample results
    print("\n=== Sample Results ===")
    for i, (model_id, metrics) in enumerate(list(model_metrics.items())[:3]):
        print(f"\nModel: {model_id}")
        print(f"  Best cluster: {metrics['best_cluster_id']} (success rate: {metrics['best_cluster_success_rate']:.3f})")
        print(f"  Clusters evaluated: {metrics['clusters_evaluated']}/500")
        print(f"  Overall success rate: {metrics['overall_success_rate']:.3f}")
    
    # Update models.json
    print("\n=== Updating models.json ===")
    models_path = base_dir.parent / 'models.json'
    update_models_json(models_path, model_metrics)
    
    print("\n✓ Complete!")

if __name__ == '__main__':
    main()
