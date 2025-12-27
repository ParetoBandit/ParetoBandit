#!/usr/bin/env python3
"""
Compute relative cluster assignments based on comparative advantage.

Instead of absolute "best cluster", identifies which cluster each model
performs best on RELATIVE to other models.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

def load_rewards(filepath: Path) -> List[dict]:
    """Load reward data from JSONL file."""
    rewards = []
    with open(filepath) as f:
        for line in f:
            rewards.append(json.loads(line))
    return rewards

def compute_cluster_performance(rewards: List[dict]) -> Dict[str, Dict[int, List[float]]]:
    """Compute performance per cluster for each model."""
    model_cluster_scores = defaultdict(lambda: defaultdict(list))
    skipped_count = 0
    
    for reward in rewards:
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

def build_performance_matrix(model_cluster_scores: Dict[str, Dict[int, List[float]]], 
                             num_clusters: int = 100) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build a matrix of model performance across clusters.
    
    Returns:
        performance_matrix: (n_models, n_clusters) array of success rates
        model_ids: List of model IDs
        valid_clusters: Boolean mask of clusters with data
    """
    model_ids = sorted(model_cluster_scores.keys())
    n_models = len(model_ids)
    
    # Initialize matrix
    performance_matrix = np.zeros((n_models, num_clusters))
    cluster_has_data = np.zeros(num_clusters, dtype=bool)
    
    for i, model_id in enumerate(model_ids):
        cluster_scores = model_cluster_scores[model_id]
        for cluster_id in range(num_clusters):
            if cluster_id in cluster_scores:
                scores = cluster_scores[cluster_id]
                performance_matrix[i, cluster_id] = np.mean(scores)
                cluster_has_data[cluster_id] = True
            else:
                # No data for this cluster - use NaN
                performance_matrix[i, cluster_id] = np.nan
    
    return performance_matrix, model_ids, cluster_has_data

def compute_relative_metrics(performance_matrix: np.ndarray, 
                             valid_clusters: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute relative performance metrics.
    
    Returns dict with:
        - z_scores: Z-score per cluster (performance relative to other models)
        - deltas: Delta from mean per cluster
        - ranks: Rank within cluster (1 = best)
        - cluster_means: Mean performance per cluster across all models
        - cluster_stds: Std dev per cluster
    """
    n_models, n_clusters = performance_matrix.shape
    
    # Compute cluster statistics (ignoring NaN values)
    cluster_means = np.nanmean(performance_matrix, axis=0)
    cluster_stds = np.nanstd(performance_matrix, axis=0)
    
    # Z-scores: (performance - mean) / std
    z_scores = np.zeros_like(performance_matrix)
    for j in range(n_clusters):
        if valid_clusters[j] and cluster_stds[j] > 0:
            z_scores[:, j] = (performance_matrix[:, j] - cluster_means[j]) / cluster_stds[j]
        else:
            z_scores[:, j] = 0
    
    # Replace NaN with very negative z-score
    z_scores = np.nan_to_num(z_scores, nan=-999)
    
    # Delta from mean
    deltas = performance_matrix - cluster_means[np.newaxis, :]
    deltas = np.nan_to_num(deltas, nan=-999)
    
    # Ranks (1 = best, higher = worse)
    ranks = np.zeros_like(performance_matrix)
    for j in range(n_clusters):
        if valid_clusters[j]:
            valid_perf = performance_matrix[:, j]
            # Higher performance = better rank (lower number)
            ranks[:, j] = n_models + 1 - np.argsort(np.argsort(-valid_perf))
        else:
            ranks[:, j] = n_models + 1
    
    return {
        'z_scores': z_scores,
        'deltas': deltas,
        'ranks': ranks,
        'cluster_means': cluster_means,
        'cluster_stds': cluster_stds,
        'cluster_difficulties': 1 - cluster_means  # Higher = harder
    }

def assign_best_clusters(performance_matrix: np.ndarray,
                         relative_metrics: Dict[str, np.ndarray],
                         model_ids: List[str],
                         valid_clusters: np.ndarray) -> Dict[str, dict]:
    """
    Assign best cluster to each model using relative metrics.
    
    Uses z-score as primary metric (comparative advantage).
    """
    n_models, n_clusters = performance_matrix.shape
    z_scores = relative_metrics['z_scores']
    
    model_assignments = {}
    
    for i, model_id in enumerate(model_ids):
        # Find cluster with highest z-score (only among valid clusters)
        valid_z_scores = z_scores[i, :].copy()
        valid_z_scores[~valid_clusters] = -999
        
        best_cluster_id = int(np.argmax(valid_z_scores))
        best_z_score = float(z_scores[i, best_cluster_id])
        best_performance = float(performance_matrix[i, best_cluster_id])
        cluster_mean = float(relative_metrics['cluster_means'][best_cluster_id])
        cluster_rank = int(relative_metrics['ranks'][i, best_cluster_id])
        
        # Also get top 3 clusters by z-score
        top_3_indices = np.argsort(valid_z_scores)[-3:][::-1]
        top_3_clusters = [
            {
                'cluster_id': int(idx),
                'z_score': float(z_scores[i, idx]),
                'success_rate': float(performance_matrix[i, idx]) if not np.isnan(performance_matrix[i, idx]) else 0.0,
                'rank': int(relative_metrics['ranks'][i, idx])
            }
            for idx in top_3_indices if valid_clusters[idx]
        ]
        
        model_assignments[model_id] = {
            'best_relative_cluster_id': best_cluster_id,
            'best_cluster_z_score': best_z_score,
            'best_cluster_success_rate': best_performance,
            'best_cluster_mean_success_rate': cluster_mean,
            'best_cluster_rank': cluster_rank,
            'best_cluster_advantage': best_performance - cluster_mean,
            'top_3_clusters': top_3_clusters,
            'cluster_success_rates': performance_matrix[i, :].tolist(),
            'cluster_z_scores': z_scores[i, :].tolist()
        }
    
    return model_assignments

def update_models_json(models_path: Path, 
                       model_assignments: Dict[str, dict],
                       relative_metrics: Dict[str, np.ndarray]) -> None:
    """Update models.json with relative cluster assignments."""
    with open(models_path) as f:
        models_data = json.load(f)
    
    updated_count = 0
    
    for model in models_data['models']:
        model_id = model.get('openrouter_id')
        
        if model_id and model_id in model_assignments:
            assignment = model_assignments[model_id]
            
            # Add new relative fields
            model['best_relative_cluster_id'] = assignment['best_relative_cluster_id']
            model['best_cluster_z_score'] = assignment['best_cluster_z_score']
            model['best_cluster_rank'] = assignment['best_cluster_rank']
            model['best_cluster_advantage'] = assignment['best_cluster_advantage']
            model['top_3_clusters'] = assignment['top_3_clusters']
            
            # Keep the existing absolute metrics too
            model['cluster_success_rates'] = assignment['cluster_success_rates']
            model['cluster_z_scores'] = assignment['cluster_z_scores']
            
            updated_count += 1
    
    # Backup and save
    backup_path = models_path.with_suffix('.json.backup2')
    models_path.rename(backup_path)
    print(f"Created backup: {backup_path}")
    
    with open(models_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    
    print(f"Updated {updated_count} models")

def main():
    base_dir = Path(__file__).parent
    
    print("=== Relative Cluster Assignment ===\n")
    
    # Load data
    print("Loading reward data...")
    train_rewards = load_rewards(base_dir / 'train_rewards.jsonl')
    test_rewards = load_rewards(base_dir / 'test_rewards.jsonl')
    all_rewards = train_rewards + test_rewards
    print(f"Total rewards: {len(all_rewards)}\n")
    
    # Compute cluster performance
    print("Computing cluster performance...")
    model_cluster_scores = compute_cluster_performance(all_rewards)
    print(f"Found {len(model_cluster_scores)} models\n")
    
    # Build performance matrix
    print("Building performance matrix...")
    performance_matrix, model_ids, valid_clusters = build_performance_matrix(
        model_cluster_scores, num_clusters=100
    )
    print(f"Matrix shape: {performance_matrix.shape}")
    print(f"Valid clusters: {valid_clusters.sum()}/100\n")
    
    # Compute relative metrics
    print("Computing relative metrics...")
    relative_metrics = compute_relative_metrics(performance_matrix, valid_clusters)
    
    # Show cluster difficulty distribution
    print("\nCluster Difficulty Distribution:")
    difficulties = relative_metrics['cluster_difficulties'][valid_clusters]
    print(f"  Easiest clusters (lowest difficulty): {np.argsort(difficulties)[:3]}")
    print(f"  Hardest clusters (highest difficulty): {np.argsort(difficulties)[-3:][::-1]}")
    print(f"  Mean difficulty: {np.mean(difficulties):.3f}")
    print(f"  Std difficulty: {np.std(difficulties):.3f}\n")
    
    # Assign best clusters
    print("Assigning best relative clusters...")
    model_assignments = assign_best_clusters(
        performance_matrix, relative_metrics, model_ids, valid_clusters
    )
    
    # Analyze distribution
    print("\n=== Assignment Distribution ===")
    best_clusters = [a['best_relative_cluster_id'] for a in model_assignments.values()]
    from collections import Counter
    cluster_dist = Counter(best_clusters)
    print(f"Unique best clusters: {len(cluster_dist)}")
    print(f"\nTop assigned clusters:")
    for cluster_id, count in cluster_dist.most_common(10):
        cluster_mean = relative_metrics['cluster_means'][cluster_id]
        cluster_diff = relative_metrics['cluster_difficulties'][cluster_id]
        print(f"  Cluster {cluster_id}: {count} models (mean: {cluster_mean:.3f}, difficulty: {cluster_diff:.3f})")
    
    # Show sample results
    print("\n=== Sample Model Assignments ===")
    for model_id, assignment in list(model_assignments.items())[:3]:
        print(f"\nModel: {model_id}")
        print(f"  Best cluster: {assignment['best_relative_cluster_id']} (z-score: {assignment['best_cluster_z_score']:.2f})")
        print(f"  Performance: {assignment['best_cluster_success_rate']:.3f} (mean: {assignment['best_cluster_mean_success_rate']:.3f})")
        print(f"  Advantage: +{assignment['best_cluster_advantage']:.3f}")
        print(f"  Rank in cluster: {assignment['best_cluster_rank']}/50")
    
    # Update models.json
    print("\n=== Updating models.json ===")
    models_path = base_dir.parent / 'models.json'
    update_models_json(models_path, model_assignments, relative_metrics)
    
    print("\n✓ Complete!")

if __name__ == '__main__':
    main()
