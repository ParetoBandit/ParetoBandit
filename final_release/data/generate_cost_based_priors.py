#!/usr/bin/env python3
"""
Cost + Latency Based Prior Generator for New Models

Generates initial cluster performance estimates based ONLY on:
1. Model cost (always available from API pricing)
2. Time-to-first-token latency (easy to measure)

This allows the bandit router to start with reasonable priors for brand new models,
then learn the actual cluster preferences through exploration.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def analyze_cost_latency_performance(models_path: Path) -> Dict:
    """
    Analyze how cost + latency correlate with cluster performance.
    
    Returns:
        Dict with 2D grid statistics (cost × latency bins)
    """
    with open(models_path) as f:
        data = json.load(f)
        models = data['models']
    
    # Filter models with cluster data, cost, and latency
    models_with_data = [
        m for m in models 
        if 'cluster_success_rates' in m 
        and 'price_1m_blended' in m
        and m.get('price_1m_blended') is not None
        and 'time_to_first_token_seconds' in m
        and m.get('time_to_first_token_seconds') is not None
    ]
    
    print(f"Analyzing {len(models_with_data)} models...")
    
    # Define 2D grid: Cost × Latency bins
    grid = defaultdict(list)  # (cost_tier, latency_tier) -> [cluster_rates]
    
    for model in models_with_data:
        cost = model['price_1m_blended']
        latency = model['time_to_first_token_seconds']
        cluster_rates = np.array(model['cluster_success_rates'])
        
        # Determine cost tier
        if cost < 0.5:
            cost_tier = 'budget'
        elif cost < 2.0:
            cost_tier = 'economy'
        elif cost < 5.0:
            cost_tier = 'standard'
        else:
            cost_tier = 'premium'
        
        # Determine latency tier
        if latency < 0.5:
            latency_tier = 'fast'
        elif latency < 2.0:
            latency_tier = 'medium'
        else:
            latency_tier = 'slow'
        
        grid[(cost_tier, latency_tier)].append(cluster_rates)
    
    # Compute statistics per grid cell
    grid_stats = {}
    for (cost_tier, latency_tier), cluster_rate_list in grid.items():
        if cluster_rate_list:
            matrix = np.array(cluster_rate_list)
            
            grid_stats[f"{cost_tier}_{latency_tier}"] = {
                'cost_tier': cost_tier,
                'latency_tier': latency_tier,
                'n_models': len(cluster_rate_list),
                'mean_cluster_rates': np.mean(matrix, axis=0).tolist(),
                'std_cluster_rates': np.std(matrix, axis=0).tolist(),
                'overall_mean': float(np.mean(matrix)),
                'overall_std': float(np.std(matrix))
            }
    
    return grid_stats

def generate_cost_latency_prior(cost: float,
                                latency: Optional[float] = None,
                                grid_stats: Dict = None,
                                exploration_bonus: float = 0.1) -> Dict:
    """
    Generate initial cluster performance estimates based on cost + latency.
    
    Args:
        cost: Model price per 1M blended tokens
        latency: Time to first token in seconds (optional)
        grid_stats: Statistics from analyze_cost_latency_performance()
        exploration_bonus: Added uncertainty to encourage exploration (default 0.1)
        
    Returns:
        Dict with:
            - cluster_priors: Initial success rate estimates per cluster
            - cost_tier: Which cost tier
            - latency_tier: Which latency tier (if available)
            - confidence: How confident we are in this prior (0-1)
    """
    # Determine cost tier
    if cost < 0.5:
        cost_tier = 'budget'
    elif cost < 2.0:
        cost_tier = 'economy'
    elif cost < 5.0:
        cost_tier = 'standard'
    else:
        cost_tier = 'premium'
    
    # Determine latency tier (if available)
    if latency is not None:
        if latency < 0.5:
            latency_tier = 'fast'
        elif latency < 2.0:
            latency_tier = 'medium'
        else:
            latency_tier = 'slow'
        
        # Try exact match first
        grid_key = f"{cost_tier}_{latency_tier}"
        stats = grid_stats.get(grid_key)
        
        # Fallback: try other latency tiers in this cost tier
        if stats is None:
            for lt in ['fast', 'medium', 'slow']:
                fallback_key = f"{cost_tier}_{lt}"
                if fallback_key in grid_stats:
                    stats = grid_stats[fallback_key]
                    latency_tier = f"{lt} (fallback)"
                    break
    else:
        # No latency - use any model in this cost tier
        latency_tier = 'unknown'
        stats = None
        for key, value in grid_stats.items():
            if value['cost_tier'] == cost_tier:
                stats = value
                break
    
    if stats is None:
        # No data - use conservative baseline
        print(f"Warning: No data for {cost_tier}/{latency_tier}, using conservative estimate")
        cluster_priors = [0.7] * 100
        confidence = 0.1
        n_models = 0
    else:
        # Use grid cell statistics
        cluster_priors = stats['mean_cluster_rates']
        
        # Add exploration bonus
        cluster_priors = [
            max(0.0, min(1.0, p + np.random.normal(0, exploration_bonus)))
            for p in cluster_priors
        ]
        
        # Confidence based on sample size
        confidence = min(1.0, stats['n_models'] / 10.0)
        n_models = stats['n_models']
    
    return {
        'cluster_priors': cluster_priors,
        'cost_tier': cost_tier,
        'latency_tier': latency_tier,
        'confidence': confidence,
        'n_models': n_models
    }

def save_cost_latency_priors(models_path: Path, output_path: Path):
    """
    Analyze cost+latency-performance relationship and save priors for future use.
    """
    print("=== Cost + Latency Based Prior Generator ===\n")
    
    # Analyze existing models
    grid_stats = analyze_cost_latency_performance(models_path)
    
    # Show grid summary
    print("\n=== Cost × Latency Grid Statistics ===")
    for grid_key, stats in sorted(grid_stats.items()):
        print(f"\n{grid_key.upper()}:")
        print(f"  Models: {stats['n_models']}")
        print(f"  Overall mean: {stats['overall_mean']:.3f}")
        print(f"  Overall std: {stats['overall_std']:.3f}")
    
    # Save for future use
    with open(output_path, 'w') as f:
        json.dump(grid_stats, f, indent=2)
    
    print(f"\n✓ Saved cost+latency priors to: {output_path}")
    
    return grid_stats

def main():
    """Demo: Generate cost+latency priors for example models."""
    base_dir = Path(__file__).parent.parent
    models_path = base_dir / 'models.json'
    priors_path = base_dir / 'data' / 'cost_latency_priors.json'
    
    # Analyze and save
    grid_stats = save_cost_latency_priors(models_path, priors_path)
    
    # Demo: Generate priors for various scenarios
    print("\n=== Example Priors for New Models ===")
    
    test_cases = [
        (0.05, 0.3, "Ultra-budget, fast model"),
        (0.75, 0.5, "Economy, fast model"),
        (2.5, 1.0, "Standard, medium latency"),
        (8.0, 2.5, "Premium, slow (reasoning) model"),
        (1.5, None, "Economy model (latency unknown)")
    ]
    
    for cost, latency, description in test_cases:
        prior = generate_cost_latency_prior(cost, latency, grid_stats)
        
        print(f"\n{description}:")
        print(f"  Cost: ${cost}/1M, Latency: {latency}s" if latency else f"  Cost: ${cost}/1M (no latency)")
        print(f"  Tier: {prior['cost_tier']}/{prior['latency_tier']}")
        print(f"  Confidence: {prior['confidence']:.2f}")
        print(f"  Cluster priors: mean={np.mean(prior['cluster_priors']):.3f}, "
              f"std={np.std(prior['cluster_priors']):.3f}")
        print(f"  Based on {prior['n_models']} similar models")
    
    print("\n" + "="*60)
    print("Integration with Bandit Router:")
    print("="*60)
    print("""
# For a new model without benchmarks:
new_model_cost = 1.50
new_model_latency = 0.65  # Measure with single test request

# Load grid statistics
with open('data/cost_latency_priors.json') as f:
    grid_stats = json.load(f)

# Generate initial priors
prior = generate_cost_latency_prior(
    cost=new_model_cost,
    latency=new_model_latency,
    grid_stats=grid_stats
)

# Initialize bandit with these priors
bandit = BanditRouter(
    model_id='new-model',
    priors=prior['cluster_priors']  # 100-element list
)

# Bandit starts with cost+latency estimates
# and LEARNS actual cluster preferences through usage
""")
    
    print("\n✓ Ready for production use!")

if __name__ == '__main__':
    main()
