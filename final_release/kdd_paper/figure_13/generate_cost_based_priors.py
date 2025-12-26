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

def analyze_cost_latency_context_performance(models_path: Path) -> Dict:
    """
    Analyze how cost + latency + context correlate with cluster performance.
    
    Returns:
        Dict with 3D grid statistics (cost × latency × context bins)
    """
    with open(models_path) as f:
        data = json.load(f)
        models = data['models']
    
    # Filter models with cluster data, cost, latency, and context
    models_with_data = [
        m for m in models 
        if 'cluster_success_rates' in m 
        and 'price_1m_blended' in m
        and m.get('price_1m_blended') is not None
        and 'time_to_first_token_seconds' in m
        and m.get('time_to_first_token_seconds') is not None
        and 'context_length' in m
        and m.get('context_length') is not None
    ]
    
    print(f"Analyzing {len(models_with_data)} models...")
    
    # Define 3D grid: Cost × Latency × Context bins
    grid = defaultdict(list)  # (cost_tier, latency_tier, context_tier) -> [cluster_rates]
    
    for model in models_with_data:
        cost = model['price_1m_blended']
        latency = model['time_to_first_token_seconds']
        context = model['context_length']
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
        
        # Determine context tier
        if context <= 32768:
            context_tier = 'small'
        elif context <= 131072:
            context_tier = 'medium'
        elif context <= 400000:
            context_tier = 'large'
        else:
            context_tier = 'xlarge'
        
        grid[(cost_tier, latency_tier, context_tier)].append(cluster_rates)
    
    # Compute statistics per grid cell
    grid_stats = {}
    for (cost_tier, latency_tier, context_tier), cluster_rate_list in grid.items():
        if cluster_rate_list:
            matrix = np.array(cluster_rate_list)
            
            grid_stats[f"{cost_tier}_{latency_tier}_{context_tier}"] = {
                'cost_tier': cost_tier,
                'latency_tier': latency_tier,
                'context_tier': context_tier,
                'n_models': len(cluster_rate_list),
                'mean_cluster_rates': np.mean(matrix, axis=0).tolist(),
                'std_cluster_rates': np.std(matrix, axis=0).tolist(),
                'overall_mean': float(np.mean(matrix)),
                'overall_std': float(np.std(matrix))
            }
    
    return grid_stats

def generate_cost_latency_context_prior(cost: float,
                                        latency: Optional[float] = None,
                                        context: Optional[int] = None,
                                        grid_stats: Dict = None,
                                        exploration_bonus: float = 0.1) -> Dict:
    """
    Generate initial cluster performance estimates based on cost + latency + context.
    
    Args:
        cost: Model price per 1M blended tokens
        latency: Time to first token in seconds (optional)
        context: Context window length in tokens (optional)
        grid_stats: Statistics from analyze_cost_latency_context_performance()
        exploration_bonus: Added uncertainty to encourage exploration (default 0.1)
        
    Returns:
        Dict with:
            - cluster_priors: Initial success rate estimates per cluster
            - cost_tier: Which cost tier
            - latency_tier: Which latency tier (if available)
            - context_tier: Which context tier (if available)
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
    else:
        latency_tier = 'unknown'
    
    # Determine context tier (if available)
    if context is not None:
        if context <= 32768:
            context_tier = 'small'
        elif context <= 131072:
            context_tier = 'medium'
        elif context <= 400000:
            context_tier = 'large'
        else:
            context_tier = 'xlarge'
    else:
        context_tier = 'unknown'
    
    # Try 5-level fallback cascade
    stats = None
    fallback_level = 0
    
    # Level 1: Exact match (cost, latency, context)
    if latency_tier != 'unknown' and context_tier != 'unknown':
        grid_key = f"{cost_tier}_{latency_tier}_{context_tier}"
        stats = grid_stats.get(grid_key)
        if stats:
            fallback_level = 1
    
    # Level 2: Same cost/latency, different context
    if stats is None and latency_tier != 'unknown':
        for ct in ['small', 'medium', 'large', 'xlarge']:
            fallback_key = f"{cost_tier}_{latency_tier}_{ct}"
            if fallback_key in grid_stats:
                stats = grid_stats[fallback_key]
                context_tier = f"{ct} (fallback)"
                fallback_level = 2
                break
    
    # Level 3: Same cost/context, different latency
    if stats is None and context_tier not in ['unknown', 'fallback']:
        for lt in ['fast', 'medium', 'slow']:
            fallback_key = f"{cost_tier}_{lt}_{context_tier}"
            if fallback_key in grid_stats:
                stats = grid_stats[fallback_key]
                latency_tier = f"{lt} (fallback)"
                fallback_level = 3
                break
    
    # Level 4: Same cost, any latency/context
    if stats is None:
        for key, value in grid_stats.items():
            if value['cost_tier'] == cost_tier:
                stats = value
                latency_tier = f"{value['latency_tier']} (fallback)"
                context_tier = f"{value['context_tier']} (fallback)"
                fallback_level = 4
                break
    
    # Level 5: Conservative baseline
    if stats is None:
        print(f"Warning: No data for {cost_tier}/{latency_tier}/{context_tier}, using conservative estimate")
        cluster_priors = [0.7] * 100
        confidence = 0.1
        n_models = 0
        fallback_level = 5
    else:
        # Use grid cell statistics
        cluster_priors = stats['mean_cluster_rates']
        
        # Add exploration bonus
        cluster_priors = [
            max(0.0, min(1.0, p + np.random.normal(0, exploration_bonus)))
            for p in cluster_priors
        ]
        
        # Confidence based on sample size and fallback level
        base_confidence = min(1.0, stats['n_models'] / 10.0)
        # Reduce confidence for fallback levels
        fallback_penalty = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.1}
        confidence = base_confidence * fallback_penalty.get(fallback_level, 0.5)
        n_models = stats['n_models']
    
    return {
        'cluster_priors': cluster_priors,
        'cost_tier': cost_tier,
        'latency_tier': latency_tier,
        'context_tier': context_tier,
        'confidence': confidence,
        'n_models': n_models,
        'fallback_level': fallback_level
    }

def save_cost_latency_context_priors(models_path: Path, output_path: Path):
    """
    Analyze cost+latency+context-performance relationship and save priors for future use.
    """
    print("=== Cost + Latency + Context Based Prior Generator ===\n")
    
    # Analyze existing models
    grid_stats = analyze_cost_latency_context_performance(models_path)
    
    # Show grid summary
    print("\n=== Cost × Latency × Context Grid Statistics ===")
    for grid_key, stats in sorted(grid_stats.items()):
        print(f"\n{grid_key.upper()}:")
        print(f"  Models: {stats['n_models']}")
        print(f"  Overall mean: {stats['overall_mean']:.3f}")
        print(f"  Overall std: {stats['overall_std']:.3f}")
    
    # Save for future use
    with open(output_path, 'w') as f:
        json.dump(grid_stats, f, indent=2)
    
        print(f"\n✓ Saved cost+latency+context priors to: {output_path}")
    
    return grid_stats

def main():
    """Demo: Generate cost+latency+context priors for example models."""
    base_dir = Path(__file__).parent.parent.parent  # Go up to final_release
    models_path = base_dir / 'models.json'
    priors_path = base_dir / 'data' / 'cost_latency_context_priors.json'
    
    # Analyze and save
    grid_stats = save_cost_latency_context_priors(models_path, priors_path)
    
    # Demo: Generate priors for various scenarios
    print("\n=== Example Priors for New Models ===")
    
    test_cases = [
        (0.05, 0.3, 64000, "Ultra-budget, fast, medium context"),
        (0.75, 0.5, 128000, "Economy, fast, large context"),
        (2.5, 1.0, 131072, "Standard, medium latency, large context"),
        (8.0, 2.5, 1000000, "Premium, slow (reasoning), xlarge context"),
        (1.5, 0.6, None, "Economy model (context unknown)"),
        (1.5, None, 100000, "Economy model (latency unknown, large context)")
    ]
    
    for cost, latency, context, description in test_cases:
        prior = generate_cost_latency_context_prior(cost, latency, context, grid_stats)
        
        print(f"\n{description}:")
        ctx_str = f"{context:,}" if context else "unknown"
        lat_str = f"{latency}s" if latency else "unknown"
        print(f"  Cost: ${cost}/1M, Latency: {lat_str}, Context: {ctx_str}")
        print(f"  Tiers: {prior['cost_tier']}/{prior['latency_tier']}/{prior['context_tier']}")
        print(f"  Fallback level: {prior['fallback_level']}")
        print(f"  Confidence: {prior['confidence']:.2f}")
        print(f"  Cluster priors: mean={np.mean(prior['cluster_priors']):.3f}, "
              f"std={np.std(prior['cluster_priors']):.3f}")
        print(f"  Based on {prior['n_models']} similar models")
    
    print("\n" + "="*70)
    print("Integration with Bandit Router:")
    print("="*70)
    print("""
# For a new model without benchmarks:
new_model_cost = 1.50
new_model_latency = 0.65  # Measure with single test request
new_model_context = 128000  # From model card/API

# Load grid statistics
with open('data/cost_latency_context_priors.json') as f:
    grid_stats = json.load(f)

# Generate initial priors
prior = generate_cost_latency_context_prior(
    cost=new_model_cost,
    latency=new_model_latency,
    context=new_model_context,
    grid_stats=grid_stats
)

# Initialize bandit with these priors
bandit = BanditRouter(
    model_id='new-model',
    priors=prior['cluster_priors']  # 100-element list
)

# Bandit starts with cost+latency+context estimates
# and LEARNS actual cluster preferences through usage
""")
    
    print("\n✓ Ready for production use!")

if __name__ == '__main__':
    main()
