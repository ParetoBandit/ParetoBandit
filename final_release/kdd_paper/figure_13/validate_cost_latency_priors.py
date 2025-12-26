#!/usr/bin/env python3
"""
Validation: Holdout Experiment for Cost+Latency Prior Generator

Hold out 10 random models, train on remaining 40, predict the held-out 10,
and measure prediction accuracy.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Import from the prior generator
import sys
sys.path.append(str(Path(__file__).parent))
from generate_cost_based_priors import (
    analyze_cost_latency_context_performance,
    generate_cost_latency_context_prior
)

def load_models_with_cluster_data(models_path: Path) -> List[Dict]:
    """Load all models with cluster performance data."""
    with open(models_path) as f:
        data = json.load(f)
        models = data['models']
    
    # Filter to models with required data (including context)
    valid_models = [
        m for m in models
        if all(k in m for k in ['cluster_success_rates', 'price_1m_blended', 
                                'time_to_first_token_seconds', 'context_length', 'openrouter_id'])
        and m.get('price_1m_blended') is not None
        and m.get('time_to_first_token_seconds') is not None
        and m.get('context_length') is not None
    ]
    
    return valid_models

def create_holdout_models_file(models: List[Dict], 
                               holdout_indices: List[int],
                               output_path: Path):
    """Create a temporary models.json with holdout models removed."""
    # Keep only training models
    training_models = [m for i, m in enumerate(models) if i not in holdout_indices]
    
    # Write to temp file
    temp_data = {'models': training_models}
    with open(output_path, 'w') as f:
        json.dump(temp_data, f, indent=2)
    
    return output_path

def evaluate_predictions(holdout_models: List[Dict],
                        grid_stats: Dict,
                        verbose: bool = True) -> Dict:
    """
    Evaluate prediction accuracy on holdout models.
    
    Returns:
        Dict with various accuracy metrics
    """
    results = {
        'cluster_rate_maes': [],
        'best_cluster_exact_matches': 0,
        'best_cluster_in_top_3': 0,
        'best_cluster_in_top_5': 0,
        'predictions': []
    }
    
    for model in holdout_models:
        # Extract features
        cost = model['price_1m_blended']
        latency = model['time_to_first_token_seconds']
        context = model['context_length']
        
        # Generate prediction
        prediction = generate_cost_latency_context_prior(
            cost=cost,
            latency=latency,
            context=context,
            grid_stats=grid_stats,
            exploration_bonus=0.0  # No randomness for eval
        )
        
        # Ground truth
        actual_rates = np.array(model['cluster_success_rates'])
        actual_best = model['best_relative_cluster_id']
        actual_z_scores = np.array(model['cluster_z_scores'])
        
        # Predicted
        pred_rates = np.array(prediction['cluster_priors'])
        
        # Predict best cluster from z-scores (simulate what we'd do)
        # Since we don't have actual z-scores in prior, use top cluster by rate
        pred_best = int(np.argmax(pred_rates))
        
        # Get actual top 3 and top 5
        top_3_actual = np.argsort(actual_z_scores)[-3:]
        top_5_actual = np.argsort(actual_z_scores)[-5:]
        
        # Metrics
        mae = np.mean(np.abs(pred_rates - actual_rates))
        exact_match = (pred_best == actual_best)
        in_top_3 = actual_best in top_3_actual
        in_top_5 = actual_best in top_5_actual
        
        results['cluster_rate_maes'].append(mae)
        results['best_cluster_exact_matches'] += int(exact_match)
        results['best_cluster_in_top_3'] += int(in_top_3)
        results['best_cluster_in_top_5'] += int(in_top_5)
        
        results['predictions'].append({
            'model_id': model['openrouter_id'],
            'cost': cost,
            'latency': latency,
            'context': context,
            'cost_tier': prediction['cost_tier'],
            'latency_tier': prediction['latency_tier'],
            'context_tier': prediction['context_tier'],
            'fallback_level': prediction['fallback_level'],
            'confidence': prediction['confidence'],
            'actual_best': actual_best,
            'predicted_best': pred_best,
            'exact_match': exact_match,
            'mae': mae
        })
    
    # Aggregate metrics
    n = len(holdout_models)
    results['summary'] = {
        'n_models': n,
        'mean_mae': np.mean(results['cluster_rate_maes']),
        'std_mae': np.std(results['cluster_rate_maes']),
        'exact_accuracy': results['best_cluster_exact_matches'] / n,
        'top_3_accuracy': results['best_cluster_in_top_3'] / n,
        'top_5_accuracy': results['best_cluster_in_top_5'] / n
    }
    
    return results

def main():
    """Run holdout validation experiment."""
    base_dir = Path(__file__).parent.parent.parent  # Go up to final_release
    models_path = base_dir / 'models.json'
    
    print("="*70)
    print("HOLDOUT VALIDATION: Cost+Latency+Context Prior Generator")
    print("="*70)
    
    # Load all models
    all_models = load_models_with_cluster_data(models_path)
    print(f"\nTotal models with data: {len(all_models)}")
    
    # Random holdout (10 models)
    random.seed(42)  # For reproducibility
    n_holdout = 10
    holdout_indices = random.sample(range(len(all_models)), n_holdout)
    
    holdout_models = [all_models[i] for i in holdout_indices]
    training_models = [all_models[i] for i in range(len(all_models)) if i not in holdout_indices]
    
    print(f"Training models: {len(training_models)}")
    print(f"Holdout models: {len(holdout_models)}")
    
    # Show holdout models
    print("\nHeld-out models:")
    for i, model in enumerate(holdout_models):
        print(f"  {i+1}. {model['openrouter_id']}")
        ctx = model.get('context_length', 0)
        print(f"     Cost: ${model['price_1m_blended']:.2f}, Latency: {model['time_to_first_token_seconds']:.2f}s, Context: {ctx:,}")
    
    # Create temporary models file (training only)
    temp_path = base_dir / 'data' / 'models_train_only.json'
    create_holdout_models_file(all_models, holdout_indices, temp_path)
    
    # Train on remaining models
    print("\n" + "="*70)
    print("Training cost-latency-context grid on 40 models...")
    print("="*70)
    grid_stats = analyze_cost_latency_context_performance(temp_path)
    
    print(f"\nGrid cells populated: {len(grid_stats)}")
    
    # Predict holdout models
    print("\n" + "="*70)
    print("Predicting held-out 10 models...")
    print("="*70)
    
    results = evaluate_predictions(holdout_models, grid_stats)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    summary = results['summary']
    print(f"\n📊 Overall Metrics (n={summary['n_models']}):")
    print(f"  Mean cluster rate MAE: {summary['mean_mae']:.3f} ± {summary['std_mae']:.3f}")
    print(f"  Best cluster exact match: {summary['exact_accuracy']:.1%}")
    print(f"  Best cluster in top-3: {summary['top_3_accuracy']:.1%}")
    print(f"  Best cluster in top-5: {summary['top_5_accuracy']:.1%}")
    
    print("\n📋 Individual Predictions:")
    for pred in results['predictions']:
        match_symbol = "✓" if pred['exact_match'] else "✗"
        print(f"\n  {match_symbol} {pred['model_id']}")
        print(f"     Tiers: {pred['cost_tier']}/{pred['latency_tier']}/{pred['context_tier']} (fallback L{pred['fallback_level']})")
        print(f"     Confidence: {pred['confidence']:.2f}")
        print(f"     Actual best: {pred['actual_best']}, Predicted: {pred['predicted_best']}")
        print(f"     MAE: {pred['mae']:.3f}")
    
    # Cleanup
    temp_path.unlink()
    
    print("\n" + "="*70)
    print("✓ Validation complete!")
    print("="*70)

if __name__ == '__main__':
    main()
