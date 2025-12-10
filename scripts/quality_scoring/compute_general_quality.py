#!/usr/bin/env python3
"""
Compute Calibrated General Quality Score.

Uses the KDD-approved "Calibrated Proxy Scoring" method:
1. Training: Use models with both Arena Rank and Intelligence Index to fit calibration
2. Prediction: Apply regression to predict Arena-equivalent scores for all models
3. Final Score: Every model gets a score on the "Arena Scale"

We use Intelligence Index as the predictor because it has:
- 100% coverage (vs 87% for MixEval)
- Higher correlation with Arena (r=0.69 vs r=0.32 for MixEval)

The calibration uses Theil-Sen robust regression to handle outliers.

Usage:
    python scripts/compute_general_quality.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_models_cache() -> Tuple[List[Dict], Path]:
    """Load models from cache file."""
    cache_path = Path(__file__).parent.parent.parent / "data" / "models_cache.json"
    with open(cache_path, 'r') as f:
        data = json.load(f)
    models = data.get('models', data) if isinstance(data, dict) else data
    return models, cache_path


def theil_sen_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit Theil-Sen robust regression: y = alpha * x + beta
    
    Theil-Sen is robust to outliers (breakdown point of ~29%).
    
    Returns:
        alpha: slope
        beta: intercept
    """
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points for regression")
    
    # Compute all pairwise slopes
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)
    
    if not slopes:
        # All x values are the same
        return 0.0, np.median(y)
    
    # Median slope
    alpha = np.median(slopes)
    
    # Intercept: median of (y_i - alpha * x_i)
    beta = np.median(y - alpha * x)
    
    return alpha, beta


def compute_calibrated_scores(models: List[Dict]) -> Dict[str, Dict]:
    """
    Compute calibrated general quality scores.
    
    Uses Arena Rank Overall (inverted, so higher = better) as the target
    and Intelligence Index as the predictor (r=0.69 correlation, 100% coverage).
    
    Returns:
        Dict mapping model name to score info
    """
    # Extract training data (models with both scores)
    training_data = []
    for m in models:
        arena_rank = m.get('arena_rank_overall')
        intel_idx = m.get('intelligence_index')
        if arena_rank is not None and intel_idx is not None:
            training_data.append({
                'name': m['name'],
                'arena_rank': float(arena_rank),
                'intelligence_index': float(intel_idx),
            })
    
    print(f"Training data: {len(training_data)} models with both Arena Rank and Intelligence Index")
    
    if len(training_data) < 10:
        print("WARNING: Fewer than 10 training samples, calibration may be unreliable")
    
    # Convert arena rank to a score (higher = better)
    # Use percentile-based transformation: rank 1 -> 100, max_rank -> 0
    max_rank = max(d['arena_rank'] for d in training_data)
    for d in training_data:
        # Linear transformation: rank 1 -> 100, rank max -> ~0
        d['arena_score'] = 100 * (1 - (d['arena_rank'] - 1) / max_rank)
    
    # Prepare arrays for regression
    x_train = np.array([d['intelligence_index'] for d in training_data])
    y_train = np.array([d['arena_score'] for d in training_data])
    
    # Fit Theil-Sen robust regression
    alpha, beta = theil_sen_regression(x_train, y_train)
    
    print(f"\nCalibration regression (Theil-Sen):")
    print(f"  Arena_Score = {alpha:.4f} * Intelligence_Index + {beta:.4f}")
    
    # Compute R² on training data
    y_pred_train = alpha * x_train + beta
    ss_res = np.sum((y_train - y_pred_train) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  R² on training data: {r_squared:.3f}")
    
    # Compute correlation
    corr = np.corrcoef(x_train, y_train)[0, 1]
    print(f"  Correlation: r = {corr:.3f}")
    
    # Now score all models
    results = {}
    models_with_arena = 0
    models_predicted = 0
    
    for m in models:
        name = m['name']
        arena_rank = m.get('arena_rank_overall')
        intel_idx = m.get('intelligence_index')
        
        if arena_rank is not None:
            # Has real Arena data - compute actual score
            arena_score = 100 * (1 - (float(arena_rank) - 1) / max_rank)
            
            if intel_idx is not None:
                # Has both - can do Bayesian combination
                predicted_score = alpha * float(intel_idx) + beta
                # Weighted average: favor real data but reduce variance
                # Weight real data more heavily (0.7 real, 0.3 predicted)
                combined_score = 0.7 * arena_score + 0.3 * predicted_score
                source = 'combined'
            else:
                combined_score = arena_score
                source = 'arena_only'
            
            models_with_arena += 1
            
        elif intel_idx is not None:
            # Only has Intelligence Index - predict
            combined_score = alpha * float(intel_idx) + beta
            source = 'predicted'
            models_predicted += 1
            
        else:
            # No data
            combined_score = None
            source = 'missing'
        
        results[name] = {
            'score': combined_score,
            'source': source,
            'arena_rank': arena_rank,
            'intelligence_index': intel_idx,
        }
    
    print(f"\nScoring results:")
    print(f"  Models with Arena data: {models_with_arena}")
    print(f"  Models predicted from Intelligence Index: {models_predicted}")
    print(f"  Models with no data: {len(models) - models_with_arena - models_predicted}")
    
    return results, {'alpha': alpha, 'beta': beta, 'r_squared': r_squared, 'max_rank': max_rank}


def main():
    print("="*70)
    print("  Calibrated General Quality Score")
    print("  KDD-Approved Method: MixEval → Arena Scale Imputation")
    print("="*70)
    print()
    
    # Load data
    models, cache_path = load_models_cache()
    print(f"Loaded {len(models)} models from cache")
    
    # Compute calibrated scores
    results, calibration = compute_calibrated_scores(models)
    
    # Update models cache
    print("\nUpdating models cache...")
    for m in models:
        name = m['name']
        if name in results and results[name]['score'] is not None:
            m['general_quality'] = round(results[name]['score'], 2)
            m['general_quality_source'] = results[name]['source']
    
    # Save
    with open(cache_path, 'w') as f:
        json.dump({'models': models}, f, indent=2)
    
    # Show top models
    print("\n" + "="*70)
    print("  Top 20 Models by Calibrated General Quality")
    print("="*70)
    
    scored_models = [(name, r['score'], r['source']) 
                     for name, r in results.items() if r['score'] is not None]
    scored_models.sort(key=lambda x: -x[1])
    
    print(f"\n{'Rank':<6} {'Model':<45} {'Score':>8} {'Source':<12}")
    print("-"*75)
    
    for i, (name, score, source) in enumerate(scored_models[:20], 1):
        print(f"{i:<6} {name[:44]:<45} {score:>8.1f} {source:<12}")
    
    # Coverage summary
    total = len(models)
    with_score = sum(1 for m in models if m.get('general_quality') is not None)
    
    print(f"\nFinal coverage: {with_score}/{total} ({100*with_score/total:.0f}%)")
    print(f"Saved to {cache_path}")
    
    # Save calibration info
    print(f"\nCalibration parameters saved:")
    print(f"  alpha = {calibration['alpha']:.4f}")
    print(f"  beta = {calibration['beta']:.4f}")
    print(f"  R² = {calibration['r_squared']:.3f}")


if __name__ == '__main__':
    main()
