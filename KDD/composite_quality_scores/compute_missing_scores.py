#!/usr/bin/env python3
"""
Compute missing composite scores for models.

This script:
1. Computes CAE for models with swebench but missing tau2/terminalbench
2. Computes CRS for GPT-3.5 Turbo (missing 1 model)
3. Shows summary of all composite scores
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import zscore

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "models_cache.json"


def load_models():
    """Load models from cache."""
    with open(CACHE_PATH, 'r') as f:
        data = json.load(f)
    return data


def save_models(data):
    """Save models to cache."""
    with open(CACHE_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved to {CACHE_PATH}")


def compute_weighted_zscore_fallback(models, benchmarks, weights, score_prefix):
    """
    Compute weighted z-score for models missing primary benchmarks.
    Uses available benchmarks as fallback.
    """
    updated_count = 0
    
    # Get all valid scores for z-score normalization
    all_scores = {b: [] for b in benchmarks}
    for m in models:
        for b in benchmarks:
            val = m.get(b)
            if val is not None:
                all_scores[b].append(float(val))
    
    # Compute means and stds
    stats = {}
    for b in benchmarks:
        if len(all_scores[b]) > 2:
            stats[b] = {
                'mean': np.mean(all_scores[b]),
                'std': np.std(all_scores[b]) if np.std(all_scores[b]) > 0 else 1.0
            }
    
    # Compute scores for models missing the score
    for m in models:
        if m.get(score_prefix) is not None:
            continue  # Already has score
        
        # Find available benchmarks for this model
        available = []
        total_weight = 0
        weighted_sum = 0
        
        for b, w in zip(benchmarks, weights):
            val = m.get(b)
            if val is not None and b in stats:
                z = (float(val) - stats[b]['mean']) / stats[b]['std']
                weighted_sum += z * w
                total_weight += w
                available.append(b)
        
        if total_weight > 0 and len(available) >= 1:
            # Compute z-score
            z_score = weighted_sum / total_weight
            
            # Transform to 0-100 scale (assuming z-score range of -3 to +3)
            score_100 = max(0, min(100, (z_score + 3) / 6 * 100))
            
            m[score_prefix] = round(z_score, 2)
            m[f'{score_prefix}_100'] = round(score_100, 1)
            m[f'{score_prefix}_method'] = 'weighted_zscore_fallback'
            m[f'{score_prefix}_sd'] = 0.5  # Higher uncertainty for fallback
            
            print(f"  ✓ {m.get('name', 'Unknown')}: {score_prefix}={z_score:.2f} (using {available})")
            updated_count += 1
    
    return updated_count


def compute_missing_cae(models):
    """
    Compute CAE for models missing it using swebench as fallback.
    Primary: tau2, terminalbench_hard
    Fallback: swebench_verified, swebench_lite
    """
    print("\n" + "=" * 70)
    print("Computing missing CAE (Agentic Execution) scores")
    print("=" * 70)
    
    # Extended benchmarks with swebench
    benchmarks = ['tau2', 'terminalbench_hard', 'swebench_verified', 'swebench_lite']
    weights = [0.35, 0.15, 0.35, 0.15]  # Prioritize tau2 and swebench_verified
    
    return compute_weighted_zscore_fallback(models, benchmarks, weights, 'cae')


def compute_missing_crs(models):
    """
    Compute CRS for models missing it.
    Benchmarks: math_500, gpqa, hle, aime, math_index
    """
    print("\n" + "=" * 70)
    print("Computing missing CRS (Reasoning) scores")
    print("=" * 70)
    
    benchmarks = ['math_500', 'gpqa', 'hle', 'aime', 'math_index']
    weights = [0.25, 0.25, 0.15, 0.15, 0.20]
    
    return compute_weighted_zscore_fallback(models, benchmarks, weights, 'crs')


def show_summary(models):
    """Show summary of all composite scores."""
    print("\n" + "=" * 70)
    print("COMPOSITE SCORE SUMMARY")
    print("=" * 70)
    
    scores = ['ccs', 'crs', 'cfs', 'css', 'cae']
    names = {
        'ccs': 'CCS (Coding)',
        'crs': 'CRS (Reasoning)',
        'cfs': 'CFS (Factual)',
        'css': 'CSS (Summarization)',
        'cae': 'CAE (Agentic)'
    }
    
    for score in scores:
        count = sum(1 for m in models if m.get(score) is not None)
        methods = set(m.get(f'{score}_method') for m in models if m.get(f'{score}_method'))
        status = '✅' if count == len(models) else '⚠️'
        print(f"{status} {names[score]:<25} {count:>3}/{len(models)} models  Methods: {methods}")


def show_top_bottom(models, score, name, n=5):
    """Show top and bottom N for a score."""
    valid = [(m.get('name', 'Unknown'), m.get(score)) for m in models if m.get(score) is not None]
    if not valid:
        return
    
    sorted_models = sorted(valid, key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 {name}")
    print("-" * 50)
    print("  TOP 5:")
    for i, (name, score_val) in enumerate(sorted_models[:n], 1):
        print(f"    {i}. {name:<35} {score_val:>6.2f}")
    print("  BOTTOM 5:")
    for i, (name, score_val) in enumerate(sorted_models[-n:], 1):
        rank = len(sorted_models) - n + i
        print(f"    {rank}. {name:<35} {score_val:>6.2f}")


def main():
    print("=" * 70)
    print("COMPUTING MISSING COMPOSITE SCORES")
    print("=" * 70)
    
    # Load data
    data = load_models()
    models = data.get('models', data)
    print(f"Loaded {len(models)} models")
    
    # Compute missing scores
    cae_updated = compute_missing_cae(models)
    crs_updated = compute_missing_crs(models)
    
    # Summary
    show_summary(models)
    
    # Show top/bottom for each
    score_names = {
        'ccs': 'CCS (Coding)',
        'crs': 'CRS (Reasoning)',
        'cfs': 'CFS (Factual)',
        'css': 'CSS (Summarization)',
        'cae': 'CAE (Agentic)'
    }
    
    for score, name in score_names.items():
        show_top_bottom(models, score, name)
    
    # Save if any updates
    if cae_updated > 0 or crs_updated > 0:
        print(f"\n\n📝 Updated {cae_updated} CAE scores, {crs_updated} CRS scores")
        
        # Ask to save
        if '--save' in sys.argv or '--dry-run' not in sys.argv:
            save_models(data)
        else:
            print("  (Use --save to persist changes)")
    else:
        print("\n✅ All scores already computed!")


if __name__ == "__main__":
    main()
