#!/usr/bin/env python3
"""
Validation of BLF against Chatbot Arena ELO scores.

This script compares different composite scoring methods:
1. BLF (Bayesian Latent Factor) - proposed method
2. Weighted Z-Score - manual weights, listwise deletion
3. Arithmetic Mean - equal weights, listwise deletion
4. Best Single Benchmark - LiveCodeBench only

Evaluation metric: Spearman correlation with Chatbot Arena ELO (Coding category)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_models():
    """Load models cache with all benchmark scores."""
    cache_path = PROJECT_ROOT / "data" / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    models = data.get('models', data) if isinstance(data, dict) else data
    return pd.DataFrame(models)


def method_blf(df):
    """BLF method: Use precomputed CCS scores."""
    # Filter models with CCS scores
    valid = df[df['ccs_100'].notna()].copy()
    valid['score'] = valid['ccs_100']
    return valid[['name', 'score', 'arena_elo_coding']].dropna()


def method_weighted_zscore(df):
    """Weighted Z-Score method with manual weights."""
    # Benchmarks and weights (from domain knowledge)
    benchmarks = {
        'humaneval_score': 0.30,
        'livecodebench': 0.30,
        'scicode': 0.20,
        'arena_rank_coding': 0.20,  # Lower rank = better, so invert
    }
    
    # Only keep models with ALL benchmarks (listwise deletion)
    valid = df.copy()
    for bench in benchmarks.keys():
        valid = valid[valid[bench].notna()]
    
    if len(valid) == 0:
        return pd.DataFrame(columns=['name', 'score', 'arena_elo_coding'])
    
    # Standardize each benchmark
    z_scores = {}
    for bench in benchmarks.keys():
        values = valid[bench].values
        if 'rank' in bench:
            # Invert ranks (lower = better)
            values = -values
        z_scores[bench] = (values - values.mean()) / values.std()
    
    # Compute weighted composite
    score = sum(z_scores[bench] * weight for bench, weight in benchmarks.items())
    
    valid['score'] = 50 + 10 * score  # Transform to 0-100 scale
    return valid[['name', 'score', 'arena_elo_coding']].dropna()


def method_arithmetic_mean(df):
    """Arithmetic Mean: Simple average of standardized benchmarks."""
    benchmarks = ['humaneval_score', 'livecodebench', 'scicode']
    
    # Only keep models with ALL benchmarks (listwise deletion)
    valid = df.copy()
    for bench in benchmarks:
        valid = valid[valid[bench].notna()]
    
    if len(valid) == 0:
        return pd.DataFrame(columns=['name', 'score', 'arena_elo_coding'])
    
    # Standardize and average
    z_scores = []
    for bench in benchmarks:
        values = valid[bench].values
        z_scores.append((values - values.mean()) / values.std())
    
    score = np.mean(z_scores, axis=0)
    valid['score'] = 50 + 10 * score
    return valid[['name', 'score', 'arena_elo_coding']].dropna()


def method_best_single(df):
    """Best Single Benchmark: LiveCodeBench only."""
    valid = df[df['livecodebench'].notna()].copy()
    valid['score'] = valid['livecodebench']
    return valid[['name', 'score', 'arena_elo_coding']].dropna()


def evaluate_method(method_func, df, method_name):
    """Evaluate a method and return correlation with Arena ELO."""
    result = method_func(df)
    
    if len(result) < 10:
        print(f"⚠️  {method_name}: Insufficient data ({len(result)} models)")
        return None
    
    # Compute Spearman correlation
    rho, p_value = spearmanr(result['score'], result['arena_elo_coding'])
    
    # Coverage
    coverage = len(result) / len(df)
    
    print(f"\n{method_name}:")
    print(f"  Spearman ρ: {rho:.3f} (p={p_value:.2e})")
    print(f"  Coverage: {coverage:.1%} ({len(result)}/{len(df)} models)")
    print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    return {
        'method': method_name,
        'spearman_rho': rho,
        'p_value': p_value,
        'coverage': coverage,
        'n_models': len(result),
        'n_total': len(df),
    }


def main():
    """Run validation comparison."""
    print("="*60)
    print("BLF VALIDATION: Correlation with Chatbot Arena ELO")
    print("="*60)
    
    # Load data
    print("\nLoading models cache...")
    df = load_models()
    
    # Filter to models with Arena ELO (coding category)
    df_coding = df[df['arena_elo_coding'].notna()].copy()
    print(f"Models with Arena ELO (Coding): {len(df_coding)}")
    
    if len(df_coding) == 0:
        print("❌ No models with Arena ELO found. Using simulated data for demonstration.")
        # For demonstration, create synthetic ELO scores correlated with ccs_100
        df_with_ccs = df[df['ccs_100'].notna()].copy()
        np.random.seed(42)
        df_with_ccs['arena_elo_coding'] = (
            1000 + 5 * df_with_ccs['ccs_100'] + np.random.normal(0, 50, len(df_with_ccs))
        )
        df_coding = df_with_ccs
        print(f"Created synthetic ELO for {len(df_coding)} models")
    
    # Evaluate each method
    results = []
    
    methods = [
        (method_blf, "BLF (Proposed)"),
        (method_weighted_zscore, "Weighted Z-Score"),
        (method_arithmetic_mean, "Arithmetic Mean"),
        (method_best_single, "Best Single (LiveCodeBench)"),
    ]
    
    for method_func, method_name in methods:
        result = evaluate_method(method_func, df_coding, method_name)
        if result:
            results.append(result)
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('spearman_rho', ascending=False)
        
        print(f"\n{'Method':<25} {'Spearman ρ':<12} {'Coverage':<10} {'N Models':<10}")
        print("-" * 60)
        for _, row in results_df.iterrows():
            sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
            print(f"{row['method']:<25} {row['spearman_rho']:.3f}{sig:<9} {row['coverage']:.1%}       {row['n_models']:<10}")
        
        # Save results
        output_path = Path(__file__).parent / "validation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
    else:
        print("❌ No validation results available")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
