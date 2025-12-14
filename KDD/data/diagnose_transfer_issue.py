#!/usr/bin/env python3
"""
Diagnose Transfer Issue

This script helps understand WHY transfer validation isn't working well.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data():
    """Load training data and results."""
    data_path = Path(__file__).parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
    df = pd.read_csv(data_path)
    
    # Load cache
    cache_path = Path(__file__).parent.parent.parent / 'data' / 'models_cache.json'
    with open(cache_path) as f:
        cache_data = json.load(f)
    cache = cache_data['models'] if 'models' in cache_data else cache_data
    
    # Add benchmark scores
    benchmarks_by_model = {m['name']: m for m in cache}
    
    # Try to match models
    for model_name in df['model'].unique():
        # Try direct match or via mappings
        found = False
        for cache_model in cache:
            if (cache_model['name'].lower() == model_name.lower() or 
                cache_model.get('slug', '').lower() == model_name.lower()):
                df.loc[df['model'] == model_name, 'model_hle'] = cache_model.get('hle', np.nan) * 100
                df.loc[df['model'] == model_name, 'model_intelligence'] = cache_model.get('intelligence_index', np.nan)
                df.loc[df['model'] == model_name, 'model_gpqa_aggregate'] = cache_model.get('gpqa', np.nan) * 100
                found = True
                break
    
    return df

def main():
    print("="*80)
    print("DIAGNOSTIC: WHY IS TRANSFER NOT WORKING WELL?")
    print("="*80)
    
    df = load_data()
    
    # Check 1: Do benchmarks correlate with success within models?
    print("\n" + "="*80)
    print("CHECK 1: Benchmark-Success Correlation Per Model")
    print("="*80)
    
    model_stats = []
    
    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name]
        
        if len(model_data) < 10:
            continue
        
        hle = model_data['model_hle'].iloc[0] if 'model_hle' in model_data else np.nan
        intelligence = model_data['model_intelligence'].iloc[0] if 'model_intelligence' in model_data else np.nan
        success_rate = model_data['success'].mean()
        n = len(model_data)
        
        model_stats.append({
            'model': model_name,
            'hle': hle,
            'intelligence': intelligence,
            'success_rate': success_rate,
            'n': n
        })
    
    stats_df = pd.DataFrame(model_stats)
    stats_df = stats_df.dropna(subset=['hle', 'intelligence'])
    
    if len(stats_df) > 3:
        # Correlation between HLE and success rate
        hle_corr, hle_p = pearsonr(stats_df['hle'], stats_df['success_rate'])
        intel_corr, intel_p = pearsonr(stats_df['intelligence'], stats_df['success_rate'])
        
        print(f"\nCorrelation: Model HLE vs. GPQA Success Rate")
        print(f"  r = {hle_corr:.3f} (p = {hle_p:.4f})")
        
        if hle_corr < 0.5:
            print(f"  ❌ WEAK! HLE is a poor predictor of GPQA performance!")
        elif hle_corr < 0.7:
            print(f"  ⚠️  MODERATE. HLE is okay but not great.")
        else:
            print(f"  ✅ STRONG! HLE is a good predictor.")
        
        print(f"\nCorrelation: Model Intelligence Index vs. GPQA Success Rate")
        print(f"  r = {intel_corr:.3f} (p = {intel_p:.4f})")
        
        if intel_corr < 0.5:
            print(f"  ❌ WEAK!")
        elif intel_corr < 0.7:
            print(f"  ⚠️  MODERATE.")
        else:
            print(f"  ✅ STRONG!")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.scatter(stats_df['hle'], stats_df['success_rate'], alpha=0.6)
        ax1.set_xlabel('HLE Score')
        ax1.set_ylabel('GPQA Success Rate')
        ax1.set_title(f'HLE vs Success (r={hle_corr:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(stats_df['hle'], stats_df['success_rate'], 1)
        p = np.poly1d(z)
        ax1.plot(stats_df['hle'], p(stats_df['hle']), "r--", alpha=0.8)
        
        ax2.scatter(stats_df['intelligence'], stats_df['success_rate'], alpha=0.6)
        ax2.set_xlabel('Intelligence Index')
        ax2.set_ylabel('GPQA Success Rate')
        ax2.set_title(f'Intelligence vs Success (r={intel_corr:.3f})')
        ax2.grid(True, alpha=0.3)
        
        z2 = np.polyfit(stats_df['intelligence'], stats_df['success_rate'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(stats_df['intelligence'], p2(stats_df['intelligence']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / 'validation_results' / 'benchmark_correlation.png', dpi=150)
        print(f"\n✓ Saved plot to validation_results/benchmark_correlation.png")
    
    # Check 2: Are proprietary models fundamentally different?
    print("\n" + "="*80)
    print("CHECK 2: Proprietary vs. Open-Source Comparison")
    print("="*80)
    
    PROPRIETARY = ['gpt-4o-mini-2024-07-18', 'gpt4o-20240806', 'gpt4o-20241120',
                   'claude-3-5-sonnet-20241022', 'claude-3-7-sonnet-20250219',
                   'gemini-1.5-pro-latest', 'gemini-2.0-flash-exp']
    
    proprietary_data = stats_df[stats_df['model'].isin(PROPRIETARY)]
    opensource_data = stats_df[~stats_df['model'].isin(PROPRIETARY)]
    
    if len(proprietary_data) > 0 and len(opensource_data) > 0:
        print(f"\nOpen-Source Models ({len(opensource_data)}):")
        print(f"  HLE range: {opensource_data['hle'].min():.1f} - {opensource_data['hle'].max():.1f}")
        print(f"  Intelligence range: {opensource_data['intelligence'].min():.1f} - {opensource_data['intelligence'].max():.1f}")
        print(f"  Success rate range: {opensource_data['success_rate'].min():.1%} - {opensource_data['success_rate'].max():.1%}")
        
        print(f"\nProprietary Models ({len(proprietary_data)}):")
        print(f"  HLE range: {proprietary_data['hle'].min():.1f} - {proprietary_data['hle'].max():.1f}")
        print(f"  Intelligence range: {proprietary_data['intelligence'].min():.1f} - {proprietary_data['intelligence'].max():.1f}")
        print(f"  Success rate range: {proprietary_data['success_rate'].min():.1%} - {proprietary_data['success_rate'].max():.1%}")
        
        # Check if proprietary models are out of distribution
        prop_hle_mean = proprietary_data['hle'].mean()
        open_hle_mean = opensource_data['hle'].mean()
        
        if prop_hle_mean > opensource_data['hle'].max():
            print(f"\n  ❌ EXTRAPOLATION! Proprietary models (HLE={prop_hle_mean:.1f}) are ABOVE training range!")
        elif prop_hle_mean < opensource_data['hle'].min():
            print(f"\n  ❌ EXTRAPOLATION! Proprietary models are BELOW training range!")
        else:
            print(f"\n  ✅ INTERPOLATION: Proprietary models are within training distribution.")
    
    # Check 3: Print full model list with scores
    print("\n" + "="*80)
    print("CHECK 3: All Models with Benchmarks")
    print("="*80)
    
    print(f"\n{'Model':<45} {'HLE':>6} {'Intel':>6} {'Success':>8} {'N':>5}")
    print("-"*80)
    
    for _, row in stats_df.sort_values('success_rate', ascending=False).iterrows():
        is_prop = "🔒" if row['model'] in PROPRIETARY else "  "
        print(f"{is_prop} {row['model']:<43} {row['hle']:>6.1f} {row['intelligence']:>6.1f} {row['success_rate']:>7.1%} {row['n']:>5}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(stats_df) > 3:
        if hle_corr < 0.5:
            print("\n❌ Problem: HLE doesn't predict GPQA well (r < 0.5)")
            print("   Solution: Use a different benchmark (intelligence_index, gpqa aggregate, or mmlu_pro)")
        
        if hle_corr < intel_corr:
            print(f"\n✅ Intelligence Index (r={intel_corr:.3f}) is better than HLE (r={hle_corr:.3f})")
            print("   Solution: Switch primary feature to intelligence_index")
        
        if len(proprietary_data) > 0:
            if proprietary_data['hle'].mean() > opensource_data['hle'].max() * 1.1:
                print("\n⚠️  Proprietary models are higher capability than training set")
                print("   This makes transfer harder (extrapolation vs interpolation)")
                print("   Solution: This is expected - acknowledge as limitation in paper")
    
    print("\n" + "="*80)
    print("BOTTOM LINE")
    print("="*80)
    
    if hle_corr >= 0.6:
        print("\n✅ Benchmarks DO correlate with GPQA (r >= 0.6)")
        print("   The transfer issue is likely due to:")
        print("   1. Limited training data (35 models)")
        print("   2. NVIDIA features not capturing prompt difficulty well")
        print("   3. Normal noise in model predictions")
        print("\n   Current validation (r=0.54) is ACCEPTABLE for paper with:")
        print("   - Clear acknowledgment of moderate transfer")
        print("   - Discussion of limitations")
        print("   - Focus on within-distribution accuracy (73%)")
    else:
        print("\n❌ Benchmarks DON'T correlate well with GPQA (r < 0.6)")
        print("   This is a DATA PROBLEM, not a model problem!")
        print("\n   Options:")
        print("   1. Use different benchmark as capability proxy")
        print("   2. Acknowledge this specific benchmark mismatch")
        print("   3. Focus paper on methodology, not specific results")

if __name__ == '__main__':
    main()
