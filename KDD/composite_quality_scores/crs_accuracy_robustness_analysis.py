#!/usr/bin/env python3
"""
Robustness Analysis: CRS vs Accuracy with Outlier Detection

This script examines how robust the CRS-accuracy correlation is by:
1. Identifying outliers using multiple methods
2. Recalculating correlations with outliers removed
3. Comparing full dataset vs outlier-removed results
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model_performance_data() -> pd.DataFrame:
    """Load model CRS scores and accuracy from validation results."""
    results_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "arc_easy_vs_challenge_results.json"
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    models = []
    for model in data['models']:
        models.append({
            'model_name': model['name'],
            'crs_score': model['crs_score'],
            'crs_rank': model['crs_rank'],
            'challenge_accuracy': model['challenge_accuracy'],
            'easy_accuracy': model['easy_accuracy'],
            'accuracy_gap': model['accuracy_gap'],
        })
    
    df = pd.DataFrame(models)
    return df


def identify_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify outliers using multiple methods."""
    
    X = df['crs_score'].values.reshape(-1, 1)
    y = df['challenge_accuracy'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate residuals
    df['predicted_accuracy'] = model.predict(X)
    df['residual'] = df['challenge_accuracy'] - df['predicted_accuracy']
    df['abs_residual'] = np.abs(df['residual'])
    
    # Method 1: Residual-based (> 2 standard deviations)
    residual_std = df['residual'].std()
    residual_threshold = 2 * residual_std
    df['is_outlier_residual'] = df['abs_residual'] > residual_threshold
    
    # Method 2: IQR method on residuals
    Q1 = df['residual'].quantile(0.25)
    Q3 = df['residual'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['is_outlier_iqr'] = (df['residual'] < lower_bound) | (df['residual'] > upper_bound)
    
    # Combined: outlier by either method
    df['is_outlier'] = df['is_outlier_residual'] | df['is_outlier_iqr']
    
    return df


def analyze_with_without_outliers(df: pd.DataFrame):
    """Compare correlations with and without outliers."""
    
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS: CRS vs ACCURACY WITH/WITHOUT OUTLIERS")
    print("="*80)
    
    # Full dataset
    print(f"\n📊 FULL DATASET (n={len(df)})")
    spearman_full, p_full = spearmanr(df['crs_score'], df['challenge_accuracy'])
    pearson_full, p_pearson_full = pearsonr(df['crs_score'], df['challenge_accuracy'])
    
    print(f"   Spearman ρ: {spearman_full:+.3f} (p={p_full:.4f})")
    print(f"   Pearson r:  {pearson_full:+.3f} (p={p_pearson_full:.4f})")
    
    # Identify outliers
    n_outliers_residual = df['is_outlier_residual'].sum()
    n_outliers_iqr = df['is_outlier_iqr'].sum()
    n_outliers_total = df['is_outlier'].sum()
    
    print(f"\n🔍 OUTLIER DETECTION")
    print(f"   Residual method (>2σ): {n_outliers_residual} outliers")
    print(f"   IQR method (1.5×IQR):   {n_outliers_iqr} outliers")
    print(f"   Combined:               {n_outliers_total} outliers")
    
    if n_outliers_total > 0:
        print(f"\n   Outliers identified:")
        outliers = df[df['is_outlier']]
        for _, row in outliers.iterrows():
            print(f"   - {row['model_name']:<45} CRS: {row['crs_score']:+.2f}, Accuracy: {row['challenge_accuracy']:.1f}%, Residual: {row['residual']:+.1f}pp")
    
    # Without outliers
    df_clean = df[~df['is_outlier']]
    
    print(f"\n📊 WITHOUT OUTLIERS (n={len(df_clean)})")
    
    if len(df_clean) >= 3:
        spearman_clean, p_clean = spearmanr(df_clean['crs_score'], df_clean['challenge_accuracy'])
        pearson_clean, p_pearson_clean = pearsonr(df_clean['crs_score'], df_clean['challenge_accuracy'])
        
        print(f"   Spearman ρ: {spearman_clean:+.3f} (p={p_clean:.4f})")
        print(f"   Pearson r:  {pearson_clean:+.3f} (p={p_pearson_clean:.4f})")
        
        # Compare
        print(f"\n📉 CHANGE IN CORRELATION")
        print(f"   Spearman Δ: {spearman_clean - spearman_full:+.3f} ({(spearman_clean - spearman_full)/spearman_full*100:+.1f}%)")
        print(f"   Pearson Δ:  {pearson_clean - pearson_full:+.3f} ({(pearson_clean - pearson_full)/pearson_full*100:+.1f}%)")
        
        if abs(spearman_clean - spearman_full) > 0.15:
            print(f"\n   ⚠️  SUBSTANTIAL CHANGE: Correlation drops by {abs(spearman_clean - spearman_full):.3f}")
            print(f"      → Relationship is DRIVEN BY OUTLIERS")
        elif abs(spearman_clean - spearman_full) > 0.05:
            print(f"\n   ⚠️  MODERATE CHANGE: Correlation changes by {abs(spearman_clean - spearman_full):.3f}")
            print(f"      → Outliers have NOTABLE influence")
        else:
            print(f"\n   ✓ MINIMAL CHANGE: Correlation only changes by {abs(spearman_clean - spearman_full):.3f}")
            print(f"      → Relationship is ROBUST to outliers")
        
        # Effect size comparison
        print(f"\n📊 PERFORMANCE METRICS")
        
        # Top vs bottom in full dataset
        df_sorted_full = df.sort_values('crs_score', ascending=False)
        top_full = df_sorted_full.iloc[:len(df_sorted_full)//2]
        bottom_full = df_sorted_full.iloc[len(df_sorted_full)//2:]
        
        print(f"\n   Full Dataset:")
        print(f"   - Top 50% CRS:    {top_full['challenge_accuracy'].mean():.1f}% accuracy")
        print(f"   - Bottom 50% CRS: {bottom_full['challenge_accuracy'].mean():.1f}% accuracy")
        print(f"   - Difference:     {top_full['challenge_accuracy'].mean() - bottom_full['challenge_accuracy'].mean():+.1f} pp")
        
        # Top vs bottom without outliers
        df_sorted_clean = df_clean.sort_values('crs_score', ascending=False)
        top_clean = df_sorted_clean.iloc[:len(df_sorted_clean)//2]
        bottom_clean = df_sorted_clean.iloc[len(df_sorted_clean)//2:]
        
        print(f"\n   Without Outliers:")
        print(f"   - Top 50% CRS:    {top_clean['challenge_accuracy'].mean():.1f}% accuracy")
        print(f"   - Bottom 50% CRS: {bottom_clean['challenge_accuracy'].mean():.1f}% accuracy")
        print(f"   - Difference:     {top_clean['challenge_accuracy'].mean() - bottom_clean['challenge_accuracy'].mean():+.1f} pp")
        
        return df_clean, spearman_full, spearman_clean, pearson_full, pearson_clean
    else:
        print(f"   ⚠️  Too few data points after removing outliers")
        return df_clean, spearman_full, None, pearson_full, None


def create_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create side-by-side plots with and without outliers."""
    
    print(f"\n📊 Creating comparison visualization...")
    
    df_clean = df[~df['is_outlier']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Full dataset
    colors_full = ['red' if x else 'gray' for x in df['is_outlier']]
    ax1.scatter(df['crs_score'], df['challenge_accuracy'], s=100, alpha=0.6, c=colors_full)
    
    # Trend line
    z = np.polyfit(df['crs_score'], df['challenge_accuracy'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['crs_score'].min(), df['crs_score'].max(), 100)
    ax1.plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2, label='Linear fit (all data)')
    
    # Mark outliers
    outliers = df[df['is_outlier']]
    if len(outliers) > 0:
        ax1.scatter(outliers['crs_score'], outliers['challenge_accuracy'], 
                   s=200, facecolors='none', edgecolors='red', linewidth=2, 
                   label='Outliers', zorder=10)
    
    spearman_r, p_val = spearmanr(df['crs_score'], df['challenge_accuracy'])
    ax1.text(0.05, 0.95, f"n = {len(df)}\nSpearman ρ = {spearman_r:.3f}\n(p = {p_val:.4f})", 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('CRS Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy on ARC-Challenge (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Full Dataset (with outliers)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Without outliers
    ax2.scatter(df_clean['crs_score'], df_clean['challenge_accuracy'], 
               s=100, alpha=0.6, c=df_clean['crs_score'], cmap='RdYlGn')
    
    # Trend line
    if len(df_clean) >= 2:
        z2 = np.polyfit(df_clean['crs_score'], df_clean['challenge_accuracy'], 1)
        p2 = np.poly1d(z2)
        x_line2 = np.linspace(df_clean['crs_score'].min(), df_clean['crs_score'].max(), 100)
        ax2.plot(x_line2, p2(x_line2), "b--", alpha=0.8, linewidth=2, label='Linear fit (clean)')
        
        spearman_r2, p_val2 = spearmanr(df_clean['crs_score'], df_clean['challenge_accuracy'])
        ax2.text(0.05, 0.95, f"n = {len(df_clean)}\nSpearman ρ = {spearman_r2:.3f}\n(p = {p_val2:.4f})", 
                 transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax2.set_xlabel('CRS Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy on ARC-Challenge (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Without Outliers', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "crs_accuracy_outlier_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved plot: {plot_path}")
    
    plt.close()


def main():
    print("="*80)
    print("ROBUSTNESS ANALYSIS: CRS vs ACCURACY")
    print("="*80)
    
    # Load data
    df = load_model_performance_data()
    
    # Identify outliers
    df_with_outliers = identify_outliers(df)
    
    # Analyze with/without outliers
    df_clean, spearman_full, spearman_clean, pearson_full, pearson_clean = analyze_with_without_outliers(df_with_outliers)
    
    # Create visualization
    output_dir = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results"
    create_comparison_plot(df_with_outliers, output_dir)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    print("\n📊 Summary:")
    print(f"   Full dataset: ρ = {spearman_full:.3f}")
    if spearman_clean is not None:
        print(f"   Without outliers: ρ = {spearman_clean:.3f}")
        print(f"   Change: {spearman_clean - spearman_full:+.3f} ({abs(spearman_clean - spearman_full)/spearman_full*100:.1f}%)")
        
        if abs(spearman_clean - spearman_full) > 0.15:
            print(f"\n   ⚠️  Correlation is HEAVILY INFLUENCED by outliers")
            print(f"      The relationship may not be as strong as it appears")
        else:
            print(f"\n   ✓ Correlation is REASONABLY ROBUST to outliers")
    
    # Save results
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'full_dataset': {
            'n': len(df_with_outliers),
            'spearman_rho': float(spearman_full),
            'pearson_r': float(pearson_full),
        },
        'outliers': {
            'n_outliers': int(df_with_outliers['is_outlier'].sum()),
            'outlier_models': df_with_outliers[df_with_outliers['is_outlier']]['model_name'].tolist(),
        },
        'clean_dataset': {
            'n': len(df_clean),
            'spearman_rho': float(spearman_clean) if spearman_clean else None,
            'pearson_r': float(pearson_clean) if pearson_clean else None,
        },
        'robustness': {
            'spearman_change': float(spearman_clean - spearman_full) if spearman_clean else None,
            'pct_change': float((spearman_clean - spearman_full)/spearman_full*100) if spearman_clean else None,
        }
    }
    
    output_path = output_dir / "crs_accuracy_robustness.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
