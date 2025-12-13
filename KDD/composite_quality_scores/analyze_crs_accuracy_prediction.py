#!/usr/bin/env python3
"""
Analyze CRS Score as Predictor of Model Accuracy

This script examines how well CRS (Composite Reasoning Score) predicts
model accuracy on reasoning tasks, and whether NVIDIA prompt features provide
additional context about task difficulty.

Key Questions:
1. How well does CRS predict model accuracy?
2. What is the relationship between CRS and accuracy?
3. Are there any outlier models?
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model_performance_data() -> pd.DataFrame:
    """Load model CRS scores and accuracy from validation results."""
    results_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "arc_easy_vs_challenge_results.json"
    
    print(f"\n📊 Loading model performance data...")
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)
    
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
            'n_correct': model['challenge_correct'],
            'n_total': model['challenge_total'],
        })
    
    df = pd.DataFrame(models)
    print(f"   ✓ Loaded {len(df)} models")
    
    return df


def analyze_crs_accuracy_relationship(df: pd.DataFrame):
    """Analyze the relationship between CRS and accuracy."""
    print("\n" + "="*80)
    print("CRS AS PREDICTOR OF MODEL ACCURACY")
    print("="*80)
    
    # Basic statistics
    print(f"\n📈 Data Summary:")
    print(f"   Models: {len(df)}")
    print(f"   CRS range: {df['crs_score'].min():.2f} to {df['crs_score'].max():.2f}")
    print(f"   Accuracy range: {df['challenge_accuracy'].min():.1f}% to {df['challenge_accuracy'].max():.1f}%")
    print(f"   Mean accuracy: {df['challenge_accuracy'].mean():.1f}% ± {df['challenge_accuracy'].std():.1f}%")
    
    # Correlations
    print(f"\n📊 Correlation Analysis:")
    
    # Challenge accuracy
    spearman_r, spearman_p = spearmanr(df['crs_score'], df['challenge_accuracy'])
    pearson_r, pearson_p = pearsonr(df['crs_score'], df['challenge_accuracy'])
    
    print(f"\n   CRS vs Challenge Accuracy:")
    print(f"   - Spearman ρ: {spearman_r:+.3f} (p={spearman_p:.4f}) {'✓✓✓' if spearman_p < 0.001 else '✓✓' if spearman_p < 0.01 else '✓' if spearman_p < 0.05 else '✗'}")
    print(f"   - Pearson r:  {pearson_r:+.3f} (p={pearson_p:.4f}) {'✓✓✓' if pearson_p < 0.001 else '✓✓' if pearson_p < 0.01 else '✓' if pearson_p < 0.05 else '✗'}")
    
    if spearman_r > 0.7:
        print(f"   → STRONG positive correlation")
    elif spearman_r > 0.4:
        print(f"   → MODERATE positive correlation")
    elif spearman_r > 0:
        print(f"   → WEAK positive correlation")
    
    # Accuracy gap
    spearman_gap, p_gap = spearmanr(df['crs_score'], df['accuracy_gap'])
    print(f"\n   CRS vs Accuracy Gap (Easy - Challenge):")
    print(f"   - Spearman ρ: {spearman_gap:+.3f} (p={p_gap:.4f})")
    if spearman_gap < -0.3:
        print(f"   → Higher CRS → Smaller gap (better reasoning transfer)")
    
    # Group analysis
    print(f"\n📊 Performance by CRS Tier:")
    
    # Split into tiers
    df_sorted = df.sort_values('crs_score', ascending=False)
    n_per_tier = len(df) // 3
    
    top_tier = df_sorted.iloc[:n_per_tier]
    mid_tier = df_sorted.iloc[n_per_tier:2*n_per_tier]
    bottom_tier = df_sorted.iloc[2*n_per_tier:]
    
    print(f"\n   {'Tier':<15} {'CRS Range':<20} {'Avg Accuracy':<15} {'Avg Gap':<12} {'N Models':<10}")
    print(f"   {'-'*15} {'-'*20} {'-'*15} {'-'*12} {'-'*10}")
    
    for tier_name, tier_df in [('Top', top_tier), ('Middle', mid_tier), ('Bottom', bottom_tier)]:
        crs_range = f"{tier_df['crs_score'].min():.2f} to {tier_df['crs_score'].max():.2f}"
        avg_acc = tier_df['challenge_accuracy'].mean()
        avg_gap = tier_df['accuracy_gap'].mean()
        n = len(tier_df)
        print(f"   {tier_name:<15} {crs_range:<20} {avg_acc:>6.1f}%          {avg_gap:>+5.1f}%       {n:<10}")
    
    # Top vs Bottom comparison
    top_10 = df_sorted.iloc[:10]
    bottom_10 = df_sorted.iloc[-10:]
    
    print(f"\n   Top 10 avg accuracy:    {top_10['challenge_accuracy'].mean():.1f}% (CRS: {top_10['crs_score'].mean():.2f})")
    print(f"   Bottom 10 avg accuracy: {bottom_10['challenge_accuracy'].mean():.1f}% (CRS: {bottom_10['crs_score'].mean():.2f})")
    print(f"   Difference:             {top_10['challenge_accuracy'].mean() - bottom_10['challenge_accuracy'].mean():+.1f} percentage points")
    
    # Outlier analysis
    print(f"\n🔍 Outlier Analysis:")
    
    # Find models that over/underperform relative to CRS
    from sklearn.linear_model import LinearRegression
    
    X = df['crs_score'].values.reshape(-1, 1)
    y = df['challenge_accuracy'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    df['predicted_accuracy'] = model.predict(X)
    df['residual'] = df['challenge_accuracy'] - df['predicted_accuracy']
    
    # Top overperformers
    overperformers = df.nlargest(3, 'residual')
    print(f"\n   Top Overperformers (accuracy > expected from CRS):")
    for _, row in overperformers.iterrows():
        print(f"   - {row['model_name']:<40} CRS: {row['crs_score']:+.2f}, Actual: {row['challenge_accuracy']:.1f}%, Expected: {row['predicted_accuracy']:.1f}% (+{row['residual']:.1f}pp)")
    
    # Top underperformers
    underperformers = df.nsmallest(3, 'residual')
    print(f"\n   Top Underperformers (accuracy < expected from CRS):")
    for _, row in underperformers.iterrows():
        print(f"   - {row['model_name']:<40} CRS: {row['crs_score']:+.2f}, Actual: {row['challenge_accuracy']:.1f}%, Expected: {row['predicted_accuracy']:.1f}% ({row['residual']:+.1f}pp)")
    
    return df, spearman_r, pearson_r


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    print(f"\n📊 Creating visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: CRS vs Accuracy scatter
    ax1.scatter(df['crs_score'], df['challenge_accuracy'], s=100, alpha=0.6, c=df['crs_score'], cmap='RdYlGn')
    
    # Add trend line
    z = np.polyfit(df['crs_score'], df['challenge_accuracy'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['crs_score'].min(), df['crs_score'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    
    # Correlation annotation
    spearman_r, spearman_p = spearmanr(df['crs_score'], df['challenge_accuracy'])
    ax1.text(0.05, 0.95, f"Spearman ρ = {spearman_r:.3f}\n(p = {spearman_p:.4f})", 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('CRS Score (Composite Reasoning)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy on ARC-Challenge (%)', fontsize=12, fontweight='bold')
    ax1.set_title('CRS Predicts Model Accuracy on Reasoning Tasks', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CRS vs Accuracy Gap
    ax2.scatter(df['crs_score'], df['accuracy_gap'], s=100, alpha=0.6, c=df['crs_score'], cmap='RdYlGn')
    
    # Add trend line
    z2 = np.polyfit(df['crs_score'], df['accuracy_gap'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_line, p2(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    
    # Horizontal line at 0
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
    
    # Correlation annotation
    spearman_gap, p_gap = spearmanr(df['crs_score'], df['accuracy_gap'])
    ax2.text(0.05, 0.95, f"Spearman ρ = {spearman_gap:.3f}\n(p = {p_gap:.4f})", 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('CRS Score (Composite Reasoning)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Gap (Easy - Challenge, %)', fontsize=12, fontweight='bold')
    ax2.set_title('Higher CRS → Smaller Performance Gap', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "crs_accuracy_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved plot: {plot_path}")
    
    plt.close()
    
    # Create tier comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_sorted = df.sort_values('crs_score', ascending=False)
    n_per_tier = len(df) // 3
    
    tiers = []
    accuracies = []
    
    for i in range(3):
        start = i * n_per_tier
        end = (i + 1) * n_per_tier if i < 2 else len(df)
        tier_df = df_sorted.iloc[start:end]
        
        tier_name = ['Top Tier', 'Middle Tier', 'Bottom Tier'][i]
        tiers.extend([tier_name] * len(tier_df))
        accuracies.extend(tier_df['challenge_accuracy'].values)
    
    tier_df_plot = pd.DataFrame({'Tier': tiers, 'Accuracy': accuracies})
    
    sns.boxplot(data=tier_df_plot, x='Tier', y='Accuracy', palette='RdYlGn', ax=ax, order=['Top Tier', 'Middle Tier', 'Bottom Tier'])
    sns.stripplot(data=tier_df_plot, x='Tier', y='Accuracy', color='black', alpha=0.3, ax=ax, order=['Top Tier', 'Middle Tier', 'Bottom Tier'])
    
    ax.set_ylabel('Accuracy on ARC-Challenge (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('CRS Tier', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance by CRS Tier', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    tier_plot_path = output_dir / "crs_tier_comparison.png"
    plt.savefig(tier_plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved plot: {tier_plot_path}")
    
    plt.close()


def main():
    print("="*80)
    print("CRS SCORE AS PREDICTOR OF MODEL ACCURACY")
    print("="*80)
    
    # Load data
    df = load_model_performance_data()
    
    # Analyze relationship
    df_with_residuals, spearman_r, pearson_r = analyze_crs_accuracy_relationship(df)
    
    # Create visualizations
    output_dir = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results"
    create_visualizations(df_with_residuals, output_dir)
    
    # Save detailed results
    print(f"\n💾 Saving detailed results...")
    
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'task': 'ARC-Challenge',
        'n_models': len(df),
        'correlation': {
            'spearman_rho': float(spearman_r),
            'pearson_r': float(pearson_r),
        },
        'models': df[['model_name', 'crs_score', 'challenge_accuracy', 'easy_accuracy', 
                      'accuracy_gap', 'predicted_accuracy', 'residual']].to_dict('records')
    }
    
    output_path = output_dir / "crs_accuracy_prediction_detailed.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Saved: {output_path}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\n📊 Key Finding:")
    print(f"   CRS has a {'strong' if abs(spearman_r) > 0.7 else 'moderate' if abs(spearman_r) > 0.4 else 'weak'} correlation")
    print(f"   with model accuracy on reasoning tasks (ρ = {spearman_r:.3f})")
    
    if spearman_r > 0.6:
        print(f"\n✓ CRS is a GOOD predictor of model reasoning accuracy")
    elif spearman_r > 0.3:
        print(f"\n⚠ CRS is a MODERATE predictor of model reasoning accuracy")
    else:
        print(f"\n✗ CRS is a WEAK predictor of model reasoning accuracy")


if __name__ == "__main__":
    main()
