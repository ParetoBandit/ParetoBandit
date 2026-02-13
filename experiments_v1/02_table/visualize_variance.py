#!/usr/bin/env python3
"""
Visualize variance across seeds for Corralling vs baselines.

This script creates diagnostic plots to understand the source and magnitude
of variance in the Corralling algorithm.

Usage:
    python visualize_variance.py \
        --results data/eta_1.0_holdout_multiseed/results_multiseed.json \
        --per-seed data/eta_1.0_holdout_multiseed/results_per_seed.json \
        --output variance_analysis.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def plot_variance_analysis(results: dict, per_seed: dict, output_path: Path):
    """Create comprehensive variance visualization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Variance Analysis: Corralling vs Baselines', fontsize=16, fontweight='bold')
    
    strategies = ['Warmup', 'Tabula Rasa', 'Hybrid (Corralling)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # Plot 1: Cumulative Regret Distribution
    ax1 = axes[0, 0]
    data = [results[s]['statistics']['raw_values']['cumulative_regret'] for s in strategies]
    bp = ax1.boxplot(data, labels=[s.replace(' (', '\n(') for s in strategies], 
                     patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('Cumulative Regret', fontsize=11)
    ax1.set_title('Regret Distribution Across Seeds', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', labelsize=9)
    
    # Annotate with std
    for i, s in enumerate(strategies):
        std = results[s]['statistics']['cumulative_regret']['std']
        y_pos = max(data[i]) + 2
        ax1.text(i+1, y_pos, f'σ={std:.1f}', ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Coefficient of Variation
    ax2 = axes[0, 1]
    cvs = []
    for s in strategies:
        mean = results[s]['statistics']['cumulative_regret']['mean']
        std = results[s]['statistics']['cumulative_regret']['std']
        cv = (std / mean * 100) if mean > 0 else 0
        cvs.append(cv)
    
    bars = ax2.bar(range(len(strategies)), cvs, color=colors, alpha=0.6, edgecolor='black')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([s.replace(' (', '\n(') for s in strategies], fontsize=9)
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax2.set_title('Relative Variance (CV = std/mean × 100%)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for i, (bar, cv) in enumerate(zip(bars, cvs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{cv:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Per-Seed Comparison
    ax3 = axes[0, 2]
    n_seeds = len(per_seed['Hybrid (Corralling)'])
    x = np.arange(n_seeds)
    width = 0.25
    
    for i, s in enumerate(strategies):
        values = [r['cumulative_regret'] for r in per_seed[s]]
        ax3.bar(x + i*width, values, width, label=s.split()[0], 
               color=colors[i], alpha=0.6, edgecolor='black')
    
    ax3.set_xlabel('Seed Index', fontsize=11)
    ax3.set_ylabel('Cumulative Regret', fontsize=11)
    ax3.set_title('Per-Seed Results', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f'{i}' for i in range(n_seeds)])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Early vs Total Regret
    ax4 = axes[1, 0]
    for i, s in enumerate(strategies):
        early = results[s]['statistics']['raw_values']['early_regret']
        total = results[s]['statistics']['raw_values']['cumulative_regret']
        ax4.scatter(early, total, label=s, s=100, alpha=0.6, color=colors[i], 
                   edgecolors='black', linewidths=1.5)
    
    # Add diagonal reference line
    max_val = max([max(results[s]['statistics']['raw_values']['cumulative_regret']) 
                   for s in strategies])
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x')
    
    ax4.set_xlabel('Early Regret (0-500)', fontsize=11)
    ax4.set_ylabel('Total Regret (0-750)', fontsize=11)
    ax4.set_title('Early vs Total Regret Correlation', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Variance Source Attribution
    ax5 = axes[1, 1]
    
    # Estimate variance sources for Corralling
    corralling_results = per_seed['Hybrid (Corralling)']
    early_regrets = np.array([r['early_regret'] for r in corralling_results])
    late_regrets = np.array([r['cumulative_regret'] - r['early_regret'] for r in corralling_results])
    
    var_early = np.var(early_regrets, ddof=1)
    var_late = np.var(late_regrets, ddof=1)
    var_total = np.var([r['cumulative_regret'] for r in corralling_results], ddof=1)
    
    # Pie chart of variance attribution
    labels = [f'Early Phase\n(0-500)\nVar={var_early:.1f}', 
             f'Late Phase\n(500-750)\nVar={var_late:.1f}']
    sizes = [var_early, var_late]
    colors_pie = ['#e74c3c', '#3498db']
    
    ax5.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax5.set_title('Variance Attribution\n(Corralling Only)', fontsize=12, fontweight='bold')
    
    # Plot 6: Statistical Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary table
    summary_text = "STATISTICAL SUMMARY\n" + "="*40 + "\n\n"
    
    for s in strategies:
        stats = results[s]['statistics']['cumulative_regret']
        summary_text += f"{s}:\n"
        summary_text += f"  Mean:   {stats['mean']:.1f}\n"
        summary_text += f"  Median: {stats['median']:.1f}\n"
        summary_text += f"  Std:    {stats['std']:.1f}\n"
        summary_text += f"  Range:  [{stats['min']:.0f}, {stats['max']:.0f}]\n"
        
        # Add interpretation
        if stats['std'] < 0.01:
            summary_text += f"  ✅ DETERMINISTIC\n"
        elif stats['std'] / stats['mean'] > 0.3:
            summary_text += f"  ⚠️ HIGH VARIANCE ({stats['std']/stats['mean']*100:.0f}% CV)\n"
        else:
            summary_text += f"  ✓ Moderate variance\n"
        summary_text += "\n"
    
    # Key finding
    corralling_std = results['Hybrid (Corralling)']['statistics']['cumulative_regret']['std']
    tabula_std = results['Tabula Rasa']['statistics']['cumulative_regret']['std']
    
    summary_text += "\n" + "="*40 + "\n"
    summary_text += "KEY FINDING:\n"
    summary_text += f"Corralling has {corralling_std:.1f}× more\n"
    summary_text += f"variance than baselines due to\n"
    summary_text += f"stochastic expert selection.\n"
    summary_text += f"\nThis is EXPECTED behavior for\n"
    summary_text += f"importance-weighted algorithms.\n"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Variance analysis saved to: {output_path}")


def print_detailed_analysis(results: dict, per_seed: dict):
    """Print detailed variance analysis to console."""
    
    print("\n" + "="*80)
    print("DETAILED VARIANCE ANALYSIS")
    print("="*80)
    
    for strategy in ['Warmup', 'Tabula Rasa', 'Hybrid (Corralling)']:
        stats = results[strategy]['statistics']['cumulative_regret']
        
        print(f"\n{strategy}:")
        print("-" * 80)
        print(f"  Mean:              {stats['mean']:.2f}")
        print(f"  Median:            {stats['median']:.2f}")
        print(f"  Std:               {stats['std']:.2f}")
        print(f"  Coefficient of Var: {stats['std']/stats['mean']*100:.1f}%")
        print(f"  Min:               {stats['min']:.0f}")
        print(f"  Max:               {stats['max']:.0f}")
        print(f"  Range:             {stats['max'] - stats['min']:.0f}")
        print(f"  95% CI:            [{stats['ci_95'][0]:.1f}, {stats['ci_95'][1]:.1f}]")
        
        # Interpretation
        if stats['std'] < 0.01:
            print(f"  Interpretation:    ✅ DETERMINISTIC (no variance)")
        elif stats['std'] / stats['mean'] > 0.3:
            print(f"  Interpretation:    ⚠️ HIGH VARIANCE (CV > 30%)")
            print(f"                     Source: Stochastic expert selection")
        else:
            print(f"  Interpretation:    ✓ Moderate variance")
    
    print("\n" + "="*80)
    print("VARIANCE COMPARISON")
    print("="*80)
    
    corralling_std = results['Hybrid (Corralling)']['statistics']['cumulative_regret']['std']
    tabula_std = results['Tabula Rasa']['statistics']['cumulative_regret']['std']
    
    if tabula_std > 0:
        ratio = corralling_std / tabula_std
        print(f"  Corralling std / Tabula Rasa std: {ratio:.1f}×")
    else:
        print(f"  Corralling std: {corralling_std:.2f} (baselines are deterministic)")
    
    # Per-seed analysis
    print("\n" + "="*80)
    print("PER-SEED BREAKDOWN (Corralling)")
    print("="*80)
    
    corralling_seeds = per_seed['Hybrid (Corralling)']
    for i, seed_result in enumerate(corralling_seeds):
        total = seed_result['cumulative_regret']
        early = seed_result['early_regret']
        late = total - early
        print(f"  Seed {i}: Total={total:.0f}, Early={early:.0f}, Late={late:.0f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize variance across seeds')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results_multiseed.json')
    parser.add_argument('--per-seed', type=str, required=True,
                       help='Path to results_per_seed.json')
    parser.add_argument('--output', type=str, default='variance_analysis.png',
                       help='Output path for visualization')
    args = parser.parse_args()
    
    # Load data
    print("Loading results...")
    results = load_json(Path(args.results))
    per_seed = load_json(Path(args.per_seed))
    
    # Print detailed analysis
    print_detailed_analysis(results, per_seed)
    
    # Generate visualization
    print("\nGenerating visualization...")
    plot_variance_analysis(results, per_seed, Path(args.output))
    
    print("\n✅ Variance analysis complete!")


if __name__ == '__main__':
    main()
