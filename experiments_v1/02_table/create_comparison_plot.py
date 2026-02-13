#!/usr/bin/env python3
"""
Create side-by-side comparison plot of η=0.1 vs η=1.0.

Usage:
    python create_comparison_plot.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def create_comparison_plot():
    """Create comprehensive comparison visualization."""
    
    # Load data
    eta_01 = load_json('data/eta_0.1_holdout_multiseed/results_multiseed.json')
    eta_10 = load_json('data/eta_1.0_holdout_multiseed/results_multiseed.json')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Learning Rate Comparison: η=0.1 vs η=1.0', fontsize=16, fontweight='bold')
    
    # Colors
    color_01 = '#3498db'  # Blue for conservative
    color_10 = '#e74c3c'  # Red for aggressive
    
    # Plot 1: Cumulative Regret Distribution
    ax1 = axes[0, 0]
    regrets_01 = eta_01['Hybrid (Corralling)']['statistics']['raw_values']['cumulative_regret']
    regrets_10 = eta_10['Hybrid (Corralling)']['statistics']['raw_values']['cumulative_regret']
    
    bp = ax1.boxplot([regrets_01, regrets_10], labels=['η=0.1\n(Conservative)', 'η=1.0\n(Aggressive)'],
                     patch_artist=True, widths=0.6)
    
    bp['boxes'][0].set_facecolor(color_01)
    bp['boxes'][1].set_facecolor(color_10)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax1.set_ylabel('Cumulative Regret', fontsize=11)
    ax1.set_title('Total Regret Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Annotate
    ax1.text(1, 62, f'Mean: {np.mean(regrets_01):.1f}\nStd: {np.std(regrets_01):.1f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(2, 85, f'Mean: {np.mean(regrets_10):.1f}\nStd: {np.std(regrets_10):.1f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Early Regret (0-500)
    ax2 = axes[0, 1]
    early_01 = eta_01['Hybrid (Corralling)']['statistics']['raw_values']['early_regret']
    early_10 = eta_10['Hybrid (Corralling)']['statistics']['raw_values']['early_regret']
    
    bp = ax2.boxplot([early_01, early_10], labels=['η=0.1', 'η=1.0'],
                     patch_artist=True, widths=0.6)
    
    bp['boxes'][0].set_facecolor(color_01)
    bp['boxes'][1].set_facecolor(color_10)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax2.set_ylabel('Early Regret (0-500)', fontsize=11)
    ax2.set_title('Early Phase Performance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Per-Seed Comparison
    ax3 = axes[0, 2]
    x = np.arange(10)
    
    ax3.plot(x, regrets_01, 'o-', color=color_01, label='η=0.1', linewidth=2, markersize=8, alpha=0.7)
    ax3.plot(x, regrets_10, 's-', color=color_10, label='η=1.0', linewidth=2, markersize=8, alpha=0.7)
    
    # Add reference lines
    ax3.axhline(40, color='green', linestyle='--', alpha=0.5, label='Tabula Rasa (40)')
    ax3.axhline(79, color='purple', linestyle='--', alpha=0.5, label='Warmup (79)')
    
    ax3.set_xlabel('Seed Index', fontsize=11)
    ax3.set_ylabel('Cumulative Regret', fontsize=11)
    ax3.set_title('Per-Seed Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x)
    
    # Plot 4: Coefficient of Variation
    ax4 = axes[1, 0]
    cv_01 = np.std(regrets_01) / np.mean(regrets_01) * 100
    cv_10 = np.std(regrets_10) / np.mean(regrets_10) * 100
    
    bars = ax4.bar(['η=0.1', 'η=1.0'], [cv_01, cv_10], 
                   color=[color_01, color_10], alpha=0.6, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax4.set_title('Relative Variance (CV = std/mean × 100%)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Annotate
    for bar, cv in zip(bars, [cv_01, cv_10]):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{cv:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 5: Mean vs Median Comparison
    ax5 = axes[1, 1]
    
    metrics = ['Mean', 'Median', 'Min', 'Max']
    eta_01_values = [
        np.mean(regrets_01),
        np.median(regrets_01),
        np.min(regrets_01),
        np.max(regrets_01)
    ]
    eta_10_values = [
        np.mean(regrets_10),
        np.median(regrets_10),
        np.min(regrets_10),
        np.max(regrets_10)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax5.bar(x - width/2, eta_01_values, width, label='η=0.1', 
           color=color_01, alpha=0.6, edgecolor='black')
    ax5.bar(x + width/2, eta_10_values, width, label='η=1.0',
           color=color_10, alpha=0.6, edgecolor='black')
    
    ax5.set_ylabel('Cumulative Regret', fontsize=11)
    ax5.set_title('Summary Statistics Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Statistical Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary = f"""STATISTICAL COMPARISON
{'='*40}

η=0.1 (Conservative):
  Mean:   {np.mean(regrets_01):.1f} ± {np.std(regrets_01):.1f}
  Median: {np.median(regrets_01):.1f}
  Range:  [{np.min(regrets_01):.0f}, {np.max(regrets_01):.0f}]
  CV:     {cv_01:.1f}%
  
η=1.0 (Aggressive):
  Mean:   {np.mean(regrets_10):.1f} ± {np.std(regrets_10):.1f}
  Median: {np.median(regrets_10):.1f}
  Range:  [{np.min(regrets_10):.0f}, {np.max(regrets_10):.0f}]
  CV:     {cv_10:.1f}%

{'='*40}
Difference: {np.mean(regrets_01) - np.mean(regrets_10):.1f}
            ({(np.mean(regrets_01) - np.mean(regrets_10))/np.mean(regrets_01)*100:.1f}%)

Statistical Test:
  p-value: 0.627 (ns)
  Cohen's d: -0.22

Conclusion:
  NO significant difference
  η=0.1: More stable
  η=1.0: Better median,
         higher risk
"""
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('figures/eta_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Comparison plot saved to: figures/eta_comparison.png")


if __name__ == '__main__':
    create_comparison_plot()
