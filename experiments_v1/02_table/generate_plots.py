#!/usr/bin/env python3
"""
Generate plots for Table 2: The Performance Gap
Visualizes the comparison between η=1.0 (aggressive) and η=0.1 (conservative).

This script creates:
1. Hybrid comparison plots (cumulative regret & average reward over time)
2. Expert weights evolution plots (showing adaptation)
3. Comparative analysis across different learning rates

Usage:
    python generate_plots.py
    python generate_plots.py --output results/custom_output
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def load_results(results_path: Path) -> Dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_hybrid_comparison(results_eta_01: Dict, results_eta_10: Dict, output_dir: Path):
    """
    Generate comparison plot showing cumulative regret for both learning rates.
    
    Args:
        results_eta_01: Results dictionary for η=0.1
        results_eta_10: Results dictionary for η=1.0
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Define colors for consistency
    colors = {
        'Warmup': '#d62728',  # Red
        'Tabula Rasa': '#2ca02c',  # Green
        'Hybrid η=0.1': '#ff7f0e',  # Orange
        'Hybrid η=1.0': '#1f77b4',  # Blue
    }
    
    # Left plot: Cumulative Regret comparison
    ax = axes[0]
    
    # Plot baselines (same for both)
    warmup_regret = results_eta_01['Warmup']['cumulative_regret']
    tr_regret = results_eta_01['Tabula Rasa']['cumulative_regret']
    hybrid_01_regret = results_eta_01['Hybrid (Corralling)']['cumulative_regret']
    hybrid_10_regret = results_eta_10['Hybrid (Corralling)']['cumulative_regret']
    
    strategies = ['Warmup', 'Tabula Rasa\n(Optimal)', 'Hybrid\nη=0.1', 'Hybrid\nη=1.0']
    regrets = [warmup_regret, tr_regret, hybrid_01_regret, hybrid_10_regret]
    bar_colors = [colors['Warmup'], colors['Tabula Rasa'], 
                  colors['Hybrid η=0.1'], colors['Hybrid η=1.0']]
    
    bars = ax.bar(strategies, regrets, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, regret in zip(bars, regrets):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{regret:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add multiplier annotations
    ax.text(1, tr_regret + 10, '1.00×', ha='center', fontsize=9, style='italic')
    ax.text(2, hybrid_01_regret + 10, f'{hybrid_01_regret/tr_regret:.2f}×', 
            ha='center', fontsize=9, style='italic')
    ax.text(3, hybrid_10_regret + 10, f'{hybrid_10_regret/tr_regret:.2f}×', 
            ha='center', fontsize=9, style='italic', color='blue', fontweight='bold')
    
    ax.set_ylabel('Cumulative Regret', fontsize=12, fontweight='bold')
    ax.set_title('Table 2: The Performance Gap', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add horizontal line at optimal
    ax.axhline(y=tr_regret, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Optimal')
    
    # Right plot: Improvement breakdown
    ax = axes[1]
    
    # Calculate improvements
    improvement_vs_warmup_01 = ((warmup_regret - hybrid_01_regret) / warmup_regret) * 100
    improvement_vs_warmup_10 = ((warmup_regret - hybrid_10_regret) / warmup_regret) * 100
    improvement_eta_tuning = ((hybrid_01_regret - hybrid_10_regret) / hybrid_01_regret) * 100
    
    improvements = [improvement_vs_warmup_01, improvement_vs_warmup_10, improvement_eta_tuning]
    labels = ['η=0.1 vs\nWarmup', 'η=1.0 vs\nWarmup', 'η=1.0 vs\nη=0.1']
    imp_colors = [colors['Hybrid η=0.1'], colors['Hybrid η=1.0'], '#9467bd']  # Purple for tuning
    
    bars = ax.bar(labels, improvements, color=imp_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Regret Reduction Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_gap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance gap comparison saved to {output_dir}/performance_gap_comparison.png")


def plot_learning_rate_sensitivity(results_eta_01: Dict, results_eta_10: Dict, output_dir: Path):
    """
    Plot showing how learning rate affects final performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data points
    learning_rates = [0.1, 1.0]
    hybrid_regrets = [
        results_eta_01['Hybrid (Corralling)']['cumulative_regret'],
        results_eta_10['Hybrid (Corralling)']['cumulative_regret']
    ]
    optimal_regret = results_eta_01['Tabula Rasa']['cumulative_regret']
    warmup_regret = results_eta_01['Warmup']['cumulative_regret']
    
    # Plot lines
    ax.axhline(y=optimal_regret, color='green', linestyle='--', linewidth=2, 
               label=f'Optimal (Tabula Rasa): {optimal_regret:.0f}', alpha=0.7)
    ax.axhline(y=warmup_regret, color='red', linestyle='--', linewidth=2, 
               label=f'Warmup (Harmful): {warmup_regret:.0f}', alpha=0.7)
    
    # Plot hybrid performance
    ax.plot(learning_rates, hybrid_regrets, 'o-', color='#1f77b4', linewidth=3, 
            markersize=12, label='Hybrid (Corralling)', markeredgecolor='black', markeredgewidth=2)
    
    # Annotate points
    for lr, regret in zip(learning_rates, hybrid_regrets):
        multiplier = regret / optimal_regret
        ax.annotate(f'η={lr}\n{regret:.0f} regret\n({multiplier:.2f}× optimal)',
                   xy=(lr, regret),
                   xytext=(lr, regret + 8),
                   ha='center',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Learning Rate (η)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Learning rate sensitivity plot saved to {output_dir}/learning_rate_sensitivity.png")


def plot_model_usage_comparison(results_eta_01: Dict, results_eta_10: Dict, output_dir: Path):
    """
    Compare model usage patterns across strategies.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract model usage percentages
    strategies = ['Warmup', 'Tabula Rasa\n(Optimal)', 'Hybrid\nη=0.1', 'Hybrid\nη=1.0']
    
    def get_gpt4_pct(result_dict, strategy_name):
        usage = result_dict[strategy_name]['model_usage']
        total = sum(usage.values())
        gpt4 = usage.get('openai/gpt-4-turbo', 0)
        return (gpt4 / total) * 100 if total > 0 else 0
    
    gpt4_percentages = [
        get_gpt4_pct(results_eta_01, 'Warmup'),
        get_gpt4_pct(results_eta_01, 'Tabula Rasa'),
        get_gpt4_pct(results_eta_01, 'Hybrid (Corralling)'),
        get_gpt4_pct(results_eta_10, 'Hybrid (Corralling)')
    ]
    
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    bars = ax.bar(strategies, gpt4_percentages, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, pct in zip(bars, gpt4_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add optimal line
    optimal_pct = gpt4_percentages[1]
    ax.axhline(y=optimal_pct, color='green', linestyle='--', linewidth=2, 
               alpha=0.5, label=f'Optimal: {optimal_pct:.1f}%')
    
    ax.set_ylabel('GPT-4-Turbo Usage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Selection Patterns', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_usage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model usage comparison saved to {output_dir}/model_usage_comparison.png")


def generate_summary_figure(results_eta_01: Dict, results_eta_10: Dict, output_dir: Path):
    """
    Generate a comprehensive summary figure with all key metrics.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    warmup_regret = results_eta_01['Warmup']['cumulative_regret']
    tr_regret = results_eta_01['Tabula Rasa']['cumulative_regret']
    hybrid_01_regret = results_eta_01['Hybrid (Corralling)']['cumulative_regret']
    hybrid_10_regret = results_eta_10['Hybrid (Corralling)']['cumulative_regret']
    
    # Subplot 1: Main comparison (larger)
    ax1 = fig.add_subplot(gs[0, :2])
    strategies = ['Warmup\n(Harmful)', 'Tabula Rasa\n(Optimal)', 'Hybrid\nη=0.1', 'Hybrid\nη=1.0']
    regrets = [warmup_regret, tr_regret, hybrid_01_regret, hybrid_10_regret]
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    
    bars = ax1.bar(strategies, regrets, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, regret in zip(bars, regrets):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{regret:.0f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax1.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
    ax1.set_title('Table 2: The Performance Gap (η=1.0 achieves 1.26× optimal)', 
                  fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Subplot 2: Multipliers
    ax2 = fig.add_subplot(gs[0, 2])
    multipliers = [r/tr_regret for r in regrets]
    bars = ax2.bar(range(4), multipliers, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, mult in zip(bars, multipliers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mult:.2f}×',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['W', 'TR', 'η=0.1', 'η=1.0'], fontsize=10)
    ax2.set_ylabel('vs Optimal', fontsize=11, fontweight='bold')
    ax2.set_title('Multipliers', fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Improvement breakdown
    ax3 = fig.add_subplot(gs[1, 0])
    improvements = [
        ((warmup_regret - hybrid_01_regret) / warmup_regret) * 100,
        ((warmup_regret - hybrid_10_regret) / warmup_regret) * 100
    ]
    labels = ['η=0.1\nvs Warmup', 'η=1.0\nvs Warmup']
    bars = ax3.bar(labels, improvements, color=['#ff7f0e', '#1f77b4'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Safety vs Warmup Failure', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Learning rate impact
    ax4 = fig.add_subplot(gs[1, 1])
    eta_improvement = ((hybrid_01_regret - hybrid_10_regret) / hybrid_01_regret) * 100
    bars = ax4.bar(['η=1.0 vs η=0.1'], [eta_improvement], color='#9467bd', 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.text(0, eta_improvement, f'{eta_improvement:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Benefit of Aggressive Learning', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(50, eta_improvement * 1.2))
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Subplot 5: Model usage
    ax5 = fig.add_subplot(gs[1, 2])
    def get_gpt4_pct(result_dict, strategy_name):
        usage = result_dict[strategy_name]['model_usage']
        total = sum(usage.values())
        gpt4 = usage.get('openai/gpt-4-turbo', 0)
        return (gpt4 / total) * 100 if total > 0 else 0
    
    gpt4_pcts = [
        get_gpt4_pct(results_eta_01, 'Tabula Rasa'),
        get_gpt4_pct(results_eta_01, 'Hybrid (Corralling)'),
        get_gpt4_pct(results_eta_10, 'Hybrid (Corralling)')
    ]
    labels = ['Optimal', 'η=0.1', 'η=1.0']
    bars = ax5.bar(labels, gpt4_pcts, color=['#2ca02c', '#ff7f0e', '#1f77b4'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, pct in zip(bars, gpt4_pcts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax5.set_ylabel('GPT-4 Usage (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Near-Optimal Selection', fontsize=12, fontweight='bold')
    ax5.set_ylim(60, 75)
    ax5.axhline(y=gpt4_pcts[0], color='green', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / 'table_2_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary figure saved to {output_dir}/table_2_summary.png")


def main():
    parser = argparse.ArgumentParser(description='Generate plots for Table 2')
    parser.add_argument('--output', type=str, default='results', help='Output directory for plots')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output
    
    print("="*80)
    print("GENERATING PLOTS FOR TABLE 2: THE PERFORMANCE GAP")
    print("="*80)
    print()
    
    # Load data
    print("📊 Loading data...")
    eta_01_path = script_dir / 'data' / 'results.json'
    eta_10_path = script_dir / 'data' / 'eta_1.0' / 'results.json'
    
    if not eta_01_path.exists():
        print(f"❌ Error: {eta_01_path} not found")
        return
    if not eta_10_path.exists():
        print(f"❌ Error: {eta_10_path} not found")
        return
    
    results_eta_01 = load_results(eta_01_path)
    results_eta_10 = load_results(eta_10_path)
    print("✓ Data loaded successfully")
    print()
    
    # Generate plots
    print("📈 Generating plots...")
    print()
    
    plot_hybrid_comparison(results_eta_01, results_eta_10, output_dir)
    plot_learning_rate_sensitivity(results_eta_01, results_eta_10, output_dir)
    plot_model_usage_comparison(results_eta_01, results_eta_10, output_dir)
    generate_summary_figure(results_eta_01, results_eta_10, output_dir)
    
    print()
    print("="*80)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    print("  - performance_gap_comparison.png")
    print("  - learning_rate_sensitivity.png")
    print("  - model_usage_comparison.png")
    print("  - table_2_summary.png")
    print()


if __name__ == '__main__':
    main()

