#!/usr/bin/env python3
"""
Visualization for Experiment 06: Prior Strength Sensitivity Analysis

Generates publication-quality plot showing the relationship between
prior strength and cumulative regret.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# KDD publication aesthetics
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def load_results():
    """Load sensitivity analysis results."""
    results_path = Path(__file__).parent / "results" / "sensitivity_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            "Please run 'python run_sensitivity.py' first."
        )
    
    with open(results_path) as f:
        data = json.load(f)
    
    return data


def plot_sensitivity_analysis(data, output_dir):
    """
    Generate publication-quality sensitivity analysis plot.
    
    Shows how cumulative regret varies with prior strength,
    with annotations for key behavioral regions.
    """
    results = data["results"]
    
    # Extract data
    prior_values = [r["prior_n_effective"] for r in results]
    regret_means = [r["regret_mean"] for r in results]
    regret_stds = [r["regret_std"] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Main line plot with error bars
    ax.errorbar(
        prior_values, regret_means, yerr=regret_stds,
        marker='o', markersize=8, linewidth=2, capsize=5,
        color='#2E86AB', label='BanditGPT', zorder=3
    )
    
    # Highlight the default value (N=10)
    default_idx = prior_values.index(10.0)
    ax.axvline(
        x=10.0, color='#A23B72', linestyle='--', linewidth=1.5,
        alpha=0.7, label='Default (N=10)', zorder=2
    )
    ax.plot(
        10.0, regret_means[default_idx],
        marker='*', markersize=15, color='#A23B72', zorder=4
    )
    
    # Add behavioral region annotations
    # Find the minimum regret to identify the sweet spot
    min_regret_idx = np.argmin(regret_means)
    min_regret_n = prior_values[min_regret_idx]
    min_regret_value = regret_means[min_regret_idx]
    
    # Annotation for cold start
    if 0 in prior_values:
        cold_idx = prior_values.index(0)
        ax.annotate(
            'Cold Start\n(No Priors)',
            xy=(0, regret_means[cold_idx]),
            xytext=(15, regret_means[cold_idx] + 5),
            fontsize=9, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
        )
    
    # Annotation for sweet spot
    ax.annotate(
        f'Sweet Spot\n(N={min_regret_n:.0f})',
        xy=(min_regret_n, min_regret_value),
        xytext=(min_regret_n - 20, min_regret_value - 8),
        fontsize=9, ha='center',
        arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='green', alpha=0.7)
    )
    
    # Annotation for over-reliance (if applicable)
    if max(prior_values) >= 100:
        high_idx = prior_values.index(max([n for n in prior_values if n >= 100]))
        high_n = prior_values[high_idx]
        high_regret = regret_means[high_idx]
        
        ax.annotate(
            'Strong Prior\n(Over-reliance)',
            xy=(high_n, high_regret),
            xytext=(high_n - 30, high_regret + 5),
            fontsize=9, ha='right',
            arrowprops=dict(arrowstyle='->', color='orange', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', edgecolor='orange', alpha=0.8)
        )
    
    # Labels and title
    ax.set_xlabel('Prior Strength (Effective Samples)', fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontweight='bold')
    ax.set_title(
        'Sensitivity Analysis: Prior Strength vs. Convergence Performance',
        fontweight='bold', pad=15
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    pdf_path = output_dir / "fig6_sensitivity_analysis.pdf"
    png_path = output_dir / "fig6_sensitivity_analysis.png"
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    
    print(f"✅ Plot saved:")
    print(f"   PDF: {pdf_path}")
    print(f"   PNG: {png_path}")
    
    return fig


def print_summary(data):
    """Print numerical summary of results."""
    results = data["results"]
    metadata = data["metadata"]
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset:")
    print(f"   Training prompts: {metadata['n_train_prompts']}")
    print(f"   Test prompts: {metadata['n_test_prompts']}")
    print(f"   Models: {metadata['n_models']}")
    print(f"   Trials per value: {metadata['n_trials']}")
    
    print(f"\n📈 Results:")
    for r in results:
        marker = "  ← DEFAULT" if r["prior_n_effective"] == 10.0 else ""
        cv = (r["regret_std"] / r["regret_mean"] * 100) if r["regret_mean"] > 0 else 0
        print(f"   N={r['prior_n_effective']:5.0f} → "
              f"Regret={r['regret_mean']:6.2f} ± {r['regret_std']:5.2f} "
              f"(CV={cv:4.1f}%){marker}")
    
    # Find optimal value
    optimal_idx = np.argmin([r["regret_mean"] for r in results])
    optimal_n = results[optimal_idx]["prior_n_effective"]
    optimal_regret = results[optimal_idx]["regret_mean"]
    
    print(f"\n🎯 Optimal Prior Strength: N={optimal_n:.0f}")
    print(f"   Minimum Regret: {optimal_regret:.2f}")
    
    # Compare to default
    default_result = next((r for r in results if r["prior_n_effective"] == 10.0), None)
    if default_result:
        default_regret = default_result["regret_mean"]
        delta = ((default_regret - optimal_regret) / optimal_regret * 100)
        if abs(delta) < 5:
            print(f"   ✅ Default (N=10) is within 5% of optimal ({delta:+.1f}%)")
        else:
            print(f"   ⚠️  Default (N=10) differs from optimal by {delta:+.1f}%")


def main():
    """Generate sensitivity analysis visualization."""
    print("="*70)
    print("VISUALIZING SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Load results
    print("\n📂 Loading results...")
    data = load_results()
    print(f"   ✓ Loaded {len(data['results'])} prior strength values")
    
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Generate plot
    print("\n📊 Generating plot...")
    plot_sensitivity_analysis(data, output_dir)
    
    # Print summary
    print_summary(data)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
