#!/usr/bin/env python3
"""
Plot Pareto curve from custom weights experiment.

Creates visualization showing:
1. Individual model cost vs quality
2. Router performance with different weight profiles
3. Pareto frontier
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6)
})


def load_results() -> Dict:
    """Load experiment results from JSON."""
    results_path = Path(__file__).parent / "results" / "custom_weights_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results not found at {results_path}. "
            f"Run run_custom_weights.py first."
        )
    
    with open(results_path) as f:
        return json.load(f)


def plot_pareto_curve(results: Dict, output_path: Path):
    """
    Create Pareto curve visualization.
    
    Args:
        results: Experiment results dict
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract data
    model_baselines = results["model_baselines"]
    profile_results = results["profile_results"]
    
    # Plot individual models (baseline points)
    model_costs = [m["cost"] * 1000 for m in model_baselines]  # Convert to per-1M
    model_qualities = [m["quality"] * 100 for m in model_baselines]  # Convert to percentage
    model_names = [m["model_id"].split("/")[-1] for m in model_baselines]
    
    ax.scatter(model_costs, model_qualities, 
              s=150, alpha=0.6, c='steelblue', 
              marker='o', edgecolors='navy', linewidth=1.5,
              label='Individual Models', zorder=2)
    
    # Annotate model names
    for cost, quality, name in zip(model_costs, model_qualities, model_names):
        ax.annotate(name, (cost, quality), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    # Plot router profiles
    profile_colors = {
        "Cost Saver": "#2ecc71",      # Green
        "High Quality": "#e74c3c",    # Red
        "Balanced": "#f39c12"          # Orange
    }
    
    profile_markers = {
        "Cost Saver": "^",
        "High Quality": "s",
        "Balanced": "D"
    }
    
    for result in profile_results:
        name = result["profile_name"]
        cost = result["avg_cost"] * 1000  # Convert to per-1M
        quality = result["avg_quality"] * 100  # Convert to percentage
        
        ax.scatter(cost, quality,
                  s=250, alpha=0.9,
                  c=profile_colors.get(name, 'purple'),
                  marker=profile_markers.get(name, '*'),
                  edgecolors='black', linewidth=2,
                  label=f'Router: {name}',
                  zorder=3)
        
        # Annotate with weights
        w_q = result["weights"]["w_q"]
        w_c = result["weights"]["w_c"]
        weight_text = f"w_q={w_q:.1f}\nw_c={w_c:.1f}"
        ax.annotate(weight_text, (cost, quality),
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=8, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=profile_colors.get(name, 'white'),
                            alpha=0.3))
    
    # Compute and plot Pareto frontier
    pareto_points = compute_pareto_frontier(model_baselines)
    if len(pareto_points) > 1:
        pareto_costs = [p["cost"] * 1000 for p in pareto_points]
        pareto_qualities = [p["quality"] * 100 for p in pareto_points]
        ax.plot(pareto_costs, pareto_qualities,
               'r--', alpha=0.4, linewidth=2,
               label='Pareto Frontier', zorder=1)
    
    # Styling
    ax.set_xlabel('Cost ($ per 1M tokens)', fontweight='bold')
    ax.set_ylabel('Quality (Success Rate %)', fontweight='bold')
    ax.set_title('Cost-Quality Tradeoff: Custom Weight Profiles\n'
                'Demonstrating how w_q and w_c control model selection',
                fontweight='bold', pad=20)
    
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation explaining the experiment
    textstr = (
        f"Experiment: Custom Weights Pareto\n"
        f"Test Samples: {results['n_test_samples']}\n"
        f"Router Mode: Greedy (α=0.0)"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Pareto curve saved to: {output_path}")
    
    return fig, ax


def compute_pareto_frontier(model_baselines: List[Dict]) -> List[Dict]:
    """
    Compute Pareto frontier from model baselines.
    
    A point is on the Pareto frontier if no other point has both
    lower cost AND higher quality.
    
    Args:
        model_baselines: List of model baseline dicts
        
    Returns:
        List of Pareto-optimal points, sorted by cost
    """
    pareto_points = []
    
    for candidate in model_baselines:
        is_dominated = False
        
        for other in model_baselines:
            if candidate == other:
                continue
            
            # Check if 'other' dominates 'candidate'
            # Dominates if: lower cost AND higher quality
            if (other["cost"] <= candidate["cost"] and 
                other["quality"] > candidate["quality"]):
                # Strict domination on quality with equal or better cost
                is_dominated = True
                break
            elif (other["cost"] < candidate["cost"] and 
                  other["quality"] >= candidate["quality"]):
                # Strict domination on cost with equal or better quality
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_points.append(candidate)
    
    # Sort by cost
    pareto_points.sort(key=lambda x: x["cost"])
    
    return pareto_points


def plot_selection_distribution(results: Dict, output_path: Path):
    """
    Create bar chart showing model selection distribution for each profile.
    
    Args:
        results: Experiment results dict
        output_path: Path to save figure
    """
    profile_results = results["profile_results"]
    
    fig, axes = plt.subplots(1, len(profile_results), 
                            figsize=(15, 5), sharey=True)
    
    if len(profile_results) == 1:
        axes = [axes]
    
    for ax, result in zip(axes, profile_results):
        selections = result["model_selections"]
        
        # Sort by count
        sorted_models = sorted(selections.items(), 
                             key=lambda x: x[1], reverse=True)
        
        models = [m.split("/")[-1] for m, _ in sorted_models]
        counts = [c for _, c in sorted_models]
        percentages = [100 * c / sum(counts) for c in counts]
        
        # Create bar chart
        bars = ax.barh(models, percentages, color='steelblue', alpha=0.7)
        
        # Add value labels
        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%',
                   ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Selection Frequency (%)')
        ax.set_title(f'{result["profile_name"]}\n'
                    f'(w_q={result["weights"]["w_q"]:.1f}, '
                    f'w_c={result["weights"]["w_c"]:.1f})',
                    fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    axes[0].set_ylabel('Model')
    
    plt.suptitle('Model Selection Distribution by Weight Profile',
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Selection distribution saved to: {output_path}")
    
    return fig, axes


def create_summary_table(results: Dict):
    """
    Print formatted summary table of results.
    
    Args:
        results: Experiment results dict
    """
    print("\n" + "="*80)
    print("SUMMARY TABLE: CUSTOM WEIGHT PROFILES")
    print("="*80)
    
    # Header
    print(f"\n{'Profile':<20} {'w_q':<6} {'w_c':<6} {'Cost ($/1M)':<15} "
          f"{'Quality':<10} {'Success Rate':<15}")
    print("-" * 80)
    
    # Profile results
    for result in results["profile_results"]:
        cost = result["avg_cost"] * 1000  # Convert to per-1M
        quality = result["avg_quality"] * 100
        success = result["success_rate"] * 100
        w_q = result["weights"]["w_q"]
        w_c = result["weights"]["w_c"]
        
        print(f"{result['profile_name']:<20} "
              f"{w_q:<6.1f} {w_c:<6.1f} "
              f"${cost:<14.5f} "
              f"{quality:<9.2f}% "
              f"{success:<14.2f}%")
    
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL BASELINES")
    print("="*80)
    
    # Model baselines
    print(f"\n{'Model':<30} {'Cost ($/1M)':<15} {'Quality':<10} {'Success Rate':<15}")
    print("-" * 80)
    
    for model in results["model_baselines"]:
        cost = model["cost"] * 1000
        quality = model["quality"] * 100
        success = model["success_rate"] * 100
        name = model["model_id"]
        
        print(f"{name:<30} ${cost:<14.5f} {quality:<9.2f}% {success:<14.2f}%")
    
    print("\n" + "="*80)


def main():
    """Generate all visualizations."""
    print("="*70)
    print("PLOTTING CUSTOM WEIGHTS PARETO EXPERIMENT")
    print("="*70)
    
    # Load results
    print("\n📊 Loading results...")
    results = load_results()
    print(f"  ✓ Loaded results for {len(results['profile_results'])} profiles")
    print(f"  ✓ {len(results['model_baselines'])} model baselines")
    
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\n📈 Generating visualizations...")
    
    # 1. Pareto curve
    pareto_path = output_dir / "pareto_curve.png"
    plot_pareto_curve(results, pareto_path)
    
    # 2. Selection distribution
    dist_path = output_dir / "selection_distribution.png"
    plot_selection_distribution(results, dist_path)
    
    # 3. Print summary table
    create_summary_table(results)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

