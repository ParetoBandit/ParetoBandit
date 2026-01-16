#!/usr/bin/env python3
"""
Plot Pareto curve from custom weights experiment.

Creates visualization showing:
1. Individual model cost vs quality (FCI-based)
2. Router performance with different weight profiles
3. Pareto frontier with indifference curves
4. How custom weights (w_q, w_c) control the quality-cost tradeoff
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
    Create Pareto curve visualization with indifference curves.
    
    Shows how custom weights (w_q, w_c) determine the quality-cost tradeoff
    and which models the router selects.
    
    Args:
        results: Experiment results dict
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data
    model_baselines = results["model_baselines"]
    profile_results = results["profile_results"]
    
    # Plot individual models (baseline points)
    model_costs = [m["cost"] * 1000 for m in model_baselines]  # Convert to per-1M
    model_qualities = [m["quality"] * 100 for m in model_baselines]  # Convert to percentage
    model_names = [m["model_id"].split("/")[-1] for m in model_baselines]
    
    ax.scatter(model_costs, model_qualities, 
              s=200, alpha=0.5, c='lightgray', 
              marker='o', edgecolors='darkgray', linewidth=2,
              label='Pareto Models', zorder=2)
    
    # Annotate model names
    for cost, quality, name in zip(model_costs, model_qualities, model_names):
        ax.annotate(name, (cost, quality), 
                   xytext=(7, 7), textcoords='offset points',
                   fontsize=10, alpha=0.8, fontweight='bold')
    
    # Compute and plot Pareto frontier first
    pareto_points = compute_pareto_frontier(model_baselines)
    if len(pareto_points) > 1:
        # Sort by cost for proper line connection
        pareto_points_sorted = sorted(pareto_points, key=lambda x: x["cost"])
        pareto_costs = [p["cost"] * 1000 for p in pareto_points_sorted]
        pareto_qualities = [p["quality"] * 100 for p in pareto_points_sorted]
        ax.plot(pareto_costs, pareto_qualities,
               'k-', alpha=0.4, linewidth=3,
               label='Pareto Frontier', zorder=1)
        
        # Highlight Pareto points
        ax.scatter(pareto_costs, pareto_qualities,
                  s=250, alpha=0.8, c='gold', 
                  marker='*', edgecolors='orange', linewidth=2,
                  zorder=4)
    
    # Plot router profiles with indifference curves
    profile_colors = {
        "Cost Saver": "#27ae60",      # Green
        "High Quality": "#c0392b",    # Red
        "Balanced": "#e67e22"          # Orange
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
                  s=400, alpha=0.95,
                  c=profile_colors.get(name, 'purple'),
                  marker=profile_markers.get(name, '*'),
                  edgecolors='black', linewidth=2.5,
                  label=f'Router: {name}',
                  zorder=5)
        
        # Calculate lambda (cost-quality tradeoff parameter)
        w_q = result["weights"]["w_q"]
        w_c = result["weights"]["w_c"]
        
        # λ = w_c/w_q determines the slope of the indifference curve
        # In quality-cost space: slope = 1/λ (how much quality per unit cost)
        if w_q > 0:
            lambda_val = w_c / w_q
            slope = 1.0 / lambda_val if lambda_val > 0 else float('inf')
        else:
            lambda_val = float('inf')
            slope = 0.0
        
        # Draw indifference curve through the router's achieved point
        # The curve shows points of equal utility for this profile
        if 0 < lambda_val < 100:  # Only draw for reasonable lambda values
            # Create line passing through (cost, quality) with slope = quality/cost
            x_range = np.array([min(model_costs) * 0.5, max(model_costs) * 1.2])
            # y = quality + slope * (x - cost)
            y_range = quality + slope * (x_range - cost)
            
            ax.plot(x_range, y_range,
                   '--', color=profile_colors.get(name, 'gray'),
                   alpha=0.4, linewidth=2, zorder=1)
            
            # Annotate with lambda
            mid_x = (x_range[0] + x_range[1]) / 2
            mid_y = quality + slope * (mid_x - cost)
            ax.text(mid_x, mid_y, f'λ={lambda_val:.1f}',
                   fontsize=9, alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=profile_colors.get(name, 'white'),
                            alpha=0.3))
        
        # Annotate with weights near the point
        weight_text = f"w_q={w_q:.1f}\nw_c={w_c:.1f}"
        ax.annotate(weight_text, (cost, quality),
                   xytext=(15, -20), textcoords='offset points',
                   fontsize=9, alpha=0.9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor=profile_colors.get(name, 'white'),
                            edgecolor=profile_colors.get(name, 'black'),
                            alpha=0.7, linewidth=1.5))
    
    # Styling
    ax.set_xlabel('Cost ($ per 1M tokens)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Quality (FCI Score, 0-100 scale)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier & Custom Weight Profiles\n'
                'How w_q and w_c determine the quality-cost tradeoff (λ = w_c/w_q)',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add annotation explaining the experiment
    textstr = (
        f"Experiment: Custom Weights Pareto\n"
        f"Test Samples: {results['n_test_samples']}\n"
        f"Router Mode: Greedy (α=0.0)\n"
        f"\n"
        f"λ = w_c/w_q (cost-quality tradeoff)\n"
        f"Dashed lines = Indifference curves\n"
        f"Gold stars = Pareto-optimal models"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', 
                edgecolor='orange', alpha=0.85, linewidth=1.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', 
           horizontalalignment='right', bbox=props)
    
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
    Print formatted summary table of results with lambda values.
    
    Args:
        results: Experiment results dict
    """
    print("\n" + "="*100)
    print("SUMMARY TABLE: CUSTOM WEIGHT PROFILES")
    print("="*100)
    
    # Header
    print(f"\n{'Profile':<18} {'λ':<8} {'w_q':<6} {'w_c':<6} "
          f"{'Cost ($/1M)':<13} {'Quality (FCI)':<15} {'Success Rate':<15}")
    print("-" * 100)
    
    # Profile results
    for result in results["profile_results"]:
        cost = result["avg_cost"] * 1000  # Convert to per-1M
        quality = result["avg_quality"] * 100
        success = result["success_rate"] * 100
        w_q = result["weights"]["w_q"]
        w_c = result["weights"]["w_c"]
        lambda_val = w_c / w_q if w_q > 0 else float('inf')
        
        print(f"{result['profile_name']:<18} "
              f"{lambda_val:<8.2f} "
              f"{w_q:<6.1f} {w_c:<6.1f} "
              f"${cost:<12.5f} "
              f"{quality:<14.2f}% "
              f"{success:<14.2f}%")
    
    print("\n" + "="*100)
    print("PARETO-OPTIMAL MODEL BASELINES")
    print("="*100)
    
    # Model baselines
    print(f"\n{'Model':<45} {'Cost ($/1M)':<15} {'Quality (FCI)':<15} {'Success Rate':<15}")
    print("-" * 100)
    
    # Sort by cost
    sorted_models = sorted(results["model_baselines"], key=lambda x: x["cost"])
    
    for model in sorted_models:
        cost = model["cost"] * 1000
        quality = model["quality"] * 100
        success = model["success_rate"] * 100
        name = model["model_id"]
        
        print(f"{name:<45} ${cost:<14.5f} {quality:<14.2f}% {success:<14.2f}%")
    
    print("\n" + "="*100)
    print("\nNote: All models shown are Pareto-optimal (no model has both lower cost AND higher quality)")
    print("      Quality measured using FCI (Frontier Capability Index)")
    print("      λ = w_c/w_q determines which model is selected for each prompt")
    print("="*100)


def main():
    """Generate all visualizations."""
    print("="*80)
    print("PLOTTING CUSTOM WEIGHTS PARETO EXPERIMENT")
    print("Visualizing how w_q and w_c control the quality-cost tradeoff")
    print("="*80)
    
    # Load results
    print("\n📊 Loading results...")
    results = load_results()
    print(f"  ✓ Loaded results for {len(results['profile_results'])} profiles")
    print(f"  ✓ {len(results['model_baselines'])} model baselines")
    
    # Show profile lambda values
    print("\n📐 Profile Tradeoff Parameters (λ = w_c/w_q):")
    for result in results['profile_results']:
        w_q = result["weights"]["w_q"]
        w_c = result["weights"]["w_c"]
        lambda_val = w_c / w_q if w_q > 0 else float('inf')
        print(f"  {result['profile_name']:<15} λ={lambda_val:<6.2f} "
              f"(w_q={w_q:.1f}, w_c={w_c:.1f})")
    print(f"\n  Note: Lower λ → quality-focused, Higher λ → cost-focused")
    
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

