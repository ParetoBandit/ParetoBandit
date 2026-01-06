#!/usr/bin/env python3
"""
Plot Pareto Frontier for Experiment 05

Generates publication-ready KDD-quality visualization showing BanditGPT's 
cost-quality tradeoff with synergy shading and professional aesthetics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Polygon


def load_results():
    """Load results from run_pareto.py"""
    results_path = Path(__file__).parent / "results" / "pareto_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results not found at {results_path}\n"
            "Run `python run_pareto.py` first to generate results."
        )
    
    with open(results_path) as f:
        return json.load(f)


def plot_pareto_frontier(results: dict, output_path: Path):
    """
    Create KDD publication-quality Pareto frontier plot.
    
    Features:
    1. Synergy Efficiency Gain (shaded region showing routing intelligence premium)
    2. Overlap-free annotations with quadrant-based logic
    3. Color-blind friendly palette (Nature/Science style)
    4. Professional typography and layout
    """
    # KDD-Quality Aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color-Blind Friendly Palette (Nature/Science standard)
    COLORS = {
        "models": "#999999",      # Gray for individual models
        "pareto": "#E69F00",      # Orange for model frontier (color-blind safe)
        "baseline": "#CC79A7",    # Rose for random baseline
        "bandit": "#0072B2",      # Blue for BanditGPT (color-blind safe)
        "synergy": "#56B4E9",     # Sky blue for efficiency gain shading
    }
    
    # Extract data
    model_baselines = results["model_baselines"]
    m_costs = np.array([m["cost"] for m in model_baselines])
    m_qualities = np.array([m["quality"] for m in model_baselines])
    
    # 1. Plot individual models
    ax.scatter(m_costs, m_qualities, color=COLORS["models"], s=80, alpha=0.6,
               label='Individual Models', zorder=2, edgecolors='white', linewidths=1)
    
    # 2. Compute Model-Only Pareto Frontier
    sorted_models = sorted(model_baselines, key=lambda x: x["cost"])
    pareto = []
    max_quality = -float('inf')
    for m in sorted_models:
        if m["quality"] > max_quality:
            pareto.append(m)
            max_quality = m["quality"]
    
    p_costs = np.array([m["cost"] for m in pareto])
    p_qualities = np.array([m["quality"] for m in pareto])
    
    # 3. BanditGPT Frontier
    frontier = results["frontier"]
    b_costs = np.array([p["cost_mean"] for p in frontier])
    b_qualities = np.array([p["quality_mean"] for p in frontier])
    b_cost_err = np.array([p["cost_std"] for p in frontier])
    b_quality_err = np.array([p["quality_std"] for p in frontier])
    
    
    # 4. Plot Model Pareto Frontier (dashed line)
    if len(pareto) > 1:
        ax.plot(p_costs, p_qualities, color=COLORS["pareto"], lw=3, 
                linestyle='--', alpha=0.8, label='Model-Only Frontier', zorder=3,
                marker='o', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
    
    # 6. Random Baseline
    avg_quality = np.mean(m_qualities)
    ax.axhline(y=avg_quality, color=COLORS["baseline"], linestyle=':', lw=2.5,
               alpha=0.7, label=f'Random Selection ({avg_quality*100:.1f}%)', zorder=1)
    
    # 7. BanditGPT Frontier (THE MAIN EVENT)
    # Error bars
    ax.errorbar(b_costs, b_qualities, xerr=b_cost_err, yerr=b_quality_err,
                color=COLORS["bandit"], fmt='none', alpha=0.25, capsize=6, 
                capthick=2, zorder=5)
    
    # Main curve with bold styling
    ax.plot(b_costs, b_qualities, color=COLORS["bandit"], lw=4.5, zorder=6,
            label='BanditGPT Adaptive Routing', marker='D', markersize=12,
            markeredgecolor='white', markeredgewidth=2.5)
    
    # 8. Overlap-Free Annotations
    # Top 3 models with quadrant-based positioning
    sorted_by_quality = sorted(model_baselines, key=lambda x: x["quality"], reverse=True)
    top_models = sorted_by_quality[:3]
    
    for i, model in enumerate(top_models):
        # Highlight with star marker
        ax.scatter(model["cost"], model["quality"], 
                  color=COLORS["bandit"], s=250, alpha=0.9, marker='*',
                  zorder=7, edgecolors='white', linewidths=2)
        
        # Get display name
        name = model.get("display_name", model.get("model", "Unknown"))
        name = name.replace("Preview", "").replace("(high)", "").strip()
        if len(name) > 25:
            name = name[:22] + "..."
        
        # Quadrant-based offset logic to avoid overlaps
        x_pos, y_pos = model["cost"], model["quality"]
        if i == 0:  # Top model - position well above and to the right, clear of Max Quality label
            xytext = (35, 30)
        elif i == 1:  # Second - position above left  
            xytext = (-70, 15)
        else:  # Third - position below right
            xytext = (20, -25)
        
        ax.annotate(name, 
                   xy=(x_pos, y_pos),
                   xytext=xytext,
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold',
                   color=COLORS["bandit"],
                   bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='white', 
                           edgecolor=COLORS["bandit"],
                           linewidth=2,
                           alpha=0.95),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.15',
                                 color=COLORS["bandit"],
                                 lw=2))
    
    # 9. BanditGPT Profile Labels (overlap-free)
    for i, p in enumerate(frontier):
        # Position labels to avoid data points
        if p["profile"] == "Max Quality":
            offset = (12, 15)
        elif p["profile"] == "Arbitrage":
            offset = (12, -25)
        else:  # Best Value
            offset = (-70, 12)
        
        ax.annotate(p["profile"], 
                   (p["cost_mean"], p["quality_mean"]),
                   xytext=offset, 
                   textcoords='offset points', 
                   fontsize=11,
                   fontweight='bold', 
                   color=COLORS["bandit"],
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='white', 
                            edgecolor=COLORS["bandit"],
                            linewidth=2,
                            alpha=0.9))
    
    # 10. Professional Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Average Cost per 1K Tokens (USD)', 
                  fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Success Rate (Quality)', 
                  fontsize=15, fontweight='bold', labelpad=10)
    ax.set_title('Cost-Quality Pareto Frontier: Intelligent Routing vs. Static Model Selection',
                 fontsize=17, fontweight='bold', pad=25)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Set y-axis limits for better visibility (75% to 105%)
    ax.set_ylim(0.75, 1.05)
    
    # Horizontal legend at bottom (maximizes data canvas)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=11, framealpha=0.98, 
              edgecolor='gray', fancybox=True)
    
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot saved to: {output_path}")


def main():
    """Generate KDD-quality Pareto frontier visualization."""
    print("="*70)
    print("PLOTTING PARETO FRONTIER (KDD Publication Quality)")
    print("="*70)
    
    # Load results
    print("\n📊 Loading results...")
    results = load_results()
    print(f"  ✓ Loaded {len(results['frontier'])} frontier points")
    print(f"  ✓ Loaded {len(results['model_baselines'])} model baselines")
    
    # Generate plot
    output_path = Path(__file__).parent / "results" / "fig5_pareto_frontier.pdf"
    print(f"\n🎨 Generating publication-quality plot...")
    plot_pareto_frontier(results, output_path)
    
    # Also save as PNG for quick preview
    png_path = output_path.with_suffix('.png')
    plot_pareto_frontier(results, png_path)
    
    print("\n✅ Done! The synergy shading shows routing's intelligence premium.")


if __name__ == "__main__":
    main()
