#!/usr/bin/env python3
"""
Plot Pareto Arbitrage Curve for Experiment 07 (Figure 1)

Generates publication-ready KDD-quality visualization proving the "Free Lunch":
BanditGPT achieves flagship quality at budget prices, lying above the 
single-model convex hull.

Key visual elements:
- Cost (log scale) vs Hard Task Accuracy (%)
- Model convex hull (baseline frontier, Orange)
- BanditGPT Pareto Curve (Blue line connecting profiles)
- Random Baseline (Rose cross)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import PercentFormatter
import seaborn as sns

def load_results():
    """Load results from run_arbitrage.py"""
    results_path = Path(__file__).parent / "results" / "arbitrage_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results not found at {results_path}\n"
            "Run `python run_arbitrage.py` first to generate results."
        )
    
    with open(results_path) as f:
        return json.load(f)

def plot_arbitrage_curve(results: dict, output_path: Path):
    """
    Create KDD publication-quality Pareto Arbitrage plot.
    """
    # KDD-Quality Aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    # Use standard reliable fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color-Blind Friendly Palette
    COLORS = {
        "models": "#999999",      # Gray for individual models
        "frontier": "#E69F00",    # Orange for model frontier
        "bandit": "#0072B2",      # Blue for BanditGPT Curve
        "random": "#CC79A7",      # Rose for random baseline
        "annotation": "#333333"
    }
    
    # Extract data
    model_baselines = results["model_baselines"]
    pareto_frontier = results["pareto_frontier"]
    bandit_curve = results["bandit_curve"]
    random_res = results["random_baseline"]
    
    # -------------------------------------------------------------------------
    # 1. Plot Individual Models
    # -------------------------------------------------------------------------
    # Filter for log scale (cost > 0)
    valid_models = [m for m in model_baselines if m["cost"] > 0]
    m_costs = [m["cost"] for m in valid_models]
    m_qualities = [m["quality"] for m in valid_models]
    
    ax.scatter(m_costs, m_qualities, color=COLORS["models"], s=60, alpha=0.4,
               label='Individual Models', zorder=2, edgecolors='white')
    
    # -------------------------------------------------------------------------
    # 2. Plot Static Pareto Frontier (Convex Hull)
    # -------------------------------------------------------------------------
    # Sort by cost for line plotting
    frontier_points = sorted([m for m in pareto_frontier if m["cost"] > 0], key=lambda x: x["cost"])
    if frontier_points:
        f_costs = [m["cost"] for m in frontier_points]
        f_qualities = [m["quality"] for m in frontier_points]
        
        ax.plot(f_costs, f_qualities, color=COLORS["frontier"], lw=2.5,
                linestyle='--', alpha=0.8, label='Static Model Frontier', zorder=3,
                marker='o', markersize=6)
                
    # -------------------------------------------------------------------------
    # 3. Plot Random Baseline
    # -------------------------------------------------------------------------
    ax.errorbar(random_res["cost_mean"], random_res["quality_mean"],
                xerr=random_res["cost_std"], yerr=random_res["quality_std"],
                color=COLORS["random"], fmt='X', markersize=12,
                capsize=5, elinewidth=2, alpha=0.9,
                label='Random Selection', zorder=4,
                markeredgecolor='white')
    
    ax.annotate("Random", 
                (random_res["cost_mean"], random_res["quality_mean"]),
                xytext=(-30, -30), textcoords='offset points',
                fontsize=10, color=COLORS["random"], fontweight='bold')

    # -------------------------------------------------------------------------
    # 4. Plot BanditGPT Arbitrage Curve
    # -------------------------------------------------------------------------
    # Sort by cost
    bandit_points = sorted(bandit_curve, key=lambda x: x["cost_mean"])
    b_costs = [b["cost_mean"] for b in bandit_points]
    b_qualities = [b["quality_mean"] for b in bandit_points]
    
    # Plot the curve
    ax.plot(b_costs, b_qualities, color=COLORS["bandit"], lw=3,
            linestyle='-', alpha=1.0, label='BanditGPT Arbitrage', zorder=5,
            marker='D', markersize=8, markeredgecolor='white')
            
    # Annotate specific profiles
    for b in bandit_points:
        name = b["profile"]
        if name in ["Arbitrage", "Max Quality", "Cost Saver"]:
            xy = (b["cost_mean"], b["quality_mean"])
            
            # Smart offset logic
            if name == "Arbitrage":
                offset = (-60, 20)
                ha = 'right'
            elif name == "Max Quality":
                offset = (-20, 15)
                ha = 'right'
            else:
                offset = (10, -20)
                ha = 'left'
                
            ax.annotate(name, xy, xytext=offset, textcoords='offset points',
                        fontsize=10, fontweight='bold', color=COLORS["bandit"],
                        arrowprops=dict(arrowstyle='->', color=COLORS["bandit"], lw=1.5),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["bandit"], alpha=0.8))

    # -------------------------------------------------------------------------
    # 5. Formatting
    # -------------------------------------------------------------------------
    ax.set_xscale('log')
    ax.set_xlabel('Cost per 1M Tokens (USD, Log Scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quality (Oracle Reward)', fontsize=12, fontweight='bold')
    
    ax.set_title("The Free Lunch: BanditGPT vs Static Frontier", fontsize=14, fontweight='bold', pad=15)
    
    # Format Y axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add Grid
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fancybox=True)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to: {output_path}")

def main():
    print("="*70)
    print("PLOTTING PARETO ARBITRAGE CURVE")
    print("="*70)
    
    results = load_results()
    print(f"Loaded results: {len(results['model_baselines'])} baselines, {len(results['bandit_curve'])} bandit points")
    
    output_dir = Path(__file__).parent / "results"
    output_path = output_dir / "fig1_arbitrage_curve.pdf"
    
    plot_arbitrage_curve(results, output_path)
    
    # PNG preview
    plot_arbitrage_curve(results, output_path.with_suffix(".png"))

if __name__ == "__main__":
    main()
