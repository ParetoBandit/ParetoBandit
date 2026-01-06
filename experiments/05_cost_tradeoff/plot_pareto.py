#!/usr/bin/env python3
"""
Plot Pareto Frontier for Experiment 05

Generates publication-ready visualization showing BanditGPT's cost-quality
tradeoff compared to individual models and random routing baselines.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import PercentFormatter


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
    Create publication-ready Pareto frontier plot.
    
    Shows:
    1. Individual models (gray scatter)
    2. Model Pareto frontier (dashed line)
    3. Random baseline (horizontal line)
    4. BanditGPT frontier (bold blue curve with "the bulge")
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color scheme
    COLORS = {
        "models": "#A19F9D",      # Gray for individual models
        "pareto": "#FF9500",      # Orange for model frontier
        "baseline": "#FF6B6B",    # Red for random baseline
        "bandit": "#0055A4",      # Blue for BanditGPT
    }
    
    # 1. Plot individual models
    model_baselines = results["model_baselines"]
    m_costs = [m["cost"] for m in model_baselines]
    m_qualities = [m["quality"] for m in model_baselines]
    
    ax.scatter(m_costs, m_qualities, color=COLORS["models"], s=60, alpha=0.5,
               label='Individual Models', zorder=2, edgecolors='white', linewidths=0.5)
    
    # Highlight and label top quality models
    # Sort models by quality (descending) to identify the best performers
    sorted_by_quality = sorted(model_baselines, key=lambda x: x["quality"], reverse=True)
    top_n = 3  # Label top 3 models
    
    for i, model in enumerate(sorted_by_quality[:top_n]):
        # Use star marker for top models
        ax.scatter(model["cost"], model["quality"], 
                  color=COLORS["bandit"], s=150, alpha=0.8, marker='*',
                  zorder=4, edgecolors='white', linewidths=1.5)
        
        # Add model name annotation
        name = model["name"].replace("Preview", "").replace("(high)", "").strip()
        # Shorten long names
        if len(name) > 25:
            name = name[:22] + "..."
            
        ax.annotate(name, 
                   xy=(model["cost"], model["quality"]),
                   xytext=(10, 5 if i % 2 == 0 else -15),
                   textcoords='offset points',
                   fontsize=9,
                   fontweight='semibold',
                   color=COLORS["bandit"],
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor=COLORS["bandit"],
                           alpha=0.9),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.2',
                                 color=COLORS["bandit"],
                                 lw=1.5))
    
    # 2. Compute and plot model Pareto frontier (the "linear baseline")
    sorted_models = sorted(model_baselines, key=lambda x: x["cost"])
    pareto = []
    max_quality = -float('inf')
    for m in sorted_models:
        if m["quality"] > max_quality:
            pareto.append(m)
            max_quality = m["quality"]
    
    if len(pareto) > 1:
        p_costs = [m["cost"] for m in pareto]
        p_qualities = [m["quality"] for m in pareto]
        ax.plot(p_costs, p_qualities, color=COLORS["pareto"], lw=2.5, 
                linestyle='--', alpha=0.8, label='Model Pareto Frontier (Linear)', zorder=3)
    
    # 3. Random baseline (average quality across all models)
    avg_quality = np.mean(m_qualities)
    ax.axhline(y=avg_quality, color=COLORS["baseline"], linestyle='-', lw=2,
               alpha=0.6, label=f'Random Baseline ({avg_quality*100:.1f}%)', zorder=1)
    
    # 4. BanditGPT frontier (THE STAR - should bulge above the line)
    frontier = results["frontier"]
    b_costs = [p["cost_mean"] for p in frontier]
    b_qualities = [p["quality_mean"] for p in frontier]
    b_cost_err = [p["cost_std"] for p in frontier]
    b_quality_err = [p["quality_std"] for p in frontier]
    
    # Error bars
    ax.errorbar(b_costs, b_qualities, xerr=b_cost_err, yerr=b_quality_err,
                color=COLORS["bandit"], fmt='none', alpha=0.3, capsize=5, zorder=5)
    
    # Main curve
    ax.plot(b_costs, b_qualities, color=COLORS["bandit"], lw=4, zorder=6,
            label='BanditGPT Frontier', marker='s', markersize=10,
            markeredgecolor='white', markeredgewidth=2)
    
    # Label profiles
    for p in frontier:
        offset = (10, 8) if p["profile"] != "Ultra Cheap" else (10, -12)
        ax.annotate(p["profile"], (p["cost_mean"], p["quality_mean"]),
                    xytext=offset, textcoords='offset points', fontsize=11,
                    fontweight='bold', color=COLORS["bandit"],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=COLORS["bandit"], alpha=0.8))
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Average Cost per 1k Tokens ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Probability (Quality)', fontsize=14, fontweight='bold')
    ax.set_title('Experiment 05: Cost-Quality Pareto Frontier\n'
                 'BanditGPT achieves higher quality than random routing at equivalent cost',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Interpretation box
    textstr = (
        "💡 The Bulge: BanditGPT curve above\n"
        "   the linear baseline proves intelligent\n"
        "   routing beats random model selection"
    )
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")


def main():
    """Generate Pareto frontier visualization."""
    print("="*70)
    print("PLOTTING PARETO FRONTIER")
    print("="*70)
    
    # Load results
    print("\n📊 Loading results...")
    results = load_results()
    print(f"  ✓ Loaded {len(results['frontier'])} frontier points")
    print(f"  ✓ Loaded {len(results['model_baselines'])} model baselines")
    
    # Generate plot
    output_path = Path(__file__).parent / "results" / "fig5_pareto_frontier.pdf"
    print(f"\n🎨 Generating plot...")
    plot_pareto_frontier(results, output_path)
    
    # Also save as PNG for quick preview
    png_path = output_path.with_suffix('.png')
    plot_pareto_frontier(results, png_path)
    
    print("\n✅ Done! View the plot to verify BanditGPT 'bulges' above the baseline.")


if __name__ == "__main__":
    main()
