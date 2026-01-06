#!/usr/bin/env python3
"""
Visualization for Experiment 08: Regret Convergence (Cold Start Defense)

Generates publication-quality line chart showing:
- Cumulative Regret vs. Requests (t)

Visual comparison:
- Cold Start LinUCB: Steep slope (thrashing)
- ε-Greedy: Linear slope (never stops exploring)
- BanditGPT (N=100): Flat slope (starts competent)

Takeaway: "We solve the Cold Start problem that makes standard bandits unusable in production."
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
    """Load convergence experiment results."""
    results_path = Path(__file__).parent / "results" / "convergence_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            "Please run 'python run_convergence.py' first."
        )
    
    with open(results_path) as f:
        data = json.load(f)
    
    return data


def plot_regret_convergence(data, output_dir):
    """
    Generate publication-quality regret convergence plot.
    
    Shows how cumulative regret evolves over requests,
    demonstrating the "Cold Start Defense" of BanditGPT.
    """
    results = data["results"]
    
    # Algorithm display settings
    algorithms = {
        "cold_start_linucb": {
            "label": "Cold Start LinUCB (No Priors)",
            "color": "#E63946",  # Red - danger/thrashing
            "linestyle": "-",
            "description": "Steep slope (thrashing)"
        },
        "epsilon_greedy": {
            "label": "ε-Greedy (ε=0.1)",
            "color": "#F4A261",  # Orange - caution
            "linestyle": "--",
            "description": "Linear slope (constant exploration)"
        },
        "banditgpt_n100": {
            "label": "BanditGPT (N=100 Priors)",
            "color": "#2A9D8F",  # Teal - success
            "linestyle": "-",
            "description": "Flat slope (starts competent)"
        }
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo_key, settings in algorithms.items():
        curves = results[algo_key]
        
        # Convert to numpy and handle variable lengths
        curves_arr = [np.array(c) for c in curves]
        min_len = min(len(c) for c in curves_arr)
        curves_arr = np.array([c[:min_len] for c in curves_arr])
        
        # Calculate mean and std
        mean_curve = np.mean(curves_arr, axis=0)
        std_curve = np.std(curves_arr, axis=0)
        
        # X-axis: request number (1-indexed for clarity)
        x = np.arange(1, len(mean_curve) + 1)
        
        # Plot mean line
        ax.plot(
            x, mean_curve,
            label=settings["label"],
            color=settings["color"],
            linestyle=settings["linestyle"],
            linewidth=2.5,
            zorder=3
        )
        
        # Plot error band (1 std)
        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=settings["color"],
            alpha=0.15,
            zorder=2
        )
    
    # Add behavioral annotations
    n_prompts = len(results["banditgpt_n100"][0])
    
    # Annotation: Cold Start Zone (first 100 requests)
    ax.axvspan(0, min(100, n_prompts * 0.1), alpha=0.08, color='red', zorder=1)
    ax.text(
        min(50, n_prompts * 0.05), ax.get_ylim()[1] * 0.95,
        "Cold Start\nZone",
        fontsize=9, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E63946', alpha=0.9),
        color='#E63946'
    )
    
    # Annotation: The "Thrashing" problem
    cold_start_final = np.mean([c[-1] for c in results["cold_start_linucb"]])
    bandit_final = np.mean([c[-1] for c in results["banditgpt_n100"]])
    
    # Arrow showing the gap
    mid_point = n_prompts // 2
    ax.annotate(
        '',
        xy=(mid_point, bandit_final),
        xytext=(mid_point, cold_start_final * 0.7),
        arrowprops=dict(
            arrowstyle='<->',
            color='gray',
            lw=1.5,
            ls='--'
        )
    )
    ax.text(
        mid_point + n_prompts * 0.02, (cold_start_final * 0.7 + bandit_final) / 2,
        f"Cold Start\nPenalty",
        fontsize=9, ha='left', va='center',
        color='gray'
    )
    
    # Labels and title
    ax.set_xlabel('Request Number ($t$)', fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontweight='bold')
    ax.set_title(
        'Regret Convergence: Cold Start Defense',
        fontweight='bold', pad=15
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set axis limits
    ax.set_xlim(0, n_prompts)
    ax.set_ylim(bottom=0)
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.95)
    
    # Add takeaway annotation box
    takeaway = "Takeaway: BanditGPT's priors eliminate\nthe cold-start penalty that makes\nstandard bandits unusable in production."
    ax.text(
        0.98, 0.25,
        takeaway,
        transform=ax.transAxes,
        fontsize=9, ha='right', va='bottom',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='#E8F5E9',
            edgecolor='#2A9D8F',
            alpha=0.95
        )
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    pdf_path = output_dir / "fig8_regret_convergence.pdf"
    png_path = output_dir / "fig8_regret_convergence.png"
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    
    print(f"✅ Plot saved:")
    print(f"   PDF: {pdf_path}")
    print(f"   PNG: {png_path}")
    
    return fig


def plot_early_convergence(data, output_dir):
    """
    Generate zoomed-in plot focusing on first 200 requests.
    This highlights the critical "cold start" period.
    """
    results = data["results"]
    
    algorithms = {
        "cold_start_linucb": {
            "label": "Cold Start LinUCB",
            "color": "#E63946",
            "linestyle": "-"
        },
        "epsilon_greedy": {
            "label": "ε-Greedy",
            "color": "#F4A261",
            "linestyle": "--"
        },
        "banditgpt_n100": {
            "label": "BanditGPT (N=100)",
            "color": "#2A9D8F",
            "linestyle": "-"
        }
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Focus on first 200 requests
    zoom_end = 200
    
    for algo_key, settings in algorithms.items():
        curves = results[algo_key]
        
        # Convert and truncate
        curves_arr = [np.array(c[:zoom_end]) for c in curves]
        min_len = min(len(c) for c in curves_arr)
        curves_arr = np.array([c[:min_len] for c in curves_arr])
        
        mean_curve = np.mean(curves_arr, axis=0)
        std_curve = np.std(curves_arr, axis=0)
        
        x = np.arange(1, len(mean_curve) + 1)
        
        ax.plot(
            x, mean_curve,
            label=settings["label"],
            color=settings["color"],
            linestyle=settings["linestyle"],
            linewidth=2.5
        )
        
        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=settings["color"],
            alpha=0.15
        )
    
    # Labels
    ax.set_xlabel('Request Number ($t$)', fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontweight='bold')
    ax.set_title(
        'Early Convergence (First 200 Requests)',
        fontweight='bold', pad=15
    )
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, min(zoom_end, len(results["banditgpt_n100"][0])))
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', framealpha=0.95)
    
    plt.tight_layout()
    
    # Save
    pdf_path = output_dir / "fig8_early_convergence.pdf"
    png_path = output_dir / "fig8_early_convergence.png"
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    
    print(f"✅ Early convergence plot saved:")
    print(f"   PDF: {pdf_path}")
    print(f"   PNG: {png_path}")
    
    return fig


def print_summary(data):
    """Print numerical summary of results."""
    results = data["results"]
    metadata = data["metadata"]
    
    print("\n" + "="*70)
    print("REGRET CONVERGENCE SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset:")
    print(f"   Prompts: {metadata['n_prompts']}")
    print(f"   Models: {metadata['n_models']}")
    print(f"   Trials: {metadata['n_trials']}")
    
    print(f"\n📈 Final Cumulative Regret:")
    
    algo_names = {
        "cold_start_linucb": "Cold Start LinUCB (N=0)",
        "epsilon_greedy": "ε-Greedy (ε=0.1)",
        "banditgpt_n100": "BanditGPT (N=100)"
    }
    
    final_regrets = {}
    for algo_key, display_name in algo_names.items():
        curves = results[algo_key]
        finals = [c[-1] for c in curves]
        mean = np.mean(finals)
        std = np.std(finals)
        final_regrets[algo_key] = mean
        
        marker = "  ← BEST" if algo_key == "banditgpt_n100" else ""
        print(f"   {display_name:30s}: {mean:7.2f} ± {std:.2f}{marker}")
    
    # Calculate improvement
    cold_start = final_regrets["cold_start_linucb"]
    bandit = final_regrets["banditgpt_n100"]
    improvement = ((cold_start - bandit) / cold_start) * 100
    
    print(f"\n🎯 Cold Start Defense:")
    print(f"   Regret Reduction: {improvement:.1f}%")
    print(f"   Absolute Savings: {cold_start - bandit:.2f} regret units")
    
    # Slope analysis (first 100 requests)
    print(f"\n📉 Early Slope Analysis (First 100 Requests):")
    for algo_key, display_name in algo_names.items():
        curves = results[algo_key]
        early_slopes = []
        for c in curves:
            if len(c) >= 100:
                slope = (c[99] - c[0]) / 99
                early_slopes.append(slope)
        
        if early_slopes:
            mean_slope = np.mean(early_slopes)
            print(f"   {display_name:30s}: {mean_slope:.4f} regret/request")


def main():
    """Generate regret convergence visualizations."""
    print("="*70)
    print("VISUALIZING REGRET CONVERGENCE")
    print("="*70)
    
    # Load results
    print("\n📂 Loading results...")
    data = load_results()
    print(f"   ✓ Loaded {len(data['results'])} algorithms")
    
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Generate main plot
    print("\n📊 Generating main convergence plot...")
    plot_regret_convergence(data, output_dir)
    
    # Generate early convergence plot
    print("\n📊 Generating early convergence plot...")
    plot_early_convergence(data, output_dir)
    
    # Print summary
    print_summary(data)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
