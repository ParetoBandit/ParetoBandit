#!/usr/bin/env python3
"""
Plot cumulative regret comparison for Experiment 01.

Generates: fig1_cumulative_regret.pdf
"""

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.plotting import apply_kdd_style, save_kdd_style_plot, COLORS


def load_results():
    """Load results from run_baselines.py."""
    results_file = Path(__file__).parent / "results" / "effectiveness_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results not found at {results_file}. "
            "Run `python run_baselines.py` first."
        )
    
    with open(results_file) as f:
        results = json.load(f)
    
    return results


def compute_statistics(regret_curves):
    """Compute mean and CI for regret curves across seeds."""
    curves = np.array(regret_curves)  # Shape: [n_seeds, n_timesteps]
    
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    
    # 95% CI = 1.96 * SEM
    sem = std_curve / np.sqrt(len(curves))
    ci = 1.96 * sem
    
    return mean_curve, ci


def create_regret_plot(results):
    """Generate cumulative regret comparison plot."""
    apply_kdd_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    # Define method order and colors
    methods = {
        "banditgpt": {"label": "Bandit GPT (Ours)", "color": COLORS["blue"], "linestyle": "-"},
        "vanilla_linucb": {"label": "LinUCB (No Features)", "color": COLORS["green"], "linestyle": "--"},
        "epsilon_greedy_0.1": {"label": "ε-greedy (ε=0.1)", "color": COLORS["orange"], "linestyle": "-."},
        "random": {"label": "Random", "color": COLORS["gray"], "linestyle": ":"},
    }
    
    # Plot each method
    for method_key, style in methods.items():
        if method_key not in results:
            print(f"⚠️ Missing results for {method_key}")
            continue
        
        mean_regret, ci = compute_statistics(results[method_key])
        timesteps = np.arange(1, len(mean_regret) + 1)
        
        # Plot line
        ax.plot(
            timesteps,
            mean_regret,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2
        )
        
        # Add CI shading
        ax.fill_between(
            timesteps,
            mean_regret - ci,
            mean_regret + ci,
            alpha=0.2,
            color=style["color"]
        )
    
    # Formatting
    ax.set_xlabel("Timestep (T)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Regret", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cumulative Regret Comparison\n(Mean ± 95% CI over 10 seeds)",
        fontsize=13,
        fontweight="bold"
    )
    ax.legend(loc="upper left", frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle="--")
    
    # Save plot
    save_kdd_style_plot(fig, "fig1_cumulative_regret.pdf", output_dir="results")
    plt.close()


def print_summary_stats(results):
    """Print final regret values for comparison."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Final Cumulative Regret)")
    print("="*70)
    
    for method_key in ["banditgpt", "vanilla_linucb", "epsilon_greedy_0.1", "random"]:
        if method_key not in results:
            continue
        
        final_regrets = [curve[-1] for curve in results[method_key]]
        mean_final = np.mean(final_regrets)
        std_final = np.std(final_regrets)
        
        print(f"{method_key:25s}: {mean_final:8.2f} ± {std_final:.2f}")
    
    print("="*70)


def main():
    """Generate regret plot."""
    print("Generating cumulative regret plot...")
    
    try:
        results = load_results()
        create_regret_plot(results)
        print_summary_stats(results)
        
        print("\n✅ Plot generated successfully!")
        print("   Output: experiments/01_effectiveness/results/fig1_cumulative_regret.pdf")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
