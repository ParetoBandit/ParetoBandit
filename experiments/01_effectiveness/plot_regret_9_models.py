#!/usr/bin/env python3
"""
Plot cumulative regret comparison for 9-Model Experiment.

Generates: fig_regret_9_models.pdf
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
    """Load results from effectiveness_results_9_models.json."""
    results_file = Path(__file__).parent / "results" / "effectiveness_results_9_models.json"
    
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results not found at {results_file}. "
            "Run `python run_baselines_9_models.py` first."
        )
    
    with open(results_file) as f:
        results = json.load(f)
    
    return results


def compute_statistics(regret_curves):
    """Compute mean and 95% CI for regret curves across seeds."""
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
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define method order and colors (Updated keys for 9-model run)
    methods = {
        "routellm_mf": {"label": "RouteLLM (MF Baseline)", "color": COLORS["purple"], "linestyle": "-", "linewidth": 2.5},
        "banditgpt_warmup": {"label": "BanditGPT Warmup (Ours)", "color": COLORS["blue"], "linestyle": "-", "linewidth": 2.5},
        "banditgpt_hle": {"label": "BanditGPT HLE", "color": "#56B4E9", "linestyle": "--", "linewidth": 2},
        "vanilla_linucb": {"label": "Vanilla LinUCB", "color": COLORS["green"], "linestyle": "-.", "linewidth": 2},
        "random": {"label": "Random", "color": COLORS["gray"], "linestyle": ":", "linewidth": 1.5},
    }
    
    # Plot each method
    for method_key, style in methods.items():
        if method_key not in results:
            print(f"⚠️  Missing results for {method_key}")
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
            linewidth=style["linewidth"]
        )
        
        # Add CI shading
        ax.fill_between(
            timesteps,
            mean_regret - ci,
            mean_regret + ci,
            alpha=0.15,
            color=style["color"]
        )
    
    # Formatting
    ax.set_xlabel("Timestep (T)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Regret", fontsize=13, fontweight="bold")
    ax.set_title(
        "Cumulative Regret (9-Model Portfolio)\\n(Mean ± 95% CI over 10 seeds)",
        fontsize=14,
        fontweight="bold"
    )
    ax.legend(loc="upper left", frameon=True, shadow=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    
    # Save plot
    save_kdd_style_plot(fig, "fig_regret_9_models.pdf", output_dir="results")
    plt.close()


def print_summary_stats(results):
    """Print final regret values for comparison."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Final Cumulative Regret)")
    print("="*70)
    
    method_order = [
        "routellm_mf",
        "banditgpt_warmup", 
        "banditgpt_hle",
        "vanilla_linucb",
        "random"
    ]
    
    for method_key in method_order:
        if method_key not in results:
            continue
        
        final_regrets = [curve[-1] for curve in results[method_key]]
        mean_final = np.mean(final_regrets)
        std_final = np.std(final_regrets)
        
        print(f"{method_key:25s}: {mean_final:8.2f} ± {std_final:.2f}")
    
    print("="*70)


def main():
    """Generate regret plot."""
    print("Generating cumulative regret plot for 9 models...")
    
    try:
        results = load_results()
        create_regret_plot(results)
        print_summary_stats(results)
        
        print("\n✅ Plot generated successfully!")
        print("   Output: experiments/01_effectiveness/results/fig_regret_9_models.pdf")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
