#!/usr/bin/env python3
"""
Plot cumulative regret for the Effectiveness Experiment.
X-Axis: Time Step (t=1 to 800)
Y-Axis: Cumulative Regret
Shading: +/- 1 Standard Deviation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_effectiveness():
    results_path = Path("experiments/01_effectiveness/results/effectiveness_results.json")
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    plt.figure(figsize=(10, 6))
    
    # Define colors and labels
    method_styles = {
        "banditgpt_warmup": {"label": "BanditGPT (Ours)", "color": "#1f77b4", "linewidth": 2.5},
        "vanilla_linucb": {"label": "Vanilla LinUCB", "color": "#2ca02c", "linewidth": 2},
        "routellm_mf": {"label": "RouteLLM (SOTA)", "color": "#9467bd", "linewidth": 2},
        "banditgpt_hle": {"label": "BanditGPT (HLE-Only)", "color": "#17becf", "linewidth": 2, "linestyle": "--"},
        "random": {"label": "Random", "color": "#7f7f7f", "linewidth": 1.5, "linestyle": ":"}
    }

    t_max = 0
    for method, data in results.items():
        if method not in method_styles:
            continue
            
        style = method_styles[method]
        curves = np.array(data) # [seeds, timesteps]
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        
        t = np.arange(1, len(mean) + 1)
        t_max = max(t_max, len(mean))
        
        plt.plot(t, mean, label=style["label"], color=style["color"], 
                 linewidth=style.get("linewidth", 2), 
                 linestyle=style.get("linestyle", "-"))
        
        plt.fill_between(t, mean - std, mean + std, color=style["color"], alpha=0.15)

    plt.xlabel("Time Step ($t$)", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title("Effectiveness Comparison: Cumulative Regret over Time", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(1, t_max)
    plt.ylim(0, None)

    output_path = Path("experiments/01_effectiveness/results/effectiveness_plot.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_effectiveness()
