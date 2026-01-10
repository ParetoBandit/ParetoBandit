#!/usr/bin/env python3
"""
Plot: Hyperparameter Tuning Heatmap (KDD Figure)

Visualizes the 3-Fold CV results to verify the robustness of hyperparameter choices.
X-Axis: Exploration Rate (Alpha)
Y-Axis: Prior Stiffness (N_eff)
Color: Average Validation Regret (Lower is Better)
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_tuning_heatmap():
    # 1. Load Data
    results_dir = Path(__file__).parent / "results"
    data_path = results_dir / "heatmap_data.json"
    
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found.")
        print("   Please run 'experiments/01_effectiveness/run_budget_experiment.py' first.")
        return

    print(f"📦 Loading tuning results from {data_path.name}...")
    with open(data_path) as f:
        data = json.load(f)

    # 2. Prepare DataFrame
    df = pd.DataFrame(data)
    
    # Pivot for Heatmap
    # Index (Y): N_eff
    # Columns (X): Alpha
    # Values: Mean Regret
    pivot_table = df.pivot(index="n_eff", columns="alpha", values="mean_regret")
    
    # Sort N_eff descending so large priors are at the top (visually intuitive)
    pivot_table = pivot_table.sort_index(ascending=False)
    
    # 3. Plot
    plt.figure(figsize=(8, 6))
    
    # Use 'viridis' colormap
    # Purple = Low Regret (Good)
    # Yellow = High Regret (Bad)
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f", # Show regret with 4 decimals to avoid "0.0" for small values
        cmap="viridis",
        linewidths=.5,
        cbar_kws={'label': 'Avg. Validation Regret'}
    )
    
    # Highlight the minimum (Best Metric)
    # Find min value location/indices if we tried, but heatmap visualizes it well.
    
    plt.title("Gold Standard Tuning: Prior Stiffness vs. Exploration")
    plt.xlabel(r"Exploration Rate ($\alpha$)")
    plt.ylabel(r"Prior Stiffness ($N_{eff}$)")
    
    # 4. Save
    output_path = results_dir / "tuning_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved heatmap to {output_path}")

if __name__ == "__main__":
    # Ensure seaborn style
    sns.set_theme(style="whitegrid")
    plot_tuning_heatmap()
