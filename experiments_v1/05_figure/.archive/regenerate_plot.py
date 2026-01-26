#!/usr/bin/env python3
"""
Regenerate Figure 4 from existing JSON results
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_pareto():
    results_path = Path("results/intermediate_pareto_results.json") # Use intermediate as it's definitely fresh
    with open(results_path) as f:
        data = json.load(f)
    
    output_dir = Path("results")
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Colors
    colors = {
        "Oracle": "#2ecc71",      # Green
        "RouteLLM-MF": "#e74c3c", # Red
        "banditGPT-Hybrid": "#3498db", # Blue
    }
    
    # Plot strategies
    for strategy, points in data['strategies'].items():
        if not points:
            continue
            
        costs = [p['cost'] for p in points]
        rewards = [p['reward'] for p in points]
        
        # Sort for line plotting
        sorted_indices = np.argsort(costs)
        costs = np.array(costs)[sorted_indices]
        rewards = np.array(rewards)[sorted_indices]
        
        if "Static" in strategy:
            label = strategy.replace("Static-", "")
            ax.scatter(costs, rewards, s=180, alpha=0.9, 
                      label=label, marker='o', edgecolors='black', linewidths=2, zorder=5)
            
        elif strategy == "Oracle":
            ax.scatter(costs, rewards, s=300, 
                      color=colors[strategy], marker='*',
                      label=strategy, edgecolors='black', linewidths=2.5, zorder=10)
            
        elif strategy == "RouteLLM-MF":
            ax.plot(costs, rewards, 
                   color=colors[strategy], linewidth=3.5, 
                   label="RouteLLM (Static)", alpha=0.85, marker='o', markersize=8, zorder=6)
            
        elif strategy == "banditGPT-Hybrid":
            ax.plot(costs, rewards, 
                   color=colors[strategy], linewidth=4.5, 
                   label="banditGPT (Adaptive)", alpha=1.0, marker='D', markersize=9, zorder=8)
            
            # Annotate peak
            peak_idx = np.argmax(rewards)
            ax.annotate(f'Peak: {rewards[peak_idx]:.4f}', 
                       xy=(costs[peak_idx], rewards[peak_idx]), 
                       xytext=(costs[peak_idx], rewards[peak_idx]+0.01),
                       arrowprops=dict(facecolor='black', shrink=0.05),
                       fontsize=12, fontweight='bold', ha='center')

    # Formatting
    production_quality = 0.80
    ax.axhline(y=production_quality, color='gray', linestyle='--', 
              linewidth=2.0, alpha=0.5, label=f'Production Standard ({production_quality:.2f})')
    
    ax.set_xlabel('Average Cost per Request ($)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Reward (Quality)', fontsize=16, fontweight='bold')
    ax.set_title(
        'Figure 4: Pareto Frontier - The Competitive Victory\n'
        'banditGPT (Online) vs RouteLLM (Offline)',
        fontsize=18, fontweight='bold', pad=20
    )
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.legend(loc='lower right', fontsize=14, framealpha=0.95, ncol=2)
    
    # Format axes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.4f}'))
    ax.set_xlim(left=-0.0005)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'figure4_final_polished.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    plot_pareto()

