#!/usr/bin/env python3
"""
Plot showing correlation between cost and throughput.
"""

import sys
sys.path.insert(0, '.')

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_data():
    """Load model data."""
    cache_path = Path("data/models_complete_composite_indices.json")
    with open(cache_path) as f:
        raw_data = json.load(f)
    
    # Filter for models with throughput data
    valid_data = [m for m in raw_data if m.get('output_tokens_per_second') and m.get('output_tokens_per_second') > 0]
    return valid_data


def plot_cost_vs_throughput(models, output_path='blog/cost_vs_throughput.png'):
    """Create a plot showing cost vs throughput correlation."""
    
    # Extract data
    names = [m['name'] for m in models]
    throughputs = [m['output_tokens_per_second'] for m in models]
    costs = [(m.get('price_1m_input', 0) * 0.75 + m.get('price_1m_output', 0) * 0.25) for m in models]
    
    # Sort by throughput ascending
    sorted_indices = np.argsort(throughputs)
    names = [names[i] for i in sorted_indices]
    throughputs = [throughputs[i] for i in sorted_indices]
    costs = [costs[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create scatter plot with color based on throughput
    scatter = ax.scatter(throughputs, costs, c=throughputs, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Throughput (tokens/sec)')
    
    # Label some interesting points
    # Fastest models
    for i in range(-5, 0):  # Top 5 fastest
        ax.annotate(names[i][:25], (throughputs[i], costs[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8,
                   alpha=0.8)
    
    # Most expensive models
    expensive_indices = np.argsort(costs)[-5:]
    for i in expensive_indices:
        if i not in range(len(names)-5, len(names)):  # Don't double-label
            ax.annotate(names[i][:25], (throughputs[i], costs[i]), 
                       xytext=(5, -10), textcoords='offset points', fontsize=8,
                       alpha=0.8, color='red')
    
    # Cheapest fast models (throughput > median, cost < median)
    median_throughput = np.median(throughputs)
    median_cost = np.median(costs)
    for i, (name, t, c) in enumerate(zip(names, throughputs, costs)):
        if t > median_throughput and c < median_cost * 0.5:
            ax.annotate(name[:25], (t, c), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8,
                       alpha=0.8, color='green')
    
    # Calculate correlation
    correlation = np.corrcoef(throughputs, costs)[0, 1]
    
    # Add trend line
    z = np.polyfit(throughputs, costs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(throughputs), max(throughputs), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2, 
            label=f'Trend (r={correlation:.2f})')
    
    # Formatting
    ax.set_xlabel('Throughput (tokens/second)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost ($/M tokens, blended)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cost vs Throughput Correlation\n({len(models)} models with throughput data)', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits with some padding
    ax.set_xlim(0, max(throughputs) * 1.1)
    ax.set_ylim(0, max(costs) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  Correlation coefficient: {correlation:.3f}")
    return correlation


def plot_throughput_bar_chart(models, output_path='blog/throughput_bar_chart.png'):
    """Create a bar chart of throughput ordered ascending."""
    
    # Extract and sort data
    data = [(m['name'], m['output_tokens_per_second'], 
             (m.get('price_1m_input', 0) * 0.75 + m.get('price_1m_output', 0) * 0.25)) 
            for m in models]
    data.sort(key=lambda x: x[1])  # Sort by throughput ascending
    
    names = [d[0] for d in data]
    throughputs = [d[1] for d in data]
    costs = [d[2] for d in data]
    
    # Truncate names for display
    names_display = [n if len(n) <= 30 else n[:27] + "..." for n in names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 16))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(names))
    
    # Color by cost (cheaper = green, expensive = red)
    norm_costs = np.array(costs)
    norm_costs = (norm_costs - norm_costs.min()) / (norm_costs.max() - norm_costs.min() + 0.001)
    colors = plt.cm.RdYlGn_r(norm_costs)  # Red for expensive, green for cheap
    
    bars = ax.barh(y_pos, throughputs, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add throughput labels
    for bar, throughput, cost in zip(bars, throughputs, costs):
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height()/2, 
                f'{throughput:.0f} tok/s (${cost:.2f})', 
                va='center', ha='left', fontsize=8)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_display, fontsize=9)
    ax.set_xlabel('Throughput (tokens/second)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Throughput (Ordered Ascending)\nColor: Green=Cheap, Red=Expensive', 
                 fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, max(throughputs) * 1.25)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == '__main__':
    print("Loading data...")
    models = load_data()
    print(f"Found {len(models)} models with throughput data")
    
    print("\nGenerating cost vs throughput scatter plot...")
    plot_cost_vs_throughput(models)
    
    print("\nGenerating throughput bar chart...")
    plot_throughput_bar_chart(models)
    
    print("\n✅ Done!")

