#!/usr/bin/env python3
"""
Visualize the FCI-based Pareto frontier.

Creates a publication-quality plot showing:
- All models with FCI scores (scatter)
- Pareto frontier (highlighted)
- Cost vs FCI tradeoff
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (12, 7),
    'lines.linewidth': 2.5,
    'lines.markersize': 10
})


def load_results() -> tuple:
    """Load FCI results and Pareto frontier."""
    results_dir = Path(__file__).parent / "results"
    
    # Load all models with FCI
    with open(results_dir / "models_with_fci.json") as f:
        all_data = json.load(f)
        all_models = all_data["models"]
        stats = all_data["normalization_stats"]
    
    # Load Pareto frontier
    with open(results_dir / "pareto_frontier_fci.json") as f:
        pareto_data = json.load(f)
        pareto_models = pareto_data["models"]
    
    return all_models, pareto_models, stats


def plot_fci_frontier(all_models: List[Dict], pareto_models: List[Dict], stats: Dict):
    """Create FCI vs Cost Pareto frontier plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract data for all models
    all_costs = []
    all_fcis = []
    all_labels = []
    
    for model in all_models:
        cost = model.get('price_1m_blended', model.get('input_cost_per_m', 0))
        fci = model['fci']
        label = model['openrouter_id'].split('/')[-1]
        
        all_costs.append(cost)
        all_fcis.append(fci)
        all_labels.append(label)
    
    # Identify which are Pareto-optimal
    pareto_ids = {m['openrouter_id'] for m in pareto_models}
    
    # Separate Pareto and non-Pareto models
    pareto_costs = []
    pareto_fcis = []
    pareto_labels = []
    
    dominated_costs = []
    dominated_fcis = []
    dominated_labels = []
    
    for model in all_models:
        cost = model.get('price_1m_blended', model.get('input_cost_per_m', 0))
        fci = model['fci']
        label = model['openrouter_id'].split('/')[-1]
        
        if model['openrouter_id'] in pareto_ids:
            pareto_costs.append(cost)
            pareto_fcis.append(fci)
            pareto_labels.append(label)
        else:
            dominated_costs.append(cost)
            dominated_fcis.append(fci)
            dominated_labels.append(label)
    
    # Plot dominated models (gray)
    if dominated_costs:
        ax.scatter(dominated_costs, dominated_fcis, 
                  s=200, alpha=0.4, c='gray', 
                  marker='o', edgecolors='black', linewidths=1.5,
                  label='Dominated Models', zorder=2)
        
        # Label dominated models
        for cost, fci, label in zip(dominated_costs, dominated_fcis, dominated_labels):
            ax.annotate(label, (cost, fci), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, alpha=0.6,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='gray', alpha=0.7))
    
    # Plot Pareto-optimal models (red)
    ax.scatter(pareto_costs, pareto_fcis,
              s=300, alpha=0.9, c='red',
              marker='*', edgecolors='darkred', linewidths=2,
              label='Pareto Frontier', zorder=5)
    
    # Label Pareto models with better visibility
    for cost, fci, label in zip(pareto_costs, pareto_fcis, pareto_labels):
        ax.annotate(label, (cost, fci),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                            edgecolor='darkred', alpha=0.8, linewidth=2))
    
    # Draw Pareto frontier line
    if len(pareto_costs) > 1:
        # Sort by cost
        sorted_pairs = sorted(zip(pareto_costs, pareto_fcis))
        frontier_costs, frontier_fcis = zip(*sorted_pairs)
        
        ax.plot(frontier_costs, frontier_fcis, 
               'r--', linewidth=2.5, alpha=0.7,
               label='Pareto Frontier Line', zorder=3)
        
        # Fill area above frontier
        ax.fill_between(frontier_costs, frontier_fcis, 
                       [1.0] * len(frontier_costs),
                       alpha=0.1, color='red', zorder=1)
    
    # Formatting
    ax.set_xlabel('Cost ($ per 1M tokens)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frontier Capability Index (FCI)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: FCI vs. Cost\n(Composite: HLE + GPQA + LiveBench)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Use log scale for cost to spread out the points
    ax.set_xscale('log')
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.05, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', 
             fancybox=True, shadow=True)
    
    # Add text box with statistics
    stats_text = (
        f"Benchmark Ranges:\n"
        f"HLE: {stats['hle']['min']:.3f}-{stats['hle']['max']:.3f}\n"
        f"GPQA: {stats['gpqa']['min']:.3f}-{stats['gpqa']['max']:.3f}\n"
        f"LiveBench: {stats['livebench']['min']:.3f}-{stats['livebench']['max']:.3f}\n\n"
        f"Total Models: {len(all_models)}\n"
        f"Pareto-Optimal: {len(pareto_models)}"
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / "results" / "fci_pareto_frontier.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    
    plt.close()


def plot_benchmark_breakdown(all_models: List[Dict], pareto_models: List[Dict]):
    """Create subplot showing individual benchmark contributions to FCI."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    pareto_ids = {m['openrouter_id'] for m in pareto_models}
    benchmarks = [
        ('hle', 'HLE (Human Level Evaluation)'),
        ('gpqa', 'GPQA (Graduate-Level Q&A)'),
        ('livebench', 'LiveBench (Contamination-Free)')
    ]
    
    for ax, (bench_key, bench_name) in zip(axes, benchmarks):
        # Separate Pareto and dominated
        pareto_costs = []
        pareto_scores = []
        pareto_labels = []
        
        dominated_costs = []
        dominated_scores = []
        dominated_labels = []
        
        for model in all_models:
            cost = model.get('price_1m_blended', model.get('input_cost_per_m', 0))
            score = model[bench_key]
            label = model['openrouter_id'].split('/')[-1]
            
            if model['openrouter_id'] in pareto_ids:
                pareto_costs.append(cost)
                pareto_scores.append(score)
                pareto_labels.append(label)
            else:
                dominated_costs.append(cost)
                dominated_scores.append(score)
                dominated_labels.append(label)
        
        # Plot
        if dominated_costs:
            ax.scatter(dominated_costs, dominated_scores,
                      s=150, alpha=0.4, c='gray',
                      marker='o', edgecolors='black', linewidths=1,
                      label='Dominated')
        
        ax.scatter(pareto_costs, pareto_scores,
                  s=250, alpha=0.9, c='red',
                  marker='*', edgecolors='darkred', linewidths=2,
                  label='Pareto-Optimal')
        
        # Labels
        for cost, score, label in zip(pareto_costs, pareto_scores, pareto_labels):
            ax.annotate(label, (cost, score),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Cost ($/1M)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{bench_key.upper()} Score', fontsize=11, fontweight='bold')
        ax.set_title(bench_name, fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle('Individual Benchmark Scores vs. Cost', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "results" / "benchmark_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    
    plt.close()


def plot_normalized_comparison(all_models: List[Dict], pareto_models: List[Dict]):
    """Bar chart comparing normalized vs raw FCI components."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    pareto_ids = {m['openrouter_id'] for m in pareto_models}
    
    # Sort models by FCI
    sorted_models = sorted(all_models, key=lambda m: m['fci'], reverse=True)
    
    model_names = [m['openrouter_id'].split('/')[-1] for m in sorted_models]
    x = np.arange(len(model_names))
    width = 0.25
    
    # Raw scores (handle None values for LiveBench)
    hle_raw = [m['hle'] for m in sorted_models]
    gpqa_raw = [m['gpqa'] for m in sorted_models]
    livebench_raw = [m['livebench'] if m['livebench'] is not None else 0 for m in sorted_models]
    
    ax1.bar(x - width, hle_raw, width, label='HLE', alpha=0.8)
    ax1.bar(x, gpqa_raw, width, label='GPQA', alpha=0.8)
    ax1.bar(x + width, livebench_raw, width, label='LiveBench', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Raw Score', fontsize=12, fontweight='bold')
    ax1.set_title('Raw Benchmark Scores', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Normalized scores (handle None values for LiveBench)
    hle_norm = [m['hle_normalized'] for m in sorted_models]
    gpqa_norm = [m['gpqa_normalized'] for m in sorted_models]
    livebench_norm = [m['livebench_normalized'] if m['livebench_normalized'] is not None else 0 for m in sorted_models]
    
    ax2.bar(x - width, hle_norm, width, label='HLE (norm)', alpha=0.8)
    ax2.bar(x, gpqa_norm, width, label='GPQA (norm)', alpha=0.8)
    ax2.bar(x + width, livebench_norm, width, label='LiveBench (norm)', alpha=0.8)
    
    # Add FCI as line
    fci_scores = [m['fci'] for m in sorted_models]
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, fci_scores, 'r*-', linewidth=3, markersize=12, 
                 label='FCI (composite)', zorder=5)
    ax2_twin.set_ylabel('FCI (Composite)', fontsize=12, fontweight='bold', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2_twin.set_ylim(0, 1.05)
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Score [0, 1]', fontsize=12, fontweight='bold')
    ax2.set_title('Normalized Benchmark Scores + FCI', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    # Highlight Pareto models
    for i, model in enumerate(sorted_models):
        if model['openrouter_id'] in pareto_ids:
            ax1.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='red', zorder=0)
            ax2.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='red', zorder=0)
    
    plt.suptitle('Benchmark Score Comparison: Raw vs. Normalized', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "results" / "normalized_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("VISUALIZING FCI PARETO FRONTIER")
    print("=" * 70)
    print()
    
    # Load data
    all_models, pareto_models, stats = load_results()
    print(f"📊 Loaded {len(all_models)} models ({len(pareto_models)} Pareto-optimal)")
    print()
    
    # Create plots
    print("📈 Creating visualizations...")
    plot_fci_frontier(all_models, pareto_models, stats)
    plot_benchmark_breakdown(all_models, pareto_models)
    plot_normalized_comparison(all_models, pareto_models)
    
    print()
    print("=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

