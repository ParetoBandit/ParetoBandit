#!/usr/bin/env python3
"""
Visualize Cost vs. Chebyshev Score with Latency Heatmap.
Highlights the trade-offs between price, performance, and speed.
"""
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LogNorm

# Ensure we can import from llm_jury (parent of blog)
sys.path.append(str(Path(__file__).parent.parent))

from llm_jury.optimization.chebyshev_scorer import ChebyshevScorer
from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import ModelMetadata, RoutingDecision, PromptCategory, ProductArchetype

def load_models():
    """Load models from the cache."""
    try:
        # Path relative to blog/ folder
        cache_path = Path(__file__).parent.parent / 'models_cache.json'
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            return cache_data.get('models', [])
    except FileNotFoundError:
        print("❌ Cache file not found!")
        return []

def plot_cost_vs_chebyshev():
    print("📊 Generating Cost vs. Chebyshev Score Scatterplot...")
    
    models_data = load_models()
    if not models_data:
        print("❌ No models found to analyze.")
        return

    # 1. Setup Scorers
    chebyshev_scorer = ChebyshevScorer(
        baseline_quality=88.7,      # GPT-4o reference
        baseline_cost=5.0,          # GPT-4o reference
        baseline_latency=500.0,     # GPT-4o reference
        baseline_trustability=2.0,  # Top ~2.5% trustability
        quality_weight=0.3,
        cost_weight=0.25,
        latency_weight=0.25,
        trustability_weight=0.2
    )
    
    quality_scorer = QualityScorer()
    
    # Use a general purpose decision for quality calculation
    decision = RoutingDecision(
        archetype=ProductArchetype.FRONTIER,
        category=PromptCategory.GENERAL,
        reason="Distribution Analysis"
    )
    
    plot_data = []
    
    print(f"   Scoring {len(models_data)} models...")
    
    for m_data in models_data:
        # Helper to safely get float value
        def get_safe_float(key, default=0.0):
            val = m_data.get(key)
            if val is None: return default
            try:
                f_val = float(val)
                return default if np.isnan(f_val) else f_val
            except (ValueError, TypeError):
                return default

        try:
            model = ModelMetadata(
                name=m_data.get('name', 'Unknown'),
                mmlu_score=get_safe_float('mmlu_score'),
                gpqa_score=get_safe_float('gpqa_score'),
                math_score=get_safe_float('math_score'),
                ifeval_score=get_safe_float('ifeval_score'),
                tool_use_ability=get_safe_float('tool_use_ability'),
                context_window_k=get_safe_float('context_window_k'),
                hallucination_rate=get_safe_float('hallucination_rate'),
                ethics_score=get_safe_float('ethics_score'),
                hf_downloads=int(get_safe_float('hf_downloads')),
                hf_likes=int(get_safe_float('hf_likes')),
                hf_created_at=str(m_data.get('hf_created_at', "")),
                archetype=ProductArchetype.FRONTIER,
                input_cost_per_m=get_safe_float('input_cost_per_m'),
                output_cost_per_m=get_safe_float('output_cost_per_m'),
                median_latency_ms=get_safe_float('median_latency_ms'),
                param_count_b=get_safe_float('param_count_b')
            )
            
            # Calculate Quality Score
            quality = quality_scorer.calculate_quality_score(model, decision)
            # if quality <= 10: continue # User requested to remove quality threshold
            
            # Calculate Blended Cost
            blended_cost = (model.input_cost_per_m * 0.75) + (model.output_cost_per_m * 0.25)
            if blended_cost <= 0: blended_cost = 0.001 # Avoid log(0)
                
            # Calculate Chebyshev Score
            score = chebyshev_scorer.score_model(
                model_name=model.name,
                quality=quality,
                cost=model.input_cost_per_m,
                latency=model.median_latency_ms,
                trustability=m_data.get('trustability_index', 0)
            )
            
            plot_data.append({
                'name': model.name,
                'chebyshev': score.chebyshev_distance,
                'cost': blended_cost,
                'latency': max(model.median_latency_ms, 1.0) # Avoid 0 latency for log scales if needed
            })
            
        except Exception:
            continue

    if not plot_data:
        print("❌ No valid data points generated.")
        return

    # 2. Sort by Chebyshev Score (Ascending)
    # This ensures better models (lower score) are plotted last if we want them on top,
    # OR if we want to visualize the progression.
    # User asked to "order the data by growing chebychev score".
    # If we plot in this order, the worst models (high score) are plotted last (on top).
    # Usually we want best models on top. Let's sort descending so best (low score) are last.
    plot_data.sort(key=lambda x: x['chebyshev'], reverse=True)
    
    # Filter for Chebyshev Score < 0.2 (Focus on high performance)
    # plot_data = [d for d in plot_data if d['chebyshev'] < 0.2] # Removed to show full distribution
    
    if not plot_data:
        print("❌ No models found with Chebyshev Score < 0.2")
        return
    
    # Print high-cost models
    high_cost_models = [d for d in plot_data if d['cost'] > 20]
    if high_cost_models:
        print(f"\n   💰 HIGH-COST MODELS (>$20):")
        for m in sorted(high_cost_models, key=lambda x: x['cost'], reverse=True):
            print(f"      • {m['name']}: ${m['cost']:.2f}")
    
    # Extract arrays for plotting
    x = np.array([d['cost'] for d in plot_data])
    y = np.array([d['chebyshev'] for d in plot_data])
    c = np.array([d['latency'] for d in plot_data])
    names = [d['name'] for d in plot_data]
    
    # Calculate data distribution to set better axis limits
    x_percentile_95 = np.percentile(x, 95)
    x_percentile_99 = np.percentile(x, 99)
    x_max = x.max()
    
    print(f"   Cost Stats: min=${x.min():.3f}, median=${np.median(x):.3f}, 95th%=${x_percentile_95:.3f}, max=${x_max:.3f}")
    print(f"   Chebyshev Stats: min={y.min():.3f}, median={np.median(y):.3f}, max={y.max():.3f}")
    
    # 3. Visualization
    sns.set_style('darkgrid')
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Scatter plot with heatmap coloring
    sc = ax.scatter(x, y, c=c, cmap='turbo', s=80, alpha=0.8, edgecolors='white', linewidths=0.5, norm=LogNorm())
    
    # Axes Scaling - Using linear but with smart limits
    # Set x-axis limit to focus on main cluster (up to 95th percentile, with some padding)
    # This prevents extreme outliers from squishing all the data to the left
    x_limit = min(x_percentile_99 * 1.2, x_max)  # Go to 99th percentile + 20% padding, or max
    ax.set_xlim(-x_limit * 0.02, x_limit)  # Small negative padding on left for aesthetics
    
    # Labels
    ax.set_xlabel('Blended Cost ($/1M Tokens)', fontsize=13, weight='bold', color='white')
    ax.set_ylabel('Chebyshev Score (Lower is Better)', fontsize=13, weight='bold', color='white')
    ax.set_title('Cost vs. Performance vs. Latency', fontsize=16, weight='bold', pad=20, color='white')
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Latency (ms) [Log Scale]', fontsize=12, weight='bold', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Grid and Ticks
    ax.grid(True, alpha=0.2, color='white', which='both')
    ax.tick_params(colors='white', which='both')
    
    # Annotate Top Models (Pareto Frontier-ish)
    # Label best models that are visible in the plot range
    top_models = sorted(plot_data, key=lambda x: x['chebyshev'])[:7]
    texts = []
    for m in top_models:
        # Only annotate if within visible range
        if m['cost'] <= x_limit:
            ax.annotate(m['name'], (m['cost'], m['chebyshev']), 
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=9, color='yellow', weight='bold',
                        arrowprops=dict(arrowstyle='->', color='yellow', alpha=0.7, lw=1.5))

    # Count outliers (models beyond the visible range)
    outliers = sum(1 for d in plot_data if d['cost'] > x_limit)
    
    # Add interpretation box
    stats_text = "Interpretation:\n"
    stats_text += "• Bottom-Left: Best Value (Low Cost, High Perf)\n"
    stats_text += "• Color: Latency (Blue=Fast, Red=Slow)\n"
    stats_text += "• Y-Axis: 0.0 is Utopia (Perfect)\n"
    if outliers > 0:
        stats_text += f"\n({outliers} high-cost models not shown)"
    
    ax.text(0.02, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', alpha=0.9, edgecolor='white'))
    
    output_file = 'cost_vs_chebyshev_heatmap.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"✅ Saved plot to {output_file}")

if __name__ == "__main__":
    plot_cost_vs_chebyshev()
