#!/usr/bin/env python3
"""
Visualize Chebyshev Value-Add (2D: Quality vs. Cost).
Demonstrates how Chebyshev scoring identifies the "Sweet Spot" (Pareto Frontier).
"""
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# Ensure we can import from llm_jury (parent of blog)
sys.path.append(str(Path(__file__).parent.parent))

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

def calculate_2d_chebyshev(quality_norm, cost_norm, w_q=0.6, w_c=0.4):
    """
    Calculate 2D Chebyshev distance to Utopia (Q=1.0, C=0.0).
    
    Args:
        quality_norm: Normalized quality (0-1, higher is better)
        cost_norm: Normalized cost (0-1, lower is better)
        w_q: Weight for quality regret
        w_c: Weight for cost regret
        
    Returns:
        Chebyshev distance (lower is better)
    """
    # Regret: Distance from ideal
    # Ideal Quality = 1.0 -> Regret = 1.0 - q
    # Ideal Cost = 0.0 -> Regret = c - 0.0 = c
    
    regret_q = max(0.0, 1.0 - quality_norm)
    regret_c = max(0.0, cost_norm) # Cost is already 0-1 where 0 is best
    
    return max(w_q * regret_q, w_c * regret_c)

def plot_chebyshev_value_add():
    print("📊 Generating 2D Chebyshev Value-Add Plot...")
    
    models_data = load_models()
    if not models_data:
        print("❌ No models found to analyze.")
        return

    quality_scorer = QualityScorer()
    
    # Use a general purpose decision for quality calculation
    decision = RoutingDecision(
        archetype=ProductArchetype.FRONTIER,
        category=PromptCategory.GENERAL,
        reason="Value Analysis"
    )
    
    plot_data = []
    
    print(f"   Scoring {len(models_data)} models...")
    
    # First pass: Collect raw values to normalize cost
    raw_costs = []
    
    temp_data = []
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
            
            # Calculate Quality Score (0-100)
            quality = quality_scorer.calculate_quality_score(model, decision)
            
            # Calculate Blended Cost
            blended_cost = (model.input_cost_per_m * 0.75) + (model.output_cost_per_m * 0.25)
            
            # Store for normalization
            temp_data.append({
                'name': model.name,
                'quality': quality,
                'cost': blended_cost
            })
            raw_costs.append(blended_cost)
            
        except Exception:
            continue
            
    if not temp_data:
        print("❌ No valid data points.")
        return
        
    # Normalize Cost (Min-Max scaling for fair comparison)
    # We want 0.0 to be the cheapest (or 0), and 1.0 to be the most expensive
    # But actually, let's normalize relative to a "reasonable max" to avoid outliers skewing everything
    # Or just use the max in the dataset.
    max_cost = max(raw_costs) if raw_costs else 1.0
    min_cost = min(raw_costs) if raw_costs else 0.0
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
    
    # Calculate Chebyshev Scores
    for d in temp_data:
        # Normalize Quality: 0-100 -> 0-1
        q_norm = d['quality'] / 100.0
        
        # Normalize Cost: 0-1
        c_norm = (d['cost'] - min_cost) / cost_range
        
        # Chebyshev Score
        # Weights: Quality 0.6, Cost 0.4 (Slight preference for quality)
        cheb_score = calculate_2d_chebyshev(q_norm, c_norm, w_q=0.6, w_c=0.4)
        
        d['chebyshev'] = cheb_score
        plot_data.append(d)

    # Sort by Chebyshev (best first)
    plot_data.sort(key=lambda x: x['chebyshev'])
    
    # Extract arrays
    x = [d['cost'] for d in plot_data]
    y = [d['quality'] for d in plot_data]
    c = [d['chebyshev'] for d in plot_data]
    
    # Visualization
    sns.set_style('darkgrid')
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Custom colormap: Purple (Best/Low Score) -> Teal -> Yellow (Worst/High Score)
    # This matches the "Chebyshev Distance" intuition (0 is best)
    cmap = sns.color_palette("viridis_r", as_cmap=True) # Reversed Viridis: Yellow(High) -> Purple(Low)
    
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=100, alpha=0.9, edgecolors='white', linewidths=0.5)
    
    # Labels
    ax.set_xlabel('Blended Cost ($/1M Tokens)', fontsize=13, weight='bold', color='white')
    ax.set_ylabel('Quality Score (0-100)', fontsize=13, weight='bold', color='white')
    ax.set_title('The "Sweet Spot": 2D Chebyshev Optimization', fontsize=16, weight='bold', pad=20, color='white')
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Chebyshev Distance (Lower is Better)', fontsize=12, weight='bold', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Annotate Top 5 Chebyshev Winners
    top_cheb = plot_data[:5]
    print("\n🏆 Top 5 Chebyshev Winners (Balanced):")
    for i, m in enumerate(top_cheb, 1):
        print(f"{i}. {m['name']} (Q: {m['quality']:.1f}, $: {m['cost']:.2f}, Score: {m['chebyshev']:.3f})")
        
        # Annotate on plot
        ax.annotate(f"{i}. {m['name']}", (m['cost'], m['quality']),
                    xytext=(15, -15), textcoords='offset points',
                    fontsize=9, color='#F1C40F', weight='bold', # Gold color
                    arrowprops=dict(arrowstyle='->', color='#F1C40F', alpha=0.8))

    # Identify "Traditional" Winners for comparison
    # Top Quality (regardless of cost)
    top_quality = sorted(plot_data, key=lambda x: x['quality'], reverse=True)[:3]
    # Top Cost (Cheapest)
    top_cheap = sorted(plot_data, key=lambda x: x['cost'])[:3]
    
    # Mark Utopia Point (Theoretical)
    # Utopia: Max Quality (100), Min Cost (min_cost)
    ax.scatter([min_cost], [100], color='#E74C3C', s=200, marker='*', label='Utopia Point', zorder=10)
    ax.annotate('Utopia Point\n(Perfect & Free)', (min_cost, 100),
                xytext=(20, -10), textcoords='offset points',
                fontsize=10, color='#E74C3C', weight='bold')

    # Grid and Limits
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white', which='both')
    
    # Add interpretation box
    stats_text = "Why Chebyshev?\n"
    stats_text += "• Finds the 'Knee' of the curve\n"
    stats_text += "• Balances Quality (60%) & Cost (40%)\n"
    stats_text += "• Avoids 'Cheap but Dumb' & 'Smart but Pricey'"
    
    ax.text(0.98, 0.03, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='bottom',
            horizontalalignment='right',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', alpha=0.9, edgecolor='white'))
    
    output_file = 'chebyshev_value_add_2d.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"\n✅ Saved plot to {output_file}")

if __name__ == "__main__":
    plot_chebyshev_value_add()
