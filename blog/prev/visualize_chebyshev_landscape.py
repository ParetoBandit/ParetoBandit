#!/usr/bin/env python3
"""
Visualize Chebyshev optimization landscape.

Shows the trade-off space between cost, quality (Chebyshev score), and speed.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from llm_jury.ranking.chebyshev import ChebyshevRanker, RankingStrategy
from llm_jury.core.models import ModelMetadata, RoutingDecision, PromptCategory, ProductArchetype


def create_model_from_dict(model_dict):
    """Create ModelMetadata from dict."""
    model = ModelMetadata(
        name=model_dict['name'],
        archetype=ProductArchetype.FRONTIER,
        input_cost_per_m=model_dict.get('price_1m_input', 1.0),
        output_cost_per_m=model_dict.get('price_1m_output', 2.0),
        median_latency_ms=1000.0,
        context_window_k=128,
        param_count_b=70.0,
        mmlu_score=0,
        gpqa_score=0,
        math_score=0,
        ifeval_score=0,
        tool_use_ability=0.5
    )
    
    # Add AA benchmark attributes
    for key in ['intelligence_index', 'coding_index', 'math_index', 
                 'mmlu_pro', 'gpqa', 'hle', 'livecodebench', 'scicode', 
                 'math_500', 'aime', 'output_tokens_per_second', 
                 'time_to_first_token_seconds']:
        setattr(model, key, model_dict.get(key))
    
    return model


def main():
    # Load data
    cache_path = Path("data/models_complete_composite_indices.json")
    
    if not cache_path.exists():
        print(f"❌ Cache not found: {cache_path}")
        return
    
    print("Loading model data...")
    with open(cache_path) as f:
        raw_data = json.load(f)
    
    # Filter to models with complete pricing
    valid_data = [
        m for m in raw_data 
        if m.get('price_1m_input') and m.get('price_1m_output')
        and m.get('price_1m_input') > 0
    ]
    
    print(f"✓ Loaded {len(valid_data)} models with complete pricing\n")
    
    # Convert to ModelMetadata
    models = [create_model_from_dict(m) for m in valid_data]
    
    # Find baseline (prefer GPT-4o)
    baseline = None
    for m in models:
        if 'gpt-4o' in m.name.lower() and 'mini' not in m.name.lower():
            baseline = m
            break
    
    if not baseline:
        # Fallback to GPT-4.1 or first model
        baseline = [m for m in models if 'gpt-4' in m.name.lower()][0]
    
    print(f"✓ Using {baseline.name} as baseline\n")
    
    # Test scenarios
    scenarios = [
        ("General Use", PromptCategory.GENERAL, RankingStrategy.BALANCED),
        ("Coding Task", PromptCategory.CODING, RankingStrategy.BALANCED),
        ("Data Science", PromptCategory.DATA_SCIENCE, RankingStrategy.BALANCED),
        ("Creative Writing", PromptCategory.CREATIVE, RankingStrategy.BALANCED),
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Chebyshev Optimization Landscape: Cost vs Quality vs Speed', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, (scenario_name, category, strategy) in enumerate(scenarios):
        print(f"Processing: {scenario_name}...")
        
        # Initialize ranker
        ranker = ChebyshevRanker(
            baseline_model=baseline,
            all_models_data=valid_data,
            strategy=strategy
        )
        
        # Create decision
        decision = RoutingDecision(
            category=category,
            archetype=ProductArchetype.FRONTIER,
            reason=f"{scenario_name}"
        )
        
        # Rank models
        ranked = ranker.rank(models, decision, top_k=len(models), return_detailed=True)
        
        # Extract data for plotting
        costs = []
        cheb_scores = []
        speeds = []
        names = []
        quality_scores = []
        
        for r in ranked:
            # Blended cost (75% input, 25% output)
            blended_cost = (r.metadata.input_cost_per_m or 0) * 0.75 + \
                          (r.metadata.output_cost_per_m or 0) * 0.25
            
            # Speed metric
            if hasattr(r.metadata, 'output_tokens_per_second') and r.metadata.output_tokens_per_second:
                speed = r.metadata.output_tokens_per_second
            elif r.metadata.median_latency_ms and r.metadata.median_latency_ms > 0:
                speed = 1000.0 / r.metadata.median_latency_ms
            else:
                speed = 50.0
            
            costs.append(blended_cost)
            cheb_scores.append(r.chebyshev_score)
            speeds.append(speed)
            names.append(r.name)
            quality_scores.append(r.quality_score)
        
        # Sort by Chebyshev score (already sorted, but ensure)
        sorted_indices = np.argsort(cheb_scores)
        costs = np.array(costs)[sorted_indices]
        cheb_scores = np.array(cheb_scores)[sorted_indices]
        speeds = np.array(speeds)[sorted_indices]
        names = [names[i] for i in sorted_indices]
        quality_scores = np.array(quality_scores)[sorted_indices]
        
        # Find GPT-4o for reference
        gpt4o_idx = None
        gpt4o_cost = None
        gpt4o_cheb = None
        gpt4o_speed = None
        for i, name in enumerate(names):
            if 'gpt-4o' in name.lower() and 'mini' not in name.lower():
                gpt4o_idx = i
                gpt4o_cost = costs[i]
                gpt4o_cheb = cheb_scores[i]
                gpt4o_speed = speeds[i]
                break
        
        # Create scatter plot
        ax = axes[idx]
        
        # Create custom colormap (green = fast, red = slow)
        colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('speed', colors, N=n_bins)
        
        # Scatter plot with speed as color
        scatter = ax.scatter(
            costs, 
            cheb_scores, 
            c=speeds, 
            s=120, 
            alpha=0.7,
            cmap=cmap,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Speed (tokens/sec)', rotation=270, labelpad=20, fontsize=10)
        
        # Highlight GPT-4o as reference
        if gpt4o_idx is not None:
            ax.scatter(
                [gpt4o_cost],
                [gpt4o_cheb],
                c='red',
                s=300,
                marker='*',
                edgecolors='black',
                linewidth=2,
                zorder=5,
                label='GPT-4o (reference)'
            )
            ax.annotate(
                'GPT-4o\n(baseline)',
                (gpt4o_cost, gpt4o_cheb),
                xytext=(-40, 20),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.8, edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2, color='red')
            )
        
        # Highlight top 5 models
        for i in range(min(5, len(names))):
            if i == gpt4o_idx:
                continue  # Skip GPT-4o, already highlighted
            ax.annotate(
                names[i],
                (costs[i], cheb_scores[i]),
                xytext=(10, 5 if i % 2 == 0 else -15),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1)
            )
        
        # Styling
        ax.set_xlabel('Blended Cost ($/1M tokens, 75% input + 25% output)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Chebyshev Score (lower = better value)', fontsize=11, fontweight='bold')
        ax.set_title(f'{scenario_name}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add reference lines
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Excellent value')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Good value')
        
        # Log scale for x-axis if costs vary widely
        if costs.max() / costs.min() > 100:
            ax.set_xscale('log')
        
        ax.legend(loc='upper right', fontsize=9)
        
        # Print top 10 for this scenario
        print(f"\n  Top 10 {scenario_name}:")
        for i in range(min(10, len(names))):
            print(f"    {i+1:2d}. {names[i]:45s} | Cheb: {cheb_scores[i]:.4f} | Cost: ${costs[i]:.2f} | Quality: {quality_scores[i]:.1f}")
    
    plt.tight_layout()
    plt.savefig('blog/chebyshev_optimization_landscape.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: blog/chebyshev_optimization_landscape.png")
    
    # Create a second figure: Single detailed view for Coding task
    print("\n" + "="*80)
    print("Creating detailed view for Coding Task...")
    print("="*80)
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 12))
    
    # Use coding data
    ranker_coding = ChebyshevRanker(
        baseline_model=baseline,
        all_models_data=valid_data,
        strategy=RankingStrategy.BALANCED
    )
    
    decision_coding = RoutingDecision(
        category=PromptCategory.CODING,
        archetype=ProductArchetype.FRONTIER,
        reason="Coding task"
    )
    
    ranked_coding = ranker_coding.rank(models, decision_coding, top_k=len(models), return_detailed=True)
    
    # Extract data
    costs_c = []
    cheb_c = []
    speeds_c = []
    names_c = []
    quality_c = []
    
    for r in ranked_coding:
        blended = (r.metadata.input_cost_per_m or 0) * 0.75 + (r.metadata.output_cost_per_m or 0) * 0.25
        
        if hasattr(r.metadata, 'output_tokens_per_second') and r.metadata.output_tokens_per_second:
            spd = r.metadata.output_tokens_per_second
        elif r.metadata.median_latency_ms and r.metadata.median_latency_ms > 0:
            spd = 1000.0 / r.metadata.median_latency_ms
        else:
            spd = 50.0
        
        costs_c.append(blended)
        cheb_c.append(r.chebyshev_score)
        speeds_c.append(spd)
        names_c.append(r.name)
        quality_c.append(r.quality_score)
    
    # Sort
    sorted_idx = np.argsort(cheb_c)
    costs_c = np.array(costs_c)[sorted_idx]
    cheb_c = np.array(cheb_c)[sorted_idx]
    speeds_c = np.array(speeds_c)[sorted_idx]
    names_c = [names_c[i] for i in sorted_idx]
    quality_c = np.array(quality_c)[sorted_idx]
    
    # Find GPT-4o
    gpt4o_idx_c = None
    gpt4o_cost_c = None
    gpt4o_cheb_c = None
    gpt4o_speed_c = None
    for i, name in enumerate(names_c):
        if 'gpt-4o' in name.lower() and 'mini' not in name.lower():
            gpt4o_idx_c = i
            gpt4o_cost_c = costs_c[i]
            gpt4o_cheb_c = cheb_c[i]
            gpt4o_speed_c = speeds_c[i]
            break
    
    # Scatter with larger points
    scatter2 = ax2.scatter(
        costs_c, 
        cheb_c, 
        c=speeds_c, 
        s=200, 
        alpha=0.7,
        cmap=cmap,
        edgecolors='black',
        linewidth=1
    )
    
    # Colorbar
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Speed (output tokens/sec)', rotation=270, labelpad=25, fontsize=12)
    
    # Highlight GPT-4o as reference
    if gpt4o_idx_c is not None:
        ax2.scatter(
            [gpt4o_cost_c],
            [gpt4o_cheb_c],
            c='red',
            s=600,
            marker='*',
            edgecolors='black',
            linewidth=3,
            zorder=10,
            label='GPT-4o (baseline reference)'
        )
        ax2.annotate(
            'GPT-4o\n(BASELINE)',
            (gpt4o_cost_c, gpt4o_cheb_c),
            xytext=(-60, 40),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.9, edgecolor='black', linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=3, color='red')
        )
    
    # Annotate top 15
    for i in range(min(15, len(names_c))):
        if i == gpt4o_idx_c:
            continue  # Skip GPT-4o, already highlighted
        ax2.annotate(
            f"{i+1}. {names_c[i]}",
            (costs_c[i], cheb_c[i]),
            xytext=(15, 8 if i % 3 == 0 else (-5 if i % 3 == 1 else -18)),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow' if i < 5 else 'lightblue', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5)
        )
    
    # Styling
    ax2.set_xlabel('Blended Cost ($/1M tokens, 75% input + 25% output)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Chebyshev Score (lower = better overall value)', fontsize=13, fontweight='bold')
    ax2.set_title('Coding Task: Multi-Objective Optimization Landscape\n(Quality + Cost + Speed)',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Reference zones
    ax2.axhline(y=0.05, color='darkgreen', linestyle='--', alpha=0.6, linewidth=2, label='Elite value')
    ax2.axhline(y=0.15, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Excellent value')
    ax2.axhline(y=0.30, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Good value')
    
    if costs_c.max() / costs_c.min() > 100:
        ax2.set_xscale('log')
    
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add text box with insights
    textstr = f'''Top 5 Models (Best Value):
1. {names_c[0]} (Cheb: {cheb_c[0]:.4f})
2. {names_c[1]} (Cheb: {cheb_c[1]:.4f})
3. {names_c[2]} (Cheb: {cheb_c[2]:.4f})
4. {names_c[3]} (Cheb: {cheb_c[3]:.4f})
5. {names_c[4]} (Cheb: {cheb_c[4]:.4f})

Lower Chebyshev = Better overall value
(optimizes quality, cost, and speed)'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('blog/chebyshev_coding_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: blog/chebyshev_coding_detailed.png")
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    print("""
KEY INSIGHTS:
  • Lower Chebyshev score = better overall value (quality + cost + speed)
  • Color indicates speed: Green (fast) to Red (slow)
  • Top-left corner = best value models (low cost, low Chebyshev)
  • Different tasks show different optimal models
  • GPT-5.1, GPT-5 Codex, and Kimi K2 consistently rank highly
    """)


if __name__ == "__main__":
    main()

