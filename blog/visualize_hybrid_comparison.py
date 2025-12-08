#!/usr/bin/env python3
"""
Visualization: Hybrid Pareto-Chebyshev Optimization Comparison

Compares the new HYBRID strategy with CHEBYSHEV and KNEE strategies
to demonstrate how the hybrid approach combines the best of both.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.ranking.optimizer import Optimizer, OptimizationStrategy
from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import PromptCategory, RoutingDecision, ModelMetadata, ProductArchetype
from llm_jury.data import ModelRegistry


# Color palette from blog plots
COLORS = {
    'bg': '#0a0e17',
    'panel': '#131a2a',
    'grid': '#1e2738',
    'text': '#e8eaed',
    'muted': '#7a8599',
    'accent': '#22c55e',
    'gold': '#ffd93d',
    'baseline': '#ff6b6b',
    'knee': '#3b82f6',
    'chebyshev': '#a855f7',
    'hybrid': '#22c55e',
    'pareto': '#f59e0b',
}

# Strategy colors
STRATEGY_COLORS = {
    OptimizationStrategy.HYBRID: '#22c55e',      # Green
    OptimizationStrategy.BALANCED: '#a855f7',    # Purple (Chebyshev)
    OptimizationStrategy.KNEE: '#3b82f6',        # Blue
}

USE_CASES = {
    PromptCategory.CODING: {'name': 'Coding', 'icon': '💻', 'color': '#22c55e'},
    PromptCategory.DATA_SCIENCE: {'name': 'Data Science', 'icon': '📊', 'color': '#3b82f6'},
    PromptCategory.CREATIVE: {'name': 'Creative', 'icon': '✨', 'color': '#a855f7'},
    PromptCategory.GENERAL: {'name': 'General', 'icon': '🎯', 'color': '#f59e0b'},
}


def load_models():
    """Load model data from cache."""
    cache_path = Path("data/models_cache.json")
    with open(cache_path) as f:
        return json.load(f)


def dict_to_model_metadata(d: dict) -> ModelMetadata:
    """Convert dict to ModelMetadata."""
    return ModelMetadata(
        name=d.get('name', ''),
        input_cost_per_m=d.get('price_1m_input') or d.get('input_cost_per_m'),
        output_cost_per_m=d.get('price_1m_output') or d.get('output_cost_per_m'),
        intelligence_index=d.get('intelligence_index'),
        coding_index=d.get('coding_index'),
        math_index=d.get('math_index'),
        mmlu_pro=d.get('mmlu_pro'),
        gpqa=d.get('gpqa'),
        hle=d.get('hle'),
        livecodebench=d.get('livecodebench'),
        scicode=d.get('scicode'),
        math_500=d.get('math_500'),
        aime=d.get('aime'),
        median_latency_ms=d.get('median_latency_ms'),
        output_tokens_per_second=d.get('output_tokens_per_second'),
        time_to_first_token_seconds=d.get('time_to_first_token_seconds'),
        measured_ttft_seconds=d.get('measured_ttft_seconds'),
        hallucination_rate=d.get('hallucination_rate'),
        factual_consistency_rate=d.get('factual_consistency_rate'),
        refusal_rate=d.get('refusal_rate'),
    )


def get_rankings_for_strategy(
    models_data: list,
    baseline_name: str,
    strategy: OptimizationStrategy,
    use_case: PromptCategory,
    top_k: int = 15
) -> list:
    """Get model rankings for a specific strategy."""
    
    # Find baseline
    baseline_dict = next((m for m in models_data if baseline_name.lower() in m.get('name', '').lower()), None)
    if not baseline_dict:
        print(f"Baseline {baseline_name} not found")
        return []
    
    baseline = dict_to_model_metadata(baseline_dict)
    
    # Convert all models to ModelMetadata
    models = [dict_to_model_metadata(m) for m in models_data 
              if m.get('price_1m_input') and m.get('price_1m_input') > 0]
    
    # Create optimizer
    optimizer = Optimizer(
        baseline_model=baseline,
        all_models_data=models_data,
        strategy=strategy,
    )
    
    # Create routing decision
    decision = RoutingDecision(
        archetype=ProductArchetype.FRONTIER,
        category=use_case,
        reason="Optimization comparison",
    )
    
    # Get rankings
    results = optimizer.rank(
        models=models,
        decision=decision,
        top_k=top_k,
        return_detailed=True,
        verbose=False
    )
    
    return results


def create_strategy_comparison_plot(
    baseline_name: str = "Gemini 3 Pro Preview (high)",
    use_case: PromptCategory = PromptCategory.CODING,
    output_path: str = "blog/hybrid_vs_strategies_coding.png"
):
    """Create a side-by-side comparison of HYBRID vs CHEBYSHEV vs KNEE."""
    
    uc_config = USE_CASES[use_case]
    
    print("=" * 70)
    print(f"STRATEGY COMPARISON: {uc_config['icon']} {uc_config['name'].upper()}")
    print("=" * 70)
    
    # Load data
    all_models = load_models()
    valid_models = [
        m for m in all_models 
        if m.get('price_1m_input') and m.get('price_1m_input') > 0
        and m.get('intelligence_index') and m.get('intelligence_index') > 0
    ]
    
    print(f"\n📊 Loaded {len(valid_models)} models")
    
    # Get rankings for each strategy
    strategies = [
        (OptimizationStrategy.HYBRID, "HYBRID\n(5D: 4D + Pareto Dominance)"),
        (OptimizationStrategy.BALANCED, "CHEBYSHEV\n(4D Balanced)"),
        (OptimizationStrategy.KNEE, "KNEE POINT\n(Best Value)"),
    ]
    
    all_rankings = {}
    for strategy, label in strategies:
        print(f"\n🔄 Computing {strategy.value} rankings...")
        rankings = get_rankings_for_strategy(
            valid_models, baseline_name, strategy, use_case, top_k=15
        )
        all_rankings[strategy] = rankings
        if rankings:
            print(f"   Top 3: {[r.name[:25] for r in rankings[:3]]}")
    
    # =========================================================================
    # CREATE VISUALIZATION
    # =========================================================================
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Title
    fig.text(0.5, 0.96, f'🔬 Strategy Comparison: {uc_config["icon"]} {uc_config["name"]}',
            fontsize=24, fontweight='bold', color=COLORS['text'], ha='center')
    fig.text(0.5, 0.93, f'HYBRID (default) vs CHEBYSHEV vs KNEE POINT optimization',
            fontsize=14, color=COLORS['muted'], ha='center')
    
    gs = GridSpec(1, 3, figure=fig, wspace=0.12,
                  left=0.04, right=0.96, top=0.88, bottom=0.08)
    
    for idx, (strategy, label) in enumerate(strategies):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(COLORS['panel'])
        
        rankings = all_rankings.get(strategy, [])
        color = STRATEGY_COLORS.get(strategy, COLORS['accent'])
        
        # Plot bars
        y_positions = list(range(len(rankings[:12])))
        
        for i, model in enumerate(rankings[:12]):
            # Bar for score (inverted since lower is better)
            score = 1 - model.chebyshev_score  # Convert to "higher is better"
            bar_width = score * 0.9
            
            # Gradient effect
            bar_color = color if i < 3 else COLORS['muted']
            alpha = 1.0 if i < 3 else 0.5
            
            # Draw bar
            rect = FancyBboxPatch(
                (0.02, i - 0.35), bar_width, 0.7,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=bar_color, alpha=alpha,
                edgecolor='none',
                transform=ax.transData
            )
            ax.add_patch(rect)
            
            # Model name
            name = model.name[:28] + "..." if len(model.name) > 31 else model.name
            ax.text(0.03, i, name, fontsize=9, fontweight='bold' if i < 3 else 'normal',
                   color=COLORS['text'] if i < 3 else COLORS['muted'],
                   va='center', ha='left')
            
            # Score on right
            ax.text(0.97, i, f"Q:{model.quality_score:.0f}", fontsize=8,
                   color=COLORS['muted'], va='center', ha='right',
                   fontfamily='monospace')
        
        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 11.5)
        ax.invert_yaxis()
        ax.axis('off')
        
        # Strategy label box
        label_box = FancyBboxPatch(
            (0.02, -1.2), 0.96, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color, alpha=0.2,
            edgecolor=color, linewidth=2,
            transform=ax.transData
        )
        ax.add_patch(label_box)
        ax.text(0.5, -0.85, label, fontsize=11, fontweight='bold',
               color=color, va='center', ha='center')
        
        # Rank numbers
        for i in range(min(12, len(rankings))):
            badge_color = COLORS['gold'] if i == 0 else (color if i < 3 else COLORS['grid'])
            circle = plt.Circle((-0.06, i), 0.25, color=badge_color, 
                               transform=ax.transData, clip_on=False)
            ax.add_patch(circle)
            ax.text(-0.06, i, str(i+1), fontsize=8, fontweight='bold',
                   color=COLORS['bg'] if i < 3 else COLORS['muted'],
                   va='center', ha='center')
    
    # =========================================================================
    # ADD COMPARISON INSIGHTS
    # =========================================================================
    
    # Get top models for each strategy
    hybrid_top = [r.name for r in all_rankings.get(OptimizationStrategy.HYBRID, [])[:5]]
    cheby_top = [r.name for r in all_rankings.get(OptimizationStrategy.BALANCED, [])[:5]]
    knee_top = [r.name for r in all_rankings.get(OptimizationStrategy.KNEE, [])[:5]]
    
    # Find unique models in hybrid top 5
    hybrid_unique = [m for m in hybrid_top if m not in cheby_top[:3] and m not in knee_top[:3]]
    
    # Add insight box at bottom
    insight_ax = fig.add_axes([0.04, 0.01, 0.92, 0.05])
    insight_ax.set_facecolor(COLORS['panel'])
    insight_ax.axis('off')
    
    insight_text = (
        f"💡 HYBRID adds Pareto dominance as 5th Chebyshev dimension. "
        f"Models dominating more others rank higher. Top pick: {hybrid_top[0] if hybrid_top else 'N/A'}"
    )
    insight_ax.text(0.5, 0.5, insight_text, fontsize=11, color=COLORS['text'],
                   va='center', ha='center', style='italic')
    
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def create_hybrid_ranking_plot(
    baseline_name: str = "Gemini 3 Pro Preview (high)",
    use_case: PromptCategory = PromptCategory.CODING,
    output_path: str = "blog/model_hybrid_ranking_coding.png"
):
    """Create a detailed ranking plot for HYBRID strategy."""
    
    uc_config = USE_CASES[use_case]
    
    print("=" * 70)
    print(f"HYBRID RANKING: {uc_config['icon']} {uc_config['name'].upper()}")
    print("=" * 70)
    
    # Load data
    all_models = load_models()
    valid_models = [
        m for m in all_models 
        if m.get('price_1m_input') and m.get('price_1m_input') > 0
        and m.get('intelligence_index') and m.get('intelligence_index') > 0
    ]
    
    print(f"\n📊 Loaded {len(valid_models)} models")
    
    # Get HYBRID rankings
    rankings = get_rankings_for_strategy(
        valid_models, baseline_name, OptimizationStrategy.HYBRID, use_case, top_k=20
    )
    
    if not rankings:
        print("❌ No rankings returned")
        return
    
    print(f"\n🏆 Top 5 HYBRID recommendations for {uc_config['name']}:")
    for i, r in enumerate(rankings[:5], 1):
        print(f"   {i}. {r.name[:40]} (Q:{r.quality_score:.0f})")
    
    # =========================================================================
    # CREATE VISUALIZATION
    # =========================================================================
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Main plot area
    ax = fig.add_axes([0.25, 0.08, 0.70, 0.82])
    ax.set_facecolor(COLORS['panel'])
    
    # Plot horizontal bars
    num_models = min(15, len(rankings))
    y_positions = list(range(num_models))
    
    # Score normalization (invert so higher = better for display)
    scores = [1 - r.chebyshev_score for r in rankings[:num_models]]
    max_score = max(scores) if scores else 1
    
    # Color based on quality score
    def get_bar_color(quality):
        if quality >= 90: return COLORS['accent']
        elif quality >= 80: return '#3b82f6'
        elif quality >= 70: return '#06b6d4'
        elif quality >= 60: return COLORS['gold']
        elif quality >= 50: return COLORS['pareto']
        else: return COLORS['baseline']
    
    for i, model in enumerate(rankings[:num_models]):
        score = scores[i]
        bar_width = (score / max_score) * 100
        color = get_bar_color(model.quality_score)
        
        # Draw bar
        ax.barh(i, bar_width, height=0.7, color=color, alpha=0.85,
               edgecolor='white', linewidth=0.5)
        
        # Score label on bar
        ax.text(bar_width + 1, i, f"Q:{model.quality_score:.0f}", fontsize=9,
               color=COLORS['muted'], va='center', fontfamily='monospace')
    
    # Model names on y-axis
    model_names = [r.name for r in rankings[:num_models]]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names, fontsize=10, color=COLORS['text'])
    
    ax.set_xlim(0, 110)
    ax.set_ylim(-0.5, num_models - 0.5)
    ax.invert_yaxis()
    
    ax.set_xlabel('Hybrid Score (Pareto-Chebyshev Fusion)', fontsize=12, 
                 color=COLORS['text'], labelpad=10)
    
    ax.tick_params(colors=COLORS['muted'], labelsize=10)
    ax.grid(True, axis='x', alpha=0.15, color=COLORS['grid'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    
    # Title
    fig.text(0.5, 0.95, f'🔬 HYBRID Optimization: {uc_config["icon"]} {uc_config["name"]}',
            fontsize=20, fontweight='bold', color=COLORS['text'], ha='center')
    fig.text(0.5, 0.915, '5D Chebyshev: Quality + Cost + Latency + Trust + Pareto Dominance',
            fontsize=12, color=COLORS['muted'], ha='center')
    
    # Legend
    legend_ax = fig.add_axes([0.02, 0.5, 0.20, 0.35])
    legend_ax.set_facecolor(COLORS['panel'])
    legend_ax.axis('off')
    
    legend_ax.text(0.5, 0.95, 'Quality Tiers', fontsize=11, fontweight='bold',
                  color=COLORS['text'], ha='center', va='top', transform=legend_ax.transAxes)
    
    tiers = [
        ('≥90', COLORS['accent'], 'Excellent'),
        ('≥80', '#3b82f6', 'Very Good'),
        ('≥70', '#06b6d4', 'Good'),
        ('≥60', COLORS['gold'], 'Above Avg'),
        ('≥50', COLORS['pareto'], 'Average'),
        ('<50', COLORS['baseline'], 'Below Avg'),
    ]
    
    for i, (score, color, label) in enumerate(tiers):
        y = 0.82 - i * 0.12
        rect = FancyBboxPatch((0.08, y - 0.04), 0.15, 0.08,
                              boxstyle="round,pad=0.01,rounding_size=0.02",
                              facecolor=color, edgecolor='none',
                              transform=legend_ax.transAxes)
        legend_ax.add_patch(rect)
        legend_ax.text(0.28, y, f'{score}', fontsize=9, color=color,
                      va='center', ha='left', transform=legend_ax.transAxes,
                      fontfamily='monospace', fontweight='bold')
        legend_ax.text(0.48, y, label, fontsize=9, color=COLORS['muted'],
                      va='center', ha='left', transform=legend_ax.transAxes)
    
    # Method explanation
    method_ax = fig.add_axes([0.02, 0.08, 0.20, 0.38])
    method_ax.set_facecolor(COLORS['panel'])
    method_ax.axis('off')
    
    method_ax.text(0.5, 0.95, 'HYBRID Method', fontsize=11, fontweight='bold',
                  color=COLORS['accent'], ha='center', va='top', transform=method_ax.transAxes)
    
    steps = [
        ('1', 'Standard 4D', 'Quality, Cost,\nLatency, Trust'),
        ('2', 'Net Dominance', 'Dominations -\nDominated by'),
        ('3', '5D Chebyshev', 'Min max regret\nacross all 5'),
    ]
    
    for i, (num, title, desc) in enumerate(steps):
        y = 0.78 - i * 0.28
        
        # Step number circle
        circle = plt.Circle((0.12, y), 0.06, color=COLORS['accent'],
                            transform=method_ax.transAxes, clip_on=False)
        method_ax.add_patch(circle)
        method_ax.text(0.12, y, num, fontsize=10, fontweight='bold',
                      color=COLORS['bg'], va='center', ha='center',
                      transform=method_ax.transAxes)
        
        method_ax.text(0.25, y + 0.02, title, fontsize=10, fontweight='bold',
                      color=COLORS['text'], va='center', ha='left',
                      transform=method_ax.transAxes)
        method_ax.text(0.25, y - 0.08, desc, fontsize=8,
                      color=COLORS['muted'], va='top', ha='left',
                      transform=method_ax.transAxes, linespacing=1.2)
    
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def main():
    """Generate hybrid optimization comparison visualizations."""
    
    baseline = "Gemini 3 Pro Preview (high)"
    
    # Create comparison plots for each use case
    for uc, config in USE_CASES.items():
        uc_name = config['name'].lower().replace(' ', '_')
        
        # Strategy comparison (3-column)
        create_strategy_comparison_plot(
            baseline_name=baseline,
            use_case=uc,
            output_path=f"blog/hybrid_vs_strategies_{uc_name}.png"
        )
        
        # Detailed hybrid ranking
        create_hybrid_ranking_plot(
            baseline_name=baseline,
            use_case=uc,
            output_path=f"blog/model_hybrid_ranking_{uc_name}.png"
        )
    
    print("\n" + "=" * 70)
    print("HYBRID VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    for config in USE_CASES.values():
        uc_name = config['name'].lower().replace(' ', '_')
        print(f"  - blog/hybrid_vs_strategies_{uc_name}.png")
        print(f"  - blog/model_hybrid_ranking_{uc_name}.png")


if __name__ == "__main__":
    main()

