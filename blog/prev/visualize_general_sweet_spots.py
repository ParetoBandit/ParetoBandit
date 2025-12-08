#!/usr/bin/env python3
"""
Create a general sweet spot visualization (not task-specific).

Shows the same constraint configuration but using general/average quality scores
instead of task-specific ones.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from llm_jury.ranking.chebyshev import ChebyshevRanker, RankingStrategy
from llm_jury.core.models import ModelMetadata, RoutingDecision, PromptCategory, ProductArchetype


def create_model(d):
    """Create ModelMetadata from dict."""
    m = ModelMetadata(
        name=d['name'], archetype=ProductArchetype.FRONTIER,
        input_cost_per_m=d.get('price_1m_input', 1.0),
        output_cost_per_m=d.get('price_1m_output', 2.0),
        median_latency_ms=1000.0, context_window_k=128, param_count_b=70.0,
        mmlu_score=0, gpqa_score=0, math_score=0, ifeval_score=0, tool_use_ability=0.5
    )
    for key in ['intelligence_index', 'coding_index', 'math_index', 'output_tokens_per_second']:
        setattr(m, key, d.get(key))
    return m


def load_data(filter_outliers=True, cost_percentile=95):
    """Load model data with optional outlier filtering."""
    cache_path = Path("data/models_complete_composite_indices.json")
    with open(cache_path) as f:
        raw_data = json.load(f)
    
    valid_data = [m for m in raw_data if m.get('price_1m_input') and m.get('price_1m_input') > 0]
    
    if filter_outliers:
        costs = [(d.get('price_1m_input', 0) * 0.75 + d.get('price_1m_output', 0) * 0.25) for d in valid_data]
        cost_threshold = np.percentile(costs, cost_percentile)
        
        filtered_data = []
        for d, cost in zip(valid_data, costs):
            if cost <= cost_threshold:
                filtered_data.append(d)
        
        print(f"  Filtered out {len(valid_data) - len(filtered_data)} cost outliers (>{cost_percentile}th percentile: ${cost_threshold:.2f})")
        valid_data = filtered_data
    
    models = [create_model(d) for d in valid_data]
    return models, valid_data


def get_quality_scores(models, valid_data):
    """Get general quality scores for all models."""
    from llm_jury.ranking.quality_scorer import QualityScorer
    
    scorer = QualityScorer(all_models_data=valid_data)
    decision = RoutingDecision(
        category=PromptCategory.GENERAL, 
        archetype=ProductArchetype.FRONTIER, 
        reason="general"
    )
    
    scores = []
    for m, d in zip(models, valid_data):
        q_score = scorer.calculate_quality_score(d, PromptCategory.GENERAL)
        scores.append(q_score)
    
    return np.array(scores)


def plot_general_sweet_spots(models, valid_data, output_path="blog/general_sweet_spots.png"):
    """
    Plot general (non-task-specific) sweet spot visualization.
    """
    baseline_name = "Gemini 3 Pro Preview (high)"
    baseline = next((m for m in models if baseline_name.lower() in m.name.lower()), models[0])
    base_cost = baseline.input_cost_per_m * 0.75 + baseline.output_cost_per_m * 0.25
    
    # Get general quality scores
    quality_scores = get_quality_scores(models, valid_data)
    costs = np.array([m.input_cost_per_m * 0.75 + m.output_cost_per_m * 0.25 for m in models])
    
    baseline_idx = models.index(baseline)
    base_quality = quality_scores[baseline_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 11))
    
    # Plot all models (darker gray for better contrast)
    ax.scatter(costs, quality_scores, c='#666666', alpha=0.4, s=60, zorder=1, label='All Models')
    
    # Get sweet spot models (no speed constraint by default due to missing data)
    ranker = ChebyshevRanker(
        baseline_model=baseline,
        all_models_data=valid_data,
        strategy=RankingStrategy.VALUE_OPTIMIZED,
        quality_range=(0.80, 0.95),
        cost_range=(0.10, 0.30),
        speed_range=None  # Optional: uncomment to add speed constraint
        # speed_range=(0.30, 10.0)  # Requires ≥30% of baseline speed
    )
    
    decision = RoutingDecision(
        category=PromptCategory.GENERAL, 
        archetype=ProductArchetype.FRONTIER, 
        reason="general"
    )
    
    results = ranker.rank(models, decision, top_k=10, return_detailed=True)
    
    # Highlight sweet spot models
    if results:
        # Plot each model individually for legend
        for i, r in enumerate(results, 1):
            cost = r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25
            label = r.name if len(r.name) <= 35 else r.name[:32] + "..."
            
            ax.scatter([cost], [r.quality_score], 
                      c='#4ECDC4', alpha=0.9, s=350, 
                      edgecolors='black', linewidth=2.5, zorder=3,
                      label=f"{i}. {label}")
            
            # Add number annotation on the dot
            ax.annotate(f"{i}", 
                       xy=(cost, r.quality_score),
                       xytext=(0, 0), textcoords='offset points',
                       fontsize=13, ha='center', va='center',
                       fontweight='bold', color='white', zorder=4)
    
    # Highlight baseline
    ax.scatter([base_cost], [base_quality], 
              c='gold', s=600, marker='*', 
              edgecolors='black', linewidth=3, zorder=5,
              label=f'Baseline: {baseline_name}')
    
    # Draw sweet spot zone
    q_min_abs = base_quality * 0.80
    q_max_abs = base_quality * 0.95
    c_min_abs = base_cost * 0.10
    c_max_abs = base_cost * 0.30
    
    rect = mpatches.Rectangle(
        (c_min_abs, q_min_abs), 
        c_max_abs - c_min_abs, 
        q_max_abs - q_min_abs,
        linewidth=3, 
        edgecolor='#4ECDC4', 
        facecolor='#4ECDC4',
        alpha=0.15,
        linestyle='--',
        label='Sweet Spot Zone'
    )
    ax.add_patch(rect)
    
    # Add reference lines from baseline
    ax.axhline(base_quality, color='gold', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvline(base_cost, color='gold', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Cost ($/M tokens)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Quality Score (General)', fontsize=16, fontweight='bold')
    ax.set_title('General Sweet Spot Models: Best Value Alternatives\n' +
                f'Constraints: 80-95% Quality, 10-30% Cost vs {baseline_name}',
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, linewidth=1)
    
    # Set reasonable limits
    max_visible_cost = min(max(costs), base_cost * 2)
    ax.set_xlim(-0.1, max_visible_cost)
    ax.set_ylim(0, 110)
    
    # Add legend with model names (split into two columns if too many)
    if results:
        ncol = 2 if len(results) > 5 else 1
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95, 
                 edgecolor='#4ECDC4', fancybox=True, ncol=ncol)
    
    # Add summary text box
    if results:
        savings_avg = np.mean([((base_cost - (r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25)) / base_cost) * 100 for r in results])
        quality_avg = np.mean([r.quality_score for r in results])
        
        summary_text = f"Sweet Spot Summary:\n"
        summary_text += f"  • {len(results)} models found\n"
        summary_text += f"  • Avg Quality: {quality_avg:.1f}/100 ({(quality_avg/base_quality)*100:.0f}% of baseline)\n"
        summary_text += f"  • Avg Savings: {savings_avg:.0f}%\n"
        summary_text += f"  • Baseline: ${base_cost:.2f}/M tokens"
        
        ax.text(0.02, 0.98, summary_text,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                        edgecolor='#4ECDC4', linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate general sweet spot visualization."""
    print("=" * 80)
    print("GENERATING GENERAL SWEET SPOT VISUALIZATION")
    print("=" * 80)
    
    print("\nLoading data...")
    models, valid_data = load_data()
    print(f"✓ Loaded {len(models)} models")
    
    print("\nGenerating visualization...")
    plot_general_sweet_spots(models, valid_data)
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated file: blog/general_sweet_spots.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

