#!/usr/bin/env python3
"""
Create compelling visualizations demonstrating the value of configurable constrained optimization.

This script generates multiple plots showing:
1. Sweet spot zones with different constraint configurations
2. Comparison of different baselines
3. Pareto frontier with feasible regions
4. Savings vs quality tradeoff
5. Task-specific optimization results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Tuple

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
    """
    Load model data with optional outlier filtering.
    
    Args:
        filter_outliers: If True, remove cost outliers
        cost_percentile: Keep models below this cost percentile (default: 95th)
    """
    cache_path = Path("data/models_complete_composite_indices.json")
    with open(cache_path) as f:
        raw_data = json.load(f)
    
    valid_data = [m for m in raw_data if m.get('price_1m_input') and m.get('price_1m_input') > 0]
    
    if filter_outliers:
        # Calculate costs
        costs = [(d.get('price_1m_input', 0) * 0.75 + d.get('price_1m_output', 0) * 0.25) for d in valid_data]
        cost_threshold = np.percentile(costs, cost_percentile)
        
        # Filter out cost outliers
        filtered_data = []
        for d, cost in zip(valid_data, costs):
            if cost <= cost_threshold:
                filtered_data.append(d)
        
        print(f"  Filtered out {len(valid_data) - len(filtered_data)} cost outliers (>{cost_percentile}th percentile: ${cost_threshold:.2f})")
        valid_data = filtered_data
    
    models = [create_model(d) for d in valid_data]
    return models, valid_data


def get_quality_scores(models, valid_data, category=PromptCategory.CODING):
    """Get quality scores for all models."""
    from llm_jury.ranking.quality_scorer import QualityScorer
    
    scorer = QualityScorer(all_models_data=valid_data)
    decision = RoutingDecision(category=category, archetype=ProductArchetype.FRONTIER, reason="test")
    
    scores = []
    for m, d in zip(models, valid_data):
        model_data = d
        q_score = scorer.calculate_quality_score(model_data, category)
        scores.append(q_score)
    
    return np.array(scores)


def plot_sweet_spot_zones(models, valid_data, baseline_name="Gemini 3 Pro Preview (high)", 
                          output_path="blog/sweet_spot_zones.png",
                          speed_constraint=None):  # Default: no speed constraint (optional)
    """
    Plot 1: Show value opportunities - models with near-Gemini quality at much lower cost AND faster speed.
    Professional visualization highlighting where users can save money while maintaining quality.
    
    Args:
        speed_constraint: Minimum speed ratio vs baseline (1.10 = 10% faster, None = no constraint)
    """
    baseline = next((m for m in models if baseline_name.lower() in m.name.lower()), models[0])
    base_cost = baseline.input_cost_per_m * 0.75 + baseline.output_cost_per_m * 0.25
    base_speed = getattr(baseline, 'output_tokens_per_second', None) or 100.0  # Default if missing
    
    # Get quality scores
    quality_scores = get_quality_scores(models, valid_data)
    
    # Get costs and speeds
    costs = np.array([m.input_cost_per_m * 0.75 + m.output_cost_per_m * 0.25 for m in models])
    speeds = np.array([getattr(m, 'output_tokens_per_second', None) or 0.0 for m in models])
    
    # Filter out outliers (above 95th percentile) for cleaner visualization
    cost_threshold = np.percentile(costs, 95)
    keep_indices = costs <= cost_threshold
    models_filtered = [m for m, keep in zip(models, keep_indices) if keep]
    quality_scores = quality_scores[keep_indices]
    costs = costs[keep_indices]
    speeds = speeds[keep_indices]
    
    # Update valid_data to match filtered models
    valid_data_filtered = [d for d, keep in zip(valid_data, keep_indices) if keep]
    
    # Calculate baseline quality
    baseline_idx = models.index(baseline)
    base_quality = quality_scores[baseline_idx]
    
    # Create professional figure with proper aspect ratio (wider than tall)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set a clean white background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    # Calculate cost ratio as percentage of baseline
    cost_ratio = (costs / base_cost) * 100
    
    # Calculate speed ratio (higher is better - faster than baseline)
    speed_ratio = speeds / base_speed
    
    # Apply speed constraint if specified
    if speed_constraint is not None:
        # Models must be at least speed_constraint times as fast as baseline
        # Models without speed data (speeds == 0) do NOT meet the constraint
        speed_meets_constraint = (speed_ratio >= speed_constraint) & (speeds > 0)
        speed_label = f", ≥{int((speed_constraint-1)*100)}% faster"
    else:
        speed_meets_constraint = np.ones(len(models_filtered), dtype=bool)
        speed_label = ""
    
    # Segment models into categories based on value proposition
    # Using absolute quality scores (0-100 scale) + speed constraint
    elite_mask = (quality_scores >= 90) & (cost_ratio <= 30) & speed_meets_constraint
    great_mask = (quality_scores >= 80) & (quality_scores < 90) & (cost_ratio <= 30) & speed_meets_constraint
    good_mask = (quality_scores >= 70) & (quality_scores < 80) & (cost_ratio <= 30) & speed_meets_constraint
    baseline_mask = np.array([m.name == baseline_name for m in models_filtered])
    other_mask = ~(elite_mask | great_mask | good_mask | baseline_mask)
    
    # Plot background models first (others) - subtle gray
    ax.scatter(costs[other_mask], quality_scores[other_mask],
              c='#CCCCCC', alpha=0.5, s=80, zorder=2, label='Other Models')
    
    # Good value (bronze) - warm brown
    if np.any(good_mask):
        ax.scatter(costs[good_mask], quality_scores[good_mask],
                  c='#CD7F32', alpha=0.9, s=180, edgecolors='#8B4513', linewidth=2,
                  zorder=4, label=f'Good Value ({np.sum(good_mask)} models)\n70-80% Quality, <30% Cost{speed_label}')
    
    # Great value (silver) - cool silver
    if np.any(great_mask):
        ax.scatter(costs[great_mask], quality_scores[great_mask],
                  c='#A8A8A8', alpha=0.9, s=220, edgecolors='#505050', linewidth=2,
                  zorder=5, label=f'Great Value ({np.sum(great_mask)} models)\n80-90% Quality, <30% Cost{speed_label}')
    
    # Elite value (gold) - bright gold
    if np.any(elite_mask):
        ax.scatter(costs[elite_mask], quality_scores[elite_mask],
                  c='#FFD700', alpha=1.0, s=280, edgecolors='#B8860B', linewidth=2.5,
                  zorder=6, label=f'🏆 Elite Value ({np.sum(elite_mask)} models)\n90%+ Quality, <30% Cost{speed_label}')
    
    # Label top 3 elite models with clean annotations
    elite_indices = np.where(elite_mask)[0]
    # Sort by quality score descending
    elite_indices = elite_indices[np.argsort(quality_scores[elite_indices])[::-1]]
    
    # Dynamic label positions to avoid overlaps - position labels away from data points
    # First model: top-right, second: bottom-right, third: left side
    label_positions = [(60, 25), (60, -50), (-120, -30)]
    for i, idx in enumerate(elite_indices[:3]):
        m = models_filtered[idx]
        q = quality_scores[idx]
        c = costs[idx]
        s = speeds[idx]
        q_pct = q  # Quality score is already 0-100
        cost_pct = (c / base_cost) * 100
        speed_pct = (s / base_speed) * 100 if s > 0 else 0
        
        # Clean model name
        name = m.name if len(m.name) <= 25 else m.name[:22] + "..."
        if s > 0:
            label_text = f'{name}\nQ={q_pct:.0f}, Cost={cost_pct:.0f}%, Speed={speed_pct:.0f}%'
        else:
            label_text = f'{name}\nQ={q_pct:.0f}, Cost={cost_pct:.0f}%'
        
        xytext = label_positions[i]
        ha = 'left' if xytext[0] > 0 else 'right'
        
        ax.annotate(label_text,
                   xy=(c, q),
                   xytext=xytext, textcoords='offset points',
                   fontsize=9, ha=ha, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFACD', 
                           edgecolor='#DAA520', linewidth=1.5, alpha=0.95),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#DAA520'),
                   zorder=10)  # Ensure labels are on top
    
    # Add shaded region for elite value zone (90%+ quality = 90 on y-axis, <30% cost)
    from matplotlib.patches import Rectangle
    elite_zone = Rectangle((0, 90), base_cost * 0.30, 15,  # From y=90 to y=105
                           facecolor='#FFD700', alpha=0.12, edgecolor='#DAA520', 
                           linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(elite_zone)
    
    # Add reference lines - subtle and professional (using absolute quality scores)
    ax.axhline(90, color='#DAA520', linestyle='--', alpha=0.6, linewidth=1.5)  # 90% quality
    ax.axhline(80, color='#808080', linestyle='--', alpha=0.4, linewidth=1)    # 80% quality
    ax.axhline(70, color='#CD7F32', linestyle='--', alpha=0.3, linewidth=1)    # 70% quality
    ax.axvline(base_cost * 0.30, color='#228B22', linestyle=':', alpha=0.5, linewidth=2)
    
    # Add zone label at the top
    ax.text(base_cost * 0.15, 97, 'ELITE VALUE ZONE',
           ha='center', va='center', fontsize=11, fontweight='bold', color='#B8860B',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#DAA520', 
                    linewidth=2, alpha=0.95))
    
    # Professional formatting
    ax.set_xlabel('Cost ($/M tokens)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Quality Score (0-100)', fontsize=13, fontweight='bold', labelpad=10)
    speed_title = f" & {int((speed_constraint-1)*100)}%+ Faster" if speed_constraint else ""
    ax.set_title(f'Value Opportunities: Near-{baseline_name.split(" (")[0]} Quality at Lower Cost{speed_title}', 
                fontsize=15, fontweight='bold', pad=15)
    
    # Clean legend inside plot
    legend = ax.legend(loc='lower right', fontsize=9, framealpha=0.95, 
                      fancybox=True, shadow=True, borderpad=1)
    legend.get_frame().set_edgecolor('#CCCCCC')
    
    # Subtle grid
    ax.grid(True, alpha=0.2, linewidth=0.5, linestyle='-')
    ax.set_axisbelow(True)
    
    # Set axis limits to show all value opportunities clearly
    ax.set_xlim(-0.05, base_cost * 0.50)  # Show up to 50% of baseline cost
    ax.set_ylim(0, 105)  # Show full quality range for proper proportions
    
    # Add axis annotations for reference lines (using absolute quality scores)
    ax.text(base_cost * 0.51, 90, '90% quality', fontsize=9, color='#DAA520', 
           va='center', fontweight='bold')
    ax.text(base_cost * 0.51, 80, '80% quality', fontsize=9, color='#808080', 
           va='center')
    ax.text(base_cost * 0.51, 70, '70% quality', fontsize=9, color='#CD7F32', 
           va='center')
    ax.text(base_cost * 0.30, 5, '30% cost', fontsize=9, color='#228B22', 
           ha='center', rotation=90)
    
    # Save with proper layout - higher DPI for clarity
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_baseline_comparison(models, valid_data, output_path="blog/baseline_comparison.png"):
    """
    Plot 2: Compare how different baselines produce different sweet spot recommendations.
    """
    baseline_names = ["Gemini 3 Pro Preview (high)", "GPT-5.1 (high)", "GPT-4o (Nov '24)"]
    
    # Get quality scores once
    quality_scores = get_quality_scores(models, valid_data)
    costs = np.array([m.input_cost_per_m * 0.75 + m.output_cost_per_m * 0.25 for m in models])
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, baseline_name in enumerate(baseline_names):
        ax = axes[idx]
        
        baseline = next((m for m in models if baseline_name.lower() in m.name.lower()), None)
        if baseline is None:
            continue
            
        base_cost = baseline.input_cost_per_m * 0.75 + baseline.output_cost_per_m * 0.25
        baseline_idx = models.index(baseline)
        base_quality = quality_scores[baseline_idx]
        
        # Plot all models
        ax.scatter(costs, quality_scores, c='lightgray', alpha=0.3, s=30, zorder=1)
        
        # Get sweet spot models (80-95% quality, 10-30% cost)
        ranker = ChebyshevRanker(
            baseline_model=baseline,
            all_models_data=valid_data,
            strategy=RankingStrategy.VALUE_OPTIMIZED,
            quality_range=(0.80, 0.95),
            cost_range=(0.10, 0.30)
        )
        
        decision = RoutingDecision(
            category=PromptCategory.CODING, 
            archetype=ProductArchetype.FRONTIER, 
            reason="test"
        )
        
        results = ranker.rank(models, decision, top_k=5, return_detailed=True)
        
        # Highlight sweet spot models
        if results:
            sweet_costs = [r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25 for r in results]
            sweet_quality = [r.quality_score for r in results]
            
            ax.scatter(sweet_costs, sweet_quality, 
                      c='#4ECDC4', alpha=0.8, s=200, 
                      edgecolors='black', linewidth=2, zorder=3,
                      label=f'{len(results)} Sweet Spot Models')
            
            # Label top 3
            for i, r in enumerate(results[:3]):
                cost = r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25
                ax.annotate(f"{i+1}. {r.name[:20]}", 
                           xy=(cost, r.quality_score),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Highlight baseline
        ax.scatter([base_cost], [base_quality], 
                  c='gold', s=400, marker='*', 
                  edgecolors='black', linewidth=2, zorder=4,
                  label=f'Baseline')
        
        # Draw sweet spot zone
        q_min_abs = base_quality * 0.80
        q_max_abs = base_quality * 0.95
        c_min_abs = base_cost * 0.10
        c_max_abs = base_cost * 0.30
        
        rect = mpatches.Rectangle(
            (c_min_abs, q_min_abs), 
            c_max_abs - c_min_abs, 
            q_max_abs - q_min_abs,
            linewidth=2, 
            edgecolor='#4ECDC4', 
            facecolor='#4ECDC4',
            alpha=0.1,
            linestyle='--'
        )
        ax.add_patch(rect)
        
        # Formatting
        ax.set_xlabel('Cost ($/M tokens)', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{baseline_name}\n(${base_cost:.2f}, Q={base_quality:.1f})', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.2)
        # Set reasonable x-limit based on baseline (outliers filtered)
        max_visible_cost = min(max(costs) * 0.5, base_cost * 2)
        ax.set_xlim(0, max_visible_cost)
        ax.set_ylim(0, 110)
    
    fig.suptitle('Different Baselines → Different Sweet Spots', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_pareto_frontier(models, valid_data, output_path="blog/pareto_frontier_sweet_spot.png"):
    """
    Plot 3: Show Pareto frontier with sweet spot highlighted.
    """
    baseline_name = "Gemini 3 Pro Preview (high)"
    baseline = next((m for m in models if baseline_name.lower() in m.name.lower()), models[0])
    base_cost = baseline.input_cost_per_m * 0.75 + baseline.output_cost_per_m * 0.25
    
    # Get quality scores
    quality_scores = get_quality_scores(models, valid_data)
    costs = np.array([m.input_cost_per_m * 0.75 + m.output_cost_per_m * 0.25 for m in models])
    
    baseline_idx = models.index(baseline)
    base_quality = quality_scores[baseline_idx]
    
    # Find Pareto frontier (higher quality OR lower cost)
    pareto_indices = []
    for i in range(len(models)):
        is_pareto = True
        for j in range(len(models)):
            if i != j:
                # j dominates i if it has both higher quality AND lower cost
                if quality_scores[j] > quality_scores[i] and costs[j] < costs[i]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    # Sort Pareto points by cost
    pareto_indices = sorted(pareto_indices, key=lambda i: costs[i])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot all models
    ax.scatter(costs, quality_scores, c='lightgray', alpha=0.3, s=50, label='All Models', zorder=1)
    
    # Plot Pareto frontier
    pareto_costs = costs[pareto_indices]
    pareto_quality = quality_scores[pareto_indices]
    ax.plot(pareto_costs, pareto_quality, 'b--', linewidth=2, alpha=0.5, label='Pareto Frontier', zorder=2)
    ax.scatter(pareto_costs, pareto_quality, c='blue', alpha=0.6, s=100, zorder=2)
    
    # Get sweet spot models
    ranker = ChebyshevRanker(
        baseline_model=baseline,
        all_models_data=valid_data,
        strategy=RankingStrategy.VALUE_OPTIMIZED,
        quality_range=(0.80, 0.95),
        cost_range=(0.10, 0.30)
    )
    
    decision = RoutingDecision(
        category=PromptCategory.CODING, 
        archetype=ProductArchetype.FRONTIER, 
        reason="test"
    )
    
    results = ranker.rank(models, decision, top_k=8, return_detailed=True)
    
    # Highlight sweet spot models
    if results:
        sweet_costs = [r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25 for r in results]
        sweet_quality = [r.quality_score for r in results]
        
        ax.scatter(sweet_costs, sweet_quality, 
                  c='#FF6B6B', alpha=0.9, s=250, 
                  edgecolors='black', linewidth=2, zorder=4,
                  label=f'Sweet Spot ({len(results)} models)')
        
        # Label top 3
        for i, r in enumerate(results[:3]):
            cost = r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25
            savings = ((base_cost - cost) / base_cost) * 100
            ax.annotate(f"#{i+1}: {r.name[:25]}\n(↓{savings:.0f}% cost)", 
                       xy=(cost, r.quality_score),
                       xytext=(15, -15 - i*15), textcoords='offset points',
                       fontsize=9, ha='left',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF6B6B', alpha=0.3),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#FF6B6B'))
    
    # Highlight baseline
    ax.scatter([base_cost], [base_quality], 
              c='gold', s=500, marker='*', 
              edgecolors='black', linewidth=2.5, zorder=5,
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
        edgecolor='#FF6B6B', 
        facecolor='none',
        linestyle='--',
        label='Sweet Spot Zone\n(80-95% quality, 10-30% cost)'
    )
    ax.add_patch(rect)
    
    # Add utopia point
    ax.scatter([0], [100], c='green', s=300, marker='D', 
              edgecolors='black', linewidth=2, zorder=5,
              label='Utopia Point (0 cost, 100 quality)')
    
    # Formatting
    ax.set_xlabel('Cost ($/M tokens)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Quality Score', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier & Sweet Spot Zone:\nConstrained Optimization Finds Best Value Models', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    # Set reasonable x-limit (outliers filtered)
    max_visible_cost = min(max(costs) * 0.6, base_cost * 2)
    ax.set_xlim(-0.2, max_visible_cost)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_savings_vs_quality(models, valid_data, output_path="blog/savings_vs_quality_tradeoff.png"):
    """
    Plot 4: Show savings percentage vs quality retention across different configurations.
    """
    baseline_name = "Gemini 3 Pro Preview (high)"
    baseline = next((m for m in models if baseline_name.lower() in m.name.lower()), models[0])
    base_cost = baseline.input_cost_per_m * 0.75 + baseline.output_cost_per_m * 0.25
    
    quality_scores = get_quality_scores(models, valid_data)
    baseline_idx = models.index(baseline)
    base_quality = quality_scores[baseline_idx]
    
    # Different configurations
    configs = [
        {"name": "Conservative", "q": (0.90, 0.98), "c": (0.10, 0.40), "color": "#FF6B6B"},  # Red
        {"name": "Balanced", "q": (0.80, 0.95), "c": (0.10, 0.30), "color": "#4ECDC4"},      # Teal
        {"name": "Aggressive", "q": (0.70, 0.90), "c": (0.05, 0.20), "color": "#9B59B6"},   # Purple
    ]
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    for config in configs:
        ranker = ChebyshevRanker(
            baseline_model=baseline,
            all_models_data=valid_data,
            strategy=RankingStrategy.VALUE_OPTIMIZED,
            quality_range=config["q"],
            cost_range=config["c"]
        )
        
        decision = RoutingDecision(
            category=PromptCategory.CODING, 
            archetype=ProductArchetype.FRONTIER, 
            reason="test"
        )
        
        results = ranker.rank(models, decision, top_k=10, return_detailed=True)
        
        if results:
            quality_retention = [(r.quality_score / base_quality) * 100 for r in results]
            cost_savings = [((base_cost - (r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25)) / base_cost) * 100 for r in results]
            
            # Add small jitter to prevent exact overlap
            jitter_x = np.random.normal(0, 0.3, len(quality_retention))
            jitter_y = np.random.normal(0, 0.3, len(cost_savings))
            quality_retention_j = [q + j for q, j in zip(quality_retention, jitter_x)]
            cost_savings_j = [c + j for c, j in zip(cost_savings, jitter_y)]
            
            # Create constraint label
            q_min, q_max = config["q"]
            c_min, c_max = config["c"]
            constraint_label = f'{config["name"]}: {q_min:.0%}-{q_max:.0%} quality, {c_min:.0%}-{c_max:.0%} cost'
            
            ax.scatter(quality_retention_j, cost_savings_j, 
                      c=config["color"], alpha=0.8, s=200, 
                      edgecolors='black', linewidth=1.5,
                      label=constraint_label, zorder=3)
            
            # Connect points in order of Chebyshev score (use original positions, not jittered)
            ax.plot(quality_retention, cost_savings, 
                   c=config["color"], alpha=0.3, linewidth=2, linestyle='--', zorder=2)
            
            # Label the best model (first in results, lowest Chebyshev score)
            # For Balanced, use second best if first is MiniMax (already shown in Aggressive)
            if results:
                # Select which model to label
                if config["name"] == "Balanced" and len(results) > 1 and "minimax" in results[0].name.lower():
                    best = results[1]  # Use second best to avoid duplicate
                else:
                    best = results[0]  # Use best model
                
                best_q = (best.quality_score / base_quality) * 100
                best_c = ((base_cost - (best.metadata.input_cost_per_m * 0.75 + best.metadata.output_cost_per_m * 0.25)) / base_cost) * 100
                label = best.name if len(best.name) <= 25 else best.name[:22] + "..."
                
                # Position labels differently for each config to avoid overlap
                if config["name"] == "Conservative":
                    xytext = (-50, -30)
                    ha = 'right'
                elif config["name"] == "Balanced":
                    xytext = (10, -35)
                    ha = 'left'
                else:  # Aggressive
                    xytext = (10, 15)
                    ha = 'left'
                
                ax.annotate(label,
                           xy=(best_q, best_c),
                           xytext=xytext, textcoords='offset points',
                           fontsize=9, ha=ha,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                   edgecolor=config["color"], linewidth=2, alpha=0.9),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color=config["color"]))
    
    # Add reference lines
    ax.axvline(100, color='gold', linestyle='--', linewidth=2, alpha=0.5, label='Baseline Quality')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Add ideal zones with different colors
    ax.axvspan(80, 95, alpha=0.08, color='#4ECDC4', label='Sweet Spot Quality Zone (80-95%)')
    ax.axhspan(70, 90, alpha=0.08, color='#FFB347', label='Sweet Spot Savings Zone (70-90%)')
    
    # Formatting with baseline model name
    baseline_name = "Gemini 3 Pro Preview (high)"
    ax.set_xlabel(f'Quality Retention (% of {baseline_name})', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Cost Savings (% vs {baseline_name})', fontsize=14, fontweight='bold')
    ax.set_title('Savings vs Quality Tradeoff: Different Constraint Configurations', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(65, 105)
    ax.set_ylim(-5, 100)
    
    # Add annotations
    ax.text(97, 85, 'High Quality\nHigh Savings', fontsize=11, ha='center', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    ax.text(72, 85, 'Lower Quality\nHigh Savings', fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_task_specific_optimization(models, valid_data, output_path="blog/task_specific_sweet_spots.png"):
    """
    Plot 5: Show how sweet spots differ across different task types.
    """
    baseline_name = "Gemini 3 Pro Preview (high)"
    baseline = next((m for m in models if baseline_name.lower() in m.name.lower()), models[0])
    base_cost = baseline.input_cost_per_m * 0.75 + baseline.output_cost_per_m * 0.25
    
    tasks = [
        (PromptCategory.CODING, "Coding", "#FF6B6B"),
        (PromptCategory.DATA_SCIENCE, "Data Science", "#4ECDC4"),
        (PromptCategory.CREATIVE, "Creative", "#95E1D3"),
        (PromptCategory.GENERAL, "General", "#F7DC6F"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, (category, name, color) in enumerate(tasks):
        ax = axes[idx]
        
        # Get task-specific quality scores
        quality_scores = get_quality_scores(models, valid_data, category)
        costs = np.array([m.input_cost_per_m * 0.75 + m.output_cost_per_m * 0.25 for m in models])
        
        baseline_idx = models.index(baseline)
        base_quality = quality_scores[baseline_idx]
        
        # Plot all models (darker gray for better contrast)
        ax.scatter(costs, quality_scores, c='#666666', alpha=0.4, s=40, zorder=1)
        
        # Get sweet spot models for this task
        ranker = ChebyshevRanker(
            baseline_model=baseline,
            all_models_data=valid_data,
            strategy=RankingStrategy.VALUE_OPTIMIZED,
            quality_range=(0.80, 0.95),
            cost_range=(0.10, 0.30),
            speed_range=None  # No speed constraint (many models lack speed data)
        )
        
        decision = RoutingDecision(
            category=category, 
            archetype=ProductArchetype.FRONTIER, 
            reason="test"
        )
        
        results = ranker.rank(models, decision, top_k=5, return_detailed=True)
        
        # Highlight sweet spot models
        if results:
            sweet_costs = [r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25 for r in results]
            sweet_quality = [r.quality_score for r in results]
            
            # Plot each model individually so we can add to legend
            for i, r in enumerate(results, 1):
                cost = r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25
                # Truncate long names for legend
                label = r.name if len(r.name) <= 30 else r.name[:27] + "..."
                
                ax.scatter([cost], [r.quality_score], 
                          c=color, alpha=0.8, s=250, 
                          edgecolors='black', linewidth=2, zorder=3,
                          label=f"{i}. {label}")
                
                # Add number annotation on the dot
                ax.annotate(f"{i}", 
                           xy=(cost, r.quality_score),
                           xytext=(0, 0), textcoords='offset points',
                           fontsize=11, ha='center', va='center',
                           fontweight='bold', color='white', zorder=4)
        
        # Highlight baseline
        ax.scatter([base_cost], [base_quality], 
                  c='gold', s=400, marker='*', 
                  edgecolors='black', linewidth=2, zorder=4)
        
        # Draw sweet spot zone
        q_min_abs = base_quality * 0.80
        q_max_abs = base_quality * 0.95
        c_min_abs = base_cost * 0.10
        c_max_abs = base_cost * 0.30
        
        rect = mpatches.Rectangle(
            (c_min_abs, q_min_abs), 
            c_max_abs - c_min_abs, 
            q_max_abs - q_min_abs,
            linewidth=2, 
            edgecolor=color, 
            facecolor=color,
            alpha=0.1,
            linestyle='--'
        )
        ax.add_patch(rect)
        
        # Formatting
        ax.set_xlabel('Cost ($/M tokens)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Quality Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} Task\n({len(results) if results else 0} sweet spot models)', 
                    fontsize=13, fontweight='bold', color=color)
        ax.grid(True, alpha=0.2)
        # Set reasonable x-limit (outliers filtered)
        max_visible_cost = min(max(costs) * 0.5, base_cost * 2)
        ax.set_xlim(0, max_visible_cost)
        ax.set_ylim(0, 110)
        
        # Add legend with model names
        if results:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.95, 
                     edgecolor=color, fancybox=True)
    
    fig.suptitle('Task-Specific Sweet Spots: Same Constraints (80-95% Quality, 10-30% Cost), Different Models', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("GENERATING CONFIGURABLE OPTIMIZATION VISUALIZATIONS")
    print("=" * 80)
    
    print("\nLoading data...")
    models, valid_data = load_data()
    print(f"✓ Loaded {len(models)} models")
    
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)
    
    print("\n1. Sweet Spot Zones (different constraint configurations)...")
    plot_sweet_spot_zones(models, valid_data)
    
    print("\n2. Baseline Comparison (different reference models)...")
    plot_baseline_comparison(models, valid_data)
    
    print("\n3. Pareto Frontier with Sweet Spot...")
    plot_pareto_frontier(models, valid_data)
    
    print("\n4. Savings vs Quality Tradeoff...")
    plot_savings_vs_quality(models, valid_data)
    
    print("\n5. Task-Specific Optimization...")
    plot_task_specific_optimization(models, valid_data)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - blog/sweet_spot_zones.png")
    print("  - blog/baseline_comparison.png")
    print("  - blog/pareto_frontier_sweet_spot.png")
    print("  - blog/savings_vs_quality_tradeoff.png")
    print("  - blog/task_specific_sweet_spots.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

