#!/usr/bin/env python3
"""
Professional Visualization: Use Case-Aware Model Selection

Shows how model recommendations change based on the use case:
- "For CODING, give me 80% of Gemini 3's quality at 25% cost"
- "For DATA SCIENCE, give me 80% of Gemini 3's quality at 25% cost"
- "For CREATIVE WRITING, give me 80% of Gemini 3's quality at 25% cost"

Different use cases have different benchmark weights, producing different rankings.
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

from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import PromptCategory


# Use case configurations with icons and colors
USE_CASES = {
    PromptCategory.CODING: {
        'name': 'Coding',
        'icon': '💻',
        'color': '#22c55e',  # Green
        'description': 'Software development, debugging, code review',
        'key_benchmarks': 'coding_index (40%), livecodebench (20%)'
    },
    PromptCategory.DATA_SCIENCE: {
        'name': 'Data Science', 
        'icon': '📊',
        'color': '#3b82f6',  # Blue
        'description': 'Analysis, ML, statistics, visualization',
        'key_benchmarks': 'math_index (35%), coding_index (20%)'
    },
    PromptCategory.CREATIVE: {
        'name': 'Creative Writing',
        'icon': '✨',
        'color': '#a855f7',  # Purple
        'description': 'Content creation, storytelling, copywriting',
        'key_benchmarks': 'intelligence_index (45%), mmlu_pro (30%)'
    },
    PromptCategory.GENERAL: {
        'name': 'General',
        'icon': '🎯',
        'color': '#f59e0b',  # Amber
        'description': 'Q&A, summarization, general assistance',
        'key_benchmarks': 'intelligence_index (30%), coding (18%)'
    },
}


def load_models():
    """Load model data from cache."""
    cache_path = Path("data/models_cache.json")
    with open(cache_path) as f:
        return json.load(f)


def calculate_constraint_distance(model, min_quality, max_cost, min_speed, base_quality, base_cost, base_speed):
    """Calculate Chebyshev distance to constraint boundaries."""
    gaps = []
    
    if model['quality'] < min_quality:
        gap = (min_quality - model['quality']) / base_quality
        gaps.append(('quality', gap))
    
    if model['cost'] > max_cost:
        gap = (model['cost'] - max_cost) / base_cost
        gaps.append(('cost', gap))
    
    if model['speed'] < min_speed:
        gap = (min_speed - model['speed']) / base_speed if base_speed > 0 else 0
        gaps.append(('speed', gap))
    
    if not gaps:
        return 0.0, []
    
    max_gap = max(g[1] for g in gaps)
    return max_gap, gaps


def rank_models_for_usecase(
    valid_models, 
    scorer, 
    use_case: PromptCategory,
    baseline_name: str,
    quality_threshold: float,
    cost_threshold: float,
    latency_threshold: float = None  # None = no speed constraint
):
    """Rank models for a specific use case."""
    
    # Find baseline
    baseline = next((m for m in valid_models if baseline_name.lower() in m['name'].lower()), None)
    if not baseline:
        return None, None, []
    
    # Calculate baseline metrics WITH use case
    base_quality = scorer.calculate_quality_score({'name': baseline['name']}, use_case)
    base_cost = baseline['price_1m_input'] * 0.75 + baseline['price_1m_output'] * 0.25
    base_speed = baseline.get('output_tokens_per_second', 0) or 100
    
    # Calculate thresholds
    min_quality = base_quality * quality_threshold
    max_cost = base_cost * cost_threshold
    
    # Speed constraint only if latency_threshold is specified
    if latency_threshold is not None:
        base_latency = 1000 / base_speed if base_speed > 0 else 10
        max_latency = base_latency * latency_threshold
        min_speed = 1000 / max_latency if max_latency > 0 else base_speed * 2
        has_speed_constraint = True
    else:
        min_speed = 0  # No speed requirement
        has_speed_constraint = False
    
    # Calculate metrics for all models
    model_data = []
    for m in valid_models:
        # Quality score depends on use case!
        quality = scorer.calculate_quality_score({'name': m['name']}, use_case)
        cost = m['price_1m_input'] * 0.75 + m['price_1m_output'] * 0.25
        speed = m.get('output_tokens_per_second', 0) or 0
        
        meets_quality = quality >= min_quality
        meets_cost = cost <= max_cost
        meets_speed = speed >= min_speed if has_speed_constraint else True  # Always true if no constraint
        meets_all = meets_quality and meets_cost and meets_speed
        
        # Count constraints (only count speed if it's a constraint)
        if has_speed_constraint:
            constraints_met = sum([meets_quality, meets_cost, meets_speed])
        else:
            constraints_met = sum([meets_quality, meets_cost])  # Only 2 constraints
        
        chebyshev_dist, gaps = calculate_constraint_distance(
            {'quality': quality, 'cost': cost, 'speed': speed},
            min_quality, max_cost, min_speed,
            base_quality, base_cost, base_speed
        )
        
        # Free models (cost=0) get maximum value
        value_score = quality / cost if cost > 0 else quality * 1000
        
        model_data.append({
            'name': m['name'],
            'quality': quality,
            'cost': cost,
            'speed': speed,
            'quality_ratio': quality / base_quality if base_quality > 0 else 0,
            'cost_ratio': cost / base_cost if base_cost > 0 else 0,
            'speed_ratio': speed / base_speed if base_speed > 0 else 0,
            'meets_quality': meets_quality,
            'meets_cost': meets_cost,
            'meets_speed': meets_speed,
            'meets_all': meets_all,
            'constraints_met': constraints_met,
            'chebyshev_distance': chebyshev_dist,
            'value_score': value_score,
            'is_baseline': baseline_name.lower() in m['name'].lower()
        })
    
    # Sort by hybrid ranking
    model_data.sort(key=lambda x: (
        x['constraints_met'],
        -x['chebyshev_distance'],
        x['value_score']
    ), reverse=True)
    
    thresholds = {
        'min_quality': min_quality,
        'max_cost': max_cost,
        'min_speed': min_speed,
        'base_quality': base_quality,
        'base_cost': base_cost,
        'base_speed': base_speed,
        'has_speed_constraint': has_speed_constraint,
        'total_constraints': 3 if has_speed_constraint else 2
    }
    
    return model_data, thresholds, [m for m in model_data if m['meets_all']]


def create_single_usecase_plot(
    baseline_name: str = "Gemini 3 Pro Preview (high)",
    use_case: PromptCategory = PromptCategory.CODING,
    quality_threshold: float = 0.80,
    cost_threshold: float = 0.25,
    latency_threshold: float = None,  # None = no speed constraint
    output_path: str = "blog/usecase_coding.png"
):
    """Create a single use-case specific visualization."""
    
    uc_config = USE_CASES[use_case]
    
    print("=" * 70)
    print(f"USE CASE: {uc_config['icon']} {uc_config['name'].upper()}")
    print("=" * 70)
    
    # Load data
    all_models = load_models()
    valid_models = [
        m for m in all_models 
        if m.get('price_1m_input') and m.get('price_1m_input') > 0
        and m.get('intelligence_index') and m.get('intelligence_index') > 0
    ]
    
    print(f"\n📊 Loaded {len(valid_models)} models")
    
    # Initialize scorer
    scorer = QualityScorer(all_models_data=valid_models)
    
    # Get rankings for this use case
    ranked_models, thresholds, winners = rank_models_for_usecase(
        valid_models, scorer, use_case, baseline_name,
        quality_threshold, cost_threshold, latency_threshold
    )
    
    if not ranked_models:
        print("❌ Could not find baseline model")
        return
    
    print(f"\n✅ Models meeting ALL constraints: {len(winners)}")
    print(f"🏆 Top recommendations for {uc_config['name']}:")
    for i, m in enumerate(ranked_models[:5], 1):
        status = "✓" if m['meets_all'] else f"{m['constraints_met']}/3"
        print(f"   {i}. [{status}] {m['name'][:35]}")
    
    # =========================================================================
    # CREATE VISUALIZATION
    # =========================================================================
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 11))
    
    # Color palette
    COLORS = {
        'bg': '#0a0e17',
        'panel': '#131a2a',
        'grid': '#1e2738',
        'text': '#e8eaed',
        'muted': '#7a8599',
        'accent': uc_config['color'],  # Use case color as accent
        'gold': '#ffd93d',
        'baseline': '#ff6b6b',
        'partial2': '#f59e0b',
        'other': '#2d3748',
    }
    
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Main plot
    ax = fig.add_axes([0.06, 0.12, 0.58, 0.78])
    ax.set_facecolor(COLORS['panel'])
    
    # Separate models
    exact_matches = [m for m in ranked_models if m['meets_all']]
    baseline_pt = next((m for m in ranked_models if m['is_baseline']), None)
    
    # Get top 10 non-baseline, non-exact models for highlighting
    top_10_partial = []
    for m in ranked_models:
        if m['is_baseline'] or m['meets_all']:
            continue
        top_10_partial.append(m)
        if len(top_10_partial) >= 10:
            break
    
    # Models not in top 10 and not exact matches
    top_10_names = {m['name'] for m in top_10_partial}
    others = [m for m in ranked_models 
              if not m['is_baseline'] and not m['meets_all'] and m['name'] not in top_10_names]
    
    def get_marker_size(speed_ratio):
        return 30 + min(speed_ratio, 3) * 60
    
    # Plot others (not in top 10)
    if others:
        sizes = [get_marker_size(m['speed_ratio']) for m in others]
        ax.scatter([m['cost'] for m in others], [m['quality'] for m in others],
                  c=COLORS['other'], s=sizes, alpha=0.35, zorder=1, edgecolors='none')
    
    # Plot top 10 partial matches as diamonds
    if top_10_partial:
        sizes = [get_marker_size(m['speed_ratio']) for m in top_10_partial]
        ax.scatter([m['cost'] for m in top_10_partial], [m['quality'] for m in top_10_partial],
                  c=COLORS['partial2'], s=sizes, alpha=0.85, zorder=3,
                  edgecolors='white', linewidth=1.0, marker='D')
    
    # Plot exact matches with glow
    if exact_matches:
        for m in exact_matches:
            ax.scatter([m['cost']], [m['quality']], c=COLORS['gold'],
                      s=get_marker_size(m['speed_ratio']) * 3, alpha=0.15, zorder=2)
        
        sizes = [get_marker_size(m['speed_ratio']) for m in exact_matches]
        ax.scatter([m['cost'] for m in exact_matches], [m['quality'] for m in exact_matches],
                  c=COLORS['gold'], s=sizes, alpha=1.0, zorder=4,
                  edgecolors=COLORS['accent'], linewidth=2.5, marker='o')
    
    # Plot baseline
    if baseline_pt:
        ax.scatter([baseline_pt['cost']], [baseline_pt['quality']],
                  c=COLORS['baseline'], s=500, alpha=1.0, zorder=5,
                  marker='*', edgecolors='white', linewidth=1.5)
    
    # Constraint zone and lines
    min_quality = thresholds['min_quality']
    max_cost = thresholds['max_cost']
    
    zone = Rectangle((0, min_quality), max_cost, 100 - min_quality + 5,
                     facecolor=COLORS['accent'], alpha=0.08, edgecolor='none', zorder=0)
    ax.add_patch(zone)
    
    ax.axhline(min_quality, color=COLORS['accent'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(max_cost, color=COLORS['accent'], linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Label top models
    for idx, m in enumerate(ranked_models[:4]):
        if m['is_baseline']:
            continue
        
        offsets = [(70, 25), (-90, 20), (65, -30), (-85, -25)]
        offset = offsets[idx % len(offsets)]
        ha = 'left' if offset[0] > 0 else 'right'
        
        name = m['name'][:25] + "..." if len(m['name']) > 28 else m['name']
        total_c = thresholds['total_constraints']
        prefix = "✓" if m['meets_all'] else f"[{m['constraints_met']}/{total_c}]"
        border_color = COLORS['gold'] if m['meets_all'] else COLORS['partial2']
        
        label = f"{prefix} {name}\nQ:{m['quality_ratio']*100:.0f}% C:{m['cost_ratio']*100:.0f}%"
        
        ax.annotate(label, xy=(m['cost'], m['quality']),
                   xytext=offset, textcoords='offset points',
                   fontsize=8, ha=ha, va='center', color=COLORS['text'],
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['panel'],
                           edgecolor=border_color, linewidth=2, alpha=0.95),
                   arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5),
                   zorder=10)
    
    # Styling
    ax.set_xlabel('Cost ($/M tokens)', fontsize=14, fontweight='bold', color=COLORS['text'], labelpad=12)
    ax.set_ylabel(f'Quality Score ({uc_config["name"]})', fontsize=14, fontweight='bold', color=COLORS['text'], labelpad=12)
    ax.grid(True, alpha=0.12, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    
    x_max = min(max_cost * 5, max([m['cost'] for m in ranked_models]) * 0.5)
    ax.set_xlim(-0.02, x_max)
    ax.set_ylim(25, 105)
    
    ax.tick_params(colors=COLORS['muted'], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1)
    
    # =========================================================================
    # RIGHT PANEL
    # =========================================================================
    
    info_ax = fig.add_axes([0.66, 0.12, 0.32, 0.78])
    info_ax.set_facecolor(COLORS['panel'])
    info_ax.axis('off')
    for spine in info_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1)
    
    y = 0.96
    
    # USER QUERY with use case
    info_ax.text(0.5, y, 'USER QUERY', fontsize=12, fontweight='bold',
                color=COLORS['accent'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.04
    
    query_box = FancyBboxPatch((0.03, y - 0.14), 0.94, 0.14,
                               boxstyle="round,pad=0.01,rounding_size=0.02",
                               facecolor='#1a2332', edgecolor=COLORS['accent'],
                               linewidth=2, transform=info_ax.transAxes)
    info_ax.add_patch(query_box)
    
    # Build query text based on what constraints are active
    if latency_threshold is None:
        # User explicitly said latency doesn't matter
        query_text = f'"For {uc_config["icon"]} {uc_config["name"].upper()},\ngive me {quality_threshold*100:.0f}% of {baseline_name.split(" (")[0]}\'s\nquality at {cost_threshold*100:.0f}% cost\n(latency not important)"'
    elif latency_threshold == 1.0:
        # Default: same speed as baseline (implicit constraint)
        query_text = f'"For {uc_config["icon"]} {uc_config["name"].upper()},\ngive me {quality_threshold*100:.0f}% of {baseline_name.split(" (")[0]}\'s\nquality at {cost_threshold*100:.0f}% cost"'
    else:
        # Explicit latency constraint
        query_text = f'"For {uc_config["icon"]} {uc_config["name"].upper()},\ngive me {quality_threshold*100:.0f}% of {baseline_name.split(" (")[0]}\'s\nquality at {cost_threshold*100:.0f}% cost, {latency_threshold*100:.0f}% latency"'
    info_ax.text(0.5, y - 0.07, query_text, fontsize=10, style='italic',
                color=COLORS['text'], ha='center', va='center', transform=info_ax.transAxes,
                linespacing=1.3)
    y -= 0.18
    
    # CONSTRAINTS section - show all active constraints
    info_ax.text(0.5, y, 'CONSTRAINTS', fontsize=10, fontweight='bold',
                color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.03
    
    # Quality constraint
    info_ax.text(0.08, y, f'• Quality ≥ {quality_threshold*100:.0f}%', fontsize=8,
                color=COLORS['text'], ha='left', va='top', transform=info_ax.transAxes)
    y -= 0.025
    
    # Cost constraint
    info_ax.text(0.08, y, f'• Cost ≤ {cost_threshold*100:.0f}%', fontsize=8,
                color=COLORS['text'], ha='left', va='top', transform=info_ax.transAxes)
    y -= 0.025
    
    # Latency constraint - highlight that it's the default
    if latency_threshold is None:
        info_ax.text(0.08, y, '• Latency: flexible (no constraint)', fontsize=8,
                    color=COLORS['muted'], ha='left', va='top', transform=info_ax.transAxes,
                    style='italic')
    elif latency_threshold == 1.0:
        info_ax.text(0.08, y, '• Latency ≤ 100% (default: same as baseline)', fontsize=8,
                    color=COLORS['text'], ha='left', va='top', transform=info_ax.transAxes)
    else:
        info_ax.text(0.08, y, f'• Latency ≤ {latency_threshold*100:.0f}%', fontsize=8,
                    color=COLORS['text'], ha='left', va='top', transform=info_ax.transAxes)
    y -= 0.035
    
    # USE CASE INFO
    info_ax.text(0.5, y, f'{uc_config["icon"]} USE CASE: {uc_config["name"].upper()}', 
                fontsize=11, fontweight='bold', color=COLORS['accent'],
                ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.03
    info_ax.text(0.5, y, uc_config['description'], fontsize=8,
                color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes,
                style='italic')
    y -= 0.025
    info_ax.text(0.5, y, f"Key: {uc_config['key_benchmarks']}", fontsize=7,
                color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.04
    
    # Divider
    info_ax.plot([0.05, 0.95], [y, y], color=COLORS['grid'], linewidth=1, transform=info_ax.transAxes)
    y -= 0.03
    
    # RESULTS - Show exact matches AND next best alternatives
    total_c = thresholds['total_constraints']
    
    # Get top 10 non-exact, non-baseline models for "Next Best" section
    next_best_models = []
    for m in ranked_models:
        if m['is_baseline'] or m['meets_all']:
            continue
        next_best_models.append(m)
        if len(next_best_models) >= 10:
            break
    
    if exact_matches:
        # Show exact matches
        info_ax.text(0.5, y, f'✓ {len(exact_matches)} MODELS FOUND', fontsize=11, fontweight='bold',
                    color=COLORS['gold'], ha='center', va='top', transform=info_ax.transAxes)
        y -= 0.035
        
        max_exact = 4 if next_best_models else 6  # Leave room for next best
        for i, m in enumerate(exact_matches[:max_exact]):
            name = m['name'][:24] + "..." if len(m['name']) > 27 else m['name']
            info_ax.text(0.06, y, f"#{i+1}", fontsize=9, fontweight='bold',
                        color=COLORS['gold'], ha='left', va='top', transform=info_ax.transAxes)
            info_ax.text(0.14, y, name, fontsize=8, color=COLORS['text'],
                        ha='left', va='top', transform=info_ax.transAxes)
            metrics = f"Q:{m['quality_ratio']*100:.0f}%  C:{m['cost_ratio']*100:.0f}%"
            info_ax.text(0.14, y - 0.025, metrics, fontsize=7, color=COLORS['muted'],
                        ha='left', va='top', transform=info_ax.transAxes)
            y -= 0.055
        
        # Show "Next Best" section with top 10 alternatives
        if next_best_models:
            y -= 0.005
            info_ax.plot([0.1, 0.9], [y, y], color=COLORS['grid'], linewidth=0.5, 
                        linestyle='--', transform=info_ax.transAxes)
            y -= 0.02
            
            info_ax.text(0.5, y, f'NEXT BEST (Top 10 alternatives)', fontsize=11, fontweight='bold',
                        color=COLORS['partial2'], ha='center', va='top', transform=info_ax.transAxes)
            y -= 0.035
            
            # Show top 10 with full names and constraint status
            for i, m in enumerate(next_best_models[:10]):
                name = m['name']  # Full name, no truncation
                
                # Build constraint status string showing ✓ or ✗ for each
                q_status = '✓Q' if m['meets_quality'] else '✗Q'
                c_status = '✓C' if m['meets_cost'] else '✗C'
                s_status = '✓S' if m['meets_speed'] else '✗S'
                
                info_ax.text(0.02, y, f"{i+1:2}.", fontsize=9, fontweight='bold',
                            color=COLORS['partial2'], ha='left', va='top', transform=info_ax.transAxes)
                info_ax.text(0.09, y, name, fontsize=9, color=COLORS['text'],
                            ha='left', va='top', transform=info_ax.transAxes)
                
                # Show each constraint with color coding
                x_pos = 0.80
                for status in [q_status, c_status, s_status]:
                    color = '#22c55e' if status.startswith('✓') else '#ef4444'
                    info_ax.text(x_pos, y, status, fontsize=7, fontweight='bold',
                                color=color, ha='left', va='top', transform=info_ax.transAxes)
                    x_pos += 0.065
                y -= 0.036
    
    else:
        # No exact matches - show top ranked models
        info_ax.text(0.5, y, 'NO EXACT MATCHES', fontsize=12, fontweight='bold',
                    color=COLORS['baseline'], ha='center', va='top', transform=info_ax.transAxes)
        y -= 0.03
        info_ax.text(0.5, y, f'TOP 10 RECOMMENDATIONS', fontsize=11, fontweight='bold',
                    color=COLORS['partial2'], ha='center', va='top', transform=info_ax.transAxes)
        y -= 0.035
        
        count = 0
        for m in ranked_models[:12]:
            if m['is_baseline']:
                continue
            if count >= 10:
                break
                
            name = m['name']  # Full name, no truncation
            
            # Build constraint status string showing ✓ or ✗ for each
            q_status = '✓Q' if m['meets_quality'] else '✗Q'
            c_status = '✓C' if m['meets_cost'] else '✗C'
            s_status = '✓S' if m['meets_speed'] else '✗S'
            
            info_ax.text(0.02, y, f"{count+1:2}.", fontsize=9, fontweight='bold',
                        color=COLORS['partial2'], ha='left', va='top', transform=info_ax.transAxes)
            info_ax.text(0.09, y, name, fontsize=9, color=COLORS['text'],
                        ha='left', va='top', transform=info_ax.transAxes)
            
            # Show each constraint with color coding
            x_pos = 0.80
            for status in [q_status, c_status, s_status]:
                color = '#22c55e' if status.startswith('✓') else '#ef4444'
                info_ax.text(x_pos, y, status, fontsize=7, fontweight='bold',
                            color=color, ha='left', va='top', transform=info_ax.transAxes)
                x_pos += 0.065
            y -= 0.036
            count += 1
    
    # Legend - positioned at bottom with adequate spacing
    y = 0.06
    info_ax.plot([0.05, 0.95], [y + 0.035, y + 0.035], color=COLORS['grid'], linewidth=1, transform=info_ax.transAxes)
    
    # Marker legend
    legend_items = [('*', COLORS['baseline'], 400, 'Baseline'),
                   ('o', COLORS['gold'], 100, 'Match'),
                   ('D', COLORS['partial2'], 80, 'Partial')]
    x_pos = 0.15
    for marker, color, size, label in legend_items:
        info_ax.scatter([x_pos], [y + 0.01], marker=marker, s=size/3, c=color,
                       transform=info_ax.transAxes, zorder=10,
                       edgecolors='white' if marker == '*' else 'none', linewidth=0.5)
        info_ax.text(x_pos + 0.04, y + 0.01, label, fontsize=8, color=COLORS['text'],
                    ha='left', va='center', transform=info_ax.transAxes)
        x_pos += 0.28
    
    # Constraint key (Q, C, S explanation)
    info_ax.text(0.5, y - 0.025, 'Q=Quality  C=Cost  S=Speed', fontsize=7, color=COLORS['muted'],
                ha='center', va='center', transform=info_ax.transAxes, style='italic')
    
    # Title
    fig.text(0.35, 0.95, f'{uc_config["icon"]} Model Selection for {uc_config["name"]}',
            fontsize=20, fontweight='bold', color=COLORS['text'], ha='center', va='bottom')
    fig.text(0.35, 0.92, f'Task-specific quality scoring vs {baseline_name}',
            fontsize=11, color=COLORS['muted'], ha='center', va='bottom')
    
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()
    
    return ranked_models


def create_usecase_comparison_grid(
    baseline_name: str = "Gemini 3 Pro Preview (high)",
    quality_threshold: float = 0.80,
    cost_threshold: float = 0.25,
    latency_threshold: float = None,  # None = no speed constraint
    output_path: str = "blog/usecase_comparison.png"
):
    """Create a 2x2 grid comparing all use cases."""
    
    print("=" * 70)
    print("USE CASE COMPARISON GRID")
    print("=" * 70)
    
    # Load data
    all_models = load_models()
    valid_models = [
        m for m in all_models 
        if m.get('price_1m_input') and m.get('price_1m_input') > 0
        and m.get('intelligence_index') and m.get('intelligence_index') > 0
    ]
    
    scorer = QualityScorer(all_models_data=valid_models)
    
    # Get rankings for all use cases
    use_case_results = {}
    for uc, config in USE_CASES.items():
        ranked, thresholds, winners = rank_models_for_usecase(
            valid_models, scorer, uc, baseline_name,
            quality_threshold, cost_threshold, latency_threshold
        )
        use_case_results[uc] = {
            'ranked': ranked,
            'thresholds': thresholds,
            'winners': winners,
            'config': config
        }
        print(f"\n{config['icon']} {config['name']}: {len(winners)} exact matches")
        if ranked:
            print(f"   Top: {ranked[0]['name'][:30]} (Q:{ranked[0]['quality_ratio']*100:.0f}%)")
    
    # =========================================================================
    # CREATE 2x2 GRID
    # =========================================================================
    
    COLORS = {
        'bg': '#0a0e17',
        'panel': '#131a2a',
        'grid': '#1e2738',
        'text': '#e8eaed',
        'muted': '#7a8599',
        'gold': '#ffd93d',
        'baseline': '#ff6b6b',
        'partial2': '#f59e0b',
        'other': '#2d3748',
    }
    
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(COLORS['bg'])
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15,
                  left=0.06, right=0.94, top=0.90, bottom=0.08)
    
    use_case_list = list(USE_CASES.keys())
    
    for idx, uc in enumerate(use_case_list):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor(COLORS['panel'])
        
        config = USE_CASES[uc]
        data = use_case_results[uc]
        ranked_models = data['ranked']
        thresholds = data['thresholds']
        winners = data['winners']
        
        if not ranked_models:
            continue
        
        # Separate models
        others = [m for m in ranked_models if m['constraints_met'] < 2 and not m['is_baseline']]
        partial = [m for m in ranked_models if m['constraints_met'] == 2 and not m['meets_all']]
        baseline_pt = next((m for m in ranked_models if m['is_baseline']), None)
        
        def get_marker_size(speed_ratio):
            return 20 + min(speed_ratio, 3) * 40
        
        # Plot
        if others:
            sizes = [get_marker_size(m['speed_ratio']) for m in others]
            ax.scatter([m['cost'] for m in others], [m['quality'] for m in others],
                      c=COLORS['other'], s=sizes, alpha=0.3, zorder=1, edgecolors='none')
        
        if partial:
            sizes = [get_marker_size(m['speed_ratio']) for m in partial]
            ax.scatter([m['cost'] for m in partial], [m['quality'] for m in partial],
                      c=COLORS['partial2'], s=sizes, alpha=0.7, zorder=2,
                      edgecolors='white', linewidth=0.5, marker='D')
        
        if winners:
            for m in winners:
                ax.scatter([m['cost']], [m['quality']], c=COLORS['gold'],
                          s=get_marker_size(m['speed_ratio']) * 2.5, alpha=0.15, zorder=2)
            sizes = [get_marker_size(m['speed_ratio']) for m in winners]
            ax.scatter([m['cost'] for m in winners], [m['quality'] for m in winners],
                      c=COLORS['gold'], s=sizes, alpha=1.0, zorder=4,
                      edgecolors=config['color'], linewidth=2, marker='o')
        
        if baseline_pt:
            ax.scatter([baseline_pt['cost']], [baseline_pt['quality']],
                      c=COLORS['baseline'], s=350, alpha=1.0, zorder=5,
                      marker='*', edgecolors='white', linewidth=1)
        
        # Constraint zone
        min_quality = thresholds['min_quality']
        max_cost = thresholds['max_cost']
        
        zone = Rectangle((0, min_quality), max_cost, 100 - min_quality + 5,
                         facecolor=config['color'], alpha=0.08, edgecolor='none', zorder=0)
        ax.add_patch(zone)
        
        ax.axhline(min_quality, color=config['color'], linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(max_cost, color=config['color'], linestyle='--', linewidth=2, alpha=0.7)
        
        # Label top 2 models
        labeled = 0
        for m in ranked_models[:5]:
            if m['is_baseline'] or labeled >= 2:
                continue
            
            offset = (50, 15) if labeled == 0 else (-60, -20)
            ha = 'left' if offset[0] > 0 else 'right'
            
            name = m['name'][:20] + "..." if len(m['name']) > 23 else m['name']
            total_c = thresholds['total_constraints']
            prefix = "✓" if m['meets_all'] else f"[{m['constraints_met']}/{total_c}]"
            border_color = COLORS['gold'] if m['meets_all'] else COLORS['partial2']
            
            ax.annotate(f"{prefix} {name}", xy=(m['cost'], m['quality']),
                       xytext=offset, textcoords='offset points',
                       fontsize=8, ha=ha, va='center', color=COLORS['text'],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['panel'],
                               edgecolor=border_color, linewidth=1.5, alpha=0.95),
                       arrowprops=dict(arrowstyle='->', color=border_color, lw=1.2),
                       zorder=10)
            labeled += 1
        
        # Styling
        ax.set_xlabel('Cost ($/M)', fontsize=11, color=COLORS['text'])
        ax.set_ylabel('Quality', fontsize=11, color=COLORS['text'])
        ax.grid(True, alpha=0.1, color=COLORS['grid'])
        ax.set_axisbelow(True)
        
        x_max = min(max_cost * 5, max([m['cost'] for m in ranked_models]) * 0.5)
        ax.set_xlim(-0.02, x_max)
        ax.set_ylim(25, 105)
        
        ax.tick_params(colors=COLORS['muted'], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
        
        # Title with results count
        match_text = f"{len(winners)} matches" if winners else "no exact matches"
        ax.set_title(f'{config["icon"]} {config["name"]} ({match_text})',
                    fontsize=14, fontweight='bold', color=config['color'], pad=10)
    
    # Main title
    fig.suptitle(f'Use Case Comparison: {quality_threshold*100:.0f}% Quality, {cost_threshold*100:.0f}% Cost',
                fontsize=18, fontweight='bold', color=COLORS['text'], y=0.96)
    fig.text(0.5, 0.93, f'Same constraints, different rankings based on task-specific benchmarks vs {baseline_name}',
            fontsize=11, color=COLORS['muted'], ha='center')
    
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def create_absolute_cost_plot(
    baseline_name: str = "Gemini 3 Pro Preview (high)",
    max_cost_per_m: float = 1.0,  # Absolute cost constraint in $/M tokens
    latency_threshold: float = 1.0,
    output_path: str = "blog/absolute_cost_selection.png"
):
    """
    Create a plot for absolute cost constraint:
    "Give me the best models (closest to baseline) where blended cost ≤ $X/M tokens"
    """
    print("=" * 70)
    print(f"ABSOLUTE COST CONSTRAINT: ≤ ${max_cost_per_m}/M tokens")
    print("=" * 70)
    
    # Load data
    all_models = load_models()
    valid_models = [
        m for m in all_models 
        if m.get('price_1m_input') and m.get('price_1m_input') > 0
        and m.get('intelligence_index') and m.get('intelligence_index') > 0
    ]
    
    print(f"\n📊 Loaded {len(valid_models)} models")
    
    scorer = QualityScorer(all_models_data=valid_models)
    
    # Find baseline
    baseline = next((m for m in valid_models if m.get('name') == baseline_name), None)
    if not baseline:
        print(f"❌ Baseline '{baseline_name}' not found")
        return
    
    # Calculate baseline metrics
    baseline_quality = scorer.calculate_quality_score(baseline, PromptCategory.GENERAL)
    baseline_cost = baseline.get('price_1m_input', 0) * 0.75 + baseline.get('price_1m_output', 0) * 0.25
    baseline_latency = baseline.get('measured_ttft_seconds') or (baseline.get('median_latency_ms', 500) / 1000)
    
    print(f"\n⭐ Baseline: {baseline_name}")
    print(f"   Quality: {baseline_quality:.1f}")
    print(f"   Blended Cost: ${baseline_cost:.2f}/M tokens")
    
    # Score and rank all models
    ranked_models = []
    for m in valid_models:
        quality = scorer.calculate_quality_score(m, PromptCategory.GENERAL)
        cost = m.get('price_1m_input', 0) * 0.75 + m.get('price_1m_output', 0) * 0.25
        latency = m.get('measured_ttft_seconds') or (m.get('median_latency_ms', 500) / 1000)
        speed_ratio = baseline_latency / latency if latency > 0 else 1.0
        
        # Check constraints
        meets_cost = cost <= max_cost_per_m
        meets_speed = latency_threshold is None or speed_ratio >= (1.0 / latency_threshold)
        meets_all = meets_cost and meets_speed
        
        ranked_models.append({
            'name': m.get('name'),
            'quality': quality,
            'quality_ratio': quality / baseline_quality if baseline_quality > 0 else 1.0,
            'cost': cost,
            'cost_ratio': cost / baseline_cost if baseline_cost > 0 else 1.0,
            'speed_ratio': speed_ratio,
            'meets_cost': meets_cost,
            'meets_speed': meets_speed,
            'meets_all': meets_all,
            'is_baseline': m.get('name') == baseline_name,
            'constraints_met': sum([meets_cost, meets_speed]),
        })
    
    # Sort by quality (descending) - we want the BEST quality models that meet cost constraint
    ranked_models.sort(key=lambda x: (-x['constraints_met'], -x['quality']))
    
    # Get exact matches and partial
    exact_matches = [m for m in ranked_models if m['meets_all'] and not m['is_baseline']]
    
    print(f"\n✅ Models meeting cost constraint (≤${max_cost_per_m}/M): {len(exact_matches)}")
    print(f"🏆 Top recommendations:")
    for i, m in enumerate(ranked_models[:5], 1):
        if m['is_baseline']:
            continue
        status = "✓" if m['meets_all'] else f"{m['constraints_met']}/2"
        print(f"   {i}. [{status}] {m['name'][:35]} (Q:{m['quality']:.1f}, ${m['cost']:.2f}/M)")
    
    # =========================================================================
    # CREATE VISUALIZATION
    # =========================================================================
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 11))
    
    COLORS = {
        'bg': '#0a0e17',
        'panel': '#131a2a',
        'grid': '#1e2738',
        'text': '#e8eaed',
        'muted': '#7a8599',
        'accent': '#22c55e',
        'gold': '#ffd93d',
        'baseline': '#ff6b6b',
        'partial2': '#f59e0b',
        'other': '#2d3748',
    }
    
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Main plot
    ax = fig.add_axes([0.06, 0.12, 0.58, 0.78])
    ax.set_facecolor(COLORS['panel'])
    
    # Separate models for plotting
    top_10 = [m for m in ranked_models if not m['is_baseline']][:10]
    top_10_names = {m['name'] for m in top_10}
    others = [m for m in ranked_models if not m['is_baseline'] and m['name'] not in top_10_names]
    baseline_pt = next((m for m in ranked_models if m['is_baseline']), None)
    
    def get_marker_size(speed_ratio):
        return 30 + min(speed_ratio, 3) * 60
    
    # Plot others
    if others:
        sizes = [get_marker_size(m['speed_ratio']) for m in others]
        ax.scatter([m['cost'] for m in others], [m['quality'] for m in others],
                  c=COLORS['other'], s=sizes, alpha=0.35, zorder=1, edgecolors='none')
    
    # Plot top 10 as diamonds
    for m in top_10:
        color = COLORS['gold'] if m['meets_all'] else COLORS['partial2']
        size = get_marker_size(m['speed_ratio'])
        ax.scatter([m['cost']], [m['quality']], c=color, s=size, alpha=0.85, zorder=3,
                  edgecolors='white', linewidth=1.0, marker='D')
    
    # Plot baseline
    if baseline_pt:
        ax.scatter([baseline_pt['cost']], [baseline_pt['quality']],
                  c=COLORS['baseline'], s=500, alpha=1.0, zorder=5,
                  marker='*', edgecolors='white', linewidth=1.5)
    
    # Cost constraint line (vertical)
    ax.axvline(max_cost_per_m, color=COLORS['accent'], linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Shade constraint zone
    zone = Rectangle((0, 0), max_cost_per_m, 105,
                     facecolor=COLORS['accent'], alpha=0.08, edgecolor='none', zorder=0)
    ax.add_patch(zone)
    
    # Label top 4 models - spread labels to avoid overlap
    # Use unique offsets for each label to prevent overlapping
    # Offsets are (x, y) in points - spread vertically to avoid overlap
    label_offsets = [
        (120, 80),     # #1: far right, high up
        (130, 20),     # #2: far right, slightly above
        (125, -40),    # #3: far right, below
        (120, -100),   # #4: far right, far below
    ]
    
    for idx, m in enumerate(top_10[:4]):
        offset = label_offsets[idx]
        ha = 'left'
        
        name = m['name'][:22] + "..." if len(m['name']) > 25 else m['name']
        prefix = "✓" if m['meets_all'] else f"[{m['constraints_met']}/2]"
        border_color = COLORS['gold'] if m['meets_all'] else COLORS['partial2']
        
        label = f"{prefix} {name}\nQ:{m['quality']:.0f}  ${m['cost']:.2f}/M"
        
        ax.annotate(label, xy=(m['cost'], m['quality']),
                   xytext=offset, textcoords='offset points',
                   fontsize=8, ha=ha, va='center', color=COLORS['text'],
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['panel'],
                           edgecolor=border_color, linewidth=2, alpha=0.95),
                   arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5),
                   zorder=10)
    
    # Styling
    ax.set_xlabel('Blended Cost ($/M tokens)', fontsize=14, fontweight='bold', color=COLORS['text'], labelpad=12)
    ax.set_ylabel('Quality Score (General)', fontsize=14, fontweight='bold', color=COLORS['text'], labelpad=12)
    ax.grid(True, alpha=0.12, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    
    x_max = min(max_cost_per_m * 6, 5)
    ax.set_xlim(-0.05, x_max)
    ax.set_ylim(25, 105)
    
    ax.tick_params(colors=COLORS['muted'], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1)
    
    # =========================================================================
    # RIGHT PANEL
    # =========================================================================
    
    info_ax = fig.add_axes([0.66, 0.12, 0.32, 0.78])
    info_ax.set_facecolor(COLORS['panel'])
    info_ax.axis('off')
    for spine in info_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1)
    
    y = 0.96
    
    # USER QUERY
    info_ax.text(0.5, y, 'USER QUERY', fontsize=12, fontweight='bold',
                color=COLORS['accent'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.04
    
    query_box = FancyBboxPatch((0.03, y - 0.14), 0.94, 0.14,
                               boxstyle="round,pad=0.01,rounding_size=0.02",
                               facecolor='#1a2332', edgecolor=COLORS['accent'],
                               linewidth=2, transform=info_ax.transAxes)
    info_ax.add_patch(query_box)
    
    query_text = f'"Give me the best models\n(closest to {baseline_name.split()[0]} 3)\nwhere cost ≤ ${max_cost_per_m:.0f}/M tokens"'
    info_ax.text(0.5, y - 0.07, query_text, fontsize=10, style='italic',
                color=COLORS['text'], ha='center', va='center', transform=info_ax.transAxes,
                linespacing=1.3)
    y -= 0.18
    
    # CONSTRAINTS section
    info_ax.text(0.5, y, 'CONSTRAINTS', fontsize=10, fontweight='bold',
                color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.03
    
    info_ax.text(0.08, y, f'• Cost ≤ ${max_cost_per_m:.2f}/M (blended)', fontsize=8,
                color=COLORS['text'], ha='left', va='top', transform=info_ax.transAxes)
    y -= 0.025
    
    if latency_threshold == 1.0:
        info_ax.text(0.08, y, '• Latency ≤ 100% (default: same as baseline)', fontsize=8,
                    color=COLORS['text'], ha='left', va='top', transform=info_ax.transAxes)
    elif latency_threshold is None:
        info_ax.text(0.08, y, '• Latency: flexible (no constraint)', fontsize=8,
                    color=COLORS['muted'], ha='left', va='top', transform=info_ax.transAxes,
                    style='italic')
    y -= 0.035
    
    # Divider
    info_ax.plot([0.05, 0.95], [y, y], color=COLORS['grid'], linewidth=1, transform=info_ax.transAxes)
    y -= 0.025
    
    # RESULTS
    total_c = 2  # Cost + Speed
    
    if exact_matches:
        info_ax.text(0.5, y, f'✓ {len(exact_matches)} MODELS FOUND', fontsize=12, fontweight='bold',
                    color=COLORS['gold'], ha='center', va='top', transform=info_ax.transAxes)
        y -= 0.025
        info_ax.text(0.5, y, 'Ranked by quality (highest first)', fontsize=8,
                    color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes,
                    style='italic')
    else:
        info_ax.text(0.5, y, 'NO EXACT MATCHES', fontsize=12, fontweight='bold',
                    color=COLORS['baseline'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.035
    
    info_ax.text(0.5, y, 'TOP 10 RECOMMENDATIONS', fontsize=11, fontweight='bold',
                color=COLORS['partial2'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.035
    
    count = 0
    for m in ranked_models[:12]:
        if m['is_baseline']:
            continue
        if count >= 10:
            break
            
        name = m['name']
        
        # Constraint status
        c_status = '✓C' if m['meets_cost'] else '✗C'
        s_status = '✓S' if m['meets_speed'] else '✗S'
        
        info_ax.text(0.02, y, f"{count+1:2}.", fontsize=9, fontweight='bold',
                    color=COLORS['partial2'], ha='left', va='top', transform=info_ax.transAxes)
        info_ax.text(0.09, y, name, fontsize=9, color=COLORS['text'],
                    ha='left', va='top', transform=info_ax.transAxes)
        
        # Show constraint status
        x_pos = 0.85
        for status in [c_status, s_status]:
            color = '#22c55e' if status.startswith('✓') else '#ef4444'
            info_ax.text(x_pos, y, status, fontsize=7, fontweight='bold',
                        color=color, ha='left', va='top', transform=info_ax.transAxes)
            x_pos += 0.07
        y -= 0.036
        count += 1
    
    # Legend
    y = 0.06
    info_ax.plot([0.05, 0.95], [y + 0.035, y + 0.035], color=COLORS['grid'], linewidth=1, transform=info_ax.transAxes)
    
    legend_items = [('*', COLORS['baseline'], 400, 'Baseline'),
                   ('o', COLORS['gold'], 100, 'Match'),
                   ('D', COLORS['partial2'], 80, 'Partial')]
    x_pos = 0.15
    for marker, color, size, label in legend_items:
        info_ax.scatter([x_pos], [y + 0.01], marker=marker, s=size/3, c=color,
                       transform=info_ax.transAxes, zorder=10,
                       edgecolors='white' if marker == '*' else 'none', linewidth=0.5)
        info_ax.text(x_pos + 0.04, y + 0.01, label, fontsize=8, color=COLORS['text'],
                    ha='left', va='center', transform=info_ax.transAxes)
        x_pos += 0.28
    
    info_ax.text(0.5, y - 0.025, 'C=Cost  S=Speed', fontsize=7, color=COLORS['muted'],
                ha='center', va='center', transform=info_ax.transAxes, style='italic')
    
    # Title
    fig.text(0.35, 0.95, f'💰 Model Selection by Absolute Cost',
            fontsize=20, fontweight='bold', color=COLORS['text'], ha='center', va='bottom')
    fig.text(0.35, 0.92, f'Best quality models where blended cost ≤ ${max_cost_per_m:.0f}/M tokens',
            fontsize=11, color=COLORS['muted'], ha='center', va='bottom')
    
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()
    
    return ranked_models


def main():
    """Generate use case-specific visualizations."""
    
    baseline = "Gemini 3 Pro Preview (high)"
    quality = 0.80  # 80% quality
    cost = 0.75     # 75% cost (increased from 25% for more model diversity)
    # Default: latency = 1.0 (must be at least as fast as baseline)
    # Set to None only when user explicitly says "latency not important"
    latency = 1.0   # Same speed as baseline (default behavior)
    
    # Individual use case plots
    for uc, config in USE_CASES.items():
        create_single_usecase_plot(
            baseline_name=baseline,
            use_case=uc,
            quality_threshold=quality,
            cost_threshold=cost,
            latency_threshold=latency,
            output_path=f"blog/usecase_{config['name'].lower().replace(' ', '_')}.png"
        )
    
    # Comparison grid
    create_usecase_comparison_grid(
        baseline_name=baseline,
        quality_threshold=quality,
        cost_threshold=cost,
        latency_threshold=latency,
        output_path="blog/usecase_comparison.png"
    )
    
    # Absolute cost constraint example
    create_absolute_cost_plot(
        baseline_name=baseline,
        max_cost_per_m=1.0,  # $1/M tokens
        latency_threshold=latency,
        output_path="blog/absolute_cost_selection.png"
    )
    
    print("\n" + "=" * 70)
    print("ALL USE CASE VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    for config in USE_CASES.values():
        print(f"  - blog/usecase_{config['name'].lower().replace(' ', '_')}.png")
    print("  - blog/usecase_comparison.png (2x2 grid)")
    print("  - blog/absolute_cost_selection.png (absolute cost constraint)")


if __name__ == "__main__":
    main()

