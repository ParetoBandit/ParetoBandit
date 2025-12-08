#!/usr/bin/env python3
"""
Professional Visualization: Constrained Model Selection

Shows which models meet user-specified constraints:
"Give me 85% of Gemini 3.0's quality at 20% of the cost and 50% of the latency"

Creates a publication-quality visualization with modern aesthetics.

RANKING STRATEGY FOR NO EXACT MATCHES:
When no models meet all constraints, we use a hybrid ranking approach:
1. Primary: Number of constraints satisfied (3 > 2 > 1 > 0)
2. Secondary: Chebyshev distance to remaining constraints (minimax regret)
3. Tertiary: Value score (quality per dollar) for tie-breaking
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import PromptCategory


def load_models():
    """Load model data from cache."""
    cache_path = Path("data/models_cache.json")
    with open(cache_path) as f:
        return json.load(f)


def calculate_constraint_distance(model, min_quality, max_cost, min_speed, base_quality, base_cost, base_speed):
    """
    Calculate Chebyshev distance to constraint boundaries.
    
    Returns the maximum normalized gap across all unmet constraints.
    This is the "minimax regret" approach - we want to minimize the worst-case deviation.
    """
    gaps = []
    
    # Quality gap (if not met)
    if model['quality'] < min_quality:
        # How far below the threshold, normalized by baseline
        gap = (min_quality - model['quality']) / base_quality
        gaps.append(('quality', gap))
    
    # Cost gap (if not met)
    if model['cost'] > max_cost:
        # How far above the threshold, normalized by baseline
        gap = (model['cost'] - max_cost) / base_cost
        gaps.append(('cost', gap))
    
    # Speed gap (if not met)
    if model['speed'] < min_speed:
        # How far below the threshold, normalized by baseline
        gap = (min_speed - model['speed']) / base_speed if base_speed > 0 else 0
        gaps.append(('speed', gap))
    
    if not gaps:
        return 0.0, []
    
    # Chebyshev distance = maximum gap
    max_gap = max(g[1] for g in gaps)
    return max_gap, gaps


def rank_models_hybrid(model_data, min_quality, max_cost, min_speed, base_quality, base_cost, base_speed):
    """
    Hybrid ranking strategy for when no exact matches exist.
    
    Ranking criteria (in order of priority):
    1. Number of constraints satisfied (more = better)
    2. Chebyshev distance to remaining constraints (lower = better)
    3. Value score: quality / cost (higher = better)
    
    Returns sorted list with ranking scores.
    """
    ranked = []
    
    for m in model_data:
        # Calculate constraint distance
        chebyshev_dist, gaps = calculate_constraint_distance(
            m, min_quality, max_cost, min_speed, 
            base_quality, base_cost, base_speed
        )
        
        # Value score for tie-breaking
        value_score = m['quality'] / m['cost'] if m['cost'] > 0 else 0
        
        # Composite ranking score
        # Primary: constraints_met (0-3), Secondary: -chebyshev_dist (negative because lower is better)
        # Tertiary: value_score
        ranking_tuple = (
            m['constraints_met'],           # Higher is better
            -chebyshev_dist,                # Lower distance is better (so negate)
            value_score,                    # Higher value is better
        )
        
        ranked.append({
            **m,
            'chebyshev_distance': chebyshev_dist,
            'value_score': value_score,
            'ranking_tuple': ranking_tuple,
            'gaps': gaps,
        })
    
    # Sort by ranking tuple (descending for all components since we negated distance)
    ranked.sort(key=lambda x: x['ranking_tuple'], reverse=True)
    
    return ranked


def create_constraint_selection_plot(
    baseline_name: str = "Gemini 3 Pro Preview (high)",
    quality_threshold: float = 0.85,  # 85% of baseline quality
    cost_threshold: float = 0.20,     # 20% of baseline cost
    latency_threshold: float = 0.50,  # 50% of baseline latency (= 2x speed)
    output_path: str = "blog/constraint_selection.png"
):
    """
    Create a professional visualization showing models meeting user constraints.
    
    The visualization answers: "Which models give me 85% of the quality 
    at 20% of the cost and 50% of the latency?"
    """
    
    print("=" * 70)
    print("CONSTRAINED MODEL SELECTION VISUALIZATION")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading model data...")
    all_models = load_models()
    
    # Filter for models with valid pricing and benchmarks
    valid_models = [
        m for m in all_models 
        if m.get('price_1m_input') and m.get('price_1m_input') > 0
        and m.get('intelligence_index') and m.get('intelligence_index') > 0
    ]
    print(f"   Found {len(valid_models)} models with valid data")
    
    # Initialize quality scorer
    print("\n🎯 Calculating quality scores...")
    scorer = QualityScorer(all_models_data=valid_models)
    
    # Find baseline model
    baseline = next((m for m in valid_models if baseline_name.lower() in m['name'].lower()), None)
    if not baseline:
        print(f"   ❌ Baseline '{baseline_name}' not found!")
        return
    
    print(f"\n📌 Baseline: {baseline['name']}")
    
    # Calculate baseline metrics
    base_quality = scorer.calculate_quality_score({'name': baseline['name']}, PromptCategory.GENERAL)
    base_cost = baseline['price_1m_input'] * 0.75 + baseline['price_1m_output'] * 0.25
    base_speed = baseline.get('output_tokens_per_second', 0) or 100  # tokens/sec (higher = faster)
    base_latency = 1000 / base_speed if base_speed > 0 else 10  # ms per token
    
    print(f"   Quality Score: {base_quality:.1f}")
    print(f"   Blended Cost: ${base_cost:.2f}/M tokens")
    print(f"   Speed: {base_speed:.1f} tokens/sec")
    print(f"   Latency: {base_latency:.2f} ms/token")
    
    # Calculate thresholds
    min_quality = base_quality * quality_threshold
    max_cost = base_cost * cost_threshold
    max_latency = base_latency * latency_threshold  # 50% of latency = faster
    min_speed = 1000 / max_latency if max_latency > 0 else base_speed * 2
    
    print(f"\n🎚️  User Request:")
    print(f"   \"Give me {quality_threshold*100:.0f}% of {baseline_name.split(' (')[0]}'s quality")
    print(f"    at {cost_threshold*100:.0f}% of the cost and {latency_threshold*100:.0f}% of the latency\"")
    print(f"\n🎚️  Translated Constraints:")
    print(f"   Quality ≥ {min_quality:.1f}")
    print(f"   Cost ≤ ${max_cost:.2f}/M")
    print(f"   Speed ≥ {min_speed:.1f} tokens/sec (latency ≤ {max_latency:.2f} ms/tok)")
    
    # Calculate metrics for all models
    model_data = []
    for m in valid_models:
        quality = scorer.calculate_quality_score({'name': m['name']}, PromptCategory.GENERAL)
        cost = m['price_1m_input'] * 0.75 + m['price_1m_output'] * 0.25
        speed = m.get('output_tokens_per_second', 0) or 0
        latency = 1000 / speed if speed > 0 else 999
        
        # Check constraints
        meets_quality = quality >= min_quality
        meets_cost = cost <= max_cost
        meets_speed = speed >= min_speed
        meets_all = meets_quality and meets_cost and meets_speed
        
        # Count how many constraints met (for partial matching)
        constraints_met = sum([meets_quality, meets_cost, meets_speed])
        
        model_data.append({
            'name': m['name'],
            'quality': quality,
            'cost': cost,
            'speed': speed,
            'latency': latency,
            'quality_ratio': quality / base_quality,
            'cost_ratio': cost / base_cost,
            'speed_ratio': speed / base_speed if base_speed > 0 else 0,
            'latency_ratio': latency / base_latency if base_latency > 0 else 1,
            'meets_quality': meets_quality,
            'meets_cost': meets_cost,
            'meets_speed': meets_speed,
            'meets_all': meets_all,
            'constraints_met': constraints_met,
            'is_baseline': baseline_name.lower() in m['name'].lower()
        })
    
    # Apply hybrid ranking
    ranked_models = rank_models_hybrid(
        model_data, min_quality, max_cost, min_speed,
        base_quality, base_cost, base_speed
    )
    
    # Separate by constraint satisfaction
    winners = [m for m in ranked_models if m['meets_all']]
    partial_2 = [m for m in ranked_models if m['constraints_met'] == 2]
    partial_1 = [m for m in ranked_models if m['constraints_met'] == 1]
    
    print(f"\n✅ Models meeting ALL constraints: {len(winners)}")
    print(f"⚠️  Models meeting 2/3 constraints: {len(partial_2)}")
    print(f"○  Models meeting 1/3 constraints: {len(partial_1)}")
    
    # Show top recommendations using hybrid ranking
    print(f"\n🏆 TOP RECOMMENDATIONS (Hybrid Ranking):")
    print("   Ranking: #Constraints → Chebyshev Distance → Value Score")
    for i, m in enumerate(ranked_models[:8], 1):
        status = "✓" if m['meets_all'] else f"{m['constraints_met']}/3"
        print(f"   {i}. [{status}] {m['name'][:35]}")
        print(f"      Q:{m['quality_ratio']*100:.0f}% C:{m['cost_ratio']*100:.0f}% S:{m['speed_ratio']*100:.0f}%")
        if m['chebyshev_distance'] > 0:
            print(f"      Distance: {m['chebyshev_distance']:.3f} | Value: {m['value_score']:.1f}")
    
    # =========================================================================
    # CREATE THE VISUALIZATION
    # =========================================================================
    print("\n🎨 Creating visualization...")
    
    # Set up figure with dark sophisticated theme
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 11))
    
    # Custom color palette - sophisticated deep tones
    COLORS = {
        'bg': '#0a0e17',           # Deep navy background
        'panel': '#131a2a',        # Slightly lighter panel
        'grid': '#1e2738',         # Subtle grid
        'text': '#e8eaed',         # Crisp white text
        'muted': '#7a8599',        # Muted gray
        'accent': '#00d4aa',       # Vibrant teal accent
        'gold': '#ffd93d',         # Gold for winners
        'baseline': '#ff6b6b',     # Coral for baseline
        'partial': '#7c3aed',      # Purple for partial matches
        'partial2': '#f59e0b',     # Amber for 2/3 matches
        'partial1': '#6366f1',     # Indigo for 1/3 matches
        'other': '#2d3748',        # Dark gray for others
        'zone': '#00d4aa',         # Zone highlight
        'quality_line': '#22c55e', # Green for quality threshold
        'cost_line': '#f97316',    # Orange for cost threshold  
        'speed_line': '#3b82f6',   # Blue for speed threshold
    }
    
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Create main axes with padding
    ax = fig.add_axes([0.06, 0.12, 0.58, 0.78])
    ax.set_facecolor(COLORS['panel'])
    
    # =========================================================================
    # PLOT DATA - Use marker size to encode speed
    # =========================================================================
    
    # Separate models by category
    others = [m for m in ranked_models if m['constraints_met'] < 2 and not m['is_baseline']]
    partial = [m for m in ranked_models if m['constraints_met'] == 2 and not m['meets_all']]
    exact_matches = [m for m in ranked_models if m['meets_all']]
    baseline_pt = next((m for m in ranked_models if m['is_baseline']), None)
    
    # Calculate marker sizes based on speed (larger = faster)
    def get_marker_size(speed_ratio):
        """Map speed ratio to marker size."""
        return 30 + min(speed_ratio, 3) * 60  # 30-210 range
    
    # Plot other models (subtle)
    if others:
        sizes = [get_marker_size(m['speed_ratio']) for m in others]
        ax.scatter(
            [m['cost'] for m in others],
            [m['quality'] for m in others],
            c=COLORS['other'],
            s=sizes,
            alpha=0.35,
            zorder=1,
            edgecolors='none'
        )
    
    # Plot partial matches (2/3 constraints) - highlighted but distinct
    if partial:
        sizes = [get_marker_size(m['speed_ratio']) for m in partial]
        ax.scatter(
            [m['cost'] for m in partial],
            [m['quality'] for m in partial],
            c=COLORS['partial2'],
            s=sizes,
            alpha=0.75,
            zorder=2,
            edgecolors='white',
            linewidth=0.8,
            marker='D'  # Diamond for partial
        )
    
    # Plot qualifying models (highlighted)
    if exact_matches:
        sizes = [get_marker_size(m['speed_ratio']) for m in exact_matches]
        
        # Glow effect
        for m in exact_matches:
            ax.scatter(
                [m['cost']], [m['quality']],
                c=COLORS['gold'],
                s=get_marker_size(m['speed_ratio']) * 3, 
                alpha=0.15, zorder=2
            )
        
        # Main points
        ax.scatter(
            [m['cost'] for m in exact_matches],
            [m['quality'] for m in exact_matches],
            c=COLORS['gold'],
            s=sizes,
            alpha=1.0,
            zorder=4,
            edgecolors=COLORS['accent'],
            linewidth=2.5,
            marker='o'
        )
    
    # Plot baseline with special marker
    if baseline_pt:
        ax.scatter(
            [baseline_pt['cost']], [baseline_pt['quality']],
            c=COLORS['baseline'],
            s=500,
            alpha=1.0,
            zorder=5,
            marker='*',
            edgecolors='white',
            linewidth=1.5
        )
    
    # =========================================================================
    # DRAW CONSTRAINT ZONE
    # =========================================================================
    
    # Draw the "qualifying zone" - shaded area where quality+cost constraints are met
    zone = mpatches.Rectangle(
        (0, min_quality),
        max_cost,
        100 - min_quality + 5,
        facecolor=COLORS['zone'],
        alpha=0.06,
        edgecolor='none',
        zorder=0
    )
    ax.add_patch(zone)
    
    # Draw constraint boundaries with colored dashed lines
    ax.axhline(min_quality, color=COLORS['quality_line'], linestyle='--', 
               linewidth=2.5, alpha=0.8, zorder=1, label=f'Quality ≥ {quality_threshold*100:.0f}%')
    ax.axvline(max_cost, color=COLORS['cost_line'], linestyle='--', 
               linewidth=2.5, alpha=0.8, zorder=1, label=f'Cost ≤ {cost_threshold*100:.0f}%')
    
    # =========================================================================
    # LABEL TOP RANKED MODELS
    # =========================================================================
    
    # Label top 5 by hybrid ranking (not just constraint count)
    to_label = ranked_models[:5]
    
    # Label positions to avoid overlaps
    label_configs = [
        {'offset': (70, 25), 'ha': 'left'},
        {'offset': (-90, 20), 'ha': 'right'},
        {'offset': (65, -30), 'ha': 'left'},
        {'offset': (-85, -25), 'ha': 'right'},
        {'offset': (60, 0), 'ha': 'left'},
    ]
    
    for idx, m in enumerate(to_label):
        if m['is_baseline']:
            continue
            
        config = label_configs[idx % len(label_configs)]
        
        # Clean name
        name = m['name']
        if len(name) > 28:
            name = name[:25] + "..."
        
        # Create label with metrics
        if m['meets_all']:
            prefix = "✓"
            border_color = COLORS['gold']
        else:
            prefix = f"[{m['constraints_met']}/3]"
            border_color = COLORS['partial2'] if m['constraints_met'] == 2 else COLORS['partial1']
        
        label = f"{prefix} {name}\n"
        label += f"Q:{m['quality_ratio']*100:.0f}% C:{m['cost_ratio']*100:.0f}% S:{m['speed_ratio']*100:.0f}%"
        
        if m['chebyshev_distance'] > 0:
            label += f"\nDist: {m['chebyshev_distance']:.2f}"
        
        ax.annotate(
            label,
            xy=(m['cost'], m['quality']),
            xytext=config['offset'], textcoords='offset points',
            fontsize=8, ha=config['ha'], va='center',
            color=COLORS['text'],
            fontweight='normal',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=COLORS['panel'],
                edgecolor=border_color,
                linewidth=2,
                alpha=0.95
            ),
            arrowprops=dict(
                arrowstyle='->',
                color=border_color,
                lw=1.5,
                connectionstyle='arc3,rad=0.1'
            ),
            zorder=10
        )
    
    # =========================================================================
    # STYLING
    # =========================================================================
    
    # Axis labels
    ax.set_xlabel('Cost ($/M tokens)', fontsize=14, fontweight='bold', 
                  color=COLORS['text'], labelpad=12)
    ax.set_ylabel('Quality Score', fontsize=14, fontweight='bold',
                  color=COLORS['text'], labelpad=12)
    
    # Grid
    ax.grid(True, alpha=0.12, color=COLORS['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Axis limits - focus on the interesting region
    x_max = min(max_cost * 5, max([m['cost'] for m in ranked_models]) * 0.5)
    ax.set_xlim(-0.02, x_max)
    ax.set_ylim(25, 105)
    
    # Tick styling
    ax.tick_params(colors=COLORS['muted'], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1)
    
    # =========================================================================
    # RIGHT PANEL - User Query + Results
    # =========================================================================
    
    info_ax = fig.add_axes([0.66, 0.12, 0.32, 0.78])
    info_ax.set_facecolor(COLORS['panel'])
    info_ax.axis('off')
    for spine in info_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1)
    
    y = 0.96
    
    # USER QUERY section
    info_ax.text(0.5, y, 'USER QUERY', fontsize=12, fontweight='bold',
                 color=COLORS['accent'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.04
    
    # Query box
    query_box = FancyBboxPatch(
        (0.03, y - 0.12), 0.94, 0.12,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor='#1a2332',
        edgecolor=COLORS['accent'],
        linewidth=2,
        transform=info_ax.transAxes
    )
    info_ax.add_patch(query_box)
    
    query_text = f'"Give me {quality_threshold*100:.0f}% of {baseline_name.split(" (")[0]}\'s\nquality at {cost_threshold*100:.0f}% cost, {latency_threshold*100:.0f}% latency"'
    info_ax.text(0.5, y - 0.06, query_text, fontsize=10, style='italic',
                 color=COLORS['text'], ha='center', va='center', transform=info_ax.transAxes,
                 linespacing=1.4)
    y -= 0.16
    
    # CONSTRAINTS section
    info_ax.text(0.5, y, 'TRANSLATED CONSTRAINTS', fontsize=10, fontweight='bold',
                 color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.035
    
    constraints = [
        (COLORS['quality_line'], f'Quality ≥ {min_quality:.1f}'),
        (COLORS['cost_line'], f'Cost ≤ ${max_cost:.2f}/M'),
        (COLORS['speed_line'], f'Speed ≥ {min_speed:.0f} tok/s'),
    ]
    
    for color, text in constraints:
        info_ax.plot([0.08, 0.14], [y - 0.015, y - 0.015], color=color, 
                    linewidth=3, linestyle='--', transform=info_ax.transAxes)
        info_ax.text(0.18, y - 0.015, text, fontsize=9, color=COLORS['text'],
                    ha='left', va='center', transform=info_ax.transAxes)
        y -= 0.035
    
    y -= 0.02
    
    # RESULTS section - draw line using plot
    info_ax.plot([0.05, 0.95], [y, y], color=COLORS['grid'], linewidth=1,
                transform=info_ax.transAxes)
    y -= 0.025
    
    # RANKING METHOD explanation
    info_ax.text(0.5, y, 'RANKING METHOD', fontsize=10, fontweight='bold',
                 color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes)
    y -= 0.03
    info_ax.text(0.5, y, '① # Constraints Met → ② Min Distance → ③ Value', fontsize=7,
                 color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes,
                 style='italic')
    y -= 0.03
    
    if exact_matches:
        info_ax.text(0.5, y, f'✓ {len(exact_matches)} EXACT MATCHES', fontsize=12, fontweight='bold',
                     color=COLORS['gold'], ha='center', va='top', transform=info_ax.transAxes)
        y -= 0.04
        
        for i, m in enumerate(exact_matches[:6]):
            name = m['name'][:28] + "..." if len(m['name']) > 31 else m['name']
            
            info_ax.text(0.06, y, f"#{i+1}", fontsize=10, fontweight='bold',
                        color=COLORS['gold'], ha='left', va='top', transform=info_ax.transAxes)
            info_ax.text(0.14, y, name, fontsize=9,
                        color=COLORS['text'], ha='left', va='top', transform=info_ax.transAxes)
            
            metrics = f"Q:{m['quality_ratio']*100:.0f}%  C:{m['cost_ratio']*100:.0f}%  S:{m['speed_ratio']*100:.0f}%"
            info_ax.text(0.14, y - 0.03, metrics, fontsize=8,
                        color=COLORS['muted'], ha='left', va='top', transform=info_ax.transAxes)
            y -= 0.065
    else:
        info_ax.text(0.5, y, 'NO EXACT MATCHES', fontsize=12, fontweight='bold',
                     color=COLORS['baseline'], ha='center', va='top', transform=info_ax.transAxes)
        y -= 0.035
        
        info_ax.text(0.5, y, f'⬡ TOP RECOMMENDATIONS', fontsize=11, fontweight='bold',
                     color=COLORS['partial2'], ha='center', va='top', transform=info_ax.transAxes)
        y -= 0.035
        info_ax.text(0.5, y, '(ranked by hybrid scoring)', fontsize=8,
                     color=COLORS['muted'], ha='center', va='top', transform=info_ax.transAxes,
                     style='italic')
        y -= 0.035
        
        # Show top ranked models regardless of constraint count
        for i, m in enumerate(ranked_models[:6]):
            if m['is_baseline']:
                continue
                
            name = m['name'][:26] + "..." if len(m['name']) > 29 else m['name']
            
            info_ax.text(0.06, y, f"#{i+1}", fontsize=9, fontweight='bold',
                        color=COLORS['partial2'], ha='left', va='top', transform=info_ax.transAxes)
            
            # Constraint status indicator
            status_color = COLORS['gold'] if m['meets_all'] else (
                COLORS['partial2'] if m['constraints_met'] == 2 else COLORS['partial1']
            )
            info_ax.text(0.12, y, f"[{m['constraints_met']}/3]", fontsize=8,
                        color=status_color, ha='left', va='top', transform=info_ax.transAxes)
            
            info_ax.text(0.22, y, name, fontsize=8,
                        color=COLORS['text'], ha='left', va='top', transform=info_ax.transAxes)
            
            # Show distance for non-exact matches
            if m['chebyshev_distance'] > 0:
                dist_text = f"Dist: {m['chebyshev_distance']:.2f}"
                info_ax.text(0.22, y - 0.025, dist_text, fontsize=7,
                            color=COLORS['muted'], ha='left', va='top', transform=info_ax.transAxes)
            
            y -= 0.055
    
    # LEGEND at bottom
    y = 0.08
    info_ax.plot([0.05, 0.95], [y + 0.02, y + 0.02], color=COLORS['grid'], linewidth=1,
                transform=info_ax.transAxes)
    
    legend_items = [
        ('*', COLORS['baseline'], 400, 'Baseline'),
        ('o', COLORS['gold'], 100, 'All Constraints'),
        ('D', COLORS['partial2'], 80, '2/3 Constraints'),
        ('o', COLORS['other'], 50, 'Other Models'),
    ]
    
    x_pos = 0.08
    for marker, color, size, label in legend_items:
        info_ax.scatter([x_pos], [y - 0.02], marker=marker, s=size/3, c=color,
                       transform=info_ax.transAxes, zorder=10, edgecolors='white' if marker == '*' else 'none',
                       linewidth=0.5)
        info_ax.text(x_pos + 0.04, y - 0.02, label, fontsize=7, color=COLORS['text'],
                    ha='left', va='center', transform=info_ax.transAxes)
        x_pos += 0.25
    
    # Size legend
    info_ax.text(0.5, y - 0.06, 'Marker size = Speed (larger = faster)', fontsize=7,
                color=COLORS['muted'], ha='center', va='center', transform=info_ax.transAxes,
                style='italic')
    
    # =========================================================================
    # TITLE
    # =========================================================================
    
    fig.text(0.35, 0.95, 'Constrained Model Selection', fontsize=20, fontweight='bold',
             color=COLORS['text'], ha='center', va='bottom')
    fig.text(0.35, 0.92, f'Finding models that match your requirements vs {baseline_name}',
             fontsize=11, color=COLORS['muted'], ha='center', va='bottom')
    
    # =========================================================================
    # SAVE
    # =========================================================================
    
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], 
                edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()
    
    return ranked_models


def main():
    """Generate the constrained selection visualization."""
    
    # Version 1: User's exact request (85% Q, 20% C, 50% L)
    # This is a very aggressive constraint set - shows near-matches
    result = create_constraint_selection_plot(
        baseline_name="Gemini 3 Pro Preview (high)",
        quality_threshold=0.85,    # 85% of baseline quality
        cost_threshold=0.20,       # 20% of baseline cost
        latency_threshold=0.50,    # 50% of baseline latency (2x faster)
        output_path="blog/constraint_selection.png"
    )
    
    # Version 2: Achievable constraints that find qualifying models
    print("\n" + "=" * 70)
    print("Creating version with models that qualify...")
    print("=" * 70)
    
    result2 = create_constraint_selection_plot(
        baseline_name="Gemini 3 Pro Preview (high)",
        quality_threshold=0.75,    # 75% of baseline quality  
        cost_threshold=0.25,       # 25% of baseline cost
        latency_threshold=1.50,    # 150% of baseline latency (can be slower)
        output_path="blog/constraint_selection_with_matches.png"
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - blog/constraint_selection.png (strict: 85% Q, 20% C, 50% L - shows near matches)")
    print("  - blog/constraint_selection_with_matches.png (75% Q, 25% C, 150% L - shows qualifying models)")


if __name__ == "__main__":
    main()
