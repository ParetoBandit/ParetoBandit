#!/usr/bin/env python3
"""
Generate publication-quality figures for the KDD paper.

Professional, compelling visualizations using ACTUAL data from the LLM Jury library.
Designed for KDD conference standards with clear storytelling.

Usage:
    python paper/generate_figures.py

Output:
    paper/figures/*.png - All paper figures
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict
from adjustText import adjust_text

# Import LLM Jury components
from llm_jury import (
    ModelRegistry,
    Optimizer,
    OptimizationStrategy,
    QualityScorer,
    PromptCategory,
)
from llm_jury.core.models import ModelMetadata, RoutingDecision
from llm_jury.optimization.chebyshev_scorer import ChebyshevScorer
from llm_jury.optimization.total_cost_inference import (
    calculate_tci,
    minmax_normalize,
    log_transform_cost,
    find_pareto_frontier_2d,
    calculate_performance_score,
    LOG_EPSILON,
)

# =============================================================================
# PROFESSIONAL KDD COLOR PALETTE
# =============================================================================
PALETTE = {
    # Background
    'bg': '#FFFFFF',
    'bg_subtle': '#FAFAFA',
    
    # Text
    'text_primary': '#1a1a2e',
    'text_secondary': '#4a4a6a',
    'text_muted': '#8888aa',
    
    # Primary accent (Deep Blue-Purple)
    'primary': '#4361ee',
    'primary_light': '#7289ef',
    'primary_dark': '#3a56d4',
    
    # Secondary (Coral/Orange for contrast)
    'secondary': '#f72585',
    'secondary_light': '#ff5da2',
    
    # Success/Value (Emerald)
    'success': '#06d6a0',
    'success_light': '#4eecc4',
    'success_dark': '#05b384',
    
    # Warning (Amber)
    'warning': '#ffc43d',
    'warning_light': '#ffd369',
    
    # Categories
    'oss': '#06d6a0',           # Open Source - Emerald
    'proprietary': '#4361ee',   # Proprietary - Blue
    'free': '#f72585',          # Free models - Pink
    'frontier': '#ff6b35',      # Frontier/Premium - Orange
    'value': '#7209b7',         # Value pick - Purple
    
    # Grid and borders
    'grid': '#e8e8f0',
    'border': '#d0d0e0',
}

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = PALETTE['border']

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "models_cache.json"


def load_models_data():
    """Load raw model data from cache."""
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)


def get_blended_cost(model_data):
    """Calculate blended cost from model data dict."""
    input_cost = model_data.get('input_cost_per_m') or model_data.get('price_1m_input') or 0
    output_cost = model_data.get('output_cost_per_m') or model_data.get('price_1m_output') or 0
    return 0.75 * input_cost + 0.25 * output_cost


def check_open_source(name):
    """Check if model is open source by name."""
    name_lower = name.lower()
    oss_patterns = ['llama', 'qwen', 'deepseek', 'mistral', 'mixtral', 'phi-', 
                    'gemma', 'falcon', 'yi-', 'internlm', 'ministral', 'gpt-oss']
    return any(p in name_lower for p in oss_patterns)


def short_name(name):
    """Create short display name for labels."""
    # Remove common suffixes
    name = name.replace(' (Reasoning)', '').replace(' (high)', '')
    name = name.replace(' Instruct', '').replace(' Preview', '')
    name = name.split("(")[0].strip()
    
    # Abbreviations
    replacements = {
        'DeepSeek': 'DS',
        'Gemini': 'Gem',
        'Claude': 'Cld',
        'Mistral': 'Mist',
        'Qwen3': 'Qw3',
        'Gemma': 'Gem',
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name[:20]


# =============================================================================
# FIGURE 1: SYSTEM ARCHITECTURE
# =============================================================================
def figure_1_architecture():
    """Generate professional system architecture diagram."""
    print("Generating Figure 1: Architecture Diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    
    def draw_box(ax, x, y, w, h, fc, ec, label, sublabel='', fontsize=10, lw=2):
        # Shadow
        shadow = FancyBboxPatch(
            (x + 0.04, y - 0.04), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor='#00000008', edgecolor='none', zorder=1
        )
        ax.add_patch(shadow)
        
        # Box
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        
        # Text
        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.15, label,
                    ha='center', va='center', fontsize=fontsize, fontweight='bold',
                    color=PALETTE['text_primary'], zorder=3)
            ax.text(x + w/2, y + h/2 - 0.18, sublabel,
                    ha='center', va='center', fontsize=fontsize-2,
                    color=PALETTE['text_secondary'], zorder=3)
        else:
            ax.text(x + w/2, y + h/2, label,
                    ha='center', va='center', fontsize=fontsize, fontweight='bold',
                    color=PALETTE['text_primary'], zorder=3)

    def draw_arrow(ax, start, end, color=PALETTE['text_secondary'], style='-', rad=0.0):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5,
                                 linestyle=style, connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=0, shrinkB=0, mutation_scale=12),
                   zorder=1)

    # Layout
    col1, col2, col3, col4 = 1.0, 4.5, 8.5, 12.5
    
    # External Data Sources
    ax.text(col1 + 1.0, 6.3, 'DATA SOURCES', ha='center', fontsize=9, 
            fontweight='bold', color=PALETTE['text_muted'])
    sources = [('Artificial Analysis', 5.5), ('Vectara Leaderboard', 4.7), ('OpenRouter API', 3.9)]
    for name, y in sources:
        draw_box(ax, col1, y, 2.0, 0.55, PALETTE['bg_subtle'], PALETTE['border'], name, fontsize=8, lw=1)
        draw_arrow(ax, (col1 + 2.05, y + 0.27), (col2 - 0.05, 4.5), 
                   color=PALETTE['text_muted'], style='--', rad=-0.1)

    # Model Registry
    draw_box(ax, col2, 4.0, 2.5, 1.5, '#eef2ff', PALETTE['primary'], 
             'Model Registry', '46 Models\nBenchmarks • Pricing • Latency')

    # User Inputs
    ax.text(col2 + 1.25, 2.7, 'USER REQUEST', ha='center', fontsize=9, 
            fontweight='bold', color=PALETTE['text_muted'])
    inputs = [('Prompt', 2.0), ('Constraints', 1.2)]
    for name, y in inputs:
        draw_box(ax, col2, y, 2.5, 0.55, '#fff7ed', PALETTE['warning'], name, fontsize=9, lw=1.5)

    # Core Processing
    ax.text(col3 + 1.25, 6.3, 'LLM JURY CORE', ha='center', fontsize=10, 
            fontweight='bold', color=PALETTE['primary'])
    
    # Container box
    container = FancyBboxPatch(
        (col3 - 0.2, 0.8), 2.9, 5.3,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor='#f8faff', edgecolor=PALETTE['primary'], linewidth=1.5,
        linestyle='--', zorder=0
    )
    ax.add_patch(container)
    
    draw_box(ax, col3, 5.0, 2.5, 0.8, '#eef2ff', PALETTE['primary'],
             'Prompt Classifier', 'Detect Task Type', fontsize=9)
    draw_box(ax, col3, 3.6, 2.5, 0.8, '#eef2ff', PALETTE['primary'],
             'Quality Scorer', 'Task-Specific Weights', fontsize=9)
    draw_box(ax, col3, 1.2, 2.5, 1.8, '#eef2ff', PALETTE['primary'],
             'Optimizer', 'Hybrid Pareto-Chebyshev', fontsize=9)
    
    # Optimizer details
    objectives = ['• Quality', '• Cost', '• Latency', '• Trust', '• Pareto Dominance']
    for i, obj in enumerate(objectives):
        ax.text(col3 + 0.2, 2.4 - i*0.22, obj, fontsize=7, color=PALETTE['text_secondary'])

    # Output
    draw_box(ax, col4, 2.8, 2.0, 1.8, '#ecfdf5', PALETTE['success'],
             'Verdict', 'Ranked Models\nCost Savings\nRationale')

    # Arrows
    draw_arrow(ax, (col2 + 2.55, 4.75), (col3 - 0.05, 4.0), PALETTE['primary'])
    draw_arrow(ax, (col2 + 2.55, 2.0), (col3 - 0.05, 2.1), PALETTE['warning'])
    draw_arrow(ax, (col3 + 1.25, 4.95), (col3 + 1.25, 4.45), PALETTE['primary'])
    draw_arrow(ax, (col3 + 1.25, 3.55), (col3 + 1.25, 3.05), PALETTE['primary'])
    draw_arrow(ax, (col3 + 2.55, 2.1), (col4 - 0.05, 3.7), PALETTE['success'])

    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'architecture.png', bbox_inches='tight', dpi=300,
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close()
    print("  ✓ Saved: figures/architecture.png")


# =============================================================================
# FIGURE 2: VALUE DISCOVERY - The Main Story
# =============================================================================
def figure_2_value_discovery():
    """
    Main figure showing LLM Jury's value proposition:
    Finding models with 85-95% quality at 10-30% cost.
    """
    print("Generating Figure 2: Value Discovery (Main Story)...")
    
    models_data = load_models_data()
    scorer = QualityScorer(all_models_data=models_data)
    
    # Collect data for CODING category
    data = []
    for m in models_data:
        quality = scorer.calculate_quality_score(m, PromptCategory.CODING)
        cost = get_blended_cost(m)
        name = m.get('name', 'Unknown')
        
        if quality > 0:
            data.append({
                'name': name,
                'quality': quality,
                'cost': cost,
                'oss': check_open_source(name),
                'free': cost == 0
            })
    
    # Find baseline (highest quality model)
    baseline = max(data, key=lambda x: x['quality'])
    baseline_quality = baseline['quality']
    baseline_cost = baseline['cost']
    
    print(f"  Baseline: {baseline['name']} (Q={baseline_quality:.1f}, C=${baseline_cost:.2f})")
    print(f"  Total models: {len(data)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    
    # Define zones
    # Value Zone: 75-100% quality, 0-35% cost
    value_zone = FancyBboxPatch(
        (0, baseline_quality * 0.75), baseline_cost * 0.35, baseline_quality * 0.25,
        boxstyle="round,pad=0.01,rounding_size=0.3",
        facecolor=PALETTE['success'] + '15', edgecolor=PALETTE['success'],
        linestyle='--', linewidth=2, zorder=0
    )
    ax.add_patch(value_zone)
    
    # Label for value zone
    ax.text(baseline_cost * 0.17, baseline_quality * 0.97, 
            "VALUE ZONE\n75-100% Quality\n<35% Cost", 
            ha='center', va='top', fontsize=10, fontweight='bold',
            color=PALETTE['success_dark'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=PALETTE['success'], alpha=0.9))
    
    # Scatter plots by category
    # Free models (most prominent)
    free_models = [d for d in data if d['free']]
    paid_oss = [d for d in data if d['oss'] and not d['free']]
    proprietary = [d for d in data if not d['oss']]
    
    # Plot proprietary first (background)
    ax.scatter([d['cost'] for d in proprietary], 
               [d['quality'] for d in proprietary],
               c=PALETTE['proprietary'], alpha=0.6, s=100, 
               label=f'Proprietary (n={len(proprietary)})', 
               edgecolors='white', linewidth=1.5, zorder=3)
    
    # Open source (paid)
    ax.scatter([d['cost'] for d in paid_oss], 
               [d['quality'] for d in paid_oss],
               c=PALETTE['oss'], alpha=0.7, s=100, marker='D',
               label=f'Open Source (n={len(paid_oss)})', 
               edgecolors='white', linewidth=1.5, zorder=4)
    
    # Free models (highlight)
    ax.scatter([d['cost'] for d in free_models], 
               [d['quality'] for d in free_models],
               c=PALETTE['free'], alpha=0.9, s=150, marker='*',
               label=f'FREE Models (n={len(free_models)})', 
               edgecolors='white', linewidth=1.5, zorder=5)
    
    # Calculate and plot Pareto frontier
    sorted_data = sorted(data, key=lambda x: x['cost'])
    frontier = []
    max_q = 0
    for d in sorted_data:
        if d['quality'] > max_q:
            frontier.append(d)
            max_q = d['quality']
    
    frontier_costs = [d['cost'] for d in frontier]
    frontier_quality = [d['quality'] for d in frontier]
    ax.step(frontier_costs, frontier_quality, where='post', 
            color=PALETTE['frontier'], linewidth=2.5, alpha=0.7, 
            label='Pareto Frontier', zorder=2)
    
    # Highlight baseline
    ax.scatter([baseline_cost], [baseline_quality], c=PALETTE['frontier'], 
               s=250, marker='P', edgecolors='white', linewidth=2, zorder=6)
    
    # Smart label placement using adjustText
    texts = []
    
    # Label key models
    # Value zone models
    value_models = [d for d in data if d['quality'] > baseline_quality * 0.75 
                    and d['cost'] < baseline_cost * 0.35 and d['cost'] > 0]
    value_models = sorted(value_models, key=lambda x: x['quality'], reverse=True)[:5]
    
    for d in value_models:
        txt = ax.annotate(short_name(d['name']), (d['cost'], d['quality']),
                         fontsize=8, fontweight='bold', color=PALETTE['value'],
                         ha='left', va='bottom')
        texts.append(txt)
    
    # Label free models
    for d in free_models:
        txt = ax.annotate(short_name(d['name']), (d['cost'] + 0.05, d['quality']),
                         fontsize=8, fontweight='bold', color=PALETTE['free'],
                         ha='left', va='center')
        texts.append(txt)
    
    # Label baseline
    ax.annotate(f"BASELINE\n{short_name(baseline['name'])}", 
                (baseline_cost, baseline_quality),
                xytext=(15, -25), textcoords='offset points',
                fontsize=9, fontweight='bold', color=PALETTE['frontier'],
                ha='left',
                arrowprops=dict(arrowstyle='->', color=PALETTE['frontier'], lw=1.5))
    
    # Adjust text positions to avoid overlap
    try:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    except:
        pass  # If adjustText fails, continue without adjustment
    
    # Add key insight annotation
    # Find best value model
    best_value = max([d for d in data if d['cost'] > 0], 
                     key=lambda x: x['quality'] / (x['cost'] + 0.01))
    savings_pct = (1 - best_value['cost'] / baseline_cost) * 100
    quality_retained = (best_value['quality'] / baseline_quality) * 100
    
    insight_text = (f"KEY FINDING\n"
                   f"Best Value: {short_name(best_value['name'])}\n"
                   f"• {quality_retained:.0f}% of baseline quality\n"
                   f"• {savings_pct:.0f}% cost savings")
    
    ax.text(0.98, 0.02, insight_text, transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=PALETTE['value'] + '15',
                     edgecolor=PALETTE['value'], linewidth=1.5),
            color=PALETTE['text_primary'])
    
    # Styling
    ax.set_xlabel('Cost per 1M Tokens ($)', fontsize=12, fontweight='bold', 
                  color=PALETTE['text_primary'], labelpad=10)
    ax.set_ylabel('Quality Score (CODING)', fontsize=12, fontweight='bold',
                  color=PALETTE['text_primary'], labelpad=10)
    ax.set_title('LLM Jury Identifies High-Value Models', fontsize=14, 
                 fontweight='bold', color=PALETTE['text_primary'], pad=15)
    
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95, 
              edgecolor=PALETTE['border'], fancybox=True)
    
    ax.grid(True, linestyle='--', alpha=0.3, color=PALETTE['grid'], zorder=0)
    ax.set_xlim(-0.3, max(d['cost'] for d in data) * 1.05)
    ax.set_ylim(0, 105)
    
    # Add model count
    ax.text(0.02, 0.98, f'n = {len(data)} models', transform=ax.transAxes,
            fontsize=9, ha='left', va='top', color=PALETTE['text_muted'])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pareto_frontier.png', bbox_inches='tight', dpi=300,
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close()
    print("  ✓ Saved: figures/pareto_frontier.png")


# =============================================================================
# FIGURE 3: CHEBYSHEV OPTIMIZATION - USING REAL LIBRARY SCORER
# =============================================================================
def figure_3_chebyshev():
    """
    Educational figure explaining Chebyshev scalarization.
    Key insight: Minimize the MAXIMUM regret (bottleneck principle).
    
    Uses the REAL ChebyshevScorer from the library.
    """
    print("Generating Figure 3: Chebyshev Algorithm Explanation...")
    
    models_data = load_models_data()
    scorer = QualityScorer(all_models_data=models_data)
    
    # Find baseline (reference point) - use a good balanced model
    baseline_data = None
    for m in models_data:
        if 'GPT-5.1' in m.get('name', ''):
            baseline_data = m
            break
    if not baseline_data:
        baseline_data = max(models_data, key=lambda x: scorer.calculate_quality_score(x, PromptCategory.CODING))
    
    baseline_quality = scorer.calculate_quality_score(baseline_data, PromptCategory.CODING)
    baseline_cost = get_blended_cost(baseline_data)
    baseline_latency = (baseline_data.get('measured_ttft_seconds') or 0.5) * 1000
    baseline_trust = 100 - (baseline_data.get('hallucination_rate') or 5)
    
    # Use the REAL ChebyshevScorer from the library
    cheb_scorer = ChebyshevScorer(
        baseline_quality=baseline_quality,
        baseline_cost=baseline_cost,
        baseline_latency=baseline_latency,
        baseline_trustability=baseline_trust,
        quality_weight=0.35,
        cost_weight=0.25,
        latency_weight=0.20,
        trustability_weight=0.20,
    )
    
    # Score all models using the REAL scorer
    model_regrets = []
    for m in models_data:
        quality = scorer.calculate_quality_score(m, PromptCategory.CODING)
        cost = get_blended_cost(m)
        latency = (m.get('measured_ttft_seconds') or 0.5) * 1000
        trust = 100 - (m.get('hallucination_rate') or 10)
        
        if quality > 0 and cost > 0:  # Need cost > 0 for regret calculation
            # Use REAL ChebyshevScorer to get the score
            cheb_result = cheb_scorer.score_model(
                model_name=m.get('name', 'Unknown'),
                quality=quality,
                cost=cost,
                latency=latency,
                trustability=trust
            )
            
            # Extract regrets from the result for visualization
            # The scorer calculates these internally, we can derive them
            q_regret = cheb_result.quality_regret
            c_regret = cheb_result.cost_regret
            l_regret = cheb_result.latency_regret
            t_regret = cheb_result.trustability_regret
            
            # Weights (as used in the library)
            w = {'q': 0.35, 'c': 0.25, 'l': 0.20, 't': 0.20}
            weighted_regrets = [w['q'] * q_regret, w['c'] * c_regret, 
                               w['l'] * l_regret, w['t'] * t_regret]
            
            model_regrets.append({
                'name': m.get('name', 'Unknown'),
                'quality': quality,
                'cost': cost,
                'regrets': [q_regret, c_regret, l_regret, t_regret],
                'weighted_regrets': weighted_regrets,
                'cheb_distance': cheb_result.chebyshev_distance,  # From REAL scorer
                'weighted_sum': sum(weighted_regrets),
            })
    
    # Find interesting models to compare
    # 1. Chebyshev pick (lowest max regret)
    cheb_pick = min(model_regrets, key=lambda x: x['cheb_distance'])
    
    # 2. Quality leader (might have high cost regret)
    quality_leader = max(model_regrets, key=lambda x: x['quality'])
    
    # 3. Find an "unbalanced" model - low on one dimension
    # A model with low average regret but high max regret
    unbalanced_candidates = [m for m in model_regrets 
                            if m['cheb_distance'] > cheb_pick['cheb_distance'] * 1.5
                            and m['weighted_sum'] < quality_leader['weighted_sum']]
    if unbalanced_candidates:
        unbalanced = min(unbalanced_candidates, key=lambda x: x['weighted_sum'])
    else:
        unbalanced = quality_leader
    
    print(f"  Chebyshev Pick: {cheb_pick['name']} (max regret: {cheb_pick['cheb_distance']:.3f})")
    print(f"  Quality Leader: {quality_leader['name']} (max regret: {quality_leader['cheb_distance']:.3f})")
    print(f"  Comparison: {unbalanced['name']} (max regret: {unbalanced['cheb_distance']:.3f})")
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    
    # ==========================================================================
    # LEFT PANEL: The Key Insight - Regret Bar Comparison
    # ==========================================================================
    ax1 = fig.add_subplot(131)
    ax1.set_facecolor(PALETTE['bg'])
    
    dimensions = ['Quality\nRegret', 'Cost\nRegret', 'Latency\nRegret', 'Trust\nRegret']
    x = np.arange(len(dimensions))
    width = 0.25
    
    # Plot weighted regrets for 3 models
    bars1 = ax1.bar(x - width, cheb_pick['weighted_regrets'], width, 
                    label=f"Chebyshev Pick\n({short_name(cheb_pick['name'])})",
                    color=PALETTE['success'], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    bars2 = ax1.bar(x, quality_leader['weighted_regrets'], width,
                    label=f"Quality Leader\n({short_name(quality_leader['name'])})", 
                    color=PALETTE['frontier'], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    bars3 = ax1.bar(x + width, unbalanced['weighted_regrets'], width,
                    label=f"Unbalanced\n({short_name(unbalanced['name'])})",
                    color=PALETTE['secondary'], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Draw horizontal lines showing Chebyshev distance (max regret) for each
    ax1.axhline(y=cheb_pick['cheb_distance'], color=PALETTE['success'], 
                linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=quality_leader['cheb_distance'], color=PALETTE['frontier'], 
                linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=unbalanced['cheb_distance'], color=PALETTE['secondary'], 
                linestyle='--', linewidth=2, alpha=0.7)
    
    # Annotate the max regret lines
    ax1.text(3.6, cheb_pick['cheb_distance'], f"Max: {cheb_pick['cheb_distance']:.2f}", 
             fontsize=8, color=PALETTE['success'], fontweight='bold', va='center')
    ax1.text(3.6, quality_leader['cheb_distance'], f"Max: {quality_leader['cheb_distance']:.2f}", 
             fontsize=8, color=PALETTE['frontier'], fontweight='bold', va='center')
    
    ax1.set_ylabel('Weighted Regret', fontsize=11, fontweight='bold')
    ax1.set_title('KEY INSIGHT: Minimize Maximum Regret', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dimensions, fontsize=9)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3, color=PALETTE['grid'])
    ax1.set_ylim(0, max(max(m['weighted_regrets']) for m in [cheb_pick, quality_leader, unbalanced]) * 1.3)
    
    # Add explanation box
    ax1.text(0.5, 0.95, 
             "Chebyshev = min(max regret)\nPrevents 'hiding' weak dimensions",
             transform=ax1.transAxes, fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PALETTE['success']+'20', 
                      edgecolor=PALETTE['success']), color=PALETTE['text_primary'])
    
    # ==========================================================================
    # MIDDLE PANEL: Why Not Weighted Sum?
    # ==========================================================================
    ax2 = fig.add_subplot(132)
    ax2.set_facecolor(PALETTE['bg'])
    
    # Compare Chebyshev vs Weighted Sum rankings
    # Sort by each method
    by_chebyshev = sorted(model_regrets, key=lambda x: x['cheb_distance'])[:10]
    by_weighted_sum = sorted(model_regrets, key=lambda x: x['weighted_sum'])[:10]
    
    # Show the difference
    models_to_show = [cheb_pick, quality_leader, unbalanced]
    model_names = [short_name(m['name']) for m in models_to_show]
    
    x_pos = np.arange(len(models_to_show))
    width = 0.35
    
    cheb_values = [m['cheb_distance'] for m in models_to_show]
    sum_values = [m['weighted_sum'] for m in models_to_show]
    
    bars1 = ax2.bar(x_pos - width/2, cheb_values, width, label='Chebyshev (max)',
                    color=PALETTE['value'], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax2.bar(x_pos + width/2, sum_values, width, label='Weighted Sum',
                    color=PALETTE['text_muted'], alpha=0.6, edgecolor='white', linewidth=1.5)
    
    # Highlight winner
    ax2.scatter([0 - width/2], [cheb_values[0] - 0.02], marker='v', s=100, 
                color=PALETTE['success'], zorder=5)
    ax2.text(0 - width/2, cheb_values[0] - 0.05, 'WINNER', ha='center', va='top',
             fontsize=8, fontweight='bold', color=PALETTE['success'])
    
    ax2.set_ylabel('Score (lower = better)', fontsize=11, fontweight='bold')
    ax2.set_title('Chebyshev vs Weighted Sum', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, fontsize=9, rotation=15, ha='right')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3, color=PALETTE['grid'])
    
    # Explanation
    ax2.text(0.5, 0.95, 
             "Weighted sum can be 'gamed'\nby one excellent dimension",
             transform=ax2.transAxes, fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PALETTE['warning']+'30', 
                      edgecolor=PALETTE['warning']), color=PALETTE['text_primary'])
    
    # ==========================================================================
    # RIGHT PANEL: 5D Radar - Visual Balance
    # ==========================================================================
    ax3 = fig.add_subplot(133, polar=True)
    ax3.set_facecolor(PALETTE['bg'])
    
    # Convert regrets to "scores" (1 - regret) for radar visualization
    categories = ['Quality', 'Cost\nEfficiency', 'Speed', 'Trust']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    def regrets_to_scores(regrets):
        # Cap regrets at 1 for visualization
        return [1 - min(r, 1) for r in regrets]
    
    cheb_scores = regrets_to_scores(cheb_pick['regrets']) + regrets_to_scores(cheb_pick['regrets'])[:1]
    quality_scores = regrets_to_scores(quality_leader['regrets']) + regrets_to_scores(quality_leader['regrets'])[:1]
    unbalanced_scores = regrets_to_scores(unbalanced['regrets']) + regrets_to_scores(unbalanced['regrets'])[:1]
    
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=10, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(['25%', '50%', '75%', '100%'], size=8, color=PALETTE['text_muted'])
    
    ax3.plot(angles, cheb_scores, 'D-', linewidth=3, color=PALETTE['success'],
             label='Chebyshev Pick', markersize=8)
    ax3.fill(angles, cheb_scores, alpha=0.3, color=PALETTE['success'])
    
    ax3.plot(angles, quality_scores, 'o-', linewidth=2, color=PALETTE['frontier'],
             label='Quality Leader', markersize=6)
    ax3.fill(angles, quality_scores, alpha=0.1, color=PALETTE['frontier'])
    
    ax3.plot(angles, unbalanced_scores, 's-', linewidth=2, color=PALETTE['secondary'],
             label='Unbalanced', markersize=6)
    ax3.fill(angles, unbalanced_scores, alpha=0.1, color=PALETTE['secondary'])
    
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax3.set_title('Balanced vs Unbalanced', fontsize=12, fontweight='bold', pad=15)
    
    # ==========================================================================
    # Main title and formula
    # ==========================================================================
    fig.suptitle('Chebyshev Scalarization: Finding Balanced Models', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    fig.text(0.5, -0.02, 
             r'$\mathbf{Chebyshev\ Distance} = \max_i \{ w_i \cdot regret_i \}$ — '
             r'Minimizes the worst-case regret across all dimensions',
             ha='center', fontsize=11, color=PALETTE['text_secondary'],
             bbox=dict(boxstyle='round,pad=0.5', facecolor=PALETTE['bg_subtle'], 
                      edgecolor=PALETTE['border']))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(FIGURES_DIR / 'chebyshev_concept.png', bbox_inches='tight', dpi=300,
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close()
    print("  ✓ Saved: figures/chebyshev_concept.png")


# =============================================================================
# FIGURE 4: COST SAVINGS BY TASK CATEGORY
# =============================================================================
def figure_4_savings_by_task():
    """
    Show cost savings achievable across different task categories.
    Compelling bar chart showing savings while maintaining quality.
    """
    print("Generating Figure 4: Cost Savings by Task Category...")
    
    models_data = load_models_data()
    scorer = QualityScorer(all_models_data=models_data)
    
    categories = [
        ('Coding', PromptCategory.CODING),
        ('Data Science', PromptCategory.DATA_SCIENCE),
        ('Creative', PromptCategory.CREATIVE),
        ('General', PromptCategory.GENERAL),
    ]
    
    results = []
    
    for cat_name, category in categories:
        # Get scores and costs
        model_data_list = []
        for m in models_data:
            quality = scorer.calculate_quality_score(m, category)
            cost = get_blended_cost(m)
            if quality > 0 and cost >= 0:
                model_data_list.append({
                    'name': m.get('name'),
                    'quality': quality,
                    'cost': cost
                })
        
        # Find frontier (best quality)
        frontier = max(model_data_list, key=lambda x: x['quality'])
        
        # Find best value (>80% quality, lowest cost)
        value_candidates = [m for m in model_data_list 
                          if m['quality'] >= frontier['quality'] * 0.80 and m['cost'] > 0]
        if value_candidates:
            best_value = min(value_candidates, key=lambda x: x['cost'])
        else:
            best_value = frontier
        
        savings_pct = (1 - best_value['cost'] / frontier['cost']) * 100 if frontier['cost'] > 0 else 0
        quality_retained = (best_value['quality'] / frontier['quality']) * 100
        
        results.append({
            'category': cat_name,
            'frontier_name': short_name(frontier['name']),
            'frontier_cost': frontier['cost'],
            'value_name': short_name(best_value['name']),
            'value_cost': best_value['cost'],
            'savings_pct': savings_pct,
            'quality_retained': quality_retained
        })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    
    x = np.arange(len(results))
    width = 0.35
    
    # Bars for frontier cost and value cost
    frontier_costs = [r['frontier_cost'] for r in results]
    value_costs = [r['value_cost'] for r in results]
    
    bars1 = ax.bar(x - width/2, frontier_costs, width, label='Frontier Model', 
                   color=PALETTE['frontier'], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, value_costs, width, label='LLM Jury Pick', 
                   color=PALETTE['success'], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add savings annotations
    for i, r in enumerate(results):
        # Draw arrow showing savings
        ax.annotate('', xy=(i + width/2, r['value_cost']), 
                   xytext=(i - width/2, r['frontier_cost']),
                   arrowprops=dict(arrowstyle='->', color=PALETTE['value'], 
                                  lw=2, connectionstyle='arc3,rad=-0.3'))
        
        # Savings label
        ax.text(i, max(r['frontier_cost'], r['value_cost']) + 0.3, 
                f"-{r['savings_pct']:.0f}%\n{r['quality_retained']:.0f}% quality",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=PALETTE['value'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor=PALETTE['value'], alpha=0.9))
        
        # Model names under bars
        ax.text(i - width/2, -0.5, r['frontier_name'], ha='center', va='top',
                fontsize=7, color=PALETTE['text_muted'], rotation=20)
        ax.text(i + width/2, -0.5, r['value_name'], ha='center', va='top',
                fontsize=7, color=PALETTE['text_muted'], rotation=20)
    
    ax.set_xlabel('Task Category', fontsize=12, fontweight='bold', labelpad=30)
    ax.set_ylabel('Cost per 1M Tokens ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Savings Across Task Categories\nwhile Maintaining >80% Quality', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([r['category'] for r in results], fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, color=PALETTE['grid'])
    
    # Key insight box
    avg_savings = np.mean([r['savings_pct'] for r in results])
    avg_quality = np.mean([r['quality_retained'] for r in results])
    ax.text(0.02, 0.98, f'Average: {avg_savings:.0f}% cost savings\nat {avg_quality:.0f}% quality',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', ha='left', color=PALETTE['success_dark'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=PALETTE['success'] + '20',
                     edgecolor=PALETTE['success']))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cost_savings.png', bbox_inches='tight', dpi=300,
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close()
    print("  ✓ Saved: figures/cost_savings.png")


# =============================================================================
# FIGURE 5: OSS vs PROPRIETARY
# =============================================================================
def figure_5_oss_vs_proprietary():
    """
    Clean comparison of Open Source vs Proprietary models.
    Shows the quality-cost trade-off landscape.
    """
    print("Generating Figure 5: OSS vs Proprietary Comparison...")
    
    models_data = load_models_data()
    scorer = QualityScorer(all_models_data=models_data)
    
    # Collect data
    oss_data = []
    prop_data = []
    
    for m in models_data:
        quality = scorer.calculate_quality_score(m, PromptCategory.CODING)
        cost = get_blended_cost(m)
        name = m.get('name', '')
        
        if quality > 0:
            entry = {'name': name, 'quality': quality, 'cost': cost, 'free': cost == 0}
            if check_open_source(name):
                oss_data.append(entry)
            else:
                prop_data.append(entry)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    
    # LEFT: Scatter plot
    ax1.set_facecolor(PALETTE['bg'])
    
    # Plot proprietary
    ax1.scatter([d['cost'] for d in prop_data], 
               [d['quality'] for d in prop_data],
               c=PALETTE['proprietary'], alpha=0.7, s=100, 
               label=f'Proprietary (n={len(prop_data)})', 
               edgecolors='white', linewidth=1.5, zorder=2)
    
    # Plot OSS (paid)
    oss_paid = [d for d in oss_data if not d['free']]
    ax1.scatter([d['cost'] for d in oss_paid], 
               [d['quality'] for d in oss_paid],
               c=PALETTE['oss'], alpha=0.7, s=100, marker='D',
               label=f'Open Source Paid (n={len(oss_paid)})', 
               edgecolors='white', linewidth=1.5, zorder=3)
    
    # Plot FREE (highlight)
    oss_free = [d for d in oss_data if d['free']]
    ax1.scatter([d['cost'] for d in oss_free], 
               [d['quality'] for d in oss_free],
               c=PALETTE['free'], alpha=0.9, s=150, marker='*',
               label=f'FREE Open Source (n={len(oss_free)})', 
               edgecolors='white', linewidth=1.5, zorder=4)
    
    # Label free models
    for d in oss_free:
        ax1.annotate(short_name(d['name']), (d['cost'] + 0.05, d['quality']),
                    fontsize=8, color=PALETTE['free'], fontweight='bold')
    
    ax1.set_xlabel('Cost ($/M tokens)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Quality Score (Coding)', fontsize=11, fontweight='bold')
    ax1.set_title('Quality vs Cost by License Type', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, color=PALETTE['grid'])
    ax1.set_xlim(-0.3, None)
    
    # RIGHT: Box plot comparison
    ax2.set_facecolor(PALETTE['bg'])
    
    categories = ['Coding', 'Data Science', 'Creative', 'General']
    cat_enums = [PromptCategory.CODING, PromptCategory.DATA_SCIENCE, 
                 PromptCategory.CREATIVE, PromptCategory.GENERAL]
    
    positions = []
    box_data = []
    colors = []
    
    for i, (cat_name, cat_enum) in enumerate(zip(categories, cat_enums)):
        oss_scores = [scorer.calculate_quality_score(m, cat_enum) 
                     for m in models_data if check_open_source(m.get('name', ''))]
        prop_scores = [scorer.calculate_quality_score(m, cat_enum) 
                      for m in models_data if not check_open_source(m.get('name', ''))]
        
        positions.extend([i * 2.5, i * 2.5 + 1])
        box_data.extend([oss_scores, prop_scores])
        colors.extend([PALETTE['oss'], PALETTE['proprietary']])
    
    bp = ax2.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    
    ax2.set_xticks([i * 2.5 + 0.5 for i in range(len(categories))])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel('Quality Score', fontsize=11, fontweight='bold')
    ax2.set_title('Quality Distribution by Task Category', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, color=PALETTE['grid'])
    
    # Legend
    oss_patch = mpatches.Patch(color=PALETTE['oss'], alpha=0.7, label='Open Source')
    prop_patch = mpatches.Patch(color=PALETTE['proprietary'], alpha=0.7, label='Proprietary')
    ax2.legend(handles=[oss_patch, prop_patch], loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'oss_vs_proprietary.png', bbox_inches='tight', dpi=300,
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close()
    print("  ✓ Saved: figures/oss_vs_proprietary.png")


# =============================================================================
# FIGURE 6: HYBRID OPTIMIZATION (5D) - Using REAL Optimizer
# =============================================================================
def figure_6_hybrid_optimization():
    """
    Show the 5D hybrid optimization including Pareto dominance.
    Uses the REAL Optimizer from the library.
    """
    print("Generating Figure 6: Hybrid 5D Optimization...")
    
    models_data = load_models_data()
    scorer = QualityScorer(all_models_data=models_data)
    registry = ModelRegistry.load_cache(verbose=False)
    
    # Find baseline (highest quality)
    baseline_data = max(models_data, key=lambda x: scorer.calculate_quality_score(x, PromptCategory.CODING))
    baseline_model = next((m for m in registry if m.name == baseline_data.get('name')), list(registry)[0])
    baseline_quality = scorer.calculate_quality_score(baseline_data, PromptCategory.CODING)
    baseline_cost = get_blended_cost(baseline_data)
    
    # Use REAL Optimizer with HYBRID strategy
    optimizer_hybrid = Optimizer(
        baseline_model=baseline_model,
        all_models_data=models_data,
        strategy=OptimizationStrategy.HYBRID,
    )
    
    # Get rankings from REAL optimizer
    decision = RoutingDecision(category=PromptCategory.CODING, archetype=None, reason='Figure')
    hybrid_results = optimizer_hybrid.rank(list(registry), decision, top_k=len(registry), verbose=False)
    
    # Collect metrics for visualization
    model_metrics = []
    for m in models_data:
        quality = scorer.calculate_quality_score(m, PromptCategory.CODING)
        cost = get_blended_cost(m)
        latency = (m.get('measured_ttft_seconds') or 0.5) * 1000
        trust = 100 - (m.get('hallucination_rate') or 10)
        
        if quality > 0 and cost >= 0:
            model_metrics.append({
                'name': m.get('name', 'Unknown'),
                'quality': quality,
                'cost': cost,
                'latency': latency,
                'trust': trust,
                'q_norm': quality / 100,
                'c_norm': 1 - min(cost / (baseline_cost + 0.1), 1),
                'l_norm': 1 - min(latency / 2000, 1),
                't_norm': trust / 100,
            })
    
    # Get Pareto dominance from optimizer's internal calculation
    # Calculate it once for visualization
    for i, m1 in enumerate(model_metrics):
        dominance = 0
        for j, m2 in enumerate(model_metrics):
            if i != j:
                if (m1['quality'] >= m2['quality'] and m1['cost'] <= m2['cost'] and
                    (m1['quality'] > m2['quality'] or m1['cost'] < m2['cost'])):
                    dominance += 1
        m1['dominance'] = dominance
    
    max_dom = max(m['dominance'] for m in model_metrics) if model_metrics else 1
    for m in model_metrics:
        m['d_norm'] = m['dominance'] / max(max_dom, 1)
    
    # Get picks from REAL optimizer results
    frontier_model = max(model_metrics, key=lambda x: x['quality'])
    hybrid_pick_name = hybrid_results[0].model_name if hybrid_results else frontier_model['name']
    value_model = next((m for m in model_metrics if m['name'] == hybrid_pick_name), frontier_model)
    budget_model = min([m for m in model_metrics if m['quality'] > 50], key=lambda x: x['cost'])
    
    print(f"  Frontier: {frontier_model['name']}")
    print(f"  Hybrid Pick: {value_model['name']} (from REAL Optimizer)")
    print(f"  Budget: {budget_model['name']}")
    
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    
    # LEFT: 5D Radar
    ax1 = fig.add_subplot(121, polar=True)
    ax1.set_facecolor(PALETTE['bg'])
    
    categories = ['Quality', 'Cost\nEfficiency', 'Speed', 'Trust', 'Pareto\nDominance']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    def get_radar_scores(m):
        return [m['q_norm'], m['c_norm'], m['l_norm'], m['t_norm'], m['d_norm']]
    
    frontier_scores = get_radar_scores(frontier_model) + get_radar_scores(frontier_model)[:1]
    value_scores = get_radar_scores(value_model) + get_radar_scores(value_model)[:1]
    budget_scores = get_radar_scores(budget_model) + get_radar_scores(budget_model)[:1]
    
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=10, fontweight='bold')
    ax1.set_ylim(0, 1)
    
    ax1.plot(angles, frontier_scores, 'o-', linewidth=2, color=PALETTE['frontier'],
             label=f'Frontier: {short_name(frontier_model["name"])}', markersize=6)
    ax1.fill(angles, frontier_scores, alpha=0.1, color=PALETTE['frontier'])
    
    ax1.plot(angles, value_scores, 'D-', linewidth=3, color=PALETTE['value'],
             label=f'Hybrid Pick: {short_name(value_model["name"])}', markersize=8)
    ax1.fill(angles, value_scores, alpha=0.25, color=PALETTE['value'])
    
    ax1.plot(angles, budget_scores, 's-', linewidth=2, color=PALETTE['success'],
             label=f'Budget: {short_name(budget_model["name"])}', markersize=6)
    ax1.fill(angles, budget_scores, alpha=0.1, color=PALETTE['success'])
    
    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax1.set_title('5D Hybrid Optimization Space', fontsize=12, fontweight='bold', pad=20)
    
    # RIGHT: Pareto dominance scatter
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor(PALETTE['bg'])
    
    cmap = LinearSegmentedColormap.from_list('dom', 
        [PALETTE['secondary'], PALETTE['warning'], PALETTE['success']])
    
    scatter = ax2.scatter(
        [m['cost'] for m in model_metrics],
        [m['quality'] for m in model_metrics],
        c=[m['d_norm'] for m in model_metrics],
        cmap=cmap, s=80, alpha=0.7, edgecolors='white', linewidth=1
    )
    
    # Highlight selected models
    for model, color, marker in [(frontier_model, PALETTE['frontier'], 'o'),
                                  (value_model, PALETTE['value'], 'D'),
                                  (budget_model, PALETTE['success'], 's')]:
        ax2.scatter([model['cost']], [model['quality']], s=250, 
                   facecolors='none', edgecolors=color, linewidth=3, marker=marker)
    
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Pareto Dominance\n(higher = dominates more models)', fontsize=10)
    
    ax2.set_xlabel('Cost ($/M tokens)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Quality Score', fontsize=11, fontweight='bold')
    ax2.set_title('Pareto Dominance as 5th Dimension', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, color=PALETTE['grid'])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hybrid_concept.png', bbox_inches='tight', dpi=300,
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close()
    print("  ✓ Saved: figures/hybrid_concept.png")


# =============================================================================
# FIGURE 7: SCORE DISTRIBUTION
# =============================================================================
def figure_7_score_distribution():
    """
    Clean violin/strip plot showing score distribution.
    """
    print("Generating Figure 7: Score Distribution...")
    
    models_data = load_models_data()
    scorer = QualityScorer(all_models_data=models_data)
    
    categories = [
        ('Coding', PromptCategory.CODING),
        ('Data Science', PromptCategory.DATA_SCIENCE),
        ('Creative', PromptCategory.CREATIVE),
        ('General', PromptCategory.GENERAL),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    
    all_scores = []
    all_positions = []
    all_colors = []
    all_labels = []
    
    for i, (cat_name, category) in enumerate(categories):
        for m in models_data:
            score = scorer.calculate_quality_score(m, category)
            is_oss = check_open_source(m.get('name', ''))
            is_free = get_blended_cost(m) == 0
            
            # Jitter
            jitter = np.random.uniform(-0.15, 0.15)
            
            all_scores.append(score)
            all_positions.append(i + jitter)
            
            if is_free:
                all_colors.append(PALETTE['free'])
            elif is_oss:
                all_colors.append(PALETTE['oss'])
            else:
                all_colors.append(PALETTE['proprietary'])
    
    ax.scatter(all_positions, all_scores, c=all_colors, alpha=0.6, s=50, 
               edgecolors='white', linewidth=0.5)
    
    # Add box plots for summary
    for i, (cat_name, category) in enumerate(categories):
        scores = [scorer.calculate_quality_score(m, category) for m in models_data]
        bp = ax.boxplot([scores], positions=[i], widths=0.5, patch_artist=True,
                       showfliers=False, zorder=0)
        bp['boxes'][0].set_facecolor('white')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][0].set_edgecolor(PALETTE['border'])
        for median in bp['medians']:
            median.set_color(PALETTE['text_primary'])
            median.set_linewidth(2)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c[0] for c in categories], fontsize=11)
    ax.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
    ax.set_title('Quality Score Distribution Across Task Categories\n(n=46 models)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, color=PALETTE['grid'])
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['proprietary'], 
               markersize=10, label='Proprietary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['oss'], 
               markersize=10, label='Open Source'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['free'], 
               markersize=10, label='FREE'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'score_distribution.png', bbox_inches='tight', dpi=300,
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close()
    print("  ✓ Saved: figures/score_distribution.png")


# =============================================================================
# FIGURE 8: WHY HYBRID? - Updated Pareto Algorithm Comparison
# =============================================================================
def figure_8_why_hybrid():
    """
    Show the Three-Phase Pareto-Chebyshev optimization algorithm.
    
    Uses the ACTUAL Optimizer from the library with PARETO_CHEBYSHEV strategy:
    - Phase 1: Shadow Price Fix (no $0 costs)
    - Phase 2: Pareto Filter (remove dominated models)
    - Phase 3: Augmented Chebyshev (rank by regret from business targets)
    
    Shows:
    - Performance vs Cost trade-offs with log scale
    - Pareto frontier with free models shadow-priced
    - The recommended model pick
    """
    print("Generating Figure 8: Three-Phase Pareto-Chebyshev Optimization...")
    
    models_data = load_models_data()
    scorer = QualityScorer(all_models_data=models_data)
    
    # Load model registry for optimizer
    registry = ModelRegistry.load_cache(verbose=False)
    
    # Use GENERAL category for broad applicability
    category = PromptCategory.GENERAL
    
    # Find baseline (highest quality model for this category)
    baseline_model = max(registry, key=lambda m: scorer.calculate_quality_score(
        next((d for d in models_data if d['name'] == m.name), {}), category))
    
    # Create optimizer with PARETO_CHEBYSHEV strategy
    optimizer = Optimizer(
        baseline_model=baseline_model,
        all_models_data=models_data,
        strategy=OptimizationStrategy.PARETO_CHEBYSHEV,
    )
    
    # Get rankings from optimizer
    decision = RoutingDecision(category=category, archetype=None, reason='Figure')
    results = optimizer.rank(list(registry), decision, top_k=len(registry), verbose=False)
    
    # Set minimum cost from population for shadow pricing (use library function)
    from llm_jury.optimization.pareto_chebyshev import set_minimum_cost_from_population, calculate_effective_cost
    min_cost = set_minimum_cost_from_population(models_data)
    print(f"  Shadow price for free models: ${min_cost:.4f}/M (90% of cheapest paid)")
    
    # Build model data for plotting - use library functions for all calculations
    models = []
    for m in models_data:
        # Use library's quality scorer
        quality = scorer.calculate_quality_score(m, category)
        
        # Use library's effective cost calculation (handles shadow pricing)
        effective_cost = calculate_effective_cost(
            input_cost_per_m=m.get('input_cost_per_m') or m.get('price_1m_input'),
            output_cost_per_m=m.get('output_cost_per_m') or m.get('price_1m_output'),
            model_name=m.get('name')
        )
        
        # Check if free (API cost = 0)
        api_input = m.get('input_cost_per_m') or m.get('price_1m_input') or 0
        api_output = m.get('output_cost_per_m') or m.get('price_1m_output') or 0
        is_free = (api_input == 0 and api_output == 0)
        
        if quality > 0:
            models.append({
                'name': m.get('name', 'Unknown'),
                'quality': quality,
                'raw_cost': 0 if is_free else effective_cost,
                'tci': effective_cost,
                'cost_log': log_transform_cost(effective_cost),
                'is_free': is_free,
            })
    
    # Normalize quality for performance calculation
    q_min = min(m['quality'] for m in models)
    q_max = max(m['quality'] for m in models)
    
    for m in models:
        m['quality_norm'] = minmax_normalize(m['quality'], q_min, q_max, 'maximize')
        m['performance'] = m['quality_norm']
    
    # Calculate Pareto frontier
    frontier_data = [{'name': m['name'], 'cost_log': m['cost_log'], 
                      'performance': m['performance']} for m in models]
    pareto_frontier = find_pareto_frontier_2d(frontier_data, x_key='cost_log', y_key='performance')
    frontier_names = {m['name'] for m in pareto_frontier}
    
    for m in models:
        m['is_pareto'] = m['name'] in frontier_names
    
    # Get the pick from optimizer
    pick_name = results[0].model_name if results else None
    pick = next((m for m in models if m['name'] == pick_name), models[0])
    
    print(f"  PICK: {pick['name']} (Q={pick['quality']:.1f}, TCI=${pick['tci']:.2f})")
    
    # Create single-panel figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    
    x_key = 'cost_log'
    
    # Plot non-Pareto models (gray)
    non_pareto = [m for m in models if not m['is_pareto'] and not m['is_free']]
    ax.scatter([m[x_key] for m in non_pareto], 
               [m['performance'] for m in non_pareto],
               c=PALETTE['text_muted'], alpha=0.3, s=60, 
               label='Dominated (removed in Phase 2)', zorder=2)
    
    # Highlight dominated free models (pink diamonds)
    dominated_free = [m for m in models if m['is_free'] and not m['is_pareto']]
    if dominated_free:
        ax.scatter([m[x_key] for m in dominated_free], 
                   [m['performance'] for m in dominated_free],
                   c=PALETTE['free'], alpha=0.5, s=70, marker='D',
                   label='Free (Dominated)', zorder=3, edgecolors='white', linewidth=0.5)
    
    # Highlight free models ON the frontier (green diamond)
    frontier_free = [m for m in models if m['is_free'] and m['is_pareto']]
    if frontier_free:
        ax.scatter([m[x_key] for m in frontier_free], 
                   [m['performance'] for m in frontier_free],
                   c=PALETTE['success'], alpha=0.9, s=120, marker='D',
                   label='Free (Pareto Frontier)', zorder=5, edgecolors='white', linewidth=1.5)
    
    # Plot Pareto frontier models (paid)
    pareto_paid = [m for m in models if m['is_pareto'] and not m['is_free']]
    ax.scatter([m[x_key] for m in pareto_paid], 
               [m['performance'] for m in pareto_paid],
               c=PALETTE['primary'], alpha=0.7, s=90, 
               label='Pareto Frontier', zorder=4, edgecolors='white', linewidth=1)
    
    # Draw Pareto frontier line
    all_pareto = [m for m in models if m['is_pareto']]
    frontier_sorted = sorted(all_pareto, key=lambda x: x[x_key])
    if len(frontier_sorted) >= 2:
        ax.plot([m[x_key] for m in frontier_sorted], 
                [m['performance'] for m in frontier_sorted],
                color=PALETTE['primary'], linewidth=2.5, 
                linestyle='--', alpha=0.6, zorder=1)
    
    # Highlight the pick with star
    ax.scatter([pick[x_key]], [pick['performance']],
               s=500, facecolors='none', edgecolors=PALETTE['success'], 
               linewidth=4, zorder=6)
    ax.scatter([pick[x_key]], [pick['performance']],
               s=250, c=PALETTE['success'], marker='*', zorder=7,
               edgecolors='white', linewidth=1.5)
    
    # Annotation for the pick
    cost_str = f"${pick['tci']:.2f}/M"
    ax.annotate(f"RECOMMENDED\n{short_name(pick['name'])}\nQ={pick['quality']:.1f}, {cost_str}",
                (pick[x_key], pick['performance']),
                xytext=(40, -40), textcoords='offset points',
                fontsize=10, fontweight='bold', color=PALETTE['success'],
                arrowprops=dict(arrowstyle='->', color=PALETTE['success'], lw=2.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor=PALETTE['success'], alpha=0.95, linewidth=2))
    
    # Labels
    ax.set_xlabel('log(Effective Cost + ε)  [← Cheaper]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (Normalized Quality)', fontsize=12, fontweight='bold')
    ax.set_title('Three-Phase Pareto-Chebyshev Optimization', 
                 fontsize=14, fontweight='bold', color=PALETTE['text_primary'], pad=15)
    ax.grid(True, alpha=0.3, color=PALETTE['grid'])
    
    # Legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'why_hybrid.png', bbox_inches='tight', dpi=300,
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close()
    print("  ✓ Saved: figures/why_hybrid.png")


# =============================================================================
# MAIN
# =============================================================================
def generate_all_figures():
    """Generate all paper figures."""
    print("=" * 70)
    print("Generating KDD Paper Figures")
    print("Professional visualizations with all 46 models")
    print("=" * 70)
    print()
    
    figure_1_architecture()
    print()
    figure_2_value_discovery()
    print()
    figure_3_chebyshev()
    print()
    figure_4_savings_by_task()
    print()
    figure_5_oss_vs_proprietary()
    print()
    figure_6_hybrid_optimization()
    print()
    figure_7_score_distribution()
    print()
    figure_8_why_hybrid()  # NEW: Explains Pareto + Chebyshev combination
    
    print()
    print("=" * 70)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_figures()
