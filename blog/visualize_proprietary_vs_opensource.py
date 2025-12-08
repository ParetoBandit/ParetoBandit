#!/usr/bin/env python3
"""
Visualization: Quality Score Distribution - Proprietary vs Open Source Models

Creates a beautiful comparison plot showing how quality scores are distributed
between proprietary (closed-source) and open source LLM models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import PromptCategory

# Color scheme - distinctive and modern
COLORS = {
    'bg': '#0d1117',
    'panel': '#161b22',
    'grid': '#21262d',
    'text': '#f0f6fc',
    'muted': '#8b949e',
    'proprietary': '#ff7b72',      # Coral red for proprietary
    'proprietary_light': '#ffa198',
    'opensource': '#7ee787',        # Green for open source  
    'opensource_light': '#aff5b4',
    'accent': '#58a6ff',
    'gold': '#f9c74f',
}

# Classification of model creators
PROPRIETARY_CREATORS = {
    'Anthropic',
    'OpenAI', 
    'Google',
    'xAI',
    'Cohere',
    'Moonshot AI',
    'Microsoft Azure',
}

OPENSOURCE_CREATORS = {
    'Meta',
    'DeepSeek',
    'Mistral',
    'Alibaba',
    'Z AI',
}


def classify_model(model: Dict) -> str:
    """Classify a model as proprietary or open source based on creator."""
    creator = model.get('creator_name', '')
    
    if creator in PROPRIETARY_CREATORS:
        return 'proprietary'
    elif creator in OPENSOURCE_CREATORS:
        return 'opensource'
    else:
        # Default heuristics for unknown creators
        name = model.get('name', '').lower()
        if any(x in name for x in ['llama', 'mixtral', 'qwen', 'deepseek', 'phi']):
            return 'opensource'
        return 'proprietary'


def load_data() -> Tuple[List[Dict], QualityScorer]:
    """Load model data and initialize scorer."""
    data_path = Path(__file__).parent.parent / 'data' / 'models_cache.json'
    
    with open(data_path) as f:
        models = json.load(f)
    
    print(f"Loaded {len(models)} models")
    
    # Initialize quality scorer
    scorer = QualityScorer(models)
    
    return models, scorer


def calculate_scores(models: List[Dict], scorer: QualityScorer) -> Dict[str, List[Tuple[str, float]]]:
    """Calculate quality scores and classify models."""
    
    proprietary_scores = []
    opensource_scores = []
    
    for model in models:
        model_type = classify_model(model)
        score = scorer.calculate_quality_score(model, category=None)  # General score
        name = model.get('name', 'Unknown')
        
        if model_type == 'proprietary':
            proprietary_scores.append((name, score))
        else:
            opensource_scores.append((name, score))
    
    return {
        'proprietary': proprietary_scores,
        'opensource': opensource_scores,
    }


def create_distribution_plot(scores: Dict[str, List[Tuple[str, float]]], output_path: str):
    """Create the distribution comparison visualization."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Extract score values
    prop_scores = [s[1] for s in scores['proprietary']]
    oss_scores = [s[1] for s in scores['opensource']]
    
    # Calculate statistics
    prop_mean, prop_std = np.mean(prop_scores), np.std(prop_scores)
    oss_mean, oss_std = np.mean(oss_scores), np.std(oss_scores)
    prop_median = np.median(prop_scores)
    oss_median = np.median(oss_scores)
    
    # =========================================================================
    # HEADER
    # =========================================================================
    
    fig.text(0.5, 0.95, 'Quality Score Distribution',
            fontsize=28, fontweight='bold', color=COLORS['text'], ha='center',
            fontname='DejaVu Sans')
    fig.text(0.5, 0.91, 'Proprietary vs Open Source Models',
            fontsize=16, color=COLORS['muted'], ha='center')
    
    # =========================================================================
    # MAIN HISTOGRAM PLOT
    # =========================================================================
    
    ax1 = fig.add_axes([0.08, 0.38, 0.60, 0.48])
    ax1.set_facecolor(COLORS['panel'])
    
    # Create overlapping histograms with transparency
    bins = np.linspace(20, 100, 25)
    
    ax1.hist(prop_scores, bins=bins, alpha=0.7, label=f'Proprietary (n={len(prop_scores)})',
             color=COLORS['proprietary'], edgecolor=COLORS['proprietary_light'], linewidth=1.5)
    ax1.hist(oss_scores, bins=bins, alpha=0.7, label=f'Open Source (n={len(oss_scores)})',
             color=COLORS['opensource'], edgecolor=COLORS['opensource_light'], linewidth=1.5)
    
    # Add median lines
    ax1.axvline(prop_median, color=COLORS['proprietary_light'], linestyle='--', 
                linewidth=2.5, label=f'Proprietary Median: {prop_median:.1f}')
    ax1.axvline(oss_median, color=COLORS['opensource_light'], linestyle='--',
                linewidth=2.5, label=f'Open Source Median: {oss_median:.1f}')
    
    # Styling
    ax1.set_xlabel('Quality Score', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax1.set_ylabel('Number of Models', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax1.tick_params(colors=COLORS['muted'], labelsize=11)
    ax1.set_xlim(20, 100)
    
    for spine in ax1.spines.values():
        spine.set_color(COLORS['grid'])
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Legend
    legend = ax1.legend(loc='upper left', fontsize=11, facecolor=COLORS['panel'],
                       edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    
    # =========================================================================
    # STATISTICS BOX
    # =========================================================================
    
    stats_ax = fig.add_axes([0.72, 0.52, 0.25, 0.34])
    stats_ax.set_facecolor(COLORS['panel'])
    stats_ax.axis('off')
    for spine in stats_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['accent'])
        spine.set_linewidth(2)
    
    stats_ax.text(0.5, 0.95, 'Statistics', fontsize=16, fontweight='bold',
                 color=COLORS['accent'], ha='center', va='top', transform=stats_ax.transAxes)
    
    # Proprietary stats
    stats_ax.text(0.5, 0.82, 'Proprietary', fontsize=13, fontweight='bold',
                 color=COLORS['proprietary'], ha='center', transform=stats_ax.transAxes)
    stats_ax.text(0.5, 0.72, f'Mean: {prop_mean:.1f}  |  Std: {prop_std:.1f}',
                 fontsize=11, color=COLORS['text'], ha='center', transform=stats_ax.transAxes)
    stats_ax.text(0.5, 0.62, f'Min: {min(prop_scores):.1f}  |  Max: {max(prop_scores):.1f}',
                 fontsize=11, color=COLORS['muted'], ha='center', transform=stats_ax.transAxes)
    
    # Divider
    stats_ax.plot([0.1, 0.9], [0.52, 0.52], color=COLORS['grid'], linewidth=1, transform=stats_ax.transAxes)
    
    # Open source stats  
    stats_ax.text(0.5, 0.42, 'Open Source', fontsize=13, fontweight='bold',
                 color=COLORS['opensource'], ha='center', transform=stats_ax.transAxes)
    stats_ax.text(0.5, 0.32, f'Mean: {oss_mean:.1f}  |  Std: {oss_std:.1f}',
                 fontsize=11, color=COLORS['text'], ha='center', transform=stats_ax.transAxes)
    stats_ax.text(0.5, 0.22, f'Min: {min(oss_scores):.1f}  |  Max: {max(oss_scores):.1f}',
                 fontsize=11, color=COLORS['muted'], ha='center', transform=stats_ax.transAxes)
    
    # Gap analysis
    gap = prop_mean - oss_mean
    gap_color = COLORS['proprietary'] if gap > 0 else COLORS['opensource']
    stats_ax.text(0.5, 0.08, f'Gap: {gap:+.1f} pts',
                 fontsize=14, fontweight='bold', color=gap_color, ha='center', transform=stats_ax.transAxes)
    
    # =========================================================================
    # BOX PLOT COMPARISON
    # =========================================================================
    
    ax2 = fig.add_axes([0.08, 0.08, 0.40, 0.22])
    ax2.set_facecolor(COLORS['panel'])
    
    bp = ax2.boxplot([prop_scores, oss_scores], 
                     tick_labels=['Proprietary', 'Open Source'],
                     patch_artist=True, widths=0.6)
    
    # Style the boxplot
    colors_box = [COLORS['proprietary'], COLORS['opensource']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(COLORS['text'])
        patch.set_linewidth(2)
    
    for whisker in bp['whiskers']:
        whisker.set_color(COLORS['muted'])
        whisker.set_linewidth(1.5)
    
    for cap in bp['caps']:
        cap.set_color(COLORS['muted'])
        cap.set_linewidth(1.5)
    
    for median in bp['medians']:
        median.set_color(COLORS['gold'])
        median.set_linewidth(3)
    
    for flier in bp['fliers']:
        flier.set_markerfacecolor(COLORS['muted'])
        flier.set_markeredgecolor(COLORS['muted'])
        flier.set_markersize(5)
    
    ax2.set_ylabel('Quality Score', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax2.tick_params(colors=COLORS['muted'], labelsize=11)
    ax2.set_ylim(20, 100)
    
    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], axis='y')
    
    # Title for boxplot
    ax2.set_title('Distribution Comparison', fontsize=13, color=COLORS['text'], 
                  fontweight='bold', pad=10)
    
    # =========================================================================
    # TOP MODELS TABLE
    # =========================================================================
    
    table_ax = fig.add_axes([0.52, 0.08, 0.45, 0.35])
    table_ax.set_facecolor(COLORS['panel'])
    table_ax.axis('off')
    for spine in table_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['gold'])
        spine.set_linewidth(2)
    
    table_ax.text(0.5, 0.97, 'Top Models by Category', fontsize=14, fontweight='bold',
                 color=COLORS['gold'], ha='center', va='top', transform=table_ax.transAxes)
    
    # Top proprietary
    sorted_prop = sorted(scores['proprietary'], key=lambda x: x[1], reverse=True)[:5]
    table_ax.text(0.25, 0.85, 'Proprietary', fontsize=12, fontweight='bold',
                 color=COLORS['proprietary'], ha='center', transform=table_ax.transAxes)
    
    for i, (name, score) in enumerate(sorted_prop):
        short_name = name[:25] + '...' if len(name) > 25 else name
        y = 0.75 - i * 0.12
        table_ax.text(0.05, y, f'{i+1}.', fontsize=10, color=COLORS['muted'],
                     ha='left', transform=table_ax.transAxes)
        table_ax.text(0.10, y, short_name, fontsize=10, color=COLORS['text'],
                     ha='left', transform=table_ax.transAxes)
        table_ax.text(0.45, y, f'{score:.1f}', fontsize=10, fontweight='bold',
                     color=COLORS['proprietary_light'], ha='right', transform=table_ax.transAxes)
    
    # Top open source
    sorted_oss = sorted(scores['opensource'], key=lambda x: x[1], reverse=True)[:5]
    table_ax.text(0.75, 0.85, 'Open Source', fontsize=12, fontweight='bold',
                 color=COLORS['opensource'], ha='center', transform=table_ax.transAxes)
    
    for i, (name, score) in enumerate(sorted_oss):
        short_name = name[:25] + '...' if len(name) > 25 else name
        y = 0.75 - i * 0.12
        table_ax.text(0.55, y, f'{i+1}.', fontsize=10, color=COLORS['muted'],
                     ha='left', transform=table_ax.transAxes)
        table_ax.text(0.60, y, short_name, fontsize=10, color=COLORS['text'],
                     ha='left', transform=table_ax.transAxes)
        table_ax.text(0.95, y, f'{score:.1f}', fontsize=10, fontweight='bold',
                     color=COLORS['opensource_light'], ha='right', transform=table_ax.transAxes)
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    
    fig.text(0.5, 0.02, 
             f'Based on {len(prop_scores) + len(oss_scores)} models | Quality scores use weighted benchmark data from Artificial Analysis',
             fontsize=10, color=COLORS['muted'], ha='center', style='italic')
    
    # Save
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("PROPRIETARY VS OPEN SOURCE QUALITY DISTRIBUTION")
    print("=" * 60)
    
    # Load data
    models, scorer = load_data()
    
    # Calculate scores
    scores = calculate_scores(models, scorer)
    
    print(f"\nClassification:")
    print(f"  Proprietary: {len(scores['proprietary'])} models")
    print(f"  Open Source: {len(scores['opensource'])} models")
    
    # Create visualization
    output_path = Path(__file__).parent / 'proprietary_vs_opensource.png'
    create_distribution_plot(scores, str(output_path))
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

