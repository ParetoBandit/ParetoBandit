#!/usr/bin/env python3
"""
Visualization: Quality Score Distribution Box Plots by Use Case

Shows how quality scores vary across different use cases (Coding, Data Science,
Creative, General, Finance) using box plots for easy comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import PromptCategory

# Color palette - vibrant and distinct
COLORS = {
    'bg': '#0f0f1a',
    'panel': '#1a1a2e',
    'grid': '#2d2d44',
    'text': '#e8e8f0',
    'muted': '#8888a0',
    'coding': '#61dafb',       # React blue
    'data_science': '#f7df1e', # JS yellow
    'creative': '#ff6b9d',     # Pink
    'general': '#98c379',      # Green
    'finance': '#c678dd',      # Purple
}

USE_CASE_CONFIG = [
    ('CODING', PromptCategory.CODING, COLORS['coding'], 'Programming & Development'),
    ('DATA_SCIENCE', PromptCategory.DATA_SCIENCE, COLORS['data_science'], 'Data Science & Math'),
    ('CREATIVE', PromptCategory.CREATIVE, COLORS['creative'], 'Creative Writing & Roleplay'),
    ('GENERAL', PromptCategory.GENERAL, COLORS['general'], 'General Assistant'),
    ('FINANCE', PromptCategory.FINANCE if hasattr(PromptCategory, 'FINANCE') else PromptCategory.GENERAL, COLORS['finance'], 'Finance & Analysis'),
]


def load_data():
    """Load model data and initialize scorer."""
    data_path = Path(__file__).parent.parent / 'data' / 'models_cache.json'
    
    with open(data_path) as f:
        models = json.load(f)
    
    print(f"Loaded {len(models)} models")
    scorer = QualityScorer(models)
    
    return models, scorer


def calculate_usecase_scores(models: List[Dict], scorer: QualityScorer) -> Dict[str, List[float]]:
    """Calculate quality scores for each use case."""
    
    scores_by_usecase = {}
    
    for name, category, color, label in USE_CASE_CONFIG:
        scores = []
        for model in models:
            score = scorer.calculate_quality_score(model, category=category)
            scores.append(score)
        scores_by_usecase[name] = {
            'scores': scores,
            'color': color,
            'label': label,
            'category': category,
        }
        print(f"  {label}: mean={np.mean(scores):.1f}, std={np.std(scores):.1f}")
    
    return scores_by_usecase


def create_boxplot_visualization(scores_data: Dict, models: List[Dict], scorer: QualityScorer, output_path: str):
    """Create the box plot visualization."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['bg'])
    
    # =========================================================================
    # HEADER
    # =========================================================================
    
    fig.text(0.5, 0.95, 'Quality Score Distribution by Use Case',
            fontsize=26, fontweight='bold', color=COLORS['text'], ha='center')
    fig.text(0.5, 0.91, 'How model quality varies across different task types',
            fontsize=14, color=COLORS['muted'], ha='center')
    
    # =========================================================================
    # MAIN BOX PLOT
    # =========================================================================
    
    ax = fig.add_axes([0.08, 0.15, 0.60, 0.70])
    ax.set_facecolor(COLORS['panel'])
    
    # Prepare data for boxplot
    use_cases = list(scores_data.keys())
    data = [scores_data[uc]['scores'] for uc in use_cases]
    colors = [scores_data[uc]['color'] for uc in use_cases]
    labels = [scores_data[uc]['label'] for uc in use_cases]
    
    # Create boxplot
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', 
                                                    markeredgecolor='white', markersize=8))
    
    # Style each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
        patch.set_linewidth(2)
    
    for whisker in bp['whiskers']:
        whisker.set_color(COLORS['muted'])
        whisker.set_linewidth(1.5)
        whisker.set_linestyle('--')
    
    for cap in bp['caps']:
        cap.set_color(COLORS['muted'])
        cap.set_linewidth(2)
    
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(3)
    
    for flier in bp['fliers']:
        flier.set_markerfacecolor(COLORS['muted'])
        flier.set_markeredgecolor(COLORS['muted'])
        flier.set_markersize(6)
        flier.set_alpha(0.6)
    
    # Styling
    ax.set_ylabel('Quality Score', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.set_xlabel('Use Case', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.tick_params(colors=COLORS['text'], labelsize=11)
    ax.set_ylim(0, 105)
    
    # Rotate x-labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'], axis='y')
    
    # Add value annotations
    for i, (uc, d) in enumerate(zip(use_cases, data)):
        median = np.median(d)
        mean = np.mean(d)
        ax.annotate(f'μ={mean:.0f}', xy=(i+1, max(d)+3), 
                   fontsize=9, color=colors[i], ha='center', fontweight='bold')
    
    # =========================================================================
    # STATISTICS PANEL
    # =========================================================================
    
    stats_ax = fig.add_axes([0.72, 0.40, 0.25, 0.45])
    stats_ax.set_facecolor(COLORS['panel'])
    stats_ax.axis('off')
    for spine in stats_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(2)
    
    stats_ax.text(0.5, 0.97, 'Statistics Summary', fontsize=14, fontweight='bold',
                 color=COLORS['text'], ha='center', va='top', transform=stats_ax.transAxes)
    
    y_pos = 0.88
    for uc in use_cases:
        d = scores_data[uc]
        scores = d['scores']
        color = d['color']
        label = d['label'].split(' ')[0]  # Short name
        
        stats_ax.text(0.05, y_pos, f"● {label}", fontsize=11, fontweight='bold',
                     color=color, ha='left', transform=stats_ax.transAxes)
        stats_ax.text(0.95, y_pos, f"Med: {np.median(scores):.1f}  IQR: {np.percentile(scores, 75) - np.percentile(scores, 25):.1f}",
                     fontsize=10, color=COLORS['muted'], ha='right', transform=stats_ax.transAxes)
        y_pos -= 0.12
    
    # =========================================================================
    # TOP MODEL BY USE CASE
    # =========================================================================
    
    top_ax = fig.add_axes([0.72, 0.15, 0.25, 0.22])
    top_ax.set_facecolor(COLORS['panel'])
    top_ax.axis('off')
    for spine in top_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(2)
    
    top_ax.text(0.5, 0.95, 'Top Model per Use Case', fontsize=12, fontweight='bold',
               color=COLORS['text'], ha='center', va='top', transform=top_ax.transAxes)
    
    y_pos = 0.78
    for uc in use_cases[:4]:  # Show top 4
        d = scores_data[uc]
        scores = d['scores']
        color = d['color']
        category = d['category']
        
        # Find top model for this use case
        best_idx = np.argmax(scores)
        best_model = models[best_idx]['name']
        best_score = scores[best_idx]
        
        short_name = best_model[:22] + '..' if len(best_model) > 22 else best_model
        top_ax.text(0.05, y_pos, f"● {short_name}", fontsize=9,
                   color=color, ha='left', transform=top_ax.transAxes)
        top_ax.text(0.95, y_pos, f"{best_score:.1f}", fontsize=9, fontweight='bold',
                   color=color, ha='right', transform=top_ax.transAxes)
        y_pos -= 0.18
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    
    fig.text(0.5, 0.02, 
             f'Based on {len(models)} models | ◆ = Mean, ─ = Median | Quality scores use task-specific benchmark weights',
             fontsize=10, color=COLORS['muted'], ha='center', style='italic')
    
    # Save
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("QUALITY SCORE DISTRIBUTION BY USE CASE")
    print("=" * 60)
    
    # Load data
    models, scorer = load_data()
    
    print("\nCalculating scores per use case...")
    scores_data = calculate_usecase_scores(models, scorer)
    
    # Create visualization
    output_path = Path(__file__).parent / 'usecase_quality_boxplots.png'
    create_boxplot_visualization(scores_data, models, scorer, str(output_path))
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

