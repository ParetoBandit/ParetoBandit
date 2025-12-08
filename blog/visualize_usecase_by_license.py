#!/usr/bin/env python3
"""
Visualization: Quality Score Distribution by Use Case - Proprietary vs Open Source

Shows side-by-side box plots comparing proprietary and open source models
across different use cases.
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

# Color palette
COLORS = {
    'bg': '#0f0f1a',
    'panel': '#1a1a2e',
    'grid': '#2d2d44',
    'text': '#e8e8f0',
    'muted': '#8888a0',
    'proprietary': '#ff7b72',
    'proprietary_light': '#ffa198',
    'opensource': '#7ee787',
    'opensource_light': '#aff5b4',
}

# Classification
PROPRIETARY_CREATORS = {'Anthropic', 'OpenAI', 'Google', 'xAI', 'Cohere', 'Moonshot AI', 'Microsoft Azure'}
OPENSOURCE_CREATORS = {'Meta', 'DeepSeek', 'Mistral', 'Alibaba', 'Z AI'}

USE_CASES = [
    ('Coding', PromptCategory.CODING),
    ('Data Science', PromptCategory.DATA_SCIENCE),
    ('Creative', PromptCategory.CREATIVE),
    ('General', PromptCategory.GENERAL),
]


def classify_model(model: Dict) -> str:
    """Classify a model as proprietary or open source."""
    creator = model.get('creator_name', '')
    if creator in PROPRIETARY_CREATORS:
        return 'proprietary'
    elif creator in OPENSOURCE_CREATORS:
        return 'opensource'
    return 'proprietary'


def load_data():
    """Load model data and initialize scorer."""
    data_path = Path(__file__).parent.parent / 'data' / 'models_cache.json'
    
    with open(data_path) as f:
        models = json.load(f)
    
    print(f"Loaded {len(models)} models")
    scorer = QualityScorer(models)
    
    return models, scorer


def calculate_scores(models: List[Dict], scorer: QualityScorer) -> Dict:
    """Calculate scores by use case and license type."""
    
    results = {}
    
    for uc_name, category in USE_CASES:
        prop_scores = []
        oss_scores = []
        
        for model in models:
            score = scorer.calculate_quality_score(model, category=category)
            if classify_model(model) == 'proprietary':
                prop_scores.append(score)
            else:
                oss_scores.append(score)
        
        results[uc_name] = {
            'proprietary': prop_scores,
            'opensource': oss_scores,
        }
        
        print(f"  {uc_name}: Prop={np.mean(prop_scores):.1f}±{np.std(prop_scores):.1f}, "
              f"OSS={np.mean(oss_scores):.1f}±{np.std(oss_scores):.1f}, "
              f"Gap={np.mean(prop_scores) - np.mean(oss_scores):+.1f}")
    
    return results


def create_visualization(results: Dict, models: List[Dict], output_path: str):
    """Create grouped box plot visualization."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Header
    fig.text(0.5, 0.95, 'Quality Score by Use Case',
            fontsize=26, fontweight='bold', color=COLORS['text'], ha='center')
    fig.text(0.5, 0.91, 'Proprietary vs Open Source Models',
            fontsize=14, color=COLORS['muted'], ha='center')
    
    # Main plot
    ax = fig.add_axes([0.08, 0.15, 0.62, 0.70])
    ax.set_facecolor(COLORS['panel'])
    
    # Prepare grouped box plot data
    use_case_names = list(results.keys())
    n_groups = len(use_case_names)
    
    positions_prop = np.arange(n_groups) * 2.5
    positions_oss = positions_prop + 0.8
    
    # Create box plots
    prop_data = [results[uc]['proprietary'] for uc in use_case_names]
    oss_data = [results[uc]['opensource'] for uc in use_case_names]
    
    bp_prop = ax.boxplot(prop_data, positions=positions_prop, widths=0.6,
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='white', 
                                       markeredgecolor='white', markersize=7))
    
    bp_oss = ax.boxplot(oss_data, positions=positions_oss, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', 
                                      markeredgecolor='white', markersize=7))
    
    # Style proprietary boxes
    for patch in bp_prop['boxes']:
        patch.set_facecolor(COLORS['proprietary'])
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
        patch.set_linewidth(2)
    for whisker in bp_prop['whiskers']:
        whisker.set_color(COLORS['proprietary_light'])
        whisker.set_linewidth(1.5)
    for cap in bp_prop['caps']:
        cap.set_color(COLORS['proprietary_light'])
        cap.set_linewidth(2)
    for median in bp_prop['medians']:
        median.set_color('white')
        median.set_linewidth(3)
    for flier in bp_prop['fliers']:
        flier.set_markerfacecolor(COLORS['proprietary'])
        flier.set_markeredgecolor(COLORS['proprietary'])
        flier.set_alpha(0.5)
    
    # Style open source boxes
    for patch in bp_oss['boxes']:
        patch.set_facecolor(COLORS['opensource'])
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
        patch.set_linewidth(2)
    for whisker in bp_oss['whiskers']:
        whisker.set_color(COLORS['opensource_light'])
        whisker.set_linewidth(1.5)
    for cap in bp_oss['caps']:
        cap.set_color(COLORS['opensource_light'])
        cap.set_linewidth(2)
    for median in bp_oss['medians']:
        median.set_color('white')
        median.set_linewidth(3)
    for flier in bp_oss['fliers']:
        flier.set_markerfacecolor(COLORS['opensource'])
        flier.set_markeredgecolor(COLORS['opensource'])
        flier.set_alpha(0.5)
    
    # X-axis labels
    ax.set_xticks((positions_prop + positions_oss) / 2)
    ax.set_xticklabels(use_case_names, fontsize=12)
    
    # Styling
    ax.set_ylabel('Quality Score', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.set_xlabel('Use Case', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.tick_params(colors=COLORS['text'], labelsize=11)
    ax.set_ylim(0, 105)
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'], axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['proprietary'], alpha=0.7, edgecolor='white', label='Proprietary'),
        Patch(facecolor=COLORS['opensource'], alpha=0.7, edgecolor='white', label='Open Source'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
              facecolor=COLORS['panel'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    
    # Add gap annotations
    for i, uc in enumerate(use_case_names):
        prop_med = np.median(results[uc]['proprietary'])
        oss_med = np.median(results[uc]['opensource'])
        gap = prop_med - oss_med
        
        mid_x = (positions_prop[i] + positions_oss[i]) / 2
        y_pos = max(max(results[uc]['proprietary']), max(results[uc]['opensource'])) + 4
        
        color = COLORS['proprietary'] if gap > 0 else COLORS['opensource']
        ax.annotate(f'Δ{gap:+.0f}', xy=(mid_x + 0.4, y_pos), fontsize=10, 
                   color=color, ha='center', fontweight='bold')
    
    # =========================================================================
    # STATISTICS PANEL
    # =========================================================================
    
    stats_ax = fig.add_axes([0.73, 0.35, 0.24, 0.50])
    stats_ax.set_facecolor(COLORS['panel'])
    stats_ax.axis('off')
    for spine in stats_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(2)
    
    stats_ax.text(0.5, 0.97, 'Gap Analysis', fontsize=14, fontweight='bold',
                 color=COLORS['text'], ha='center', va='top', transform=stats_ax.transAxes)
    
    stats_ax.text(0.5, 0.88, '(Proprietary - Open Source)', fontsize=9,
                 color=COLORS['muted'], ha='center', transform=stats_ax.transAxes)
    
    y_pos = 0.78
    gaps = []
    for uc in use_case_names:
        prop_mean = np.mean(results[uc]['proprietary'])
        oss_mean = np.mean(results[uc]['opensource'])
        gap = prop_mean - oss_mean
        gaps.append(gap)
        
        color = COLORS['proprietary'] if gap > 0 else COLORS['opensource']
        stats_ax.text(0.10, y_pos, f"{uc}:", fontsize=11, color=COLORS['text'],
                     ha='left', transform=stats_ax.transAxes)
        stats_ax.text(0.90, y_pos, f"{gap:+.1f} pts", fontsize=11, fontweight='bold',
                     color=color, ha='right', transform=stats_ax.transAxes)
        y_pos -= 0.12
    
    # Average gap
    avg_gap = np.mean(gaps)
    stats_ax.plot([0.1, 0.9], [y_pos + 0.04, y_pos + 0.04], 
                  color=COLORS['grid'], linewidth=1, transform=stats_ax.transAxes)
    stats_ax.text(0.10, y_pos - 0.04, "Average:", fontsize=11, fontweight='bold',
                 color=COLORS['text'], ha='left', transform=stats_ax.transAxes)
    stats_ax.text(0.90, y_pos - 0.04, f"{avg_gap:+.1f} pts", fontsize=12, fontweight='bold',
                 color=COLORS['proprietary'], ha='right', transform=stats_ax.transAxes)
    
    # Insight
    stats_ax.text(0.5, 0.18, "Insight:", fontsize=10, fontweight='bold',
                 color=COLORS['muted'], ha='center', transform=stats_ax.transAxes)
    
    # Find biggest/smallest gap
    max_gap_uc = use_case_names[np.argmax(gaps)]
    min_gap_uc = use_case_names[np.argmin(gaps)]
    
    stats_ax.text(0.5, 0.08, f"Largest gap: {max_gap_uc}", fontsize=9,
                 color=COLORS['text'], ha='center', transform=stats_ax.transAxes)
    stats_ax.text(0.5, 0.00, f"Smallest gap: {min_gap_uc}", fontsize=9,
                 color=COLORS['text'], ha='center', transform=stats_ax.transAxes)
    
    # =========================================================================
    # SAMPLE SIZE
    # =========================================================================
    
    n_prop = len([m for m in models if classify_model(m) == 'proprietary'])
    n_oss = len([m for m in models if classify_model(m) == 'opensource'])
    
    fig.text(0.5, 0.02, 
             f'Proprietary: {n_prop} models | Open Source: {n_oss} models | ◆ = Mean, ─ = Median',
             fontsize=10, color=COLORS['muted'], ha='center', style='italic')
    
    # Save
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("QUALITY BY USE CASE: PROPRIETARY VS OPEN SOURCE")
    print("=" * 60)
    
    models, scorer = load_data()
    
    print("\nCalculating scores...")
    results = calculate_scores(models, scorer)
    
    output_path = Path(__file__).parent / 'usecase_proprietary_vs_opensource.png'
    create_visualization(results, models, str(output_path))
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

