#!/usr/bin/env python3
"""
Distribution plot of model quality scores.
"""

import sys
sys.path.insert(0, '.')

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import ModelMetadata, PromptCategory, ProductArchetype


def create_model(d: dict) -> ModelMetadata:
    """Create ModelMetadata from raw data dict."""
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


def load_data():
    """Load model data."""
    cache_path = Path("data/models_complete_composite_indices.json")
    with open(cache_path) as f:
        raw_data = json.load(f)
    
    # Filter for models with intelligence_index (the main quality metric)
    valid_data = [m for m in raw_data if m.get('intelligence_index') is not None]
    models = [create_model(d) for d in valid_data]
    return models, valid_data


def plot_quality_distribution(models, valid_data, output_path='blog/quality_score_distribution.png'):
    """Create a histogram showing the distribution of quality scores."""
    
    # Initialize scorer
    scorer = QualityScorer(valid_data)
    
    # Calculate quality scores for all models
    scores = []
    for m, d in zip(models, valid_data):
        score = scorer.calculate_quality_score(d, PromptCategory.GENERAL)
        scores.append(score)
    
    scores = np.array(scores)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Histogram
    ax1 = axes[0]
    bins = np.linspace(0, 100, 21)  # 20 bins from 0 to 100
    n, bins_edges, patches = ax1.hist(scores, bins=bins, color='#3498db', edgecolor='white', 
                                       linewidth=1, alpha=0.8)
    
    # Color bins by position (matching bar chart - high scores = purple/blue, low scores = yellow/green)
    cm = plt.cm.viridis
    n_bins = len(patches)
    for i, p in enumerate(patches):
        # Reverse the color mapping: first bins (low scores) get high viridis values (yellow)
        # last bins (high scores) get low viridis values (purple)
        color_val = 0.9 - (i / n_bins) * 0.8  # Maps from 0.9 (yellow) to 0.1 (purple)
        plt.setp(p, 'facecolor', cm(color_val))
    
    ax1.set_xlabel('Quality Score (0-100)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Models', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Quality Scores\n(All Models)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 100)
    
    # Add statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    
    stats_text = f'Mean: {mean_score:.1f}\nMedian: {median_score:.1f}\nStd Dev: {std_score:.1f}\nN: {len(scores)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add vertical lines for mean and median
    ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean ({mean_score:.1f})')
    ax1.axvline(median_score, color='orange', linestyle='-', linewidth=2, label=f'Median ({median_score:.1f})')
    ax1.legend(loc='upper right', fontsize=10)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Right plot: Box plot with swarm
    ax2 = axes[1]
    
    # Create box plot
    bp = ax2.boxplot(scores, vert=True, widths=0.5, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.6)
    bp['medians'][0].set_color('orange')
    bp['medians'][0].set_linewidth(2)
    
    # Add jittered scatter points - color by score (high scores = purple, low = yellow)
    jitter = np.random.normal(0, 0.04, len(scores))
    # Reverse the colormap so high scores are purple (like bar chart)
    ax2.scatter(1 + jitter, scores, alpha=0.5, s=30, c=100-scores, cmap='viridis', zorder=3)
    
    ax2.set_ylabel('Quality Score (0-100)', fontsize=12, fontweight='bold')
    ax2.set_title('Quality Score Distribution\n(Box Plot with Data Points)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['All Models'])
    
    # Add percentile annotations
    percentiles = [25, 50, 75, 90, 95]
    pct_values = np.percentile(scores, percentiles)
    for pct, val in zip(percentiles, pct_values):
        ax2.annotate(f'{pct}th: {val:.1f}', xy=(1.3, val), fontsize=9,
                    verticalalignment='center')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == '__main__':
    print("Loading data...")
    models, valid_data = load_data()
    
    print("Generating quality score distribution plot...")
    plot_quality_distribution(models, valid_data)
    
    print("✅ Done!")

