#!/usr/bin/env python3
"""
Bar chart visualization of model quality scores.
"""

import sys
sys.path.insert(0, '.')

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import ModelMetadata, PromptCategory


def create_model(d: dict) -> ModelMetadata:
    """Create ModelMetadata from raw data dict."""
    from llm_jury.core.models import ProductArchetype
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


def plot_quality_bar_chart(models, valid_data, output_path='blog/quality_scores_bar_chart.png'):
    """Create a bar chart of quality scores sorted in descending order."""
    
    # Initialize scorer
    scorer = QualityScorer(valid_data)
    
    # Calculate quality scores for all models
    model_scores = []
    for m, d in zip(models, valid_data):
        score = scorer.calculate_quality_score(d, PromptCategory.GENERAL)
        model_scores.append((m.name, score))
    
    # Sort by score descending
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 30 models
    model_scores = model_scores[:30]
    
    names = [x[0] for x in model_scores]
    scores = [x[1] for x in model_scores]
    
    # Show full names (no truncation)
    names_display = names
    
    # Create figure - sized for 30 bars with room for full names
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # Generate distinct colors using a colormap
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(names)))
    
    # Create vertical bar chart
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, scores, color=colors, edgecolor='white', linewidth=0.5)
    
    # Customize
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_display, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Quality Score (0-100)', fontsize=14, fontweight='bold')
    ax.set_title('Top 30 Models by Quality Score\n(General Category)', 
                 fontsize=18, fontweight='bold', pad=15)
    
    # Add score labels on top of bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f'{score:.1f}', va='bottom', ha='center', fontsize=9, color='#333333')
    
    # Set y-axis limits
    ax.set_ylim(0, 110)
    
    # Add gridlines
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == '__main__':
    print("Loading data...")
    models, valid_data = load_data()
    
    print("Generating quality score bar chart...")
    plot_quality_bar_chart(models, valid_data)
    
    print("✅ Done!")
