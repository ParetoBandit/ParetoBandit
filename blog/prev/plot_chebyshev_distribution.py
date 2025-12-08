#!/usr/bin/env python3
"""
Plot the distribution of Chebyshev scores (4D optimization) across all models.
"""
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure we can import from llm_jury (parent of blog)
sys.path.append(str(Path(__file__).parent.parent))

from llm_jury.optimization.chebyshev_scorer import ChebyshevScorer
from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import ModelMetadata, RoutingDecision, PromptCategory, ProductArchetype

def load_models():
    """Load models from the cache."""
    try:
        # Path relative to blog/ folder
        cache_path = Path(__file__).parent.parent / 'model_registry_cache_enhanced.json'
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            return cache_data.get('models', [])
    except FileNotFoundError:
        print("❌ Cache file not found!")
        return []

def plot_chebyshev_distribution():
    print("📊 Generating 4D Chebyshev Score Distribution...")
    
    models_data = load_models()
    if not models_data:
        print("❌ No models found to analyze.")
        return

    # 1. Setup Scorers
    chebyshev_scorer = ChebyshevScorer(
        baseline_quality=88.7,      # GPT-4o reference
        baseline_cost=5.0,          # GPT-4o reference
        baseline_latency=500.0,     # GPT-4o reference
        baseline_trustability=2.0,  # Top ~2.5% trustability
        quality_weight=0.3,
        cost_weight=0.25,
        latency_weight=0.25,
        trustability_weight=0.2
    )
    
    quality_scorer = QualityScorer()
    
    # Use a general purpose decision for quality calculation
    decision = RoutingDecision(
        archetype=ProductArchetype.FRONTIER,
        category=PromptCategory.GENERAL,
        reason="Distribution Analysis"
    )
    
    scores = []
    valid_models = []
    
    print(f"   Scoring {len(models_data)} models...")
    
    for m_data in models_data:
        # Helper to safely get float value, defaulting to 0 if None or NaN
        def get_safe_float(key, default=0.0):
            val = m_data.get(key)
            if val is None:
                return default
            try:
                f_val = float(val)
                if np.isnan(f_val):
                    return default
                return f_val
            except (ValueError, TypeError):
                return default

        # Convert dict to ModelMetadata for QualityScorer
        try:
            model = ModelMetadata(
                name=m_data.get('name', 'Unknown'),
                mmlu_score=get_safe_float('mmlu_score'),
                gpqa_score=get_safe_float('gpqa_score'),
                math_score=get_safe_float('math_score'),
                ifeval_score=get_safe_float('ifeval_score'),
                tool_use_ability=get_safe_float('tool_use_ability'),
                context_window_k=get_safe_float('context_window_k'),
                hallucination_rate=get_safe_float('hallucination_rate'),
                ethics_score=get_safe_float('ethics_score'),
                hf_downloads=int(get_safe_float('hf_downloads')),
                hf_likes=int(get_safe_float('hf_likes')),
                hf_created_at=str(m_data.get('hf_created_at', "")),
                archetype=ProductArchetype.FRONTIER, # Default for scoring
                input_cost_per_m=get_safe_float('input_cost_per_m'),
                output_cost_per_m=get_safe_float('output_cost_per_m'),
                median_latency_ms=get_safe_float('median_latency_ms'),
                param_count_b=get_safe_float('param_count_b')
            )
        except Exception:
            continue

        # Calculate Quality Score on the fly
        quality = quality_scorer.calculate_quality_score(model, decision)
        
        # Filter out invalid models (e.g. 0 quality)
        if quality <= 10: # Basic threshold to filter garbage
            continue
            
        # Calculate Chebyshev Score
        score = chebyshev_scorer.score_model(
            model_name=model.name,
            quality=quality,
            cost=model.input_cost_per_m,
            latency=model.median_latency_ms,
            trustability=m_data.get('trustability_index', 0)
        )
        scores.append(score.chebyshev_distance)
        valid_models.append(score)

    if not scores:
        print("❌ No valid scores generated.")
        return

    # Print top 5 models (lowest distance)
    print("\n🏆 Top 5 Models (Lowest Chebyshev Distance):")
    valid_models.sort(key=lambda x: x.chebyshev_distance)
    
    for i, s in enumerate(valid_models[:5], 1):
        print(f"{i}. {s.model_name}: {s.chebyshev_distance:.4f}")
        print(f"   Q: {s.quality_score:.1f} | C: ${s.cost_per_m:.2f} | L: {s.latency_ms:.0f}ms | T: {s.trustability_index:.2f}")

    # 2. Create Plot with Dark Theme
    sns.set_style('darkgrid')
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Histogram with KDE
    # Using a purple color for Chebyshev to distinguish from Trustability (Teal)
    sns.histplot(scores, bins=50, kde=True, color='#9B59B6', ax=ax, alpha=0.7)
    
    # Calculate stats
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    # Add reference lines
    ax.axvline(mean_score, color='#E74C3C', linestyle='--', linewidth=2.5, label=f'Mean ({mean_score:.4f})', alpha=0.9)
    ax.axvline(median_score, color='#F39C12', linestyle='--', linewidth=2.5, label=f'Median ({median_score:.4f})', alpha=0.9)
    
    # Styling with white text
    ax.set_xlabel('Chebyshev Distance (Lower is Better)', fontsize=13, weight='bold', color='white')
    ax.set_ylabel('Number of Models', fontsize=13, weight='bold', color='white')
    ax.set_title('Model Performance Distribution (4D Optimization)', fontsize=15, weight='bold', pad=20, color='white')
    ax.legend(fontsize=11, framealpha=0.9, facecolor='#1a1a1a', edgecolor='white')
    ax.grid(True, alpha=0.2, color='white')
    
    # Style tick labels
    ax.tick_params(colors='white', which='both')
    
    # Add statistics text box
    stats_text = f'Total Models: {len(scores)}\n'
    stats_text += f'Mean Distance: {mean_score:.4f}\n'
    stats_text += f'Median Distance: {median_score:.4f}\n'
    stats_text += f'Best (Min): {min_score:.4f}\n'
    stats_text += f'Worst (Max): {max_score:.4f}'
    
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', alpha=0.9, edgecolor='white'))
    
    output_file = 'chebyshev_score_distribution.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"✅ Saved plot to {output_file}")
    


if __name__ == "__main__":
    plot_chebyshev_distribution()
