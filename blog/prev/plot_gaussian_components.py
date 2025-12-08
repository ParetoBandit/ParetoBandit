#!/usr/bin/env python3
"""
Analyze Gaussian components of the Chebyshev score distribution.
Identifies distinct performance clusters (Frontier, Mid-tier, Legacy) using GMM.
"""
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

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

def plot_gaussian_components():
    print("📊 Analyzing Gaussian Components of Chebyshev Scores...")
    
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
    
    print(f"   Scoring {len(models_data)} models...")
    
    for m_data in models_data:
        # Helper to safely get float value
        def get_safe_float(key, default=0.0):
            val = m_data.get(key)
            if val is None: return default
            try:
                f_val = float(val)
                return default if np.isnan(f_val) else f_val
            except (ValueError, TypeError):
                return default

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
                archetype=ProductArchetype.FRONTIER,
                input_cost_per_m=get_safe_float('input_cost_per_m'),
                output_cost_per_m=get_safe_float('output_cost_per_m'),
                median_latency_ms=get_safe_float('median_latency_ms'),
                param_count_b=get_safe_float('param_count_b')
            )
            
            # Calculate Quality Score
            quality = quality_scorer.calculate_quality_score(model, decision)
            if quality <= 10: continue
                
            # Calculate Chebyshev Score
            score = chebyshev_scorer.score_model(
                model_name=model.name,
                quality=quality,
                cost=model.input_cost_per_m,
                latency=model.median_latency_ms,
                trustability=m_data.get('trustability_index', 0)
            )
            scores.append(score.chebyshev_distance)
            
        except Exception:
            continue

    if not scores:
        print("❌ No valid scores generated.")
        return

    scores = np.array(scores).reshape(-1, 1)
    
    # 2. Fit Gaussian Mixture Model (3 components)
    print("   Fitting Gaussian Mixture Model (3 components)...")
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(scores)
    
    # Sort components by mean (Cluster 0 = Best/Lowest Score)
    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    covariances = gmm.covariances_.flatten()
    
    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    weights = weights[sorted_indices]
    covariances = covariances[sorted_indices]
    stds = np.sqrt(covariances)
    
    # 3. Visualization
    sns.set_style('darkgrid')
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Plot histogram of actual data
    sns.histplot(scores.flatten(), bins=50, stat='density', kde=False, 
                 color='white', alpha=0.1, label='Actual Distribution', ax=ax)
    
    # Plot individual Gaussian components
    x_range = np.linspace(min(scores), max(scores), 1000).flatten()
    colors = ['#2ECC71', '#F1C40F', '#E74C3C'] # Green (Best), Yellow (Mid), Red (Worst)
    labels = ['Frontier (Best)', 'Competent (Mid)', 'Legacy/Niche (Worst)']
    
    total_pdf = np.zeros_like(x_range)
    
    for i in range(3):
        pdf = weights[i] * norm.pdf(x_range, means[i], stds[i])
        total_pdf += pdf
        ax.plot(x_range, pdf, color=colors[i], linewidth=2, linestyle='--', 
                label=f'{labels[i]}\n(μ={means[i]:.3f}, {weights[i]*100:.1f}%)')
        
        # Shade area under curve
        ax.fill_between(x_range, pdf, alpha=0.2, color=colors[i])

    # Plot combined GMM density
    ax.plot(x_range, total_pdf, color='white', linewidth=2.5, label='Combined GMM Fit')
    
    # Styling
    ax.set_xlabel('Chebyshev Distance (Lower is Better)', fontsize=13, weight='bold', color='white')
    ax.set_ylabel('Density', fontsize=13, weight='bold', color='white')
    ax.set_title('Performance Clusters: Gaussian Mixture Analysis', fontsize=16, weight='bold', pad=20, color='white')
    
    # Legend
    ax.legend(fontsize=11, framealpha=0.9, facecolor='#1a1a1a', edgecolor='white', loc='upper right')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white', which='both')
    
    # Add interpretation box
    stats_text = "Cluster Interpretation:\n"
    stats_text += f"• Frontier: Top {weights[0]*100:.1f}% models (Low distance)\n"
    stats_text += f"• Competent: Middle {weights[1]*100:.1f}% models\n"
    stats_text += f"• Legacy: Bottom {weights[2]*100:.1f}% models"
    
    ax.text(0.02, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', alpha=0.9, edgecolor='white'))
    
    output_file = 'chebyshev_gaussian_components.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"✅ Saved plot to {output_file}")
    
    # Print stats
    print("\nGaussian Components (Sorted by Performance):")
    for i in range(3):
        print(f"Cluster {i+1} ({labels[i]}):")
        print(f"   Mean: {means[i]:.4f}")
        print(f"   Std:  {stds[i]:.4f}")
        print(f"   Weight: {weights[i]*100:.1f}%")

if __name__ == "__main__":
    plot_gaussian_components()
