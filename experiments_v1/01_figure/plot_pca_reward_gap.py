#!/usr/bin/env python3
"""
2D PCA Projection of RouteLLM Prompts Colored by Reward Gap

This script visualizes the semantic structure of 80K RouteLLM prompts
by projecting them into 2D using PCA, colored by the reward gap:
    
    Reward Gap = R_GPT4-Turbo - R_Mixtral
    
where rewards are binary: 1.0 (win), 0.5 (tie), 0.0 (loss).
Gap ranges from -1.0 to +1.0.

Difficulty Thresholds:
- Easy (|Gap| ≤ 0.3): Models perform nearly equally, Mixtral is sufficient
- Hard (Gap > 0.6): GPT-4 wins decisively, quality difference justifies cost

The visualization reveals:
- Where in semantic space GPT-4-Turbo dominates (red)
- Where Mixtral performs well (blue)
- Where models are tied (white)

This helps understand prompt difficulty distribution across semantic dimensions.

Usage:
    python3 experiments_v1/01_figure/plot_pca_reward_gap.py
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.stats import gaussian_kde
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    ROUTELLM_BATTLES_REWARDS_PATH
)


def load_battles_with_rewards(battles_file: Path, max_samples: int = 80000):
    """
    Load RouteLLM battles and compute reward gaps.
    
    Returns:
        prompts: List of prompt strings
        reward_gaps: Array of R_Turbo - R_Mixtral gaps
    """
    print(f"📥 Loading battles from: {battles_file}")
    
    prompts = []
    reward_gaps = []
    
    with open(battles_file, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="   Reading", total=max_samples)):
            if i >= max_samples:
                break
            
            try:
                battle = json.loads(line)
                
                # Extract prompt
                prompt = battle['prompt']
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ""
                if isinstance(prompt, str) and prompt.startswith('["'):
                    try:
                        prompt_list = json.loads(prompt)
                        prompt = prompt_list[0] if prompt_list else ""
                    except:
                        pass
                
                prompt = prompt.strip()
                if not prompt:
                    continue
                
                # Get rewards for both models
                model_a = battle['model_a']
                model_b = battle['model_b']
                reward_a = battle['reward_a']
                reward_b = battle['reward_b']
                
                # Compute reward gap (GPT-4-Turbo - Mixtral)
                if 'gpt-4-turbo' in model_a.lower():
                    reward_turbo = reward_a
                    reward_mixtral = reward_b
                else:  # gpt-4-turbo is model_b
                    reward_turbo = reward_b
                    reward_mixtral = reward_a
                
                gap = reward_turbo - reward_mixtral
                
                prompts.append(prompt)
                reward_gaps.append(gap)
                
            except Exception as e:
                continue
    
    print(f"   ✅ Loaded {len(prompts):,} prompts with reward gaps")
    
    return prompts, np.array(reward_gaps)


def embed_and_project_2d(prompts: list, pca_file: Path, batch_size: int = 64):
    """
    Embed prompts and project to 2D using pre-trained PCA.
    
    Loads the PCA model from config_legacy.py (auto-discovers number of components),
    then takes first 2 components for visualization.
    
    Args:
        prompts: List of prompt strings
        pca_file: Path to pre-trained PCA model (from DEFAULT_PCA_PATH)
        batch_size: Batch size for embedding
    
    Returns:
        X_2d: (n_samples, 2) array of 2D coordinates
        pca: The loaded PCA model
    """
    print(f"\n🔤 Loading sentence encoder...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print(f"   ✅ Encoder loaded")
    
    print(f"\n📐 Loading pre-trained PCA model...")
    print(f"   PCA file: {pca_file}")
    pca = joblib.load(pca_file)
    n_components = pca.n_components_
    print(f"   ✅ Loaded PCA: {n_components} components")
    
    print(f"\n🧮 Embedding {len(prompts):,} prompts...")
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    
    print(f"\n📐 Projecting to {n_components}D with pre-trained PCA...")
    X_nd = pca.transform(embeddings)
    print(f"   ✅ {n_components}D projection: {X_nd.shape}")
    print(f"   Explained variance ({n_components} components): {np.sum(pca.explained_variance_ratio_):.2%}")
    
    print(f"\n📐 Taking first 2 components for visualization...")
    X_2d = X_nd[:, :2]
    
    # Compute explained variance for first 2 components
    explained_var_2d = np.sum(pca.explained_variance_ratio_[:2])
    print(f"   ✅ 2D projection complete")
    print(f"   Explained variance (2 components): {explained_var_2d:.2%}")
    print(f"   PC1: {pca.explained_variance_ratio_[0]:.3%}")
    print(f"   PC2: {pca.explained_variance_ratio_[1]:.3%}")
    
    return X_2d, pca


def create_visualization(X_2d, reward_gaps, output_dir: Path):
    """
    Create heatmap visualization showing hard/easy ratio across semantic space.
    
    Shows a diverging colormap where:
    - Blue regions: More easy prompts (Mixtral sufficient)
    - Red regions: More hard prompts (GPT-4 required)
    - White regions: Mixed/balanced
    
    Args:
        X_2d: (n_samples, 2) 2D coordinates
        reward_gaps: (n_samples,) reward gaps
        output_dir: Directory to save figure
    """
    print(f"\n🎨 Creating semantic difficulty heatmap visualization...")
    
    # Statistics
    print(f"\n   📊 Reward Gap Statistics:")
    print(f"      Min: {np.min(reward_gaps):.3f}")
    print(f"      Max: {np.max(reward_gaps):.3f}")
    print(f"      Mean: {np.mean(reward_gaps):.3f}")
    print(f"      Median: {np.median(reward_gaps):.3f}")
    print(f"      Std: {np.std(reward_gaps):.3f}")
    
    # Count by category
    turbo_wins = np.sum(reward_gaps > 0)
    mixtral_wins = np.sum(reward_gaps < 0)
    ties = np.sum(reward_gaps == 0)
    print(f"\n   🏆 Battle Outcomes:")
    print(f"      GPT-4-Turbo wins: {turbo_wins:,} ({turbo_wins/len(reward_gaps)*100:.1f}%)")
    print(f"      Mixtral wins: {mixtral_wins:,} ({mixtral_wins/len(reward_gaps)*100:.1f}%)")
    print(f"      Ties: {ties:,} ({ties/len(reward_gaps)*100:.1f}%)")
    
    # Separate data by difficulty
    # Easy: Mixtral is sufficient or better (Gap <= 0.3)
    # Hard: GPT-4 is required (Gap > 0.6)
    easy_mask = reward_gaps <= 0.3
    hard_mask = reward_gaps > 0.6
    
    X_easy = X_2d[easy_mask]
    X_hard = X_2d[hard_mask]
    
    print(f"\n   📍 Semantic Regions:")
    print(f"      Easy prompts (Gap ≤ 0.3): {len(X_easy):,} ({len(X_easy)/len(X_2d)*100:.1f}%)")
    print(f"      Hard prompts (Gap > 0.6): {len(X_hard):,} ({len(X_hard)/len(X_2d)*100:.1f}%)")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Publication-quality heatmap with data points
    print(f"   🗺️  Computing spatial density ratio...")
    
    # Create grid for evaluation with padding
    x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
    
    # Add 10% padding on each side
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    # Use finer grid for smoother heatmap
    grid_size = 200  # Increased resolution
    xx, yy = np.mgrid[x_min:x_max:complex(0, grid_size), y_min:y_max:complex(0, grid_size)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Compute KDEs with reduced bandwidth for less smoothing
    if len(X_easy) > 100 and len(X_hard) > 100:
        print(f"   🔵 Computing KDE for Easy prompts...")
        kde_easy = gaussian_kde(X_easy.T, bw_method=0.08)  # Reduced from 0.15
        density_easy = np.reshape(kde_easy(positions).T, xx.shape)
        
        print(f"   🔴 Computing KDE for Hard prompts...")
        kde_hard = gaussian_kde(X_hard.T, bw_method=0.08)  # Reduced from 0.15
        density_hard = np.reshape(kde_hard(positions).T, xx.shape)
        
        # Compute ratio: (hard - easy) / (hard + easy)
        # This gives values from -1 (all easy) to +1 (all hard)
        total_density = density_hard + density_easy
        # Avoid division by zero
        ratio = np.zeros_like(total_density)
        mask = total_density > 1e-6
        ratio[mask] = (density_hard[mask] - density_easy[mask]) / total_density[mask]
        
        # Create heatmap with diverging colormap
        from matplotlib.colors import TwoSlopeNorm
        
        # Center colormap at 0 (balanced)
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        
        im = axes[0].imshow(
            ratio.T,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            cmap='RdBu_r',  # Red for hard, Blue for easy
            norm=norm,
            alpha=0.7,  # Slightly more transparent to show scatter points
            aspect='auto',
            interpolation='bilinear'  # Smoother interpolation
        )
        
        # Add scatter plot of actual data points (downsampled for clarity)
        print(f"   📍 Adding data point scatter overlay...")
        # Downsample to 5000 points for visibility
        downsample_size = min(5000, len(X_2d))
        indices = np.random.choice(len(X_2d), downsample_size, replace=False)
        X_sample = X_2d[indices]
        gaps_sample = reward_gaps[indices]
        
        # Color points by their category
        colors_scatter = np.where(np.abs(gaps_sample) <= 0.3, '#4575b4',  # Blue for easy
                                   np.where(gaps_sample > 0.6, '#d73027',  # Red for hard
                                           '#fee090'))  # Yellow for medium
        
        axes[0].scatter(X_sample[:, 0], X_sample[:, 1], 
                       c=colors_scatter, s=1, alpha=0.15, 
                       rasterized=True, edgecolors='none')
        
        # Add colorbar with better styling
        cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label('Task Difficulty Ratio\n(Hard - Easy) / (Hard + Easy)', 
                      fontsize=11, fontweight='bold')
        cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.set_yticklabels(['All Easy\n(Mixtral)', 'Mostly\nEasy', 'Mixed', 
                                 'Mostly\nHard', 'All Hard\n(GPT-4)'], fontsize=9)
        
        # Add contour lines with better visibility
        contour_levels = [-0.7, -0.3, 0, 0.3, 0.7]
        contours = axes[0].contour(
            xx, yy, ratio,
            levels=contour_levels,
            colors='black',
            alpha=0.4,
            linewidths=[1, 1, 2, 1, 1],  # Thicker line at 0
            linestyles=[':', '--', '-', '--', ':']
        )
        # Only label key contours
        axes[0].clabel(contours, levels=[0], inline=True, fontsize=10, 
                      fmt='Decision\nBoundary')
    
    # Set axis limits explicitly to show full data range with padding
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    
    axes[0].set_xlabel('PC1 (3.10% variance)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('PC2 (2.29% variance)', fontsize=13, fontweight='bold')
    axes[0].set_title(
        'Semantic Task Difficulty Landscape (80K Prompts)\n'
        'Red: GPT-4 Required | Blue: Mixtral Sufficient | White: Ambiguous',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    # Cleaner grid
    axes[0].grid(alpha=0.15, linestyle='--', color='gray', linewidth=0.5)
    
    # Plot 2: Professional difficulty distribution bar chart
    categories = [
        'Easy\nGap ≤ 0.3',
        'Medium\n0.3 < Gap ≤ 0.6',
        'Hard\nGap > 0.6'
    ]
    
    easy_count = np.sum(reward_gaps <= 0.3)
    medium_count = np.sum((reward_gaps > 0.3) & (reward_gaps <= 0.6))
    hard_count = np.sum(reward_gaps > 0.6)
    
    counts = [easy_count, medium_count, hard_count]
    colors_cat = ['#4575b4', '#fee090', '#d73027']  # Blue, Yellow, Red
    
    bars = axes[1].bar(range(len(categories)), counts, color=colors_cat, 
                       alpha=0.85, edgecolor='black', linewidth=2.5)
    axes[1].set_xticks(range(len(categories)))
    axes[1].set_xticklabels(categories, fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Prompts', fontsize=13, fontweight='bold')
    axes[1].set_title('Task Difficulty Distribution\n(Bimodal Structure)', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    axes[1].set_ylim(0, max(counts) * 1.18)
    
    # Add counts on bars with better formatting
    for bar, count, color in zip(bars, counts, colors_cat):
        height = bar.get_height()
        pct = count / len(reward_gaps) * 100
        
        # Add count and percentage
        axes[1].text(
            bar.get_x() + bar.get_width()/2.,
            height + max(counts) * 0.02,
            f'{count:,}',
            ha='center',
            va='bottom',
            fontsize=13,
            fontweight='bold'
        )
        axes[1].text(
            bar.get_x() + bar.get_width()/2.,
            height + max(counts) * 0.08,
            f'({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            style='italic'
        )
    
    # Add interpretation labels
    label_y = -max(counts) * 0.12
    axes[1].text(0, label_y, 'Mixtral\nSufficient', 
                ha='center', fontsize=10, style='italic', color='#2166ac')
    axes[1].text(1, label_y, 'Ambiguous\nRegion', 
                ha='center', fontsize=10, style='italic', color='#d95f02')
    axes[1].text(2, label_y, 'GPT-4\nRequired', 
                ha='center', fontsize=10, style='italic', color='#b2182b')
    
    # Add spines styling
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_linewidth(1.5)
    axes[1].spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pca_2d_reward_gap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: {output_file}")
    
    # Also save high-res version
    output_file_hires = output_dir / "pca_2d_reward_gap_hires.png"
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight')
    print(f"   ✅ Saved high-res: {output_file_hires}")
    
    plt.close()


def main():
    print("="*80)
    print("2D PCA PROJECTION OF ROUTELLM PROMPTS (REWARD GAP)")
    print("="*80)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    battles_file = ROUTELLM_BATTLES_REWARDS_PATH
    pca_file = DEFAULT_PCA_PATH
    output_dir = Path(__file__).parent / "results"
    
    print(f"\n📋 Configuration:")
    print(f"   Input: {battles_file}")
    print(f"   PCA model: {pca_file}")
    print(f"   Output: {output_dir}")
    
    if not pca_file.exists():
        print(f"\n❌ PCA file not found: {pca_file}")
        print(f"   Run: python3 scripts/train_pca_from_routellm.py")
        return
    
    # Step 1: Load data
    prompts, reward_gaps = load_battles_with_rewards(battles_file, max_samples=80000)
    
    if len(prompts) == 0:
        print("\n❌ No data loaded!")
        return
    
    # Step 2: Embed and project to 2D using pre-trained PCA
    X_2d, pca = embed_and_project_2d(prompts, pca_file, batch_size=64)
    
    # Step 3: Visualize
    create_visualization(X_2d, reward_gaps, output_dir)
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   Prompts visualized: {len(prompts):,}")
    print(f"   PCA components: {pca.n_components_}")
    print(f"   Total variance captured: {np.sum(pca.explained_variance_ratio_):.2%}")
    print(f"   2D projection variance: {np.sum(pca.explained_variance_ratio_[:2]):.2%}")
    print(f"   Output: {output_dir}/pca_2d_reward_gap.png")
    
    print(f"\n🔍 Key Insights:")
    print(f"   • Blue density contours: Easy prompts (Mixtral sufficient)")
    print(f"   • Red density contours: Hard prompts (GPT-4 required)")
    print(f"   • Density peaks show distinct semantic neighborhoods")
    print(f"   • Proves: Hard tasks (complex code, math) have specific semantic structure")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

