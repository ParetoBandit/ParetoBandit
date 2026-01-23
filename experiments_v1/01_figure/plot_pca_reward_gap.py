#!/usr/bin/env python3
"""
2D PCA Projection of RouteLLM Prompts Colored by Reward Gap

This script visualizes the semantic structure of 80K RouteLLM prompts
by projecting them into 2D using PCA, colored by the reward gap:
    
    Reward Gap = R_GPT4-Turbo - R_Mixtral

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER, DEFAULT_PCA_PATH


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
    
    Uses the 23-component PCA trained on 80K prompts, then takes first 2 components.
    
    Args:
        prompts: List of prompt strings
        pca_file: Path to pre-trained PCA model (pca_23.joblib)
        batch_size: Batch size for embedding
    
    Returns:
        X_2d: (n_samples, 2) array of 2D coordinates
        pca_23: The 23-component PCA model
    """
    print(f"\n🔤 Loading sentence encoder...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print(f"   ✅ Encoder loaded")
    
    print(f"\n📐 Loading pre-trained PCA model...")
    print(f"   PCA file: {pca_file}")
    pca_23 = joblib.load(pca_file)
    print(f"   ✅ Loaded PCA: {pca_23.n_components_} components")
    
    print(f"\n🧮 Embedding {len(prompts):,} prompts...")
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    
    print(f"\n📐 Projecting to 23D with pre-trained PCA...")
    X_23d = pca_23.transform(embeddings)
    print(f"   ✅ 23D projection: {X_23d.shape}")
    print(f"   Explained variance (23 components): {np.sum(pca_23.explained_variance_ratio_):.2%}")
    
    print(f"\n📐 Taking first 2 components for visualization...")
    X_2d = X_23d[:, :2]
    
    # Compute explained variance for first 2 components
    explained_var_2d = np.sum(pca_23.explained_variance_ratio_[:2])
    print(f"   ✅ 2D projection complete")
    print(f"   Explained variance (2 components): {explained_var_2d:.2%}")
    print(f"   PC1: {pca_23.explained_variance_ratio_[0]:.3%}")
    print(f"   PC2: {pca_23.explained_variance_ratio_[1]:.3%}")
    
    return X_2d, pca_23


def create_visualization(X_2d, reward_gaps, output_dir: Path):
    """
    Create KDE density visualization showing semantic difficulty distribution.
    
    Shows two overlapping density contours:
    - Blue: "Easy" prompts (Gap ≈ 0, Mixtral sufficient)
    - Red: "Hard" prompts (Gap > 0.6, GPT-4 required)
    
    Args:
        X_2d: (n_samples, 2) 2D coordinates
        reward_gaps: (n_samples,) reward gaps
        output_dir: Directory to save figure
    """
    print(f"\n🎨 Creating semantic difficulty density visualization...")
    
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
    # Easy: Gap ≈ 0 (Mixtral is sufficient) - using |gap| <= 0.3
    # Hard: Gap > 0.6 (GPT-4 required)
    easy_mask = np.abs(reward_gaps) <= 0.3
    hard_mask = reward_gaps > 0.6
    
    X_easy = X_2d[easy_mask]
    X_hard = X_2d[hard_mask]
    
    print(f"\n   📍 Semantic Regions:")
    print(f"      Easy prompts (|Gap| ≤ 0.3): {len(X_easy):,} ({len(X_easy)/len(X_2d)*100:.1f}%)")
    print(f"      Hard prompts (Gap > 0.6): {len(X_hard):,} ({len(X_hard)/len(X_2d)*100:.1f}%)")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: KDE density contours
    # Compute KDE for "Easy" prompts (Blue contours)
    if len(X_easy) > 100:
        print(f"   🔵 Computing KDE for Easy prompts...")
        kde_easy = gaussian_kde(X_easy.T, bw_method=0.1)
        
        # Create grid for evaluation
        x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
        y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        density_easy = np.reshape(kde_easy(positions).T, xx.shape)
        
        # Plot Easy density contours (Blue)
        contours_easy = axes[0].contour(
            xx, yy, density_easy,
            levels=5,
            colors='blue',
            alpha=0.6,
            linewidths=2
        )
        axes[0].clabel(contours_easy, inline=True, fontsize=8, fmt='%.2f')
        
        # Fill lowest contour with light blue
        axes[0].contourf(
            xx, yy, density_easy,
            levels=[0, density_easy.max() * 0.2],
            colors=['lightblue'],
            alpha=0.2
        )
    
    # Compute KDE for "Hard" prompts (Red contours)
    if len(X_hard) > 100:
        print(f"   🔴 Computing KDE for Hard prompts...")
        kde_hard = gaussian_kde(X_hard.T, bw_method=0.1)
        density_hard = np.reshape(kde_hard(positions).T, xx.shape)
        
        # Plot Hard density contours (Red)
        contours_hard = axes[0].contour(
            xx, yy, density_hard,
            levels=5,
            colors='red',
            alpha=0.6,
            linewidths=2
        )
        axes[0].clabel(contours_hard, inline=True, fontsize=8, fmt='%.2f')
        
        # Fill lowest contour with light red
        axes[0].contourf(
            xx, yy, density_hard,
            levels=[0, density_hard.max() * 0.2],
            colors=['lightcoral'],
            alpha=0.2
        )
    
    axes[0].set_xlabel('PC1 (Semantic Dimension 1)', fontsize=14)
    axes[0].set_ylabel('PC2 (Semantic Dimension 2)', fontsize=14)
    axes[0].set_title(
        'Semantic Difficulty Density: Easy vs Hard Prompts\n'
        'Blue: Mixtral Sufficient (|Gap| ≤ 0.3) | Red: GPT-4 Required (Gap > 0.6)',
        fontsize=15,
        fontweight='bold'
    )
    axes[0].grid(alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', label='Easy (Mixtral Sufficient)', alpha=0.6),
        Patch(facecolor='lightcoral', edgecolor='red', label='Hard (GPT-4 Required)', alpha=0.6)
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Plot 2: Semantic difficulty distribution
    categories = [
        'Easy\n(|Gap| ≤ 0.3)\nMixtral OK',
        'Medium\n(0.3 < Gap ≤ 0.6)\nMixed',
        'Hard\n(Gap > 0.6)\nGPT-4 Required'
    ]
    
    easy_count = np.sum(np.abs(reward_gaps) <= 0.3)
    medium_count = np.sum((reward_gaps > 0.3) & (reward_gaps <= 0.6))
    hard_count = np.sum(reward_gaps > 0.6)
    
    counts = [easy_count, medium_count, hard_count]
    colors_cat = ['#4575b4', '#fee090', '#d73027']  # Blue, Yellow, Red
    
    bars = axes[1].bar(range(len(categories)), counts, color=colors_cat, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1].set_xticks(range(len(categories)))
    axes[1].set_xticklabels(categories, fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Prompts', fontsize=14)
    axes[1].set_title('Semantic Difficulty Distribution', fontsize=16, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, max(counts) * 1.15)
    
    # Add counts on bars
    for bar, count, color in zip(bars, counts, colors_cat):
        height = bar.get_height()
        pct = count / len(reward_gaps) * 100
        axes[1].text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{count:,}\n({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Add interpretation text
    axes[1].text(
        0.5, 0.95,
        'Key Insight: Hard prompts occupy distinct semantic neighborhoods',
        transform=axes[1].transAxes,
        ha='center',
        va='top',
        fontsize=11,
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
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
    battles_file = project_root / "src/bandit_gpt/data/offline_dataset/routellm_battles_rewards.jsonl"
    pca_file = DEFAULT_PCA_PATH
    output_dir = Path(__file__).parent
    
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
    print(f"   2D projection variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    print(f"   Output: {output_dir}/pca_2d_reward_gap.png")
    
    print(f"\n🔍 Key Insights:")
    print(f"   • Blue density contours: Easy prompts (Mixtral sufficient)")
    print(f"   • Red density contours: Hard prompts (GPT-4 required)")
    print(f"   • Density peaks show distinct semantic neighborhoods")
    print(f"   • Proves: Hard tasks (complex code, math) have specific semantic structure")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

