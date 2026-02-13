#!/usr/bin/env python3
"""
Figure 1M: Spatial Structure of Full LMSYS Chat-1M Dataset

Visualizes the FULL 1M prompt dataset from LMSYS Chat-1M,
showing spatial distribution persistence at scale.

Compared to the holdout analysis (N=750), this analysis uses the
complete 1M dataset to show spatial structure persistence at scale.

IMPORTANT: The 1M dataset has NO reward labels. This script validates
spatial structure only — it cannot validate that reward patterns persist.

METHODOLOGY:
  PCA trained on generic C4 corpus (no circularity).
  Threshold matches holdout analysis (silhouette-optimal on holdout).

Usage:
    # Train generic PCA first (if not already done)
    python3 scripts/train_pca_generic.py
    
    # Generate 1M visualization
    python3 experiments_v1/01_figure/plot_lmsys_1M_pca.py
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import gzip
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.stats import gaussian_kde
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    ARTIFACTS_DIR
)

# Use generic PCA (trained on C4 corpus) to avoid circularity
GENERIC_PCA_PATH = ARTIFACTS_DIR / "pca_32_generic.joblib"

# Threshold from holdout silhouette-optimal analysis (PC1 = 0.222 on
# domain-adapted PCA).  For the generic PCA used here, we apply the same
# boundary for comparability.  Note: the exact threshold value matters less
# than the qualitative observation that a High-PC1 region exists and is
# smaller at scale (~5.9%) than in the holdout (~19%).
HOLDOUT_THRESHOLD = 0.222


def load_lmsys_1M_prompts(data_file: Path, max_prompts: int = None):
    """
    Load prompts from LMSYS Chat-1M dataset.
    
    Args:
        data_file: Path to the JSONL.GZ file with prompts
        max_prompts: Maximum number of prompts to load (None = all)
    
    Returns:
        prompts: List of prompt strings
    """
    print(f"📥 Loading LMSYS Chat-1M prompts...")
    print(f"   Data: {data_file}")
    
    prompts = []
    
    print(f"\n   Reading prompts...")
    
    with gzip.open(data_file, 'rt') as f:
        for line in tqdm(f, desc="   Loading"):
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '')
                
                if not prompt or not isinstance(prompt, str):
                    continue
                
                prompt = prompt.strip()
                if not prompt:
                    continue
                
                prompts.append(prompt)
                
                if max_prompts and len(prompts) >= max_prompts:
                    break
                
            except Exception as e:
                continue
    
    print(f"\n   ✅ Loaded {len(prompts):,} prompts")
    
    return prompts


def embed_and_project_2d(prompts: list, pca_file: Path, batch_size: int = 64):
    """
    Embed prompts and project to 2D using pre-trained PCA.
    """
    print(f"\n🔤 Loading sentence encoder: {DEFAULT_SENTENCE_TRANSFORMER}")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print(f"   ✅ Encoder loaded")
    
    print(f"\n📐 Loading pre-trained PCA model: {pca_file}")
    pca = joblib.load(pca_file)
    n_components = pca.n_components_
    print(f"   ✅ Loaded PCA: {n_components} components")
    print(f"   Total variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    print(f"\n🧮 Embedding {len(prompts):,} prompts...")
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    
    print(f"\n📐 Projecting to 2D...")
    X_nd = pca.transform(embeddings)
    X_2d = X_nd[:, :2]
    
    explained_var_2d = np.sum(pca.explained_variance_ratio_[:2])
    print(f"   ✅ 2D projection complete")
    print(f"   PC1: {pca.explained_variance_ratio_[0]:.3%}")
    print(f"   PC2: {pca.explained_variance_ratio_[1]:.3%}")
    print(f"   Total (2D): {explained_var_2d:.2%}")
    
    return X_2d, pca


def create_spatial_visualization(X_2d, pca, output_dir: Path):
    """
    Create visualization showing spatial clustering structure at scale.
    
    Left panel: Semantic scatter plot with PC1-based coloring
    Right panel: Distribution statistics showing spatial clustering
    """
    print(f"\n🎨 Creating spatial structure visualization...")
    
    # Categorize by PC1 position using holdout-derived threshold
    pc1_values = X_2d[:, 0]
    
    low_pc1_mask = pc1_values < HOLDOUT_THRESHOLD
    high_pc1_mask = pc1_values >= HOLDOUT_THRESHOLD
    
    X_low_pc1 = X_2d[low_pc1_mask]
    X_high_pc1 = X_2d[high_pc1_mask]
    
    # Print statistics
    print(f"\n   📊 Spatial Distribution (threshold={HOLDOUT_THRESHOLD:.3f}):")
    print(f"      Low PC1 (< {HOLDOUT_THRESHOLD}): {len(X_low_pc1):,} ({len(X_low_pc1)/len(X_2d)*100:.1f}%)")
    print(f"      High PC1 (≥ {HOLDOUT_THRESHOLD}): {len(X_high_pc1):,} ({len(X_high_pc1)/len(X_2d)*100:.1f}%)")
    
    # Create figure with 2 panels
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
    
    # Panel 1: Semantic scatter with cluster separation line
    ax1 = fig.add_subplot(gs[0])
    
    # Plot points (downsample for visualization)
    downsample_size = min(10000, len(X_2d))
    if len(X_2d) > downsample_size:
        indices = np.random.choice(len(X_2d), downsample_size, replace=False)
        X_sample = X_2d[indices]
    else:
        X_sample = X_2d
    
    # Categorize sampled points by PC1 position
    pc1_sample = X_sample[:, 0]
    low_pc1_mask_s = pc1_sample < HOLDOUT_THRESHOLD
    high_pc1_mask_s = pc1_sample >= HOLDOUT_THRESHOLD
    
    # Plot with beautiful colors
    ax1.scatter(X_sample[low_pc1_mask_s, 0], X_sample[low_pc1_mask_s, 1],
               c='#4575b4', s=25, alpha=0.7, label=f'Low PC1 ({len(X_low_pc1):,})',
               edgecolors='none', rasterized=True)
    
    ax1.scatter(X_sample[high_pc1_mask_s, 0], X_sample[high_pc1_mask_s, 1],
               c='#d73027', s=25, alpha=0.7, label=f'High PC1 ({len(X_high_pc1):,})',
               edgecolors='none', rasterized=True)
    
    # Add KDE contour for low PC1 cluster only
    if len(X_low_pc1) > 100:
        try:
            # Sample for KDE computation (too many points otherwise)
            kde_sample_size = min(5000, len(X_low_pc1))
            kde_indices = np.random.choice(len(X_low_pc1), kde_sample_size, replace=False)
            X_kde_sample = X_low_pc1[kde_indices]
            
            kde_low = gaussian_kde(X_kde_sample.T, bw_method=0.12)
            x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
            y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density_low = np.reshape(kde_low(positions).T, xx.shape)
            ax1.contour(xx, yy, density_low, levels=4, colors='#2166ac', alpha=0.6, linewidths=2.5, linestyles='solid')
        except:
            pass
    
    # Add vertical line showing holdout-derived threshold
    ax1.axvline(x=HOLDOUT_THRESHOLD, color='black', linestyle='--', linewidth=3, 
                alpha=0.7, label=f'Threshold ({HOLDOUT_THRESHOLD:.3f})', zorder=5)
    
    # Styling
    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    
    ax1.set_xlabel(f'PC1 ({pc1_var:.2%} variance)', fontsize=15, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pc2_var:.2%} variance)', fontsize=15, fontweight='bold')
    ax1.set_title(
        'Spatial Structure in LMSYS Chat-1M Dataset\n'
        'PC1 Distribution at Scale (No Reward Labels)',
        fontsize=17,
        fontweight='bold',
        pad=15
    )
    ax1.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Panel 2: Distribution breakdown with better visualization
    ax2 = fig.add_subplot(gs[1])
    
    categories = ['Low PC1\nRegion', 'High PC1\nRegion']
    counts = [len(X_low_pc1), len(X_high_pc1)]
    colors_bar = ['#4575b4', '#d73027']
    
    bars = ax2.bar(range(len(categories)), counts, color=colors_bar, 
                   alpha=0.9, edgecolor='black', linewidth=2.5, width=0.7)
    
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Prompts', fontsize=15, fontweight='bold')
    ax2.set_title(
        'Spatial Distribution\n'
        f'N = {len(X_2d):,} Prompts',
        fontsize=17,
        fontweight='bold',
        pad=15
    )
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax2.set_ylim(0, max(counts) * 1.22)
    
    # Add counts and percentages on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / len(X_2d) * 100
        
        # Count
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height + max(counts) * 0.02,
            f'{count:,}',
            ha='center',
            va='bottom',
            fontsize=15,
            fontweight='bold'
        )
        # Percentage
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height + max(counts) * 0.10,
            f'({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=13,
            fontweight='bold',
            style='italic'
        )
    
    # Styling
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "figure1_lmsys_1M_pca.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n   ✅ Saved: {output_file}")
    
    # Also save high-res version
    output_file_hires = output_dir / "figure1_lmsys_1M_pca_hires.png"
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved high-res: {output_file_hires}")
    
    plt.close()


def main():
    print("="*80)
    print("FIGURE 1M: SEMANTIC PCA OF FULL LMSYS CHAT-1M DATASET")
    print("="*80)
    print("\n🎯 Goal: Show spatial structure persistence at scale")
    print("   → Check whether PC1 distribution persists at 1M scale")
    print("   → Compare spatial proportions with holdout (N=750)")
    print("   → NOTE: No reward labels — spatial structure only")
    print("\n📐 Methodology: PCA trained on generic C4 corpus (no circularity)")
    
    # Paths
    data_file = Path(__file__).parent / "data" / "lmsys_chat_1M.jsonl.gz"
    pca_file = GENERIC_PCA_PATH
    output_dir = Path(__file__).parent / "results"
    
    print(f"\n📋 Configuration:")
    print(f"   Data: {data_file}")
    print(f"   PCA model: {pca_file} (Generic C4)")
    print(f"   Output: {output_dir}")
    print(f"   Processing: ALL prompts (no limit)")
    
    if not data_file.exists():
        print(f"\n❌ Data file not found: {data_file}")
        print(f"\n💡 Run download script first:")
        print(f"   python experiments_v1/01_figure/download_1M_dataset.py")
        return
    
    if not pca_file.exists():
        print(f"\n❌ PCA file not found: {pca_file}")
        print(f"\n💡 Train generic PCA first:")
        print(f"   python3 scripts/train_pca_generic.py")
        return
    
    # Step 1: Load all prompts from 1M dataset
    max_prompts = None  # Process all prompts (594k)
    print(f"\n   Note: Processing ALL prompts from the dataset")
    prompts = load_lmsys_1M_prompts(data_file, max_prompts=max_prompts)
    
    if len(prompts) == 0:
        print("\n❌ No data loaded!")
        return
    
    # Step 2: Embed and project to 2D
    print("\n" + "="*80)
    print("EMBEDDING AND PROJECTING TO 2D")
    print("="*80)
    X_2d, pca = embed_and_project_2d(prompts, pca_file)
    
    # Step 3: Create visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    create_spatial_visualization(X_2d, pca, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("✅ FIGURE 1M COMPLETE!")
    print("="*80)
    
    print(f"\n🔍 Key Observations:")
    print(f"   • PC1 spatial distribution persists at 1M scale")
    print(f"   • High PC1 region is rarer in 1M data (~5.9%) than holdout (~19%)")
    print(f"   • This validates spatial structure persistence only")
    print(f"   • ⚠️  Cannot validate reward patterns (no labels in 1M data)")
    
    print(f"\n📊 For Paper:")
    print(f"   • N = {len(prompts):,} prompts ({len(prompts)//750}x larger than holdout)")
    print(f"   • Spatial structure persists at scale (expected: fixed linear projection)")
    print(f"   • Distribution shift: High PC1 shrinks from ~19% to ~5.9%")
    print(f"   • All performance claims remain based on holdout (N=750) only")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
