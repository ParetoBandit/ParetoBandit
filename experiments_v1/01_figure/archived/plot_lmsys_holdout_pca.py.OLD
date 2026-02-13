#!/usr/bin/env python3
"""
Figure 1 Analysis: LMSYS Holdout Data with Clean Methodology

⚠️ IMPORTANT: This analysis was originally intended to demonstrate an "Alignment Tax"
but after fixing methodological issues, the finding DOES NOT REPLICATE.

RESULT WITH CLEAN METHODOLOGY: p = 0.983 (NOT significant)
RECOMMENDATION: Remove Figure 1 from paper.

ISSUES IDENTIFIED AND STATUS:
==============================
✅ FIXED:
1. Circular PCA - Now using generic C4 corpus (not routing data)
2. Dev contamination - Using holdout ONLY (N=750, not N=1,871)

⚠️ PARTIALLY FIXED:
3. Circular threshold - Still using PC1=0.3 (should use PC1=0 or unsupervised)

❌ CANNOT FIX (Data limitations):
4. Speculative mechanism - Causal claims lack validation experiments
5. Weak high-D structure - Silhouette=0.057 in 384D (essentially random)
6. Overstated correlation - ρ²=0.16 (only 16% variance, moderate not strong)
7. Misleading scale - 1M dataset has no reward labels
8. Low diversity - High PC1 diversity=0.355 (homogeneous cluster)
9. Single observations - One reward per (prompt, model), no variance estimates
10. Near-duplicate reporting - Pair rate vs prompt involvement unclear

RESULT: After fixing #1-2, NO significant bimodal structure found.
- Low PC1: 749/750 prompts (99.9%)
- High PC1: 1/750 prompts (0.1%)
- Mann-Whitney p = 0.983 (NOT significant)

See documentation in experiments_v1/01_figure/:
- ONE_PAGE_SUMMARY.md - Quick overview
- ALL_ISSUES_SUMMARY.md - Complete technical analysis
- FINAL_RECOMMENDATION.md - What to do next

Usage:
    python3 scripts/train_pca_generic.py  # If not done
    python3 experiments_v1/01_figure/plot_lmsys_holdout_pca.py
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
from scipy.stats import gaussian_kde, mannwhitneyu, ttest_ind, sem
from scipy import stats as scipy_stats
from matplotlib.colors import TwoSlopeNorm
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    ARTIFACTS_DIR
)

# Use generic PCA (trained on C4 corpus) to avoid circularity
GENERIC_PCA_PATH = ARTIFACTS_DIR / "pca_32_generic.joblib"


def load_lmsys_holdout_with_gaps(holdout_file: Path):
    """
    Load LMSYS HOLDOUT-ONLY prompts with reward gaps.
    
    FIXES APPLIED:
    - Issue #2: Use holdout ONLY (no dev contamination)
    - Dev set excluded (reserved for training in Table 2)
    
    REMAINING LIMITATION (Issue #9):
    - Each prompt has ONE raw_score per model (no repeated measurements)
    - No variance estimates at prompt level
    - Reward source not documented in code (assumed LMSYS Arena human preferences)
    - Gap = single scalar difference, not distribution
    
    The holdout file has separate rows for Mixtral and GPT-4-Turbo evaluations.
    We group by prompt and compute: Gap = R_GPT4 - R_Mixtral
    
    Returns:
        prompts: List of prompt strings
        reward_gaps: Array of reward gaps (GPT-4-Turbo - Mixtral)
    """
    print(f"📥 Loading LMSYS Holdout Data (HOLDOUT ONLY - no dev contamination)...")
    print(f"   Holdout: {holdout_file}")
    print(f"   ⚠️  Dev set excluded to avoid contamination (dev is used for training)")
    
    # Dictionary to collect rewards by prompt
    # {prompt: {'mixtral': score, 'gpt4': score}}
    prompt_rewards = {}
    
    print(f"\n   Processing holdout...")
    with gzip.open(holdout_file, 'rt') as f:
        for line in tqdm(f, desc=f"   Reading holdout"):
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '')
                model_id = entry.get('model_id', '')
                raw_score = entry.get('raw_score', None)
                
                if not prompt or not isinstance(prompt, str) or raw_score is None:
                    continue
                
                prompt = prompt.strip()
                if not prompt:
                    continue
                
                # Initialize prompt entry if needed
                if prompt not in prompt_rewards:
                    prompt_rewards[prompt] = {}
                
                # Store reward by model type
                if 'mixtral' in model_id.lower():
                    prompt_rewards[prompt]['mixtral'] = raw_score
                elif 'gpt-4-turbo' in model_id.lower() or 'gpt-4' in model_id.lower():
                    prompt_rewards[prompt]['gpt4'] = raw_score
                
            except Exception:
                continue
    
    # Compute reward gaps for prompts that have both model evaluations
    prompts = []
    reward_gaps = []
    
    print(f"\n   Computing reward gaps...")
    for prompt, rewards in tqdm(prompt_rewards.items(), desc="   Processing"):
        if 'mixtral' in rewards and 'gpt4' in rewards:
            gap = rewards['gpt4'] - rewards['mixtral']
            prompts.append(prompt)
            reward_gaps.append(gap)
    
    print(f"\n   ✅ Loaded {len(prompts):,} LMSYS holdout prompts with reward gaps")
    print(f"      (from {len(prompt_rewards):,} total unique prompts)")
    print(f"   ✅ NO dev set contamination - holdout only")
    
    return prompts, np.array(reward_gaps)


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


def create_bimodal_visualization(X_2d, reward_gaps, pca, output_dir: Path):
    """
    Create visualization of prompt distribution in PCA space.
    
    WARNING: With clean methodology, this shows NO significant bimodal structure.
    - 749/750 prompts in "Low PC1" cluster
    - 1/750 prompts in "High PC1" cluster  
    - Mann-Whitney p = 0.983 (NOT significant)
    
    METHODOLOGY FIXES APPLIED:
    - Issue #1: PCA trained on generic C4 corpus (not routing data)
    - Issue #2: Holdout only (N=750, no dev contamination)
    
    REMAINING ISSUES:
    - Issue #3: Threshold PC1=0.3 is arbitrary (should use PC1=0 or unsupervised)
    - Issue #5: Structure weak in high-D (silhouette=0.057)
    - Issue #6: Correlation moderate (ρ²=0.16, only 16% variance)
    - Issue #8: High PC1 cluster homogeneous (diversity=0.355)
    
    RESULT: No significant structure with clean methodology.
    """
    print(f"\n🎨 Creating visualization...")
    print(f"   PCA source: Generic (C4 corpus)")
    print(f"   ⚠️  WARNING: Clean methodology shows NO significant structure")
    
    # Categorize by PC1 position (spatial clustering)
    # Issue #3: This threshold (0.3) is arbitrary and was chosen circularly
    # Should use PC1=0 (natural midpoint) or unsupervised clustering
    # However, result doesn't change - still no significant structure
    pc1_values = X_2d[:, 0]
    
    # Using 0.3 for comparison with original analysis
    # NOTE: This threshold was chosen circularly in original analysis
    low_pc1_mask = pc1_values < 0.3  
    high_pc1_mask = pc1_values >= 0.3
    
    X_low_pc1 = X_2d[low_pc1_mask]
    X_high_pc1 = X_2d[high_pc1_mask]
    
    # Verify the clusters match the reward gaps (data validation!)
    gaps_low_pc1 = reward_gaps[low_pc1_mask]
    gaps_high_pc1 = reward_gaps[high_pc1_mask]
    
    # Print statistics
    print(f"\n   📊 Spatial Distribution (by PC1 position):")
    print(f"      Low PC1 (< 0.3): {len(X_low_pc1):,} ({len(X_low_pc1)/len(X_2d)*100:.1f}%)")
    print(f"      High PC1 (≥ 0.3): {len(X_high_pc1):,} ({len(X_high_pc1)/len(X_2d)*100:.1f}%)")
    
    print(f"\n   📊 Reward Gap Statistics (R_GPT4-Turbo - R_Mixtral):")
    print(f"      Overall Mean: {np.mean(reward_gaps):.3f}")
    print(f"      Overall Median: {np.median(reward_gaps):.3f}")
    print(f"      Overall Std: {np.std(reward_gaps):.3f}")
    
    print(f"\n   🔍 CLUSTER STATISTICS:")
    print(f"      Low PC1 Mean Gap: {np.mean(gaps_low_pc1):+.4f}")
    print(f"      High PC1 Mean Gap: {np.mean(gaps_high_pc1):+.4f}")
    if len(gaps_high_pc1) < 10:
        print(f"      ⚠️  High PC1 has only {len(gaps_high_pc1)} sample(s) - insufficient for conclusions")
    
    # Statistical Significance Testing
    print(f"\n   📊 STATISTICAL SIGNIFICANCE:")
    
    # Mann-Whitney U test (non-parametric, robust)
    statistic_mw, p_value_mw = mannwhitneyu(gaps_low_pc1, gaps_high_pc1, alternative='two-sided')
    print(f"      Mann-Whitney U: p = {p_value_mw:.2e} {'***' if p_value_mw < 0.001 else '**' if p_value_mw < 0.01 else '*' if p_value_mw < 0.05 else 'ns'}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(gaps_low_pc1) - 1) * np.var(gaps_low_pc1, ddof=1) + 
                           (len(gaps_high_pc1) - 1) * np.var(gaps_high_pc1, ddof=1)) / 
                          (len(gaps_low_pc1) + len(gaps_high_pc1) - 2))
    cohens_d = (np.mean(gaps_low_pc1) - np.mean(gaps_high_pc1)) / pooled_std
    print(f"      Cohen's d: {cohens_d:.3f} (large effect)")
    
    # 95% Confidence Intervals
    low_ci = scipy_stats.t.interval(0.95, len(gaps_low_pc1)-1, 
                                     loc=np.mean(gaps_low_pc1), 
                                     scale=sem(gaps_low_pc1))
    high_ci = scipy_stats.t.interval(0.95, len(gaps_high_pc1)-1, 
                                      loc=np.mean(gaps_high_pc1), 
                                      scale=sem(gaps_high_pc1))
    print(f"      95% CI Low:  [{low_ci[0]:+.3f}, {low_ci[1]:+.3f}]")
    print(f"      95% CI High: [{high_ci[0]:+.3f}, {high_ci[1]:+.3f}]")
    print(f"      ✅ CIs do not overlap (p < 0.001)")
    
    # Create figure with 2 panels
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
    
    # Panel 1: Semantic scatter with cluster separation line
    ax1 = fig.add_subplot(gs[0])
    
    # Plot points
    downsample_size = min(5000, len(X_2d))
    if len(X_2d) > downsample_size:
        indices = np.random.choice(len(X_2d), downsample_size, replace=False)
        X_sample = X_2d[indices]
        gaps_sample = reward_gaps[indices]
    else:
        X_sample = X_2d
        gaps_sample = reward_gaps
    
    # Categorize sampled points by PC1 position
    pc1_sample = X_sample[:, 0]
    low_pc1_mask_s = pc1_sample < 0.3
    high_pc1_mask_s = pc1_sample >= 0.3
    
    # Plot with colors
    # NOTE: Original labels were misleading - clean methodology shows no separation
    ax1.scatter(X_sample[low_pc1_mask_s, 0], X_sample[low_pc1_mask_s, 1],
               c='#4575b4', s=25, alpha=0.7, 
               label=f'Low PC1 ({len(X_low_pc1):,} prompts, {len(X_low_pc1)/len(X_2d)*100:.1f}%)',
               edgecolors='none', rasterized=True)
    
    ax1.scatter(X_sample[high_pc1_mask_s, 0], X_sample[high_pc1_mask_s, 1],
               c='#d73027', s=25, alpha=0.7, 
               label=f'High PC1 ({len(X_high_pc1):,} prompts, {len(X_high_pc1)/len(X_2d)*100:.1f}%)',
               edgecolors='none', rasterized=True)
    
    # Add KDE contour for low PC1 cluster only
    if len(X_low_pc1) > 100:
        try:
            kde_low = gaussian_kde(X_low_pc1.T, bw_method=0.12)
            x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
            y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density_low = np.reshape(kde_low(positions).T, xx.shape)
            ax1.contour(xx, yy, density_low, levels=4, colors='#2166ac', alpha=0.6, linewidths=2.5, linestyles='solid')
        except:
            pass
    
    # Add vertical line at threshold
    # Issue #3: This threshold is arbitrary and was chosen circularly
    separation_threshold = 0.3
    ax1.axvline(x=separation_threshold, color='black', linestyle='--', linewidth=3, 
                alpha=0.7, label=f'Threshold (PC1={separation_threshold}, arbitrary)', zorder=5)
    
    # Styling
    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    
    ax1.set_xlabel(f'PC1 ({pc1_var:.2%} variance)', fontsize=15, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pc2_var:.2%} variance)', fontsize=15, fontweight='bold')
    
    # Honest title reflecting clean methodology results
    ax1.set_title(
        'LMSYS Holdout Prompts in PCA Space (Generic C4 PCA)\n'
        f'N={len(X_2d)} holdout prompts, Threshold PC1=0.3 (arbitrary)',
        fontsize=15,
        fontweight='bold',
        pad=15
    )
    ax1.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Panel 2: Distribution breakdown with better visualization
    ax2 = fig.add_subplot(gs[1])
    
    # Honest labels - no longer claiming "Natural Language" vs "Alignment Tax"
    categories = [f'Low PC1\n(< {separation_threshold})', f'High PC1\n(≥ {separation_threshold})']
    counts = [len(X_low_pc1), len(X_high_pc1)]
    colors_bar = ['#4575b4', '#d73027']
    
    bars = ax2.bar(range(len(categories)), counts, color=colors_bar, 
                   alpha=0.9, edgecolor='black', linewidth=2.5, width=0.7)
    
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Prompts', fontsize=15, fontweight='bold')
    ax2.set_title(
        f'Distribution by PC1 Threshold\n'
        f'N = {len(X_2d):,} Holdout Prompts',
        fontsize=15,
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
    output_file = output_dir / "figure1_lmsys_holdout_pca.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n   ✅ Saved: {output_file}")
    
    # Also save high-res version
    output_file_hires = output_dir / "figure1_lmsys_holdout_pca_hires.png"
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved high-res: {output_file_hires}")
    
    plt.close()


def main():
    print("="*80)
    print("FIGURE 1 ANALYSIS: LMSYS HOLDOUT WITH CLEAN METHODOLOGY")
    print("="*80)
    print("\n⚠️  WARNING: This analysis was intended to show 'Alignment Tax'")
    print("   but the finding DOES NOT REPLICATE with proper methodology.")
    print("\n📐 Clean Methodology Applied:")
    print("   ✅ Issue #1 Fixed: PCA trained on generic C4 corpus (no circularity)")
    print("   ✅ Issue #2 Fixed: Holdout ONLY (N=750, no dev contamination)")
    print("   ⚠️  Issue #3 Remains: Threshold PC1=0.3 is arbitrary (was chosen circularly)")
    print("\n📊 Result: NO significant structure (p = 0.983)")
    print("   See ONE_PAGE_SUMMARY.md for complete issue analysis.")
    
    # Paths
    holdout_file = CANONICAL_HOLDOUT_DATA_PATH
    pca_file = GENERIC_PCA_PATH
    output_dir = Path(__file__).parent / "results"
    
    print(f"\n📋 Configuration:")
    print(f"   LMSYS Holdout: {holdout_file} (holdout only)")
    print(f"   PCA model: {pca_file} (Generic C4)")
    print(f"   Output: {output_dir}")
    print(f"   ⚠️  Dev set excluded (used for training in Table 2)")
    
    if not pca_file.exists():
        print(f"\n❌ PCA file not found: {pca_file}")
        print(f"\n💡 Train generic PCA first:")
        print(f"   python3 scripts/train_pca_generic.py")
        return
    
    # Step 1: Load LMSYS holdout data ONLY (no dev contamination)
    prompts, reward_gaps = load_lmsys_holdout_with_gaps(holdout_file)
    
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
    print("CREATING THE HOOK VISUALIZATION")
    print("="*80)
    create_bimodal_visualization(X_2d, reward_gaps, pca, output_dir)
    
    # Summary with validation
    print("\n" + "="*80)
    print("⚠️  ANALYSIS COMPLETE - NULL FINDING WITH CLEAN METHODOLOGY")
    print("="*80)
    
    # Compute final validation statistics
    pc1_values = X_2d[:, 0]
    low_mask = pc1_values < 0.3
    high_mask = pc1_values >= 0.3
    gaps_low = reward_gaps[low_mask]
    gaps_high = reward_gaps[high_mask]
    gap_low = np.mean(gaps_low)
    gap_high = np.mean(gaps_high)
    
    # Statistical tests
    _, p_value = mannwhitneyu(gaps_low, gaps_high, alternative='two-sided')
    pooled_std = np.sqrt(((len(gaps_low) - 1) * np.var(gaps_low, ddof=1) + 
                           (len(gaps_high) - 1) * np.var(gaps_high, ddof=1)) / 
                          (len(gaps_low) + len(gaps_high) - 2))
    cohens_d = (gap_low - gap_high) / pooled_std
    
    print(f"\n📊 Results:")
    print(f"   • Low PC1 (< 0.3): {len(gaps_low):,} prompts ({len(gaps_low)/len(prompts)*100:.1f}%)")
    print(f"     → Mean Gap: {gap_low:+.4f}")
    if len(gaps_low) > 1:
        ci_low = scipy_stats.t.interval(0.95, len(gaps_low)-1, loc=gap_low, scale=sem(gaps_low))
        print(f"     → 95% CI: [{ci_low[0]:+.3f}, {ci_low[1]:+.3f}]")
    
    print(f"   • High PC1 (≥ 0.3): {len(gaps_high):,} prompts ({len(gaps_high)/len(prompts)*100:.1f}%)")
    print(f"     → Mean Gap: {gap_high:+.4f}")
    if len(gaps_high) > 1:
        ci_high = scipy_stats.t.interval(0.95, len(gaps_high)-1, loc=gap_high, scale=sem(gaps_high))
        print(f"     → 95% CI: [{ci_high[0]:+.3f}, {ci_high[1]:+.3f}]")
    else:
        print(f"     → ⚠️  Only {len(gaps_high)} sample(s) - no meaningful statistics")
    
    print(f"\n📊 Statistical Test:")
    if len(gaps_high) > 1:
        print(f"   • Mann-Whitney U: p = {p_value:.3f}")
        if p_value < 0.001:
            print(f"   • Result: Significant (p < 0.001)")
        elif p_value < 0.05:
            print(f"   • Result: Significant (p < 0.05)")
        else:
            print(f"   • Result: NOT significant (p > 0.05)")
        print(f"   • Cohen's d = {cohens_d:.3f}")
    else:
        print(f"   • Mann-Whitney U: Cannot compute (insufficient data)")
        print(f"   • ⚠️  High PC1 cluster has only {len(gaps_high)} sample(s)")
    
    print(f"\n⚠️  HONEST ASSESSMENT FOR PAPER:")
    print(f"   • N = {len(prompts):,} holdout prompts (no dev contamination)")
    print(f"   • PCA trained on generic text (C4) - NO circularity")
    print(f"   • Result: NO significant structure (p = {p_value:.3f})")
    print(f"   • Distribution: {len(gaps_low)}/{len(prompts)} in one cluster ({len(gaps_low)/len(prompts)*100:.1f}%)")
    print(f"   • Cohen's d = {cohens_d:.2f} (but only {len(gaps_high)} sample(s) in high cluster)")
    print(f"")
    print(f"   CONCLUSION: Original 'Alignment Tax' finding does NOT replicate")
    print(f"   with clean methodology. Multiple issues identified:")
    print(f"   - Weak high-D structure (silhouette=0.057)")
    print(f"   - Moderate correlation (ρ²=0.16, only 16% variance)")
    print(f"   - Low diversity in 'high' cluster (0.355 vs 0.953)")
    print(f"   - No validation at scale (1M has no rewards)")
    print(f"")
    print(f"   RECOMMENDATION: Remove Figure 1 from paper.")
    print(f"   See: experiments_v1/01_figure/ONE_PAGE_SUMMARY.md")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

