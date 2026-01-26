#!/usr/bin/env python3
"""
Feature Distribution Shift Analysis (Figure 1.2)

This script analyzes covariate shift between Source/Prior data and RouteLLM data
by comparing the distribution of the first principal component (PCA_0).

The analysis:
1. Loads prompts from both Source data (dev/holdout) and RouteLLM battles
2. Embeds prompts and projects them onto the first principal component
3. Creates 1D density plots to visualize distribution shift
4. Computes Population Stability Index (PSI) to quantify the shift
5. Shows whether RouteLLM data is shifted toward "Easy" or "Hard" clusters

The Proof: If the RouteLLM density is shifted toward the "Easy" cluster 
compared to the Prior data, we have visual proof of Covariate Shift.

Usage:
    python3 experiments_v1/01.5_figure/plot_distribution_shift.py
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
    DEFAULT_PCA_PATH,
    ROUTELLM_BATTLES_REWARDS_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)


def load_source_prompts_from_datasets(dev_file: Path, holdout_file: Path, max_samples: int = 10000):
    """
    Load prompts from dev/holdout datasets (without requiring reward gaps).
    
    This represents the "Source/Prior" distribution from training data.
    We load prompts only; reward gaps not available in this format.
    
    Args:
        dev_file: Path to dev data file
        holdout_file: Path to holdout data file
        max_samples: Maximum prompts to load per file
    
    Returns:
        prompts: List of prompt strings
    """
    print(f"📥 Loading Source/Prior prompts (dev + holdout)...")
    print(f"   Dev: {dev_file}")
    print(f"   Holdout: {holdout_file}")
    
    prompts = []
    
    for file_path, name in [(dev_file, "dev"), (holdout_file, "holdout")]:
        if not file_path.exists():
            print(f"   ⚠️  File not found: {file_path}")
            continue
        
        # Handle both gzipped and plain JSONL
        opener = gzip.open if file_path.suffix == '.gz' else open
        
        with opener(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"   Reading {name}", total=max_samples)):
                if i >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    prompt = data.get('prompt', '')
                    
                    # Handle list-formatted prompts
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else ""
                    
                    prompt = prompt.strip()
                    if prompt:
                        prompts.append(prompt)
                        
                except Exception as e:
                    continue
    
    print(f"   ✅ Loaded {len(prompts):,} source prompts")
    return prompts


def load_routellm_prompts(battles_file: Path, start_idx: int = 0, max_samples: int = 10000):
    """
    Load prompts from RouteLLM battles dataset.
    
    This represents the "target" distribution we're deploying on.
    
    Args:
        battles_file: Path to battles JSONL file
        start_idx: Starting index to read from
        max_samples: Maximum prompts to load
    
    Returns:
        prompts: List of prompt strings
        reward_gaps: Array of R_Turbo - R_Mixtral gaps (for clustering info)
    """
    print(f"\n📥 Loading RouteLLM deployment prompts...")
    print(f"   File: {battles_file}")
    print(f"   Range: {start_idx:,} to {start_idx + max_samples:,}")
    
    prompts = []
    reward_gaps = []
    
    with open(battles_file, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="   Reading", total=start_idx + max_samples)):
            # Skip lines until we reach start_idx
            if i < start_idx:
                continue
            
            # Stop after we've collected max_samples
            if i >= start_idx + max_samples:
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
                
                # Get reward gap (for difficulty annotation)
                model_a = battle['model_a']
                model_b = battle['model_b']
                reward_a = battle['reward_a']
                reward_b = battle['reward_b']
                
                # Compute reward gap (GPT-4-Turbo - Mixtral)
                if 'gpt-4-turbo' in model_a.lower():
                    reward_turbo = reward_a
                    reward_mixtral = reward_b
                else:
                    reward_turbo = reward_b
                    reward_mixtral = reward_a
                
                gap = reward_turbo - reward_mixtral
                
                prompts.append(prompt)
                reward_gaps.append(gap)
                
            except Exception as e:
                continue
    
    print(f"   ✅ Loaded {len(prompts):,} RouteLLM prompts")
    
    return prompts, np.array(reward_gaps)


def project_to_pc1(prompts: list, pca_file: Path, batch_size: int = 64):
    """
    Embed prompts and project onto first principal component (PC1).
    
    Args:
        prompts: List of prompt strings
        pca_file: Path to pre-trained PCA model
        batch_size: Batch size for embedding
    
    Returns:
        pc1_values: Array of PC1 coordinates (1D)
    """
    print(f"\n🔤 Loading sentence encoder...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print(f"   ✅ Encoder loaded")
    
    print(f"\n📐 Loading pre-trained PCA model...")
    pca = joblib.load(pca_file)
    print(f"   ✅ Loaded PCA: {pca.n_components_} components")
    print(f"   PC1 explains: {pca.explained_variance_ratio_[0]:.3%} of variance")
    
    print(f"\n🧮 Embedding {len(prompts):,} prompts...")
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    
    print(f"\n📐 Projecting to PC1...")
    X_pca = pca.transform(embeddings)
    pc1_values = X_pca[:, 0]  # Extract first component only
    
    print(f"   ✅ PC1 projection complete")
    print(f"   PC1 range: [{np.min(pc1_values):.3f}, {np.max(pc1_values):.3f}]")
    print(f"   PC1 mean: {np.mean(pc1_values):.3f}")
    print(f"   PC1 std: {np.std(pc1_values):.3f}")
    
    return pc1_values


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10):
    """
    Compute Population Stability Index (PSI) to quantify distribution shift.
    
    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    
    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 ≤ PSI < 0.2: Moderate change
    - PSI ≥ 0.2: Significant change (may need model retraining)
    
    Args:
        expected: Distribution from source/prior data
        actual: Distribution from RouteLLM/target data
        n_bins: Number of bins for histogram
    
    Returns:
        psi: PSI value
        bins: Bin edges
        expected_percents: Percentage in each bin for expected
        actual_percents: Percentage in each bin for actual
    """
    print(f"\n📊 Computing Population Stability Index (PSI)...")
    
    # Determine bin edges based on expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    # Compute histograms
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Convert to percentages (avoid zeros by adding small epsilon)
    epsilon = 1e-6
    expected_percents = (expected_counts + epsilon) / (len(expected) + epsilon * n_bins)
    actual_percents = (actual_counts + epsilon) / (len(actual) + epsilon * n_bins)
    
    # Compute PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    print(f"   ✅ PSI = {psi:.4f}")
    
    if psi < 0.1:
        print(f"   ✅ No significant distribution shift")
    elif psi < 0.2:
        print(f"   ⚠️  Moderate distribution shift detected")
    else:
        print(f"   🚨 Significant distribution shift! Consider retraining.")
    
    return psi, bins, expected_percents, actual_percents


def create_visualization(pc1_source, pc1_routellm, reward_gaps_routellm, output_dir: Path):
    """
    Create compelling visualization comparing Source vs RouteLLM distributions on PC1.
    
    Shows overall shift in top plot, then explains Source bimodal structure in bottom plot.
    
    Args:
        pc1_source: PC1 values for source/prior data
        pc1_routellm: PC1 values for RouteLLM data
        reward_gaps_routellm: Reward gaps for RouteLLM prompts (for difficulty annotation)
        output_dir: Directory to save figure
    """
    print(f"\n🎨 Creating compelling distribution shift visualization...")
    
    # Compute PSI
    psi, bins, expected_percents, actual_percents = compute_psi(
        pc1_source, pc1_routellm, n_bins=20
    )
    
    # We need to infer Source difficulty by using RouteLLM as a proxy
    # Create a simple heuristic: PC1 > 0.2 → Hard, PC1 < 0.0 → Easy (based on RouteLLM clustering)
    # This explains the bimodal structure in Source data
    source_easy_mask = pc1_source < 0.0
    source_hard_mask = pc1_source > 0.2
    
    pc1_source_easy = pc1_source[source_easy_mask]
    pc1_source_hard = pc1_source[source_hard_mask]
    
    source_easy_pct = len(pc1_source_easy) / len(pc1_source) * 100
    source_hard_pct = len(pc1_source_hard) / len(pc1_source) * 100
    
    print(f"\n   📊 Source Difficulty Distribution (inferred from PC1):")
    print(f"      Easy (PC1 < 0.0): {len(pc1_source_easy):,} ({source_easy_pct:.1f}%)")
    print(f"      Hard (PC1 > 0.2): {len(pc1_source_hard):,} ({source_hard_pct:.1f}%)")
    
    # Also separate ROUTELLM data by difficulty for comparison
    easy_mask_routellm = reward_gaps_routellm <= 0.3
    hard_mask_routellm = reward_gaps_routellm > 0.6
    
    pc1_routellm_easy = pc1_routellm[easy_mask_routellm]
    pc1_routellm_hard = pc1_routellm[hard_mask_routellm]
    
    routellm_easy_pct = len(pc1_routellm_easy) / len(pc1_routellm) * 100
    routellm_hard_pct = len(pc1_routellm_hard) / len(pc1_routellm) * 100
    
    print(f"\n   📊 RouteLLM Difficulty Distribution:")
    print(f"      Easy (Gap ≤ 0.3): {len(pc1_routellm_easy):,} ({routellm_easy_pct:.1f}%)")
    print(f"      Hard (Gap > 0.6): {len(pc1_routellm_hard):,} ({routellm_hard_pct:.1f}%)")
    
    # Statistics
    print(f"\n   📊 Distribution Statistics:")
    print(f"      Source PC1: mean={np.mean(pc1_source):.3f}, std={np.std(pc1_source):.3f}")
    print(f"      RouteLLM PC1: mean={np.mean(pc1_routellm):.3f}, std={np.std(pc1_routellm):.3f}")
    print(f"      Mean shift: {np.mean(pc1_routellm) - np.mean(pc1_source):.3f}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # === Plot 1: Overall Distribution Comparison ===
    ax1 = axes[0]
    
    # Compute KDEs for smooth density curves
    print(f"   🔵 Computing KDE for Source data...")
    kde_source = gaussian_kde(pc1_source, bw_method=0.1)
    
    print(f"   🔴 Computing KDE for RouteLLM data...")
    kde_routellm = gaussian_kde(pc1_routellm, bw_method=0.1)
    
    # Create x-axis for plotting
    x_min = min(pc1_source.min(), pc1_routellm.min())
    x_max = max(pc1_source.max(), pc1_routellm.max())
    x_range = x_max - x_min
    x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
    
    # Plot density curves
    density_source = kde_source(x)
    density_routellm = kde_routellm(x)
    
    ax1.plot(x, density_source, label='Source/Prior Data', 
            color='#4575b4', linewidth=3, alpha=0.8)
    ax1.plot(x, density_routellm, label='RouteLLM Data', 
            color='#d73027', linewidth=3, alpha=0.8)
    
    # Fill areas for visual emphasis
    ax1.fill_between(x, 0, density_source, color='#4575b4', alpha=0.2)
    ax1.fill_between(x, 0, density_routellm, color='#d73027', alpha=0.2)
    
    # Add mean lines
    ax1.axvline(np.mean(pc1_source), color='#4575b4', linestyle='--', 
               linewidth=2, alpha=0.6, label=f'Source Mean: {np.mean(pc1_source):.3f}')
    ax1.axvline(np.mean(pc1_routellm), color='#d73027', linestyle='--', 
               linewidth=2, alpha=0.6, label=f'RouteLLM Mean: {np.mean(pc1_routellm):.3f}')
    
    ax1.set_xlabel('First Principal Component (PC1)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax1.set_title(
        f'Feature Distribution Shift: Source vs RouteLLM Data\n'
        f'Population Stability Index (PSI) = {psi:.4f}',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add PSI interpretation text
    psi_color = 'green' if psi < 0.1 else ('orange' if psi < 0.2 else 'red')
    psi_text = ('No Shift' if psi < 0.1 else 
                ('Moderate Shift' if psi < 0.2 else 'Significant Shift!'))
    
    ax1.text(0.02, 0.98, f'PSI Interpretation: {psi_text}',
            transform=ax1.transAxes,
            fontsize=12,
            fontweight='bold',
            color=psi_color,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=psi_color, linewidth=2))
    
    # === Plot 2: Source Data - Easy vs Hard Clustering (Explains Bimodal Structure) ===
    ax2 = axes[1]
    
    # Use different colors to avoid confusion with top plot
    # Green for Easy, Purple for Hard
    color_easy = '#1b9e77'  # Teal/green
    color_hard = '#7570b3'  # Purple
    
    # Plot SOURCE difficulty-based densities
    if len(pc1_source_easy) > 50 and len(pc1_source_hard) > 50:
        print(f"   🟢 Computing KDE for Source Easy prompts...")
        kde_source_easy = gaussian_kde(pc1_source_easy, bw_method=0.1)
        
        print(f"   🟣 Computing KDE for Source Hard prompts...")
        kde_source_hard = gaussian_kde(pc1_source_hard, bw_method=0.1)
        
        density_source_easy = kde_source_easy(x)
        density_source_hard = kde_source_hard(x)
        
        ax2.plot(x, density_source_easy, label=f'Easy (PC1 < 0.0): {source_easy_pct:.1f}%', 
                color=color_easy, linewidth=3, alpha=0.8)
        ax2.plot(x, density_source_hard, label=f'Hard (PC1 > 0.2): {source_hard_pct:.1f}%', 
                color=color_hard, linewidth=3, alpha=0.8)
        
        ax2.fill_between(x, 0, density_source_easy, color=color_easy, alpha=0.2)
        ax2.fill_between(x, 0, density_source_hard, color=color_hard, alpha=0.2)
        
        # Add mean lines
        ax2.axvline(np.mean(pc1_source_easy), color=color_easy, linestyle='--', 
                   linewidth=2, alpha=0.6)
        ax2.axvline(np.mean(pc1_source_hard), color=color_hard, linestyle='--', 
                   linewidth=2, alpha=0.6)
    
    ax2.set_xlabel('First Principal Component (PC1)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax2.set_title(
        'Source/Prior Data: Easy vs Hard Task Distribution\n'
        f'Bimodal Structure Explained by Two Distinct Task Clusters',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add interpretation text
    ax2.text(0.02, 0.98, 
            f'Source Training Distribution\n'
            f'Left Peak (Easy): Centered at {np.mean(pc1_source_easy):.3f}\n'
            f'Right Peak (Hard): Centered at {np.mean(pc1_source_hard):.3f}\n'
            f'→ Bimodal = Two Task Types',
            transform=ax2.transAxes,
            fontsize=12,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#1b9e77', linewidth=2))
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "distribution_shift_pc1.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: {output_file}")
    
    # Also save high-res version
    output_file_hires = output_dir / "distribution_shift_pc1_hires.png"
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight')
    print(f"   ✅ Saved high-res: {output_file_hires}")
    
    plt.close()
    
    # Compute mean shift for return value
    mean_shift = np.mean(pc1_routellm) - np.mean(pc1_source)
    
    return psi, mean_shift


def main():
    print("="*80)
    print("FEATURE DISTRIBUTION SHIFT ANALYSIS")
    print("="*80)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    dev_file = CANONICAL_DEV_DATA_PATH
    holdout_file = CANONICAL_HOLDOUT_DATA_PATH
    battles_file = ROUTELLM_BATTLES_REWARDS_PATH
    pca_file = DEFAULT_PCA_PATH
    output_dir = Path(__file__).parent / "results"
    
    print(f"\n📋 Configuration:")
    print(f"   Source data (dev): {dev_file}")
    print(f"   Source data (holdout): {holdout_file}")
    print(f"   RouteLLM data: {battles_file}")
    print(f"   PCA model: {pca_file}")
    print(f"   Output: {output_dir}")
    
    if not pca_file.exists():
        print(f"\n❌ PCA file not found: {pca_file}")
        print(f"   Run: python3 scripts/train_pca_from_routellm.py")
        return
    
    # Step 1: Load Source/Prior data (dev + holdout)
    source_prompts = load_source_prompts_from_datasets(
        dev_file, holdout_file, max_samples=10000
    )
    
    if len(source_prompts) == 0:
        print("\n❌ No source prompts loaded!")
        return
    
    # Step 2: Load RouteLLM deployment data
    routellm_prompts, reward_gaps_routellm = load_routellm_prompts(
        battles_file, start_idx=0, max_samples=10000
    )
    
    if len(routellm_prompts) == 0:
        print("\n❌ No RouteLLM prompts loaded!")
        return
    
    # Step 3: Project to PC1
    print("\n" + "="*80)
    print("PROJECTING SOURCE DATA TO PC1")
    print("="*80)
    pc1_source = project_to_pc1(source_prompts, pca_file, batch_size=64)
    
    print("\n" + "="*80)
    print("PROJECTING ROUTELLM DATA TO PC1")
    print("="*80)
    pc1_routellm = project_to_pc1(routellm_prompts, pca_file, batch_size=64)
    
    # Step 4: Visualize
    psi, mean_shift = create_visualization(pc1_source, pc1_routellm, reward_gaps_routellm, output_dir)
    
    # Step 5: Summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Distribution Shift Summary:")
    print(f"   Source prompts: {len(source_prompts):,}")
    print(f"   RouteLLM prompts: {len(routellm_prompts):,}")
    print(f"   PSI: {psi:.4f}")
    print(f"   Mean shift: {mean_shift:.4f}")
    
    print(f"\n🔍 Interpretation:")
    if psi < 0.1:
        print(f"   ✅ No significant covariate shift detected")
        print(f"   → Source and RouteLLM distributions are similar")
    elif psi < 0.2:
        print(f"   ⚠️  Moderate covariate shift detected")
        print(f"   → Some distribution differences, monitor performance")
    else:
        print(f"   🚨 Significant covariate shift detected!")
        print(f"   → Source and RouteLLM distributions differ substantially")
        print(f"   → Consider: domain adaptation, retraining, or transfer learning")
    
    if abs(mean_shift) > 0.1:
        shift_dir = "Easy" if mean_shift < 0 else "Hard"
        print(f"\n   📍 Semantic Shift: RouteLLM data shifted toward {shift_dir} cluster")
        if mean_shift < 0:
            print(f"      → More easy prompts in RouteLLM vs Source")
            print(f"      → Mixtral may be more cost-effective for this distribution")
        else:
            print(f"      → More hard prompts in RouteLLM vs Source")
            print(f"      → GPT-4 usage may be higher than expected")
    
    print(f"\n   Output: {output_dir}/distribution_shift_pc1.png")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

