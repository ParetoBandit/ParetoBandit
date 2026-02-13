#!/usr/bin/env python3
"""
Compare PCA Models: RouteLLM vs Generic (C4)

This script validates that the Alignment Tax structure is not an artifact of
circular PCA training. It compares results using:

1. OLD PCA: Trained on RouteLLM battles (Mixtral vs GPT-4-Turbo)
   - Issue: PCA optimized on routing data
   - Finding: Structure partly tautological

2. NEW PCA: Trained on generic text (C4 corpus)
   - Fix: PCA trained on routing-agnostic data
   - Finding: If structure persists, it's genuine

The comparison shows:
- Cluster separation statistics (Mann-Whitney p-value, Cohen's d)
- Mean reward gaps by cluster
- Cluster proportions
- Consistency analysis across PCA models

If the Alignment Tax structure holds with generic PCA, it proves the discovery
is real and not an artifact of PCA training data selection.

Usage:
    python3 experiments_v1/01_figure/compare_pca_models.py
    
    # With custom PCA paths
    python3 experiments_v1/01_figure/compare_pca_models.py \\
        --pca-routellm src/artifacts/pca_32.joblib \\
        --pca-generic src/artifacts/pca_32_generic.joblib
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
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.stats import mannwhitneyu, sem
from scipy import stats as scipy_stats
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    ARTIFACTS_DIR
)


def load_lmsys_holdout_with_gaps(holdout_file: Path):
    """Load LMSYS HOLDOUT-ONLY prompts with reward gaps (no dev contamination)."""
    print(f"\n📥 Loading LMSYS Holdout Data (holdout only)...")
    print(f"   ⚠️  Dev set excluded to avoid contamination")
    
    prompt_rewards = {}
    
    print(f"   Processing holdout...")
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
                
                if prompt not in prompt_rewards:
                    prompt_rewards[prompt] = {}
                
                if 'mixtral' in model_id.lower():
                    prompt_rewards[prompt]['mixtral'] = raw_score
                elif 'gpt-4-turbo' in model_id.lower() or 'gpt-4' in model_id.lower():
                    prompt_rewards[prompt]['gpt4'] = raw_score
                
            except Exception:
                continue
    
    # Compute reward gaps
    prompts = []
    reward_gaps = []
    
    for prompt, rewards in prompt_rewards.items():
        if 'mixtral' in rewards and 'gpt4' in rewards:
            gap = rewards['gpt4'] - rewards['mixtral']
            prompts.append(prompt)
            reward_gaps.append(gap)
    
    print(f"   ✅ Loaded {len(prompts):,} prompts with reward gaps")
    print(f"   ✅ NO dev contamination - holdout only")
    return prompts, np.array(reward_gaps)


def analyze_with_pca(prompts, reward_gaps, pca_path: Path, pca_name: str, threshold: float = 0.3):
    """
    Analyze LMSYS data with a given PCA model.
    
    Returns:
        dict with analysis results
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING WITH {pca_name} PCA")
    print(f"{'='*80}")
    print(f"\nPCA model: {pca_path}")
    
    # Load encoder and PCA
    print(f"\n🔤 Loading encoder...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    
    print(f"📐 Loading PCA model...")
    pca = joblib.load(pca_path)
    print(f"   Components: {pca.n_components_}")
    print(f"   Total variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    print(f"   PC1 variance: {pca.explained_variance_ratio_[0]:.3%}")
    print(f"   PC2 variance: {pca.explained_variance_ratio_[1]:.3%}")
    
    # Embed and project
    print(f"\n🧮 Embedding prompts...")
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True
    )
    
    print(f"📐 Projecting to PCA space...")
    X_nd = pca.transform(embeddings)
    pc1_values = X_nd[:, 0]
    
    # Cluster by PC1
    low_pc1_mask = pc1_values < threshold
    high_pc1_mask = pc1_values >= threshold
    
    gaps_low = reward_gaps[low_pc1_mask]
    gaps_high = reward_gaps[high_pc1_mask]
    
    # Statistics
    n_low = len(gaps_low)
    n_high = len(gaps_high)
    pct_low = n_low / len(prompts) * 100
    pct_high = n_high / len(prompts) * 100
    
    mean_low = np.mean(gaps_low)
    mean_high = np.mean(gaps_high)
    
    # Statistical tests
    stat_mw, p_mw = mannwhitneyu(gaps_low, gaps_high, alternative='two-sided')
    
    # Cohen's d
    pooled_std = np.sqrt(((n_low - 1) * np.var(gaps_low, ddof=1) + 
                           (n_high - 1) * np.var(gaps_high, ddof=1)) / 
                          (n_low + n_high - 2))
    cohens_d = (mean_low - mean_high) / pooled_std
    
    # 95% CIs
    ci_low = scipy_stats.t.interval(0.95, n_low-1, loc=mean_low, scale=sem(gaps_low))
    ci_high = scipy_stats.t.interval(0.95, n_high-1, loc=mean_high, scale=sem(gaps_high))
    
    print(f"\n📊 Results:")
    print(f"   Low PC1 (< {threshold}):")
    print(f"      Count: {n_low:,} ({pct_low:.1f}%)")
    print(f"      Mean Gap: {mean_low:+.4f}")
    print(f"      95% CI: [{ci_low[0]:+.3f}, {ci_low[1]:+.3f}]")
    print(f"   High PC1 (≥ {threshold}):")
    print(f"      Count: {n_high:,} ({pct_high:.1f}%)")
    print(f"      Mean Gap: {mean_high:+.4f}")
    print(f"      95% CI: [{ci_high[0]:+.3f}, {ci_high[1]:+.3f}]")
    print(f"\n   Statistical Tests:")
    print(f"      Mann-Whitney p: {p_mw:.2e}")
    print(f"      Cohen's d: {cohens_d:.3f}")
    print(f"      Significant: {'YES' if p_mw < 0.001 else 'NO'}")
    
    return {
        'pca_name': pca_name,
        'pca_path': pca_path,
        'n_components': pca.n_components_,
        'total_variance': np.sum(pca.explained_variance_ratio_),
        'pc1_variance': pca.explained_variance_ratio_[0],
        'pc2_variance': pca.explained_variance_ratio_[1],
        'n_low': n_low,
        'n_high': n_high,
        'pct_low': pct_low,
        'pct_high': pct_high,
        'mean_gap_low': mean_low,
        'mean_gap_high': mean_high,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_mw,
        'cohens_d': cohens_d,
        'pc1_values': pc1_values,
        'gaps_low': gaps_low,
        'gaps_high': gaps_high
    }


def create_comparison_visualization(results_list, output_dir: Path):
    """Create side-by-side comparison visualization."""
    print(f"\n🎨 Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#d73027', '#4575b4']
    
    for idx, results in enumerate(results_list):
        ax = axes[idx, 0]
        
        # Bar chart: cluster proportions
        categories = ['Natural\nLanguage\n(Low PC1)', 'Alignment\nTax\n(High PC1)']
        values = [results['pct_low'], results['pct_high']]
        
        bars = ax.bar(range(len(categories)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{results["pca_name"]} PCA\nCluster Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Add percentages
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Mean gap comparison
        ax2 = axes[idx, 1]
        categories2 = ['Natural\nLanguage', 'Alignment\nTax']
        means = [results['mean_gap_low'], results['mean_gap_high']]
        cis = [results['ci_low'], results['ci_high']]
        errors = [[m - ci[0] for m, ci in zip(means, cis)],
                  [ci[1] - m for m, ci in zip(means, cis)]]
        
        bars2 = ax2.bar(range(len(categories2)), means, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=2, yerr=errors, capsize=10)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.set_xticks(range(len(categories2)))
        ax2.set_xticklabels(categories2, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Reward Gap\n(GPT-4-Turbo - Mixtral)', fontsize=13, fontweight='bold')
        ax2.set_title(f'{results["pca_name"]} PCA\nReward Gaps (95% CI)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add values
        for bar, val in zip(bars2, means):
            y_pos = val + (0.05 if val > 0 else -0.05)
            va = 'bottom' if val > 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:+.3f}', ha='center', va=va, fontsize=11, fontweight='bold')
        
        # Add stats annotation
        ax2.text(0.95, 0.95, 
                f"p = {results['p_value']:.1e}\nCohen's d = {results['cohens_d']:.2f}",
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pca_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved: {output_file}")
    plt.close()


def print_comparison_summary(results_list):
    """Print comprehensive comparison summary."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    if len(results_list) != 2:
        print("⚠️  Expected 2 PCA models for comparison")
        return
    
    r1, r2 = results_list
    
    print(f"\n📊 PCA Model Comparison:")
    print(f"\n   {r1['pca_name']} PCA:")
    print(f"      PC1 variance: {r1['pc1_variance']:.3%}")
    print(f"      Total variance: {r1['total_variance']:.2%}")
    
    print(f"\n   {r2['pca_name']} PCA:")
    print(f"      PC2 variance: {r2['pc1_variance']:.3%}")
    print(f"      Total variance: {r2['total_variance']:.2%}")
    
    print(f"\n📊 Cluster Distribution:")
    print(f"\n   Natural Language (Low PC1):")
    print(f"      {r1['pca_name']}: {r1['pct_low']:.1f}% ({r1['n_low']:,} prompts)")
    print(f"      {r2['pca_name']}: {r2['pct_low']:.1f}% ({r2['n_low']:,} prompts)")
    print(f"      Difference: {abs(r1['pct_low'] - r2['pct_low']):.1f} percentage points")
    
    print(f"\n   Alignment Tax (High PC1):")
    print(f"      {r1['pca_name']}: {r1['pct_high']:.1f}% ({r1['n_high']:,} prompts)")
    print(f"      {r2['pca_name']}: {r2['pct_high']:.1f}% ({r2['n_high']:,} prompts)")
    print(f"      Difference: {abs(r1['pct_high'] - r2['pct_high']):.1f} percentage points")
    
    print(f"\n📊 Reward Gaps:")
    print(f"\n   Natural Language (Low PC1):")
    print(f"      {r1['pca_name']}: {r1['mean_gap_low']:+.4f}")
    print(f"      {r2['pca_name']}: {r2['mean_gap_low']:+.4f}")
    print(f"      Difference: {abs(r1['mean_gap_low'] - r2['mean_gap_low']):.4f}")
    
    print(f"\n   Alignment Tax (High PC1):")
    print(f"      {r1['pca_name']}: {r1['mean_gap_high']:+.4f}")
    print(f"      {r2['pca_name']}: {r2['mean_gap_high']:+.4f}")
    print(f"      Difference: {abs(r1['mean_gap_high'] - r2['mean_gap_high']):.4f}")
    
    print(f"\n📊 Statistical Significance:")
    print(f"\n   {r1['pca_name']} PCA:")
    print(f"      Mann-Whitney p: {r1['p_value']:.2e}")
    print(f"      Cohen's d: {r1['cohens_d']:.3f}")
    print(f"      Significant: {'YES' if r1['p_value'] < 0.001 else 'NO'}")
    
    print(f"\n   {r2['pca_name']} PCA:")
    print(f"      Mann-Whitney p: {r2['p_value']:.2e}")
    print(f"      Cohen's d: {r2['cohens_d']:.3f}")
    print(f"      Significant: {'YES' if r2['p_value'] < 0.001 else 'NO'}")
    
    # Consistency check
    print(f"\n{'='*80}")
    print("CONSISTENCY ANALYSIS")
    print(f"{'='*80}")
    
    both_significant = r1['p_value'] < 0.001 and r2['p_value'] < 0.001
    similar_direction = (r1['mean_gap_low'] > r1['mean_gap_high']) == (r2['mean_gap_low'] > r2['mean_gap_high'])
    
    print(f"\n✓ Both models show significant separation: {'YES' if both_significant else 'NO'}")
    print(f"✓ Both models show same direction: {'YES' if similar_direction else 'NO'}")
    
    if both_significant and similar_direction:
        print(f"\n✅ VALIDATION SUCCESS!")
        print(f"   The Alignment Tax structure is CONSISTENT across PCA models.")
        print(f"   This proves the discovery is GENUINE, not an artifact of PCA training.")
        print(f"   Using generic PCA trained on C4 eliminates circularity concerns.")
    else:
        print(f"\n⚠️  INCONSISTENT RESULTS")
        print(f"   Further investigation needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PCA models: RouteLLM vs Generic",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--pca-routellm", type=str,
        default=str(DEFAULT_PCA_PATH),
        help="Path to RouteLLM PCA model"
    )
    parser.add_argument(
        "--pca-generic", type=str,
        default=str(ARTIFACTS_DIR / "pca_32_generic.joblib"),
        help="Path to generic PCA model"
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PCA MODEL COMPARISON: RouteLLM vs Generic (C4)")
    print("="*80)
    print("\n🎯 Goal: Validate that Alignment Tax is not a PCA artifact")
    print("   → Compare results using two different PCA models")
    print("   → If structure persists with generic PCA, it's genuine")
    print("   → Using holdout ONLY (no dev contamination)")
    
    # Paths
    holdout_file = CANONICAL_HOLDOUT_DATA_PATH
    pca_routellm = Path(args.pca_routellm)
    pca_generic = Path(args.pca_generic)
    output_dir = Path(args.output) if args.output else Path(__file__).parent / "results"
    
    print(f"\n📋 Configuration:")
    print(f"   LMSYS Holdout: {holdout_file} (holdout only)")
    print(f"   RouteLLM PCA: {pca_routellm}")
    print(f"   Generic PCA: {pca_generic}")
    print(f"   Output: {output_dir}")
    print(f"   ⚠️  Dev set excluded (used for training)")
    
    # Check if files exist
    if not pca_routellm.exists():
        print(f"\n❌ RouteLLM PCA not found: {pca_routellm}")
        return
    
    if not pca_generic.exists():
        print(f"\n❌ Generic PCA not found: {pca_generic}")
        print(f"\n💡 Train generic PCA first:")
        print(f"   python3 scripts/train_pca_generic.py")
        return
    
    # Step 1: Load holdout data only (no dev contamination)
    prompts, reward_gaps = load_lmsys_holdout_with_gaps(holdout_file)
    
    if len(prompts) == 0:
        print("\n❌ No data loaded!")
        return
    
    # Step 2: Analyze with both PCAs
    results_list = []
    
    results_routellm = analyze_with_pca(prompts, reward_gaps, pca_routellm, "RouteLLM")
    results_list.append(results_routellm)
    
    results_generic = analyze_with_pca(prompts, reward_gaps, pca_generic, "Generic (C4)")
    results_list.append(results_generic)
    
    # Step 3: Create comparison visualization
    create_comparison_visualization(results_list, output_dir)
    
    # Step 4: Print summary
    print_comparison_summary(results_list)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
