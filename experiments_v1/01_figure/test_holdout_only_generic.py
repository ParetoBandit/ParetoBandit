#!/usr/bin/env python3
"""
Final Validation: Test with Generic C4 PCA (100K samples)

This is the KEY test to determine if Issue #1 (PCA circularity) matters.

We now know:
- Routing PCA + holdout-only + unsupervised → p<0.0001, d=1.53 (REAL)
- Generic C4 PCA (10K) + holdout-only → p=0.983 (FAIL - undertrained)

This test:
- Generic C4 PCA (100K samples, properly trained)
- Holdout only (N=750, no dev contamination)
- Unsupervised threshold (k-means or silhouette)

If significant:
  → Finding is FULLY VALIDATED (routing and generic both work)
  → Keep Figure 1 with all fixes
  → Scientific integrity confirmed

If NOT significant:
  → Issue #1 (circularity) is critical
  → Routing PCA finding is tautological
  → Need to decide: report with routing PCA + caveats, or remove

Usage:
    python3 experiments_v1/01_figure/test_holdout_only_generic.py
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
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu, sem
from scipy import stats as scipy_stats
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    CANONICAL_HOLDOUT_DATA_PATH,
    ARTIFACTS_DIR
)


def load_holdout_only(holdout_file: Path):
    """Load holdout data ONLY."""
    print(f"\n📥 Loading LMSYS Holdout (holdout only)...")
    
    prompt_rewards = {}
    
    with gzip.open(holdout_file, 'rt') as f:
        for line in tqdm(f, desc="   Reading"):
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '').strip()
                model_id = entry.get('model_id', '')
                raw_score = entry.get('raw_score', None)
                
                if not prompt or raw_score is None:
                    continue
                
                if prompt not in prompt_rewards:
                    prompt_rewards[prompt] = {}
                
                if 'mixtral' in model_id.lower():
                    prompt_rewards[prompt]['mixtral'] = raw_score
                elif 'gpt-4-turbo' in model_id.lower() or 'gpt-4' in model_id.lower():
                    prompt_rewards[prompt]['gpt4'] = raw_score
            except:
                continue
    
    prompts = []
    reward_gaps = []
    
    for prompt, rewards in prompt_rewards.items():
        if 'mixtral' in rewards and 'gpt4' in rewards:
            gap = rewards['gpt4'] - rewards['mixtral']
            prompts.append(prompt)
            reward_gaps.append(gap)
    
    print(f"   ✅ Loaded {len(prompts):,} holdout prompts")
    return prompts, np.array(reward_gaps)


def embed_and_project_2d(prompts, pca_path):
    """Embed and project to 2D."""
    print(f"\n🔤 Embedding prompts...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True
    )
    
    print(f"\n📐 Loading PCA: {pca_path}")
    pca = joblib.load(pca_path)
    print(f"   Components: {pca.n_components_}")
    print(f"   PC1 variance: {pca.explained_variance_ratio_[0]:.3%}")
    print(f"   PC2 variance: {pca.explained_variance_ratio_[1]:.3%}")
    
    X_pca = pca.transform(embeddings)
    X_2d = X_pca[:, :2]
    
    return X_2d, pca


def find_unsupervised_threshold(X_2d, method='kmeans'):
    """Find threshold using unsupervised methods."""
    pc1 = X_2d[:, 0]
    
    if method == 'kmeans':
        print(f"\n📐 Finding threshold with k-means (k=2, unsupervised)...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_2d)
        
        cluster_0_pc1 = pc1[labels == 0]
        cluster_1_pc1 = pc1[labels == 1]
        
        mean_0 = np.mean(cluster_0_pc1)
        mean_1 = np.mean(cluster_1_pc1)
        threshold = (mean_0 + mean_1) / 2
        
        print(f"   Cluster 0 mean PC1: {mean_0:.3f}")
        print(f"   Cluster 1 mean PC1: {mean_1:.3f}")
        print(f"   Threshold (midpoint): {threshold:.3f}")
        
        sil_score = silhouette_score(X_2d, labels)
        print(f"   Silhouette score: {sil_score:.3f}")
        
        return threshold, {'method': 'kmeans', 'silhouette': sil_score}
    
    elif method == 'silhouette':
        print(f"\n📐 Finding silhouette-optimal threshold (unsupervised)...")
        
        thresholds = np.linspace(pc1.min() + 0.05, pc1.max() - 0.05, 50)
        best_threshold = None
        best_silhouette = -1
        
        for threshold in tqdm(thresholds, desc="   Testing"):
            labels = (pc1 >= threshold).astype(int)
            
            if len(np.unique(labels)) < 2:
                continue
            
            smaller_cluster_size = min(np.sum(labels == 0), np.sum(labels == 1))
            if smaller_cluster_size < len(pc1) * 0.05:
                continue
            
            try:
                sil = silhouette_score(X_2d, labels)
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_threshold = threshold
            except:
                continue
        
        print(f"   Best threshold: {best_threshold:.3f}")
        print(f"   Best silhouette: {best_silhouette:.3f}")
        
        return best_threshold, {'method': 'silhouette', 'silhouette': best_silhouette}


def test_cluster_separation(X_2d, reward_gaps, threshold):
    """Test if unsupervised clusters have different reward gaps."""
    pc1 = X_2d[:, 0]
    
    low_mask = pc1 < threshold
    high_mask = pc1 >= threshold
    
    gaps_low = reward_gaps[low_mask]
    gaps_high = reward_gaps[high_mask]
    
    n_low = len(gaps_low)
    n_high = len(gaps_high)
    
    print(f"\n📊 Cluster Statistics:")
    print(f"   Low PC1 (< {threshold:.3f}): {n_low:,} prompts ({n_low/len(reward_gaps)*100:.1f}%)")
    print(f"   High PC1 (≥ {threshold:.3f}): {n_high:,} prompts ({n_high/len(reward_gaps)*100:.1f}%)")
    
    if n_high < 10:
        print(f"\n   ⚠️  WARNING: High cluster has only {n_high} samples")
        return None
    
    mean_low = np.mean(gaps_low)
    mean_high = np.mean(gaps_high)
    
    print(f"\n📊 Reward Gaps:")
    print(f"   Low PC1 mean: {mean_low:+.4f}")
    print(f"   High PC1 mean: {mean_high:+.4f}")
    print(f"   Difference: {abs(mean_low - mean_high):.4f}")
    
    stat, p_value = mannwhitneyu(gaps_low, gaps_high, alternative='two-sided')
    
    pooled_std = np.sqrt(((n_low - 1) * np.var(gaps_low, ddof=1) + 
                           (n_high - 1) * np.var(gaps_high, ddof=1)) / 
                          (n_low + n_high - 2))
    cohens_d = (mean_low - mean_high) / pooled_std
    
    ci_low = scipy_stats.t.interval(0.95, n_low-1, loc=mean_low, scale=sem(gaps_low))
    ci_high = scipy_stats.t.interval(0.95, n_high-1, loc=mean_high, scale=sem(gaps_high))
    
    print(f"\n📊 Statistical Tests:")
    print(f"   Mann-Whitney p: {p_value:.4f}")
    print(f"   Cohen's d: {cohens_d:.3f}")
    print(f"   95% CI Low: [{ci_low[0]:+.3f}, {ci_low[1]:+.3f}]")
    print(f"   95% CI High: [{ci_high[0]:+.3f}, {ci_high[1]:+.3f}]")
    
    print(f"\n📊 Interpretation:")
    if p_value < 0.001:
        sig_str = "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig_str = "Significant (p < 0.01)"
    elif p_value < 0.05:
        sig_str = "Significant (p < 0.05)"
    else:
        sig_str = "NOT significant (p ≥ 0.05)"
    
    print(f"   Significance: {sig_str}")
    
    if abs(cohens_d) < 0.2:
        effect_str = "Negligible effect"
    elif abs(cohens_d) < 0.5:
        effect_str = "Small effect"
    elif abs(cohens_d) < 0.8:
        effect_str = "Medium effect"
    else:
        effect_str = "Large effect"
    
    print(f"   Effect size: {effect_str} (|d| = {abs(cohens_d):.2f})")
    
    return {
        'n_low': n_low,
        'n_high': n_high,
        'mean_low': mean_low,
        'mean_high': mean_high,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant': p_value < 0.05,
        'meaningful_effect': abs(cohens_d) >= 0.5
    }


def main():
    print("="*80)
    print("FINAL VALIDATION: GENERIC C4 PCA (100K) + HOLDOUT-ONLY")
    print("="*80)
    
    print("\n🎯 Goal: Test if finding persists with generic PCA")
    print("\n📐 Test Setup:")
    print("   ✅ Fix Issue #1: Generic C4 PCA (100K samples, not routing PCA)")
    print("   ✅ Fix Issue #2: Holdout only (N=750, no dev)")
    print("   ✅ Fix Issue #3: Unsupervised threshold")
    print("\n🔍 Why This Matters:")
    print("   If significant → Finding is FULLY VALIDATED")
    print("   If NOT → Issue #1 (circularity) is critical")
    
    # Paths
    holdout_file = CANONICAL_HOLDOUT_DATA_PATH
    pca_file = ARTIFACTS_DIR / "pca_32_generic.joblib"
    
    print(f"\n📋 Configuration:")
    print(f"   Data: {holdout_file} (holdout only)")
    print(f"   PCA: {pca_file} (GENERIC C4, 100K samples)")
    print(f"   Threshold: Unsupervised (k-means)")
    
    if not pca_file.exists():
        print(f"\n❌ PCA file not found: {pca_file}")
        print(f"\nTrain it first:")
        print(f"   python3 scripts/train_pca_generic.py --max-samples 100000")
        return
    
    # Load data
    prompts, reward_gaps = load_holdout_only(holdout_file)
    
    if len(prompts) < 100:
        print(f"\n❌ Insufficient data: {len(prompts)} prompts")
        return
    
    # Embed and project
    X_2d, pca = embed_and_project_2d(prompts, pca_file)
    
    # Find thresholds
    print("\n" + "="*80)
    print("UNSUPERVISED THRESHOLD SELECTION")
    print("="*80)
    
    threshold_kmeans, info_kmeans = find_unsupervised_threshold(X_2d, method='kmeans')
    threshold_sil, info_sil = find_unsupervised_threshold(X_2d, method='silhouette')
    
    print(f"\n📊 Unsupervised Threshold Results:")
    print(f"   K-means: {threshold_kmeans:.3f} (silhouette: {info_kmeans['silhouette']:.3f})")
    print(f"   Silhouette-optimal: {threshold_sil:.3f} (silhouette: {info_sil['silhouette']:.3f})")
    
    # Test separation
    print("\n" + "="*80)
    print("TESTING CLUSTER SEPARATION")
    print("="*80)
    
    results = {}
    
    for threshold, method_name in [(threshold_kmeans, 'K-means'), (threshold_sil, 'Silhouette-optimal')]:
        print(f"\n{'='*80}")
        print(f"Method: {method_name} (threshold = {threshold:.3f})")
        print(f"{'='*80}")
        
        result = test_cluster_separation(X_2d, reward_gaps, threshold)
        results[method_name] = result
    
    # Summary
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    print(f"\n📊 Summary of Results:")
    print(f"\n{'Method':<20} {'Threshold':<12} {'p-value':<12} {'Cohen\'s d':<12} {'Significant?':<15} {'Effect?'}")
    print("-" * 90)
    
    for method_name, result in results.items():
        if result is None:
            print(f"{method_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'Insufficient data':<15}")
        else:
            threshold_val = threshold_kmeans if method_name == 'K-means' else threshold_sil
            sig_str = 'Yes' if result['significant'] else 'No'
            effect_str = 'Yes (|d|≥0.5)' if result['meaningful_effect'] else 'No (|d|<0.5)'
            print(f"{method_name:<20} {threshold_val:<12.3f} {result['p_value']:<12.4f} {result['cohens_d']:<12.3f} {sig_str:<15} {effect_str}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    any_significant = any(r and r['significant'] and r['meaningful_effect'] 
                         for r in results.values() if r is not None)
    
    if any_significant:
        print("\n✅ FINDING FULLY VALIDATED")
        print("\n   Generic C4 PCA (100K) + holdout-only + unsupervised → SIGNIFICANT")
        print("\n   Interpretation:")
        print("   • Finding is REAL and ROBUST")
        print("   • Works with BOTH routing PCA AND generic C4 PCA")
        print("   • Issue #1 (circularity) is NOT critical")
        print("   • All three core issues (#1-3) are FIXED")
        print("\n   Recommendation: KEEP FIGURE 1")
        print("\n   Required Actions:")
        print("   1. Use generic C4 PCA (100K) for Figure 1")
        print("   2. Use holdout only (N=750)")
        print("   3. Use unsupervised threshold (k-means or silhouette)")
        print("   4. Fix presentation issues (#4-10):")
        print("      - Remove causal claims")
        print("      - Report effect sizes honestly (ρ²=0.16)")
        print("      - Acknowledge limitations (2D projection, weak high-D)")
        print("      - Remove scale extrapolations")
        print("\n   Scientific Status: VALIDATED")
    else:
        print("\n⚠️  FINDING DOES NOT REPLICATE WITH GENERIC PCA")
        print("\n   Generic C4 PCA → NOT significant")
        print("   Routing PCA → Significant (p<0.0001, d=1.53)")
        print("\n   Interpretation:")
        print("   • Issue #1 (PCA circularity) IS CRITICAL")
        print("   • Routing PCA finding is partly tautological")
        print("   • Generic PCA doesn't capture the structure")
        print("\n   Recommendation: REMOVE FIGURE 1 or REPORT WITH CAVEATS")
        print("\n   Options:")
        print("   A. Remove Figure 1 entirely (cleanest)")
        print("   B. Report with routing PCA + strong caveats:")
        print("      - 'PCA trained on routing data'")
        print("      - 'Finding may be tautological'")
        print("      - 'Does not replicate with generic PCA'")
        print("   C. Investigate why generic PCA fails:")
        print("      - 100K samples sufficient?")
        print("      - C4 distribution match?")
        print("\n   Scientific Status: QUESTIONABLE (circularity)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
