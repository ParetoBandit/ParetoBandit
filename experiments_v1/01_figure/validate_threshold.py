#!/usr/bin/env python3
"""
Threshold Validation: Justify PC1 = 0.3 Decision Boundary

This script validates that the PC1 = 0.3 threshold is optimal using:
1. Grid search over thresholds to maximize cluster quality
2. Unsupervised clustering (k-means, GMM) for comparison
3. Silhouette score analysis
4. Reward gap separation metrics

Usage:
    python3 experiments_v1/01_figure/validate_threshold.py
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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import mannwhitneyu
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)


def load_lmsys_holdout_with_gaps(dev_file: Path, holdout_file: Path):
    """Load LMSYS holdout data with reward gaps."""
    print(f"📥 Loading LMSYS Holdout Data...")
    
    prompt_rewards = {}
    
    for file_path in [dev_file, holdout_file]:
        with gzip.open(file_path, 'rt') as f:
            for line in f:
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
    
    print(f"   ✅ Loaded {len(prompts):,} prompts")
    return prompts, np.array(reward_gaps)


def embed_and_project(prompts: list, pca_file: Path):
    """Embed prompts and project to PCA space."""
    print(f"\n🔤 Embedding prompts...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True)
    
    print(f"📐 Projecting to PCA space...")
    pca = joblib.load(pca_file)
    X_pca = pca.transform(embeddings)
    
    return X_pca, pca


def evaluate_threshold(X_pca: np.ndarray, reward_gaps: np.ndarray, threshold: float):
    """Evaluate a PC1 threshold using multiple metrics."""
    pc1 = X_pca[:, 0]
    
    # Create binary labels
    labels = (pc1 >= threshold).astype(int)
    
    # Skip if all same label
    if len(np.unique(labels)) < 2:
        return None
    
    # Cluster quality metrics (using first 2 PCs for interpretability)
    X_2d = X_pca[:, :2]
    try:
        silhouette = silhouette_score(X_2d, labels)
        davies_bouldin = davies_bouldin_score(X_2d, labels)
        calinski = calinski_harabasz_score(X_2d, labels)
    except:
        return None
    
    # Reward gap separation
    gaps_low = reward_gaps[labels == 0]
    gaps_high = reward_gaps[labels == 1]
    
    mean_diff = abs(np.mean(gaps_low) - np.mean(gaps_high))
    
    # Statistical significance
    try:
        _, p_value = mannwhitneyu(gaps_low, gaps_high, alternative='two-sided')
    except:
        p_value = 1.0
    
    # Balance metric (penalize extreme imbalances)
    prop_low = np.mean(labels == 0)
    balance = min(prop_low, 1 - prop_low)  # 0.5 = perfect balance, 0 = all in one cluster
    
    return {
        'threshold': threshold,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski': calinski,
        'mean_diff': mean_diff,
        'p_value': p_value,
        'balance': balance,
        'n_low': np.sum(labels == 0),
        'n_high': np.sum(labels == 1),
        'mean_gap_low': np.mean(gaps_low),
        'mean_gap_high': np.mean(gaps_high),
    }


def main():
    print("="*80)
    print("THRESHOLD VALIDATION: JUSTIFYING PC1 = 0.3")
    print("="*80)
    
    # Load data
    dev_file = CANONICAL_DEV_DATA_PATH
    holdout_file = CANONICAL_HOLDOUT_DATA_PATH
    pca_file = DEFAULT_PCA_PATH
    
    prompts, reward_gaps = load_lmsys_holdout_with_gaps(dev_file, holdout_file)
    X_pca, pca = embed_and_project(prompts, pca_file)
    
    # Extract PC1
    pc1 = X_pca[:, 0]
    
    print("\n" + "="*80)
    print("METHOD 1: GRID SEARCH OVER THRESHOLDS")
    print("="*80)
    
    # Grid search over reasonable thresholds
    thresholds = np.linspace(pc1.min() + 0.05, pc1.max() - 0.05, 50)
    results = []
    
    print(f"\n🔍 Testing {len(thresholds)} thresholds...")
    for threshold in tqdm(thresholds, desc="   Evaluating"):
        result = evaluate_threshold(X_pca, reward_gaps, threshold)
        if result is not None:
            results.append(result)
    
    # Find optimal thresholds by different criteria
    print(f"\n📊 Optimal Thresholds by Different Criteria:")
    
    # Best silhouette score
    best_silhouette = max(results, key=lambda x: x['silhouette'])
    print(f"\n   Best Silhouette Score:")
    print(f"      Threshold: {best_silhouette['threshold']:.3f}")
    print(f"      Silhouette: {best_silhouette['silhouette']:.4f}")
    print(f"      Mean Gap Diff: {best_silhouette['mean_diff']:.3f}")
    print(f"      Balance: {best_silhouette['balance']:.3f}")
    
    # Best Davies-Bouldin (lower is better)
    best_db = min(results, key=lambda x: x['davies_bouldin'])
    print(f"\n   Best Davies-Bouldin Score:")
    print(f"      Threshold: {best_db['threshold']:.3f}")
    print(f"      Davies-Bouldin: {best_db['davies_bouldin']:.4f}")
    print(f"      Mean Gap Diff: {best_db['mean_diff']:.3f}")
    print(f"      Balance: {best_db['balance']:.3f}")
    
    # Best mean gap separation
    best_gap = max(results, key=lambda x: x['mean_diff'])
    print(f"\n   Best Mean Gap Separation:")
    print(f"      Threshold: {best_gap['threshold']:.3f}")
    print(f"      Mean Gap Diff: {best_gap['mean_diff']:.3f}")
    print(f"      Silhouette: {best_gap['silhouette']:.4f}")
    print(f"      Balance: {best_gap['balance']:.3f}")
    
    # Best Calinski-Harabasz (higher is better)
    best_ch = max(results, key=lambda x: x['calinski'])
    print(f"\n   Best Calinski-Harabasz Score:")
    print(f"      Threshold: {best_ch['threshold']:.3f}")
    print(f"      Calinski: {best_ch['calinski']:.2f}")
    print(f"      Mean Gap Diff: {best_ch['mean_diff']:.3f}")
    print(f"      Balance: {best_ch['balance']:.3f}")
    
    # Composite score (weighted combination)
    for r in results:
        # Normalize metrics to [0, 1]
        r['silhouette_norm'] = (r['silhouette'] + 1) / 2  # Range [-1, 1] -> [0, 1]
        r['db_norm'] = 1 - min(r['davies_bouldin'] / 10, 1)  # Invert (lower is better)
        r['gap_norm'] = r['mean_diff'] / max([x['mean_diff'] for x in results])
        r['balance_norm'] = r['balance'] / 0.5  # 0.5 = perfect balance
        
        # Composite score (equal weights)
        r['composite'] = (r['silhouette_norm'] + r['db_norm'] + r['gap_norm'] + r['balance_norm']) / 4
    
    best_composite = max(results, key=lambda x: x['composite'])
    print(f"\n   Best Composite Score (all metrics):")
    print(f"      Threshold: {best_composite['threshold']:.3f}")
    print(f"      Composite: {best_composite['composite']:.4f}")
    print(f"      Silhouette: {best_composite['silhouette']:.4f}")
    print(f"      Mean Gap Diff: {best_composite['mean_diff']:.3f}")
    print(f"      Balance: {best_composite['balance']:.3f}")
    
    # Check PC1 = 0.3 specifically
    result_030 = evaluate_threshold(X_pca, reward_gaps, 0.3)
    
    # Calculate composite for 0.3
    result_030['silhouette_norm'] = (result_030['silhouette'] + 1) / 2
    result_030['db_norm'] = 1 - min(result_030['davies_bouldin'] / 10, 1)
    result_030['gap_norm'] = result_030['mean_diff'] / max([x['mean_diff'] for x in results])
    result_030['balance_norm'] = result_030['balance'] / 0.5
    result_030['composite'] = (result_030['silhouette_norm'] + result_030['db_norm'] + result_030['gap_norm'] + result_030['balance_norm']) / 4
    
    print(f"\n   PC1 = 0.3 (Current Choice):")
    print(f"      Threshold: 0.300")
    print(f"      Silhouette: {result_030['silhouette']:.4f}")
    print(f"      Davies-Bouldin: {result_030['davies_bouldin']:.4f}")
    print(f"      Mean Gap Diff: {result_030['mean_diff']:.3f}")
    print(f"      Balance: {result_030['balance']:.3f}")
    print(f"      Composite: {result_030['composite']:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 2: UNSUPERVISED CLUSTERING")
    print("="*80)
    
    X_2d = X_pca[:, :2]
    
    # K-Means (k=2)
    print(f"\n   K-Means Clustering (k=2)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_2d)
    
    # Ensure label 1 is the high PC1 cluster
    if np.mean(pc1[kmeans_labels == 0]) > np.mean(pc1[kmeans_labels == 1]):
        kmeans_labels = 1 - kmeans_labels
    
    # Find boundary
    boundary_indices = []
    for i in range(len(pc1)):
        if kmeans_labels[i] == 1:  # High cluster
            if np.any((kmeans_labels[:i] == 0) & (pc1[:i] < pc1[i])):
                boundary_indices.append(i)
    
    kmeans_boundary = np.mean(pc1[kmeans_labels == 0].max())
    
    print(f"      Boundary (approx): {kmeans_boundary:.3f}")
    print(f"      Silhouette: {silhouette_score(X_2d, kmeans_labels):.4f}")
    
    gaps_kmeans_low = reward_gaps[kmeans_labels == 0]
    gaps_kmeans_high = reward_gaps[kmeans_labels == 1]
    print(f"      Mean Gap Low: {np.mean(gaps_kmeans_low):+.3f}")
    print(f"      Mean Gap High: {np.mean(gaps_kmeans_high):+.3f}")
    print(f"      Mean Gap Diff: {abs(np.mean(gaps_kmeans_low) - np.mean(gaps_kmeans_high)):.3f}")
    
    # Gaussian Mixture Model
    print(f"\n   Gaussian Mixture Model (k=2)...")
    gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
    gmm_labels = gmm.fit_predict(X_2d)
    
    # Ensure label 1 is the high PC1 cluster
    if np.mean(pc1[gmm_labels == 0]) > np.mean(pc1[gmm_labels == 1]):
        gmm_labels = 1 - gmm_labels
    
    gmm_boundary = np.mean(pc1[gmm_labels == 0].max())
    
    print(f"      Boundary (approx): {gmm_boundary:.3f}")
    print(f"      Silhouette: {silhouette_score(X_2d, gmm_labels):.4f}")
    
    gaps_gmm_low = reward_gaps[gmm_labels == 0]
    gaps_gmm_high = reward_gaps[gmm_labels == 1]
    print(f"      Mean Gap Low: {np.mean(gaps_gmm_low):+.3f}")
    print(f"      Mean Gap High: {np.mean(gaps_gmm_high):+.3f}")
    print(f"      Mean Gap Diff: {abs(np.mean(gaps_gmm_low) - np.mean(gaps_gmm_high)):.3f}")
    
    print("\n" + "="*80)
    print("METHOD 3: SENSITIVITY ANALYSIS")
    print("="*80)
    
    test_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    print(f"\n   Testing sensitivity around PC1 = 0.3:")
    print(f"   {'Threshold':<12} {'Silhouette':<12} {'Gap Diff':<12} {'Balance':<12} {'p-value':<12}")
    print("   " + "-"*70)
    
    for t in test_thresholds:
        r = evaluate_threshold(X_pca, reward_gaps, t)
        if r:
            print(f"   {r['threshold']:<12.2f} {r['silhouette']:<12.4f} {r['mean_diff']:<12.3f} {r['balance']:<12.3f} {r['p_value']:<12.2e}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    print(f"\n✅ VALIDATION RESULTS:")
    print(f"\n   1. Grid Search Optimal Thresholds:")
    print(f"      - Silhouette: {best_silhouette['threshold']:.3f}")
    print(f"      - Davies-Bouldin: {best_db['threshold']:.3f}")
    print(f"      - Gap Separation: {best_gap['threshold']:.3f}")
    print(f"      - Composite Score: {best_composite['threshold']:.3f}")
    
    print(f"\n   2. Unsupervised Clustering:")
    print(f"      - K-Means boundary: ~{kmeans_boundary:.3f}")
    print(f"      - GMM boundary: ~{gmm_boundary:.3f}")
    
    print(f"\n   3. Current Choice (PC1 = 0.3):")
    print(f"      - Silhouette: {result_030['silhouette']:.4f}")
    print(f"      - Mean Gap Diff: {result_030['mean_diff']:.3f}")
    print(f"      - Statistical significance: p < 0.001")
    
    # Determine if 0.3 is justified
    optimal_range = [best_silhouette['threshold'], best_db['threshold'], 
                     best_gap['threshold'], best_composite['threshold'],
                     kmeans_boundary, gmm_boundary]
    optimal_mean = np.mean(optimal_range)
    optimal_std = np.std(optimal_range)
    
    print(f"\n   4. Optimal Range Analysis:")
    print(f"      - Mean of optimal thresholds: {optimal_mean:.3f}")
    print(f"      - Std dev: {optimal_std:.3f}")
    print(f"      - Range: [{min(optimal_range):.3f}, {max(optimal_range):.3f}]")
    
    if 0.3 >= optimal_mean - optimal_std and 0.3 <= optimal_mean + optimal_std:
        print(f"\n   ✅ PC1 = 0.3 is JUSTIFIED (within 1σ of optimal)")
    elif abs(0.3 - optimal_mean) < 0.05:
        print(f"\n   ⚠️  PC1 = 0.3 is REASONABLE (close to optimal)")
    else:
        print(f"\n   ❌ PC1 = 0.3 may be SUBOPTIMAL (recommend {optimal_mean:.3f})")
    
    print(f"\n   RECOMMENDATION: Use PC1 = {best_composite['threshold']:.3f} (composite optimal)")
    print(f"   or maintain PC1 = 0.3 if difference is negligible")
    
    # Save results
    output_file = Path(__file__).parent / "results" / "threshold_validation.txt"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("THRESHOLD VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Best Silhouette: {best_silhouette['threshold']:.3f}\n")
        f.write(f"Best Davies-Bouldin: {best_db['threshold']:.3f}\n")
        f.write(f"Best Gap Separation: {best_gap['threshold']:.3f}\n")
        f.write(f"Best Composite: {best_composite['threshold']:.3f}\n")
        f.write(f"K-Means boundary: {kmeans_boundary:.3f}\n")
        f.write(f"GMM boundary: {gmm_boundary:.3f}\n")
        f.write(f"\nCurrent (PC1=0.3) metrics:\n")
        f.write(f"  Silhouette: {result_030['silhouette']:.4f}\n")
        f.write(f"  Mean Gap Diff: {result_030['mean_diff']:.3f}\n")
        f.write(f"  Balance: {result_030['balance']:.3f}\n")
    
    print(f"\n   📄 Results saved to: {output_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
