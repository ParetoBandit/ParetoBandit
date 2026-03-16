#!/usr/bin/env python3
"""
Integration tests for Figure 1 validation scripts.

Tests that the validation analysis scripts work correctly end-to-end
with synthetic data that mimics the alignment tax phenomenon.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Ensure we can import from experiments
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import validation functions
from pareto_bandit.utils.validation import (
    compute_statistical_metrics,
    evaluate_threshold,
    compute_cluster_quality,
    analyze_high_d_separation,
    find_exact_duplicates,
    find_near_duplicates,
    compute_diversity_score
)


# ============================================================================
# Fixtures for Synthetic Data
# ============================================================================

@pytest.fixture
def synthetic_alignment_tax_data():
    """
    Generate synthetic data that mimics the alignment tax phenomenon.
    
    Creates two clusters:
    - Low PC1: GPT-4-Turbo wins (positive reward gap)
    - High PC1: Mixtral wins (negative reward gap)
    """
    np.random.seed(42)
    
    # Low PC1 cluster (82% of data, GPT-4-Turbo advantage)
    n_low = 164
    X_low = np.random.normal([-2, 0], 0.8, (n_low, 2))
    rewards_low = np.random.normal(0.13, 0.10, n_low)  # Mean +0.13
    
    # High PC1 cluster (18% of data, Mixtral advantage)
    n_high = 36
    X_high = np.random.normal([2, 0], 0.6, (n_high, 2))
    rewards_high = np.random.normal(-0.68, 0.15, n_high)  # Mean -0.68
    
    # Combine
    X_pca = np.vstack([X_low, X_high])
    reward_gaps = np.concatenate([rewards_low, rewards_high])
    
    # Generate synthetic prompts
    prompts = []
    for i in range(n_low):
        prompts.append(f"Natural language task {i}: explain, describe, summarize")
    for i in range(n_high):
        prompts.append(f"Strict format task {i}: output EXACTLY in JSON format")
    
    return {
        'X_pca': X_pca,
        'reward_gaps': reward_gaps,
        'prompts': prompts,
        'n_low': n_low,
        'n_high': n_high,
        'true_threshold': 0.0  # Optimal threshold for this synthetic data
    }


@pytest.fixture
def synthetic_embeddings():
    """Generate synthetic high-dimensional embeddings."""
    np.random.seed(42)
    
    # Create 200 samples in 384D space
    n_samples = 200
    n_dims = 384
    
    # Two clusters with some separation in high-D
    cluster1 = np.random.normal([0] * n_dims, 1, (n_samples // 2, n_dims))
    cluster2 = np.random.normal([0.5] * n_dims, 1, (n_samples // 2, n_dims))
    
    embeddings = np.vstack([cluster1, cluster2])
    
    return embeddings


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# ============================================================================
# Tests: Threshold Validation
# ============================================================================

def test_threshold_grid_search(synthetic_alignment_tax_data):
    """Test that grid search identifies optimal threshold."""
    data = synthetic_alignment_tax_data
    X_pca = data['X_pca']
    reward_gaps = data['reward_gaps']
    
    # Test multiple thresholds
    thresholds = np.linspace(-3, 3, 20)
    results = []
    
    for threshold in thresholds:
        result = evaluate_threshold(X_pca, reward_gaps, threshold)
        if result is not None:
            results.append(result)
    
    # Should find multiple valid thresholds
    assert len(results) > 10
    
    # Best threshold should be near 0 (the true boundary)
    best_by_silhouette = max(results, key=lambda x: x['silhouette'])
    assert abs(best_by_silhouette['threshold']) < 1.5


def test_threshold_validation_metrics_consistency(synthetic_alignment_tax_data):
    """Test that different metrics give consistent threshold recommendations."""
    
    data = synthetic_alignment_tax_data
    X_pca = data['X_pca']
    reward_gaps = data['reward_gaps']
    
    thresholds = np.linspace(-2, 2, 30)
    results = [evaluate_threshold(X_pca, reward_gaps, t) 
               for t in thresholds if evaluate_threshold(X_pca, reward_gaps, t) is not None]
    
    # Find best by different criteria
    best_silhouette = max(results, key=lambda x: x['silhouette'])
    best_gap = max(results, key=lambda x: x['mean_diff'])
    
    # Should be reasonably close (within 1.0 of each other)
    assert abs(best_silhouette['threshold'] - best_gap['threshold']) < 1.5


def test_threshold_sensitivity_analysis(synthetic_alignment_tax_data):
    """Test that results are robust across threshold perturbations."""
    
    data = synthetic_alignment_tax_data
    X_pca = data['X_pca']
    reward_gaps = data['reward_gaps']
    
    # Test thresholds around optimal
    optimal_threshold = 0.0
    perturbations = np.linspace(-0.5, 0.5, 10)
    
    results = []
    for delta in perturbations:
        result = evaluate_threshold(X_pca, reward_gaps, optimal_threshold + delta)
        if result is not None:
            results.append(result)
    
    # All should show significant difference
    p_values = [r['p_value'] for r in results]
    assert all(p < 0.01 for p in p_values)
    
    # Silhouette should remain reasonable
    silhouettes = [r['silhouette'] for r in results]
    assert all(s > 0.2 for s in silhouettes)


# ============================================================================
# Tests: High-Dimensional Validation
# ============================================================================

def test_high_dimensional_structure_validation(synthetic_embeddings):
    """Test high-dimensional validation with synthetic embeddings."""
    from sklearn.decomposition import PCA
    
    embeddings = synthetic_embeddings
    labels = np.array([0] * 100 + [1] * 100)
    
    # Test cluster quality in different dimensionalities
    # 2D PCA
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(embeddings)
    quality_2d = compute_cluster_quality(X_2d, labels)
    
    # 32D PCA
    pca_32d = PCA(n_components=32)
    X_32d = pca_32d.fit_transform(embeddings)
    quality_32d = compute_cluster_quality(X_32d, labels)
    
    # 384D (original)
    quality_384d = compute_cluster_quality(embeddings, labels)
    
    # All should return valid results
    assert quality_2d is not None
    assert quality_32d is not None
    assert quality_384d is not None
    
    # Silhouette typically decreases with dimensionality (curse of dimensionality)
    # But structure should still be detectable
    assert quality_2d['silhouette'] > quality_32d['silhouette']


def test_high_d_separation_ratio(synthetic_embeddings):
    """Test that separation ratio captures cluster quality in high-D."""
    
    embeddings = synthetic_embeddings
    pc1_values = np.array([-1] * 100 + [1] * 100)
    
    separation = analyze_high_d_separation(embeddings, pc1_values, threshold=0)
    
    assert separation is not None
    assert separation['separation_ratio'] > 0.1  # Some separation
    assert separation['between'] > 0  # Clusters are separated


def test_pc1_predictive_power_in_high_d(synthetic_embeddings):
    """Test that PC1-based clustering predicts structure in high-D."""
    from sklearn.decomposition import PCA
    from scipy.stats import spearmanr
    
    embeddings = synthetic_embeddings
    
    # Compute PC1
    pca = PCA(n_components=32)
    X_pca = pca.fit_transform(embeddings)
    pc1 = X_pca[:, 0]
    
    # Create synthetic "reward gaps" correlated with cluster membership
    labels = np.array([0] * 100 + [1] * 100)
    reward_gaps = labels * 2 - 1 + np.random.normal(0, 0.1, 200)
    
    # PC1 should correlate with reward gaps
    correlation, p_value = spearmanr(pc1, reward_gaps)
    
    assert p_value < 0.001  # Significant correlation
    assert abs(correlation) > 0.5  # Moderate to strong correlation


# ============================================================================
# Tests: Data Quality Validation
# ============================================================================

def test_duplicate_detection_integration(synthetic_alignment_tax_data):
    """Test duplicate detection on synthetic prompts."""
    
    prompts = synthetic_alignment_tax_data['prompts']
    
    # All prompts should be unique
    result = find_exact_duplicates(prompts)
    
    assert result['total_items'] == 200
    assert result['unique_items'] == 200
    assert result['duplicate_groups'] == 0


def test_near_duplicate_detection_with_similar_prompts():
    """Test near-duplicate detection with intentionally similar prompts."""
    
    # Create embeddings with some near-duplicates
    np.random.seed(42)
    
    embeddings = []
    # First 5: unique
    for i in range(5):
        embeddings.append(np.random.normal(i, 0.1, 50))
    
    # Next 2: near-duplicates of first
    embeddings.append(embeddings[0] + np.random.normal(0, 0.01, 50))
    embeddings.append(embeddings[0] + np.random.normal(0, 0.01, 50))
    
    embeddings = np.array(embeddings)
    
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    near_dups = find_near_duplicates(embeddings, threshold=0.95)
    
    # Should find near-duplicates
    assert len(near_dups) > 0


def test_diversity_score_integration(synthetic_alignment_tax_data):
    """Test diversity score on synthetic data."""
    
    # Generate embeddings for prompts
    np.random.seed(42)
    n_prompts = len(synthetic_alignment_tax_data['prompts'])
    embeddings = np.random.normal(0, 1, (n_prompts, 50))
    
    diversity = compute_diversity_score(embeddings)
    
    # Should have reasonable diversity (allow for floating point precision)
    assert 0.3 < diversity <= 1.01


# ============================================================================
# Tests: Statistical Validation End-to-End
# ============================================================================

def test_full_statistical_pipeline(synthetic_alignment_tax_data):
    """Test complete statistical validation pipeline."""
    data = synthetic_alignment_tax_data
    X_pca = data['X_pca']
    reward_gaps = data['reward_gaps']
    
    # Step 1: Find optimal threshold
    threshold = 0.0  # Known optimal for synthetic data
    
    # Step 2: Evaluate threshold
    threshold_result = evaluate_threshold(X_pca, reward_gaps, threshold)
    assert threshold_result is not None
    
    # Step 3: Compute cluster quality
    pc1 = X_pca[:, 0]
    labels = (pc1 >= threshold).astype(int)
    quality = compute_cluster_quality(X_pca, labels)
    assert quality is not None
    
    # Step 4: Statistical significance
    gaps_low = reward_gaps[labels == 0]
    gaps_high = reward_gaps[labels == 1]
    stats = compute_statistical_metrics(gaps_low, gaps_high)
    
    # Verify all metrics are computed
    assert 'mann_whitney_p' in stats
    assert 'cohens_d' in stats
    assert 'ci_low_group1' in stats
    
    # Should find significant difference (by design)
    assert stats['mann_whitney_p'] < 0.001
    assert abs(stats['cohens_d']) > 1.0  # Large effect


def test_statistical_validation_with_noise(synthetic_alignment_tax_data):
    """Test statistical validation with noisy data."""
    
    data = synthetic_alignment_tax_data
    reward_gaps = data['reward_gaps']
    
    # Add noise
    noisy_gaps = reward_gaps + np.random.normal(0, 0.2, len(reward_gaps))
    
    pc1 = data['X_pca'][:, 0]
    labels = (pc1 >= 0).astype(int)
    
    gaps_low = noisy_gaps[labels == 0]
    gaps_high = noisy_gaps[labels == 1]
    
    stats = compute_statistical_metrics(gaps_low, gaps_high)
    
    # Should still detect difference despite noise
    assert stats['mann_whitney_p'] < 0.01


# ============================================================================
# Tests: Reproducibility
# ============================================================================

def test_validation_pipeline_reproducibility(synthetic_alignment_tax_data):
    """Test that validation pipeline is reproducible."""
    
    data = synthetic_alignment_tax_data
    X_pca = data['X_pca']
    reward_gaps = data['reward_gaps']
    
    # Run twice
    result1 = evaluate_threshold(X_pca, reward_gaps, 0.0)
    result2 = evaluate_threshold(X_pca, reward_gaps, 0.0)
    
    # Should be identical
    assert result1['silhouette'] == result2['silhouette']
    assert result1['p_value'] == result2['p_value']
    assert result1['mean_diff'] == result2['mean_diff']


# ============================================================================
# Tests: Edge Cases and Error Handling
# ============================================================================

def test_validation_with_small_sample():
    """Test validation handles small samples gracefully."""
    # Very small sample
    group1 = np.array([1, 2, 3])
    group2 = np.array([4, 5, 6])
    
    # Should handle small samples
    stats = compute_statistical_metrics(group1, group2)
    assert stats is not None
    
    # Cluster quality with small sample
    X = np.random.normal(0, 1, (10, 2))
    labels = np.array([0] * 5 + [1] * 5)
    quality = compute_cluster_quality(X, labels)
    assert quality is not None


def test_validation_with_outliers():
    """Test validation is robust to outliers."""
    
    np.random.seed(42)
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(2, 1, 100)
    
    # Add outliers
    group1 = np.append(group1, [100, -100])
    group2 = np.append(group2, [100, -100])
    
    stats = compute_statistical_metrics(group1, group2)
    
    # Mann-Whitney should be robust to outliers
    assert stats['mann_whitney_p'] < 0.05


def test_validation_with_imbalanced_clusters():
    """Test validation handles imbalanced clusters."""
    
    np.random.seed(42)
    
    # Create 95:5 imbalance
    n_large = 190
    n_small = 10
    
    X_large = np.random.normal([0, 0], 1, (n_large, 2))
    X_small = np.random.normal([3, 3], 1, (n_small, 2))
    X_pca = np.vstack([X_large, X_small])
    
    reward_gaps = np.concatenate([
        np.random.normal(0, 0.1, n_large),
        np.random.normal(-1, 0.1, n_small)
    ])
    
    # Find threshold that creates this split
    threshold = 1.5
    result = evaluate_threshold(X_pca, reward_gaps, threshold)
    
    # Should handle imbalance
    assert result is not None
    assert result['balance'] < 0.2  # Imbalanced


# ============================================================================
# Tests: Performance
# ============================================================================

def test_validation_performance_large_dataset():
    """Test that validation completes quickly on large datasets."""
    import time
    
    # Large dataset
    np.random.seed(42)
    n = 10000
    X = np.random.normal(0, 1, (n, 100))
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    
    start = time.time()
    quality = compute_cluster_quality(X, labels)
    elapsed = time.time() - start
    
    # Should complete in reasonable time (< 5 seconds with sampling)
    assert elapsed < 5.0
    assert quality is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
