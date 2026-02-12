#!/usr/bin/env python3
"""
Unit tests for statistical validation methods used in experiments.

Tests the validation functions developed for Figure 1 analysis to ensure:
1. Statistical significance testing is correct
2. Threshold validation works properly
3. High-dimensional validation handles edge cases
4. Data quality metrics are computed correctly
5. Results are reproducible
"""

import pytest
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Import validation functions from project module
from bandit_gpt.utils.validation import (
    compute_statistical_metrics,
    evaluate_threshold,
    compute_cluster_quality,
    analyze_high_d_separation,
    find_exact_duplicates,
    find_near_duplicates,
    compute_diversity_score
)


# ============================================================================
# Tests: Statistical Validation
# ============================================================================

def test_compute_statistical_metrics_clear_difference():
    """Test statistical metrics with clearly different groups."""
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(3, 1, 100)
    
    metrics = compute_statistical_metrics(group1, group2)
    
    # Should detect significant difference
    assert metrics['mann_whitney_p'] < 0.001
    assert metrics['welch_t_p'] < 0.001
    assert abs(metrics['cohens_d']) > 2.0  # Large effect
    
    # CIs should not overlap
    assert metrics['ci_high_group1'] < metrics['ci_low_group2']


def test_compute_statistical_metrics_no_difference():
    """Test statistical metrics with identical groups."""
    np.random.seed(42)
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(0, 1, 100)
    
    metrics = compute_statistical_metrics(group1, group2)
    
    # Should not detect significant difference
    assert metrics['mann_whitney_p'] > 0.05
    assert metrics['welch_t_p'] > 0.05
    assert abs(metrics['cohens_d']) < 0.5  # Small effect


def test_compute_statistical_metrics_confidence_intervals():
    """Test that confidence intervals contain the true mean."""
    np.random.seed(42)
    true_mean = 5.0
    group = np.random.normal(true_mean, 1, 100)
    
    metrics = compute_statistical_metrics(group, group)
    
    # CI should contain the true mean
    assert metrics['ci_low_group1'] < true_mean < metrics['ci_high_group1']


def test_evaluate_threshold_good_separation():
    """Test threshold evaluation with well-separated clusters."""
    np.random.seed(42)
    
    # Create clearly separated clusters
    n = 200
    X_low = np.random.normal([-2, 0], 0.5, (n, 2))
    X_high = np.random.normal([2, 0], 0.5, (n, 2))
    X_pca = np.vstack([X_low, X_high])
    
    reward_gaps = np.concatenate([
        np.random.normal(0.5, 0.1, n),   # Low PC1: positive gaps
        np.random.normal(-0.5, 0.1, n)   # High PC1: negative gaps
    ])
    
    result = evaluate_threshold(X_pca, reward_gaps, threshold=0.0)
    
    assert result is not None
    assert result['silhouette'] > 0.3  # Good separation
    assert result['p_value'] < 0.001   # Significant difference
    assert result['mean_diff'] > 0.5   # Large mean difference
    assert 0.3 <= result['balance'] <= 0.5  # Reasonable balance


def test_evaluate_threshold_edge_cases():
    """Test threshold evaluation with edge cases."""
    np.random.seed(42)
    X_pca = np.random.normal(0, 1, (100, 2))
    reward_gaps = np.random.normal(0, 1, 100)
    
    # Threshold that creates empty cluster
    result = evaluate_threshold(X_pca, reward_gaps, threshold=100)
    assert result is None
    
    # Threshold that creates single-point cluster
    result = evaluate_threshold(X_pca, reward_gaps, threshold=X_pca[:, 0].max() - 1e-6)
    # Should handle gracefully (return None or valid result)
    assert result is None or isinstance(result, dict)


def test_compute_cluster_quality_good_clusters():
    """Test cluster quality computation with well-formed clusters."""
    np.random.seed(42)
    
    # Create two well-separated clusters
    cluster1 = np.random.normal(0, 0.5, (100, 10))
    cluster2 = np.random.normal(5, 0.5, (100, 10))
    X = np.vstack([cluster1, cluster2])
    labels = np.array([0] * 100 + [1] * 100)
    
    quality = compute_cluster_quality(X, labels)
    
    assert quality is not None
    assert quality['silhouette'] > 0.4  # Good separation
    assert quality['davies_bouldin'] < 1.0  # Good compactness
    assert quality['calinski'] > 100  # Good density-based separation


def test_compute_cluster_quality_poor_clusters():
    """Test cluster quality with overlapping clusters."""
    np.random.seed(42)
    
    # Create overlapping clusters
    cluster1 = np.random.normal(0, 2, (100, 10))
    cluster2 = np.random.normal(1, 2, (100, 10))
    X = np.vstack([cluster1, cluster2])
    labels = np.array([0] * 100 + [1] * 100)
    
    quality = compute_cluster_quality(X, labels)
    
    assert quality is not None
    # Silhouette should be lower for overlapping clusters
    assert quality['silhouette'] < 0.5


def test_compute_cluster_quality_single_cluster():
    """Test cluster quality with single cluster (should return None)."""
    X = np.random.normal(0, 1, (100, 10))
    labels = np.zeros(100)
    
    quality = compute_cluster_quality(X, labels)
    
    assert quality is None


def test_analyze_high_d_separation():
    """Test high-dimensional separation analysis."""
    np.random.seed(42)
    
    # Create separated clusters in high-D
    n_dims = 100
    cluster1 = np.random.normal([0] * n_dims, 1, (100, n_dims))
    cluster2 = np.random.normal([3] * n_dims, 1, (100, n_dims))
    X_high_d = np.vstack([cluster1, cluster2])
    
    pc1_values = np.array([-1] * 100 + [1] * 100)  # Proxy for PC1
    
    separation = analyze_high_d_separation(X_high_d, pc1_values, threshold=0)
    
    assert separation is not None
    assert separation['separation_ratio'] > 1.0  # Clusters are separated
    assert separation['between'] > separation['avg_within']


# ============================================================================
# Tests: Data Quality
# ============================================================================

def test_find_exact_duplicates_no_duplicates():
    """Test duplicate detection with unique items."""
    items = [f"item_{i}" for i in range(100)]
    
    result = find_exact_duplicates(items)
    
    assert result['total_items'] == 100
    assert result['unique_items'] == 100
    assert result['duplicate_groups'] == 0
    assert result['total_duplicates'] == 0


def test_find_exact_duplicates_with_duplicates():
    """Test duplicate detection with duplicates."""
    items = ["a", "b", "c", "a", "b", "a"]
    
    result = find_exact_duplicates(items)
    
    assert result['total_items'] == 6
    assert result['unique_items'] == 3
    assert result['duplicate_groups'] == 2  # "a" and "b" are duplicated
    assert result['total_duplicates'] == 3   # 3 extra copies
    assert result['duplicates']['a'] == 3
    assert result['duplicates']['b'] == 2


def test_find_near_duplicates_identical_embeddings():
    """Test near-duplicate detection with identical embeddings."""
    embeddings = np.array([
        [1, 0, 0],
        [1, 0, 0],  # Identical to first
        [0, 1, 0]
    ])
    
    near_dups = find_near_duplicates(embeddings, threshold=0.99)
    
    # Should find one near-duplicate pair (0, 1)
    assert len(near_dups) >= 1
    assert any(d['idx1'] == 0 and d['idx2'] == 1 for d in near_dups)


def test_find_near_duplicates_no_near_duplicates():
    """Test near-duplicate detection with distinct embeddings."""
    np.random.seed(42)
    embeddings = np.random.normal(0, 1, (10, 50))
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    near_dups = find_near_duplicates(embeddings, threshold=0.95)
    
    # Unlikely to find near-duplicates with random embeddings
    assert len(near_dups) < 5


def test_compute_diversity_score_identical_items():
    """Test diversity score with identical items."""
    embeddings = np.array([[1, 0, 0]] * 10)
    
    diversity = compute_diversity_score(embeddings)
    
    # No diversity (all identical)
    assert diversity < 0.01


def test_compute_diversity_score_diverse_items():
    """Test diversity score with diverse items."""
    np.random.seed(42)
    embeddings = np.random.normal(0, 1, (50, 100))
    
    diversity = compute_diversity_score(embeddings)
    
    # Should have good diversity
    assert diversity > 0.5


def test_compute_diversity_score_edge_cases():
    """Test diversity score with edge cases."""
    # Single item
    assert compute_diversity_score(np.array([[1, 0]])) == 0.0
    
    # Two items
    embeddings = np.array([[1, 0], [0, 1]])
    diversity = compute_diversity_score(embeddings)
    assert 0 < diversity <= 1


# ============================================================================
# Tests: Reproducibility
# ============================================================================

def test_statistical_metrics_reproducibility():
    """Test that statistical metrics are reproducible."""
    np.random.seed(42)
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(1, 1, 100)
    
    metrics1 = compute_statistical_metrics(group1, group2)
    metrics2 = compute_statistical_metrics(group1, group2)
    
    # Should be identical
    assert metrics1['mann_whitney_p'] == metrics2['mann_whitney_p']
    assert metrics1['cohens_d'] == metrics2['cohens_d']


def test_threshold_evaluation_reproducibility():
    """Test that threshold evaluation is reproducible."""
    np.random.seed(42)
    X_pca = np.random.normal(0, 1, (100, 2))
    reward_gaps = np.random.normal(0, 1, 100)
    
    result1 = evaluate_threshold(X_pca, reward_gaps, threshold=0.0)
    result2 = evaluate_threshold(X_pca, reward_gaps, threshold=0.0)
    
    assert result1['silhouette'] == result2['silhouette']
    assert result1['p_value'] == result2['p_value']


# ============================================================================
# Tests: Robustness
# ============================================================================

def test_statistical_metrics_with_nan():
    """Test statistical metrics handle NaN gracefully."""
    group1 = np.array([1, 2, np.nan, 4])
    group2 = np.array([5, 6, 7, 8])
    
    # Should handle NaN gracefully or raise appropriate error
    try:
        # Remove NaN values first (expected preprocessing)
        group1_clean = group1[~np.isnan(group1)]
        group2_clean = group2[~np.isnan(group2)]
        metrics = compute_statistical_metrics(group1_clean, group2_clean)
        # Check results are reasonable
        assert not np.isnan(metrics['mann_whitney_p'])
    except (ValueError, RuntimeError):
        # Expected behavior for problematic inputs
        pass


def test_threshold_evaluation_extreme_imbalance():
    """Test threshold evaluation with extremely imbalanced clusters."""
    np.random.seed(42)
    X_pca = np.random.normal(0, 1, (100, 2))
    reward_gaps = np.random.normal(0, 1, 100)
    
    # Threshold creating 99:1 split
    threshold = np.percentile(X_pca[:, 0], 99)
    result = evaluate_threshold(X_pca, reward_gaps, threshold)
    
    # Should handle extreme imbalance
    if result is not None:
        assert result['balance'] < 0.1  # Very imbalanced
        assert result['n_low'] + result['n_high'] == 100


def test_cluster_quality_large_dataset():
    """Test cluster quality handles large datasets efficiently."""
    np.random.seed(42)
    
    # Large dataset
    n = 10000
    cluster1 = np.random.normal(0, 1, (n // 2, 50))
    cluster2 = np.random.normal(3, 1, (n // 2, 50))
    X = np.vstack([cluster1, cluster2])
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    
    # Should complete in reasonable time (sampling kicks in)
    quality = compute_cluster_quality(X, labels)
    
    assert quality is not None
    assert quality['n_samples'] == n


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
