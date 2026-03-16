"""
Statistical validation utilities for experiment analysis.

Functions for computing statistical metrics, cluster quality, and data quality measures.
Used across experiments to ensure methodological rigor.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from collections import Counter


# ============================================================================
# Statistical Validation Functions
# ============================================================================

def compute_statistical_metrics(group1, group2):
    """
    Compute comprehensive statistical metrics comparing two groups.
    
    Parameters:
    -----------
    group1 : array-like
        First group of values
    group2 : array-like
        Second group of values
    
    Returns:
    --------
    dict with:
        - mann_whitney_p: Mann-Whitney U test p-value (non-parametric)
        - welch_t_p: Welch's t-test p-value (parametric)
        - cohens_d: Effect size (Cohen's d)
        - mean_group1/2: Group means
        - ci_low/high_group1/2: 95% confidence intervals
    """
    # Mann-Whitney U test (non-parametric)
    _, mann_whitney_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Welch's t-test (parametric, unequal variances)
    _, welch_t_p = stats.ttest_ind(group1, group2, equal_var=False)
    
    # Cohen's d (effect size)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    
    # 95% Confidence Intervals
    ci_low_group1, ci_high_group1 = stats.t.interval(
        0.95, len(group1) - 1,
        loc=np.mean(group1),
        scale=stats.sem(group1)
    )
    ci_low_group2, ci_high_group2 = stats.t.interval(
        0.95, len(group2) - 1,
        loc=np.mean(group2),
        scale=stats.sem(group2)
    )
    
    return {
        'mann_whitney_p': mann_whitney_p,
        'welch_t_p': welch_t_p,
        'cohens_d': cohens_d,
        'mean_group1': mean1,
        'mean_group2': mean2,
        'ci_low_group1': ci_low_group1,
        'ci_high_group1': ci_high_group1,
        'ci_low_group2': ci_low_group2,
        'ci_high_group2': ci_high_group2,
    }


def evaluate_threshold(X_pca, reward_gaps, threshold):
    """
    Evaluate a threshold on PC1 for cluster separation quality.
    
    Parameters:
    -----------
    X_pca : np.ndarray
        PCA-transformed data (N x d)
    reward_gaps : np.ndarray
        Reward gaps for each sample (N,)
    threshold : float
        Threshold value for PC1
    
    Returns:
    --------
    dict with clustering metrics and statistical significance, or None if invalid
    """
    pc1 = X_pca[:, 0]
    labels = (pc1 >= threshold).astype(int)
    
    # Check if we have both clusters
    if len(np.unique(labels)) < 2:
        return None
    
    X_2d = X_pca[:, :2]
    
    # Compute cluster quality metrics
    try:
        silhouette = silhouette_score(X_2d, labels)
        davies_bouldin = davies_bouldin_score(X_2d, labels)
        calinski = calinski_harabasz_score(X_2d, labels)
    except:
        return None
    
    # Compute reward gap statistics
    gaps_low = reward_gaps[labels == 0]
    gaps_high = reward_gaps[labels == 1]
    
    if len(gaps_low) == 0 or len(gaps_high) == 0:
        return None
    
    mean_diff = abs(np.mean(gaps_low) - np.mean(gaps_high))
    
    # Statistical significance
    try:
        _, p_value = stats.mannwhitneyu(gaps_low, gaps_high, alternative='two-sided')
    except:
        p_value = 1.0
    
    # Cluster balance
    prop_low = np.mean(labels == 0)
    balance = min(prop_low, 1 - prop_low)
    
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


def compute_cluster_quality(X, labels):
    """
    Compute cluster quality metrics for given embeddings and labels.
    
    Parameters:
    -----------
    X : np.ndarray
        Data points (N x d)
    labels : np.ndarray
        Cluster labels (N,)
    
    Returns:
    --------
    dict with silhouette, davies_bouldin, and calinski scores, or None if invalid
    """
    if len(np.unique(labels)) < 2:
        return None
    
    try:
        # Use sampling for large datasets
        sample_size = min(5000, len(X))
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        silhouette = silhouette_score(X_sample, labels_sample)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        
        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski': calinski,
            'n_samples': len(X),
            'n_clusters': len(np.unique(labels))
        }
    except Exception as e:
        return None


def analyze_high_d_separation(X_high_d, pc1_values, threshold):
    """
    Analyze cluster separation in high-dimensional space.
    
    Parameters:
    -----------
    X_high_d : np.ndarray
        High-dimensional embeddings (N x d)
    pc1_values : np.ndarray
        PC1 values for each sample (N,)
    threshold : float
        Threshold for PC1 clustering
    
    Returns:
    --------
    dict with separation metrics, or None if invalid
    """
    labels = (pc1_values >= threshold).astype(int)
    
    if len(np.unique(labels)) < 2:
        return None
    
    # Compute centroids
    centroid_low = X_high_d[labels == 0].mean(axis=0)
    centroid_high = X_high_d[labels == 1].mean(axis=0)
    
    # Within-cluster distances
    within_low = cdist(X_high_d[labels == 0], centroid_low.reshape(1, -1), 
                       metric='euclidean').mean()
    within_high = cdist(X_high_d[labels == 1], centroid_high.reshape(1, -1), 
                        metric='euclidean').mean()
    
    # Between-cluster distance
    between = np.linalg.norm(centroid_high - centroid_low)
    
    # Average within-cluster distance
    n_low = np.sum(labels == 0)
    n_high = np.sum(labels == 1)
    avg_within = (within_low * n_low + within_high * n_high) / len(labels)
    
    # Separation ratio
    separation_ratio = between / avg_within if avg_within > 0 else 0
    
    return {
        'within_low': within_low,
        'within_high': within_high,
        'between': between,
        'separation_ratio': separation_ratio,
        'avg_within': avg_within
    }


# ============================================================================
# Data Quality Functions
# ============================================================================

def find_exact_duplicates(items):
    """
    Find exact duplicates in a list.
    
    Parameters:
    -----------
    items : list
        List of items (strings, etc.)
    
    Returns:
    --------
    dict with counts and duplicate items
    """
    item_counts = Counter(items)
    duplicates = {item: count for item, count in item_counts.items() if count > 1}
    unique_items = set(items)
    
    return {
        'total_items': len(items),
        'unique_items': len(unique_items),
        'duplicate_groups': len(duplicates),
        'total_duplicates': len(items) - len(unique_items),
        'duplicates': duplicates
    }


def find_near_duplicates(embeddings, threshold=0.95):
    """
    Find near-duplicates based on cosine similarity.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embedding vectors (N x d)
    threshold : float
        Similarity threshold (default: 0.95)
    
    Returns:
    --------
    list of near-duplicate pairs with indices and similarity scores
    """
    # Compute pairwise cosine similarities
    similarities = 1 - squareform(pdist(embeddings, metric='cosine'))
    
    near_duplicates = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarities[i, j] >= threshold:
                near_duplicates.append({
                    'idx1': i,
                    'idx2': j,
                    'similarity': similarities[i, j]
                })
    
    return near_duplicates


def compute_diversity_score(embeddings):
    """
    Compute diversity score as 1 - average pairwise similarity.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embedding vectors (N x d)
    
    Returns:
    --------
    float between 0 (no diversity) and 1 (maximum diversity)
    """
    if len(embeddings) < 2:
        return 0.0
    
    # Compute average pairwise cosine similarity
    similarities = 1 - squareform(pdist(embeddings, metric='cosine'))
    
    # Get upper triangle (exclude diagonal)
    upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
    avg_similarity = np.mean(upper_triangle)
    
    # Diversity is 1 - similarity
    diversity = 1 - avg_similarity
    
    return diversity
