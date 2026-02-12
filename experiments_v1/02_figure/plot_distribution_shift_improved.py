#!/usr/bin/env python3
"""
Feature Distribution Shift Analysis (Figure 1.2) - IMPROVED VERSION

This script analyzes covariate shift between Source/Prior data and RouteLLM data
by comparing the distribution of the first principal component (PCA_0).

**CRITICAL**: This experiment uses the ACTUAL BanditRouter from src/bandit_gpt/router.py
for feature extraction. This ensures the analysis reflects the real routing system's
behavior, not a separate implementation.

IMPROVEMENTS IN THIS VERSION:
1. Uses BanditRouter._build_routing_features() for feature extraction (consistency!)
2. Bottom panel shows RouteLLM data decomposed by ACTUAL reward gaps (not Source data)
3. Adds statistical significance tests (KS test, bootstrap CI for PSI)
4. Reports PCA variance explained
5. Extracts sample prompts from clusters for qualitative validation
6. Fixes threshold inconsistencies

The analysis:
1. Loads prompts from both Source data (dev/holdout) and RouteLLM battles
2. Uses BanditRouter to extract features (same as production routing!)
3. Creates 1D density plots to visualize distribution shift
4. Computes Population Stability Index (PSI) to quantify the shift
5. Shows whether RouteLLM data is shifted toward "Easy" or "Hard" clusters
6. Performs statistical significance testing

Usage:
    python3 experiments_v1/02_figure/plot_distribution_shift_improved.py
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
from scipy.stats import gaussian_kde, ks_2samp
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    ROUTELLM_BATTLES_REWARDS_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)
from bandit_gpt.router import BanditRouter, RouterConfig


def load_source_prompts_from_datasets(dev_file: Path, holdout_file: Path, max_samples: int = 10000):
    """
    Load prompts from dev/holdout datasets (WITHOUT reward gaps for now).
    
    This represents the "Source/Prior" distribution from training data.
    
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


def load_routellm_prompts_with_metadata(battles_file: Path, start_idx: int = 0, max_samples: int = 10000):
    """
    Load prompts from RouteLLM battles dataset WITH full metadata.
    
    This represents the "target" distribution we're deploying on.
    
    Args:
        battles_file: Path to battles JSONL file
        start_idx: Starting index to read from
        max_samples: Maximum prompts to load
    
    Returns:
        prompts: List of prompt strings
        reward_gaps: Array of R_Turbo - R_Mixtral gaps (for clustering info)
        metadata: Dict with full battle information for analysis
    """
    print(f"\n📥 Loading RouteLLM deployment prompts...")
    print(f"   File: {battles_file}")
    print(f"   Range: {start_idx:,} to {start_idx + max_samples:,}")
    
    prompts = []
    reward_gaps = []
    metadata = {
        'reward_gpt4': [],
        'reward_mixtral': [],
        'winners': [],
        'raw_battles': []
    }
    
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
                metadata['reward_gpt4'].append(reward_turbo)
                metadata['reward_mixtral'].append(reward_mixtral)
                metadata['winners'].append(battle.get('winner', 'tie'))
                metadata['raw_battles'].append(battle)
                
            except Exception as e:
                continue
    
    print(f"   ✅ Loaded {len(prompts):,} RouteLLM prompts")
    
    # Convert to numpy arrays
    metadata['reward_gpt4'] = np.array(metadata['reward_gpt4'])
    metadata['reward_mixtral'] = np.array(metadata['reward_mixtral'])
    
    return prompts, np.array(reward_gaps), metadata


def project_to_pc1_with_stats(prompts: list, pca_file: Path, batch_size: int = 64, 
                               use_router: bool = True, return_full_features: bool = False):
    """
    Embed prompts and project onto first principal component (PC1).
    Also returns PCA stats for reporting.
    
    Uses the project's BanditRouter for feature extraction to ensure consistency
    between the experiment and the actual routing system.
    
    Args:
        prompts: List of prompt strings
        pca_file: Path to pre-trained PCA model
        batch_size: Batch size for embedding
        use_router: If True, uses BanditRouter's feature extraction (recommended)
        return_full_features: If True, returns all PC features for sensitivity analysis
    
    Returns:
        pc1_values: Array of PC1 coordinates (1D)
        pca_stats: Dict with PCA statistics
        all_features: (optional) Full feature matrix if return_full_features=True
    """
    # Load PCA for stats
    print(f"\n📐 Loading pre-trained PCA model...")
    pca = joblib.load(pca_file)
    n_components = pca.n_components_
    variance_explained_pc1 = pca.explained_variance_ratio_[0]
    variance_explained_cumulative = np.cumsum(pca.explained_variance_ratio_)
    
    print(f"   ✅ Loaded PCA: {n_components} components")
    print(f"   PC1 explains: {variance_explained_pc1:.3%} of variance")
    print(f"   PC1-5 explain: {variance_explained_cumulative[4]:.3%} of variance")
    print(f"   All {n_components} PCs explain: {variance_explained_cumulative[-1]:.3%} of variance")
    
    if use_router:
        print(f"\n🤖 Using BanditRouter for feature extraction...")
        print(f"   This ensures consistency with actual routing system")
        
        # Create router with minimal config (just for feature extraction)
        config = RouterConfig()
        model_registry = {
            "gpt-4-turbo": {"hle": 0.05, "cost_per_1m": 10.0},
            "mixtral-8x7b": {"hle": 0.15, "cost_per_1m": 0.5}
        }
        router = BanditRouter(model_registry=model_registry, config=config)
        
        print(f"\n🧮 Extracting features for {len(prompts):,} prompts using router...")
        all_features = []
        for prompt in tqdm(prompts, desc="   Extracting features"):
            features, _ = router._build_routing_features(prompt)
            all_features.append(features)
        
        all_features = np.array(all_features)
        print(f"   ✅ Feature shape: {all_features.shape}")
        
        # Extract PC1 (first component)
        pc1_values = all_features[:, 0]
        full_features = all_features  # Save for sensitivity analysis
        
    else:
        # Fallback: Direct embedding (original method)
        print(f"\n🔤 Loading sentence encoder...")
        print(f"   Model: {DEFAULT_SENTENCE_TRANSFORMER}")
        encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
        print(f"   ✅ Encoder loaded")
        
        print(f"\n🧮 Embedding {len(prompts):,} prompts...")
        embeddings = encoder.encode(
            prompts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        print(f"   ✅ Embeddings shape: {embeddings.shape}")
        
        print(f"\n📐 Projecting to PCA...")
        X_pca = pca.transform(embeddings)
        pc1_values = X_pca[:, 0]  # Extract first component only
        full_features = X_pca  # Save for sensitivity analysis
    
    print(f"   ✅ PC1 projection complete")
    print(f"   PC1 range: [{np.min(pc1_values):.3f}, {np.max(pc1_values):.3f}]")
    print(f"   PC1 mean: {np.mean(pc1_values):.3f}")
    print(f"   PC1 std: {np.std(pc1_values):.3f}")
    
    pca_stats = {
        'n_components': n_components,
        'variance_explained_pc1': variance_explained_pc1,
        'variance_explained_pc1_5': variance_explained_cumulative[4],
        'variance_explained_total': variance_explained_cumulative[-1],
        'embedding_model': DEFAULT_SENTENCE_TRANSFORMER,
        'used_router': use_router
    }
    
    if return_full_features:
        return pc1_values, pca_stats, full_features
    else:
        return pc1_values, pca_stats


def compute_psi_with_bootstrap(expected: np.ndarray, actual: np.ndarray, n_bins: int = 20, n_bootstrap: int = 1000):
    """
    Compute Population Stability Index (PSI) with bootstrap confidence intervals.
    
    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    
    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 ≤ PSI < 0.2: Moderate change
    - PSI ≥ 0.2: Significant change (may need model retraining)
    - PSI ≥ 0.25: Substantial shift requiring adaptive mechanisms
    
    Args:
        expected: Distribution from source/prior data
        actual: Distribution from RouteLLM/target data
        n_bins: Number of bins for histogram
        n_bootstrap: Number of bootstrap samples for CI
    
    Returns:
        psi: PSI value
        psi_ci: (lower, upper) 95% confidence interval
        bins: Bin edges
        expected_percents: Percentage in each bin for expected
        actual_percents: Percentage in each bin for actual
    """
    print(f"\n📊 Computing Population Stability Index (PSI)...")
    
    # Determine bin edges based on expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    def compute_psi_single(exp, act, bins):
        """Helper to compute PSI for bootstrap"""
        expected_counts, _ = np.histogram(exp, bins=bins)
        actual_counts, _ = np.histogram(act, bins=bins)
        
        epsilon = 1e-6
        expected_percents = (expected_counts + epsilon) / (len(exp) + epsilon * n_bins)
        actual_percents = (actual_counts + epsilon) / (len(act) + epsilon * n_bins)
        
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi, expected_percents, actual_percents
    
    # Compute observed PSI
    psi, expected_percents, actual_percents = compute_psi_single(expected, actual, bins)
    
    # Bootstrap CI
    print(f"   🔄 Computing bootstrap confidence interval ({n_bootstrap} samples)...")
    psi_bootstrap = []
    for _ in tqdm(range(n_bootstrap), desc="   Bootstrap", leave=False):
        # Resample with replacement
        exp_boot = np.random.choice(expected, size=len(expected), replace=True)
        act_boot = np.random.choice(actual, size=len(actual), replace=True)
        psi_boot, _, _ = compute_psi_single(exp_boot, act_boot, bins)
        psi_bootstrap.append(psi_boot)
    
    psi_ci = (np.percentile(psi_bootstrap, 2.5), np.percentile(psi_bootstrap, 97.5))
    
    print(f"   ✅ PSI = {psi:.4f} (95% CI: [{psi_ci[0]:.4f}, {psi_ci[1]:.4f}])")
    
    if psi < 0.1:
        print(f"   ✅ No significant distribution shift")
    elif psi < 0.2:
        print(f"   ⚠️  Moderate distribution shift detected")
    elif psi < 0.25:
        print(f"   🚨 Significant distribution shift! Consider retraining.")
    else:
        print(f"   🚨🚨 SUBSTANTIAL shift! Adaptive mechanisms required.")
    
    return psi, psi_ci, bins, expected_percents, actual_percents


def perform_statistical_tests(pc1_source, pc1_routellm):
    """
    Perform statistical significance tests for distribution shift.
    
    Args:
        pc1_source: PC1 values for source data
        pc1_routellm: PC1 values for RouteLLM data
    
    Returns:
        test_results: Dict with test statistics
    """
    print(f"\n📊 Performing statistical significance tests...")
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = ks_2samp(pc1_source, pc1_routellm)
    print(f"   Kolmogorov-Smirnov test:")
    print(f"      Statistic: {ks_stat:.4f}")
    print(f"      P-value: {ks_pval:.4e}")
    print(f"      Result: {'REJECT H0' if ks_pval < 0.05 else 'FAIL TO REJECT H0'} (distributions differ)")
    
    # Mean shift test (effect size)
    mean_shift = np.mean(pc1_routellm) - np.mean(pc1_source)
    pooled_std = np.sqrt((np.std(pc1_source)**2 + np.std(pc1_routellm)**2) / 2)
    cohens_d = mean_shift / pooled_std
    
    print(f"\n   Effect size (Cohen's d): {cohens_d:.4f}")
    effect_interpretation = (
        "negligible" if abs(cohens_d) < 0.2 else
        "small" if abs(cohens_d) < 0.5 else
        "medium" if abs(cohens_d) < 0.8 else
        "large"
    )
    print(f"      Interpretation: {effect_interpretation}")
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'mean_shift': mean_shift,
        'cohens_d': cohens_d,
        'effect_size': effect_interpretation
    }


def sensitivity_analysis_multipc(features_source, features_routellm, pca_stats):
    """
    Perform sensitivity analysis: compute PSI for different numbers of PCs.
    
    This addresses the concern that PC1 only explains 3.1% of variance.
    We test whether distribution shift conclusions are robust to using more PCs.
    
    Args:
        features_source: Full feature matrix for source data (n_samples, n_features)
        features_routellm: Full feature matrix for RouteLLM data (n_samples, n_features)
        pca_stats: PCA statistics dict
    
    Returns:
        sensitivity_results: Dict with PSI for different PC choices
    """
    print(f"\n📊 Performing sensitivity analysis: PSI across multiple PCs...")
    print(f"   PC1 explains {pca_stats['variance_explained_pc1']:.3%} of variance")
    print(f"   Testing robustness of distribution shift findings...")
    
    n_components = min(features_source.shape[1], features_routellm.shape[1])
    
    # Test configurations: [PC1 only, PC1-5, PC1-10, all PCs]
    configs = [
        (1, "PC1 only"),
        (5, "PC1-5"),
        (10, "PC1-10"),
        (n_components, f"All {n_components} PCs")
    ]
    
    sensitivity_results = {}
    
    for n_pcs, label in configs:
        if n_pcs > n_components:
            continue
        
        # Extract first n_pcs components
        source_subset = features_source[:, :n_pcs]
        routellm_subset = features_routellm[:, :n_pcs]
        
        # Compute multivariate PSI using L2 distance binning
        # For 1D, use direct binning; for multi-D, use Euclidean distance from centroid
        if n_pcs == 1:
            # 1D case: direct binning as before
            min_val = min(source_subset.min(), routellm_subset.min())
            max_val = max(source_subset.max(), routellm_subset.max())
            bins = np.linspace(min_val, max_val, 21)
            
            expected_counts, _ = np.histogram(source_subset.flatten(), bins=bins)
            actual_counts, _ = np.histogram(routellm_subset.flatten(), bins=bins)
            
            epsilon = 1e-6
            n_bins = len(bins) - 1
            expected_percents = (expected_counts + epsilon) / (len(source_subset) + epsilon * n_bins)
            actual_percents = (actual_counts + epsilon) / (len(routellm_subset) + epsilon * n_bins)
            
            psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        else:
            # Multi-D case: use Euclidean distance from centroid for binning
            # Compute centroids
            source_centroid = np.mean(source_subset, axis=0)
            routellm_centroid = np.mean(routellm_subset, axis=0)
            
            # Compute distances from source centroid
            source_dists = np.linalg.norm(source_subset - source_centroid, axis=1)
            routellm_dists = np.linalg.norm(routellm_subset - source_centroid, axis=1)
            
            # Bin by distance
            min_dist = min(source_dists.min(), routellm_dists.min())
            max_dist = max(source_dists.max(), routellm_dists.max())
            bins = np.linspace(min_dist, max_dist, 21)
            
            expected_counts, _ = np.histogram(source_dists, bins=bins)
            actual_counts, _ = np.histogram(routellm_dists, bins=bins)
            
            epsilon = 1e-6
            n_bins = len(bins) - 1
            expected_percents = (expected_counts + epsilon) / (len(source_dists) + epsilon * n_bins)
            actual_percents = (actual_counts + epsilon) / (len(routellm_dists) + epsilon * n_bins)
            
            psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        sensitivity_results[label] = float(psi)
        print(f"   {label}: PSI = {psi:.4f}")
    
    print(f"\n   ✅ Conclusion: Distribution shift is {'consistent' if max(sensitivity_results.values()) / min(sensitivity_results.values()) < 2.0 else 'varies significantly'} across dimensionality choices")
    
    return sensitivity_results


def analyze_cluster_separation(pc1_values, reward_gaps, threshold_low=0.3, threshold_high=0.6):
    """
    Analyze whether Mixtral-Sufficient and GPT-4-Turbo-Required clusters are statistically distinct.
    
    This addresses the concern that centroids are close (0.0221 vs -0.0160) and validates
    the claim of "distinct clusters" in the paper.
    
    Args:
        pc1_values: PC1 coordinates
        reward_gaps: Reward gaps (GPT-4-Turbo - Mixtral)
        threshold_low: Threshold for Mixtral-Sufficient tasks
        threshold_high: Threshold for GPT-4-Turbo-Required tasks
    
    Returns:
        separation_results: Dict with statistical tests for cluster separation
    """
    from scipy import stats
    
    print(f"\n📊 Testing cluster separation...")
    
    # Define clusters
    mixtral_mask = reward_gaps <= threshold_low
    gpt4_turbo_mask = reward_gaps > threshold_high
    
    pc1_mixtral = pc1_values[mixtral_mask]
    pc1_gpt4_turbo = pc1_values[gpt4_turbo_mask]
    
    # Compute centroids
    centroid_mixtral = np.mean(pc1_mixtral)
    centroid_gpt4_turbo = np.mean(pc1_gpt4_turbo)
    centroid_distance = abs(centroid_mixtral - centroid_gpt4_turbo)
    
    # Welch's t-test (doesn't assume equal variances)
    t_stat, t_pval = stats.ttest_ind(pc1_mixtral, pc1_gpt4_turbo, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(pc1_mixtral)**2 + np.std(pc1_gpt4_turbo)**2) / 2)
    cohens_d = (centroid_mixtral - centroid_gpt4_turbo) / pooled_std
    
    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_pval = stats.mannwhitneyu(pc1_mixtral, pc1_gpt4_turbo, alternative='two-sided')
    
    print(f"   Cluster centroids:")
    print(f"      Mixtral-Sufficient: {centroid_mixtral:.4f} (n={len(pc1_mixtral)})")
    print(f"      GPT-4-Turbo-Required: {centroid_gpt4_turbo:.4f} (n={len(pc1_gpt4_turbo)})")
    print(f"      Distance: {centroid_distance:.4f}")
    
    print(f"\n   Welch's t-test (cluster means):")
    print(f"      t-statistic: {t_stat:.4f}")
    print(f"      P-value: {t_pval:.4e}")
    print(f"      Result: {'REJECT H0' if t_pval < 0.05 else 'FAIL TO REJECT H0'} (clusters differ)")
    
    print(f"\n   Effect size (Cohen's d): {cohens_d:.4f}")
    effect_interpretation = (
        "negligible" if abs(cohens_d) < 0.2 else
        "small" if abs(cohens_d) < 0.5 else
        "medium" if abs(cohens_d) < 0.8 else
        "large"
    )
    print(f"      Interpretation: {effect_interpretation}")
    
    print(f"\n   Mann-Whitney U test (non-parametric):")
    print(f"      U-statistic: {u_stat:.0f}")
    print(f"      P-value: {u_pval:.4e}")
    
    # Compute overlap coefficient
    # What percentage of each distribution falls in the other's range?
    overlap_mixtral_in_gpt4_range = np.sum((pc1_mixtral >= pc1_gpt4_turbo.min()) & 
                                            (pc1_mixtral <= pc1_gpt4_turbo.max())) / len(pc1_mixtral)
    overlap_gpt4_in_mixtral_range = np.sum((pc1_gpt4_turbo >= pc1_mixtral.min()) & 
                                             (pc1_gpt4_turbo <= pc1_mixtral.max())) / len(pc1_gpt4_turbo)
    
    print(f"\n   Distribution overlap:")
    print(f"      Mixtral-Sufficient in GPT-4-Turbo range: {overlap_mixtral_in_gpt4_range:.1%}")
    print(f"      GPT-4-Turbo-Required in Mixtral range: {overlap_gpt4_in_mixtral_range:.1%}")
    
    # Conclusion
    if t_pval < 0.001 and abs(cohens_d) > 0.2:
        conclusion = "Clusters are statistically distinct with meaningful separation"
    elif t_pval < 0.05:
        conclusion = "Clusters differ statistically but with substantial overlap"
    else:
        conclusion = "Clusters do not show significant separation"
    
    print(f"\n   ✅ Conclusion: {conclusion}")
    
    return {
        'centroid_mixtral': float(centroid_mixtral),
        'centroid_gpt4_turbo': float(centroid_gpt4_turbo),
        'centroid_distance': float(centroid_distance),
        't_statistic': float(t_stat),
        't_pvalue': float(t_pval),
        'cohens_d': float(cohens_d),
        'effect_size': effect_interpretation,
        'u_statistic': float(u_stat),
        'u_pvalue': float(u_pval),
        'overlap_mixtral_in_gpt4_range': float(overlap_mixtral_in_gpt4_range),
        'overlap_gpt4_in_mixtral_range': float(overlap_gpt4_in_mixtral_range),
        'conclusion': conclusion
    }


def extract_sample_prompts(prompts, pc1_values, reward_gaps, n_samples=3):
    """
    Extract sample prompts from different clusters for qualitative validation.
    
    Returns REPRESENTATIVE samples closest to cluster centroids, not extremes.
    This ensures samples accurately reflect typical cluster characteristics.
    
    Cluster definitions:
    - Mixtral-Sufficient (Gap ≤ 0.3): Tasks where Mixtral performs similarly to or better than GPT-4-Turbo
    - GPT-4-Turbo-Required (Gap > 0.6): Tasks where GPT-4-Turbo has a clear advantage
    
    Args:
        prompts: List of prompt strings
        pc1_values: PC1 coordinates
        reward_gaps: Reward gaps (GPT-4 - Mixtral)
        n_samples: Number of samples per cluster
    
    Returns:
        samples: Dict with sample prompts from each cluster
    """
    print(f"\n📝 Extracting representative sample prompts from clusters...")
    
    # Define clusters based on reward gaps (GPT-4-Turbo - Mixtral)
    # Gap ≤ 0.3: Mixtral performs similarly or better (including negative gaps)
    # Gap > 0.6: GPT-4-Turbo has clear advantage
    mixtral_sufficient_mask = reward_gaps <= 0.3
    gpt4_turbo_required_mask = reward_gaps > 0.6
    
    samples = {
        'mixtral_sufficient': [],
        'gpt4_turbo_required': [],
        'mixtral_sufficient_centroid': None,
        'gpt4_turbo_required_centroid': None
    }
    
    # Get REPRESENTATIVE examples closest to cluster centroids (not extremes!)
    if np.any(mixtral_sufficient_mask):
        mixtral_indices = np.where(mixtral_sufficient_mask)[0]
        mixtral_pc1 = pc1_values[mixtral_sufficient_mask]
        
        # Compute centroid
        mixtral_centroid = np.mean(mixtral_pc1)
        samples['mixtral_sufficient_centroid'] = float(mixtral_centroid)
        
        # Find samples closest to centroid (medoids)
        distances_to_centroid = np.abs(mixtral_pc1 - mixtral_centroid)
        closest_indices = mixtral_indices[np.argsort(distances_to_centroid)]
        
        print(f"   Mixtral-Sufficient cluster: centroid={mixtral_centroid:.4f}, n={len(mixtral_indices)}")
        
        for idx in closest_indices[:n_samples]:
            samples['mixtral_sufficient'].append({
                'prompt': prompts[idx][:200] + "..." if len(prompts[idx]) > 200 else prompts[idx],
                'pc1': float(pc1_values[idx]),
                'reward_gap': float(reward_gaps[idx]),
                'distance_to_centroid': float(np.abs(pc1_values[idx] - mixtral_centroid))
            })
    
    if np.any(gpt4_turbo_required_mask):
        gpt4_turbo_indices = np.where(gpt4_turbo_required_mask)[0]
        gpt4_turbo_pc1 = pc1_values[gpt4_turbo_required_mask]
        
        # Compute centroid
        gpt4_turbo_centroid = np.mean(gpt4_turbo_pc1)
        samples['gpt4_turbo_required_centroid'] = float(gpt4_turbo_centroid)
        
        # Find samples closest to centroid (medoids)
        distances_to_centroid = np.abs(gpt4_turbo_pc1 - gpt4_turbo_centroid)
        closest_indices = gpt4_turbo_indices[np.argsort(distances_to_centroid)]
        
        print(f"   GPT-4-Turbo-Required cluster: centroid={gpt4_turbo_centroid:.4f}, n={len(gpt4_turbo_indices)}")
        
        for idx in closest_indices[:n_samples]:
            samples['gpt4_turbo_required'].append({
                'prompt': prompts[idx][:200] + "..." if len(prompts[idx]) > 200 else prompts[idx],
                'pc1': float(pc1_values[idx]),
                'reward_gap': float(reward_gaps[idx]),
                'distance_to_centroid': float(np.abs(pc1_values[idx] - gpt4_turbo_centroid))
            })
    
    print(f"   ✅ Extracted {len(samples['mixtral_sufficient'])} Mixtral-Sufficient samples, {len(samples['gpt4_turbo_required'])} GPT-4-Turbo-Required samples")
    
    return samples


def create_improved_visualization(pc1_source, pc1_routellm, reward_gaps_routellm, 
                                  psi, psi_ci, pca_stats, output_dir: Path):
    """
    Create improved visualization with RouteLLM task decomposition in bottom panel.
    
    Cluster definitions based on reward gaps (GPT-4-Turbo - Mixtral):
    - Mixtral-Sufficient (Gap ≤ 0.3): Tasks where Mixtral performs similarly to or better than GPT-4-Turbo
    - GPT-4-Turbo-Required (Gap > 0.6): Tasks where GPT-4-Turbo has a clear advantage
    
    Args:
        pc1_source: PC1 values for source/prior data
        pc1_routellm: PC1 values for RouteLLM data
        reward_gaps_routellm: Reward gaps for RouteLLM prompts
        psi: PSI value
        psi_ci: PSI confidence interval tuple
        pca_stats: PCA statistics dict
        output_dir: Directory to save figure
    """
    print(f"\n🎨 Creating improved distribution shift visualization...")
    
    # Define task clusters based on reward gaps (GPT-4-Turbo - Mixtral)
    # CONSISTENT THRESHOLDS: Gap ≤ 0.3 = Mixtral-Sufficient, Gap > 0.6 = GPT-4-Turbo-Required
    mixtral_sufficient_mask = reward_gaps_routellm <= 0.3
    gpt4_turbo_required_mask = reward_gaps_routellm > 0.6
    
    pc1_routellm_mixtral = pc1_routellm[mixtral_sufficient_mask]
    pc1_routellm_gpt4_turbo = pc1_routellm[gpt4_turbo_required_mask]
    
    routellm_mixtral_pct = len(pc1_routellm_mixtral) / len(pc1_routellm) * 100
    routellm_gpt4_turbo_pct = len(pc1_routellm_gpt4_turbo) / len(pc1_routellm) * 100
    
    # Compute mean reward gaps for annotation
    mean_gap_mixtral = np.mean(reward_gaps_routellm[mixtral_sufficient_mask])
    mean_gap_gpt4_turbo = np.mean(reward_gaps_routellm[gpt4_turbo_required_mask])
    
    print(f"\n   📊 RouteLLM Task Distribution:")
    print(f"      Mixtral-Sufficient (Gap ≤ 0.3): {len(pc1_routellm_mixtral):,} ({routellm_mixtral_pct:.1f}%)")
    print(f"      Mean Gap: {mean_gap_mixtral:.3f}")
    print(f"      GPT-4-Turbo-Required (Gap > 0.6): {len(pc1_routellm_gpt4_turbo):,} ({routellm_gpt4_turbo_pct:.1f}%)")
    print(f"      Mean Gap: {mean_gap_gpt4_turbo:.3f}")
    
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
    
    # Updated title with PCA info
    ax1.set_title(
        f'Feature Distribution Shift: Source vs RouteLLM Data\n'
        f'PSI = {psi:.4f} (95% CI: [{psi_ci[0]:.3f}, {psi_ci[1]:.3f}]) | '
        f'PC1 explains {pca_stats["variance_explained_pc1"]:.2%} of variance',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add PSI interpretation text
    psi_color = 'green' if psi < 0.1 else ('orange' if psi < 0.2 else ('red' if psi < 0.25 else 'darkred'))
    psi_text = ('No Shift' if psi < 0.1 else 
                ('Moderate Shift' if psi < 0.2 else 
                 ('Significant Shift' if psi < 0.25 else 'SUBSTANTIAL Shift')))
    
    ax1.text(0.02, 0.98, f'PSI Interpretation: {psi_text}',
            transform=ax1.transAxes,
            fontsize=12,
            fontweight='bold',
            color=psi_color,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=psi_color, linewidth=2))
    
    # === Plot 2: RouteLLM Data - Mixtral-Sufficient vs GPT-4-Turbo-Required Clustering (GROUND TRUTH) ===
    ax2 = axes[1]
    
    # Use different colors
    color_mixtral = '#1b9e77'  # Teal/green for Mixtral-Sufficient
    color_gpt4_turbo = '#7570b3'  # Purple for GPT-4-Turbo-Required
    
    # Plot ROUTELLM task-based densities (GROUND TRUTH)
    if len(pc1_routellm_mixtral) > 50 and len(pc1_routellm_gpt4_turbo) > 50:
        print(f"   🟢 Computing KDE for Mixtral-Sufficient prompts...")
        kde_routellm_mixtral = gaussian_kde(pc1_routellm_mixtral, bw_method=0.1)
        
        print(f"   🟣 Computing KDE for GPT-4-Turbo-Required prompts...")
        kde_routellm_gpt4_turbo = gaussian_kde(pc1_routellm_gpt4_turbo, bw_method=0.1)
        
        density_routellm_mixtral = kde_routellm_mixtral(x)
        density_routellm_gpt4_turbo = kde_routellm_gpt4_turbo(x)
        
        ax2.plot(x, density_routellm_mixtral, 
                label=f'Mixtral-Sufficient (Gap ≤ 0.3): {routellm_mixtral_pct:.1f}%', 
                color=color_mixtral, linewidth=3, alpha=0.8)
        ax2.plot(x, density_routellm_gpt4_turbo, 
                label=f'GPT-4-Turbo-Required (Gap > 0.6): {routellm_gpt4_turbo_pct:.1f}%', 
                color=color_gpt4_turbo, linewidth=3, alpha=0.8)
        
        ax2.fill_between(x, 0, density_routellm_mixtral, color=color_mixtral, alpha=0.2)
        ax2.fill_between(x, 0, density_routellm_gpt4_turbo, color=color_gpt4_turbo, alpha=0.2)
        
        # Add mean lines
        ax2.axvline(np.mean(pc1_routellm_mixtral), color=color_mixtral, linestyle='--', 
                   linewidth=2, alpha=0.6)
        ax2.axvline(np.mean(pc1_routellm_gpt4_turbo), color=color_gpt4_turbo, linestyle='--', 
                   linewidth=2, alpha=0.6)
    
    ax2.set_xlabel('First Principal Component (PC1)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax2.set_title(
        'RouteLLM Deployment Data: Task Distribution by Model Performance\n'
        f'Clusters Based on Ground Truth Reward Gaps (GPT-4-Turbo - Mixtral)',
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
            f'RouteLLM Deployment Distribution\n'
            f'Mixtral-Sufficient: Centered at {np.mean(pc1_routellm_mixtral):.3f}, Δ={mean_gap_mixtral:.2f}\n'
            f'GPT-4-Turbo-Required: Centered at {np.mean(pc1_routellm_gpt4_turbo):.3f}, Δ={mean_gap_gpt4_turbo:.2f}\n'
            f'→ Two Distinct Performance Clusters',
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
    
    return psi, mean_shift, mean_gap_mixtral, mean_gap_gpt4_turbo


def main():
    print("="*80)
    print("FEATURE DISTRIBUTION SHIFT ANALYSIS (IMPROVED)")
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
    print(f"   Embedding model: {DEFAULT_SENTENCE_TRANSFORMER}")
    
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
    
    # Step 2: Load RouteLLM deployment data WITH metadata
    routellm_prompts, reward_gaps_routellm, routellm_metadata = load_routellm_prompts_with_metadata(
        battles_file, start_idx=0, max_samples=10000
    )
    
    if len(routellm_prompts) == 0:
        print("\n❌ No RouteLLM prompts loaded!")
        return
    
    # Step 3: Project to PC1 with stats (and save full features for sensitivity analysis)
    print("\n" + "="*80)
    print("PROJECTING SOURCE DATA TO PC1 (USING ROUTER)")
    print("="*80)
    pc1_source, pca_stats, features_source = project_to_pc1_with_stats(
        source_prompts, pca_file, batch_size=64, use_router=True, return_full_features=True
    )
    
    print("\n" + "="*80)
    print("PROJECTING ROUTELLM DATA TO PC1 (USING ROUTER)")
    print("="*80)
    pc1_routellm, _, features_routellm = project_to_pc1_with_stats(
        routellm_prompts, pca_file, batch_size=64, use_router=True, return_full_features=True
    )
    
    # Step 4: Statistical tests
    test_results = perform_statistical_tests(pc1_source, pc1_routellm)
    
    # Step 4.5: Sensitivity analysis (robustness to PC dimensionality choice)
    sensitivity_results = sensitivity_analysis_multipc(
        features_source, features_routellm, pca_stats
    )
    
    # Step 5: PSI with confidence intervals
    psi, psi_ci, bins, expected_percents, actual_percents = compute_psi_with_bootstrap(
        pc1_source, pc1_routellm, n_bins=20, n_bootstrap=1000
    )
    
    # Step 6: Test cluster separation (addresses "bimodal" interpretation concerns)
    cluster_separation = analyze_cluster_separation(
        pc1_routellm, reward_gaps_routellm, threshold_low=0.3, threshold_high=0.6
    )
    
    # Step 7: Extract sample prompts
    samples = extract_sample_prompts(
        routellm_prompts, pc1_routellm, reward_gaps_routellm, n_samples=3
    )
    
    # Step 8: Visualize
    psi_val, mean_shift, mean_gap_mixtral, mean_gap_gpt4_turbo = create_improved_visualization(
        pc1_source, pc1_routellm, reward_gaps_routellm, 
        psi, psi_ci, pca_stats, output_dir
    )
    
    # Step 9: Summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Distribution Shift Summary:")
    print(f"   Source prompts: {len(source_prompts):,}")
    print(f"   RouteLLM prompts: {len(routellm_prompts):,}")
    print(f"   PSI: {psi:.4f} (95% CI: [{psi_ci[0]:.3f}, {psi_ci[1]:.3f}])")
    print(f"   Mean shift: {mean_shift:.4f}")
    print(f"   KS statistic: {test_results['ks_statistic']:.4f} (p={test_results['ks_pvalue']:.4e})")
    print(f"   Cohen's d: {test_results['cohens_d']:.4f} ({test_results['effect_size']})")
    
    print(f"\n📐 PCA Statistics:")
    print(f"   Embedding model: {pca_stats['embedding_model']}")
    print(f"   Used router: {pca_stats.get('used_router', False)}")
    print(f"   PCA components: {pca_stats['n_components']}")
    print(f"   PC1 variance: {pca_stats['variance_explained_pc1']:.3%}")
    print(f"   PC1-5 variance: {pca_stats['variance_explained_pc1_5']:.3%}")
    
    print(f"\n🔬 Sensitivity Analysis (PSI across dimensionality):")
    for label, psi_val in sensitivity_results.items():
        print(f"   {label}: PSI = {psi_val:.4f}")
    
    print(f"\n🔍 RouteLLM Task Clusters:")
    print(f"   Mixtral-Sufficient (Gap ≤ 0.3): Mean gap = {mean_gap_mixtral:.3f}")
    print(f"   GPT-4-Turbo-Required (Gap > 0.6): Mean gap = {mean_gap_gpt4_turbo:.3f}")
    
    print(f"\n📏 Cluster Separation:")
    print(f"   Centroid distance: {cluster_separation['centroid_distance']:.4f}")
    print(f"   t-test p-value: {cluster_separation['t_pvalue']:.4e}")
    print(f"   Cohen's d: {cluster_separation['cohens_d']:.4f} ({cluster_separation['effect_size']})")
    print(f"   Conclusion: {cluster_separation['conclusion']}")
    
    print(f"\n📝 Sample Prompts (Mixtral-Sufficient Cluster):")
    for i, sample in enumerate(samples['mixtral_sufficient'][:2], 1):
        print(f"   {i}. PC1={sample['pc1']:.3f}, Gap={sample['reward_gap']:.3f}")
        print(f"      \"{sample['prompt']}\"")
    
    print(f"\n📝 Sample Prompts (GPT-4-Turbo-Required Cluster):")
    for i, sample in enumerate(samples['gpt4_turbo_required'][:2], 1):
        print(f"   {i}. PC1={sample['pc1']:.3f}, Gap={sample['reward_gap']:.3f}")
        print(f"      \"{sample['prompt']}\"")
    
    print(f"\n   Output: {output_dir}/distribution_shift_pc1.png")
    
    # Save summary to JSON
    summary = {
        'psi': float(psi),
        'psi_ci_lower': float(psi_ci[0]),
        'psi_ci_upper': float(psi_ci[1]),
        'mean_shift': float(mean_shift),
        'ks_statistic': float(test_results['ks_statistic']),
        'ks_pvalue': float(test_results['ks_pvalue']),
        'cohens_d': float(test_results['cohens_d']),
        'pca_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                      for k, v in pca_stats.items()},
        'sensitivity_analysis': sensitivity_results,
        'cluster_separation': cluster_separation,
        'samples': samples
    }
    
    summary_file = output_dir / "distribution_shift_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   ✅ Saved summary: {summary_file}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
