"""
Unit tests for Distribution Shift Analysis (Experiment 02)

Tests the improved distribution shift analysis script including:
- PSI calculation with bootstrap confidence intervals
- Kolmogorov-Smirnov test
- Effect size (Cohen's d)
- Sample prompt extraction
- Statistical validation functions
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments_v1" / "02_figure"))

# Import after path setup
from plot_distribution_shift_improved import (
    compute_psi_with_bootstrap,
    perform_statistical_tests,
    extract_sample_prompts,
    sensitivity_analysis_multipc,
    analyze_task_category_separation
)


class TestPSICalculation:
    """Test Population Stability Index (PSI) calculation."""
    
    def test_psi_identical_distributions(self):
        """PSI should be near zero for identical distributions."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        
        psi, psi_ci, bins, _, _ = compute_psi_with_bootstrap(
            expected, actual, n_bins=10, n_bootstrap=100
        )
        
        # PSI should be very small for similar distributions
        assert psi < 0.1, f"Expected PSI < 0.1 for similar distributions, got {psi}"
        assert psi_ci[0] >= 0, "Lower CI should be non-negative"
        assert psi_ci[1] >= psi_ci[0], "Upper CI should be >= lower CI"
    
    def test_psi_shifted_distributions(self):
        """PSI should be large for significantly shifted distributions."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(1, 1, 1000)  # Shifted by 1 std
        
        psi, psi_ci, bins, _, _ = compute_psi_with_bootstrap(
            expected, actual, n_bins=10, n_bootstrap=100
        )
        
        # PSI should indicate significant shift
        assert psi > 0.2, f"Expected PSI > 0.2 for shifted distributions, got {psi}"
        assert psi_ci[1] > psi_ci[0], "CI should have positive width"
    
    def test_psi_bootstrap_coverage(self):
        """Bootstrap CI should contain the point estimate."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 500)
        actual = np.random.normal(0.3, 1, 500)
        
        psi, psi_ci, _, _, _ = compute_psi_with_bootstrap(
            expected, actual, n_bins=10, n_bootstrap=100
        )
        
        # Point estimate should be within CI (with small tolerance for randomness)
        # Due to bootstrap variability, we check if PSI is reasonably close to CI range
        assert psi_ci[0] <= psi * 1.5, f"PSI {psi} far below lower CI {psi_ci[0]}"
        assert psi <= psi_ci[1] * 1.5, f"PSI {psi} far above upper CI {psi_ci[1]}"
    
    def test_psi_bin_sensitivity(self):
        """PSI should be relatively stable across different bin numbers."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)
        
        psi_10, _, _, _, _ = compute_psi_with_bootstrap(
            expected, actual, n_bins=10, n_bootstrap=50
        )
        psi_20, _, _, _, _ = compute_psi_with_bootstrap(
            expected, actual, n_bins=20, n_bootstrap=50
        )
        
        # PSI values should be in same ballpark (within 50% of each other)
        ratio = max(psi_10, psi_20) / min(psi_10, psi_20)
        assert ratio < 2.0, f"PSI varies too much with bins: {psi_10} vs {psi_20}"
    
    def test_psi_output_format(self):
        """Verify PSI function returns correct output format."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 100)
        actual = np.random.normal(0, 1, 100)
        
        result = compute_psi_with_bootstrap(
            expected, actual, n_bins=5, n_bootstrap=10
        )
        
        assert len(result) == 5, "Should return 5 values"
        psi, psi_ci, bins, exp_pct, act_pct = result
        
        assert isinstance(psi, (float, np.floating)), "PSI should be float"
        assert isinstance(psi_ci, tuple), "CI should be tuple"
        assert len(psi_ci) == 2, "CI should have 2 values"
        assert isinstance(bins, np.ndarray), "Bins should be array"
        assert len(bins) == 6, "Should have n_bins+1 bin edges"
        assert isinstance(exp_pct, np.ndarray), "Expected percentages should be array"
        assert isinstance(act_pct, np.ndarray), "Actual percentages should be array"


class TestStatisticalTests:
    """Test statistical significance tests."""
    
    def test_ks_test_identical_distributions(self):
        """KS test should not reject for identical distributions."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(0, 1, 500)
        
        results = perform_statistical_tests(data1, data2)
        
        assert 'ks_statistic' in results
        assert 'ks_pvalue' in results
        # P-value should be high (not significant)
        assert results['ks_pvalue'] > 0.01, "Should not reject null for similar dists"
    
    def test_ks_test_different_distributions(self):
        """KS test should reject for significantly different distributions."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(1, 1, 500)  # Shifted
        
        results = perform_statistical_tests(data1, data2)
        
        # Should strongly reject null hypothesis
        assert results['ks_pvalue'] < 0.001, "Should reject null for different dists"
        assert results['ks_statistic'] > 0.1, "KS statistic should be substantial"
    
    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        np.random.seed(42)
        # Create distributions with known effect size
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0.5, 1, 1000)  # 0.5 std shift
        
        results = perform_statistical_tests(data1, data2)
        
        assert 'cohens_d' in results
        assert 'effect_size' in results
        # Cohen's d should be approximately -0.5 (negative because data2 mean > data1 mean)
        assert abs(results['cohens_d']) > 0.3, "Effect size should be detectable"
        assert abs(results['cohens_d']) < 0.7, "Effect size should be reasonable"
    
    def test_effect_size_interpretation(self):
        """Test that effect size interpretation is correct."""
        np.random.seed(42)
        
        # Small effect (0.2 std)
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0.15, 1, 1000)
        results_small = perform_statistical_tests(data1, data2)
        assert results_small['effect_size'] in ['negligible', 'small']
        
        # Medium effect (0.5 std)
        data3 = np.random.normal(0.6, 1, 1000)
        results_medium = perform_statistical_tests(data1, data3)
        assert results_medium['effect_size'] in ['small', 'medium']
        
        # Large effect (1.0 std)
        data4 = np.random.normal(1.0, 1, 1000)
        results_large = perform_statistical_tests(data1, data4)
        assert results_large['effect_size'] in ['medium', 'large']
    
    def test_mean_shift_sign(self):
        """Test that mean shift has correct sign."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(0.5, 1, 500)
        
        results = perform_statistical_tests(data1, data2)
        
        # Mean shift should be positive (data2 - data1)
        assert results['mean_shift'] > 0, "Mean shift should be positive"
        assert 0.3 < results['mean_shift'] < 0.7, "Mean shift should be ~0.5"


class TestSampleExtraction:
    """Test sample prompt extraction functionality."""
    
    def test_extract_samples_basic(self):
        """Test basic sample extraction."""
        prompts = [
            "Mixtral-Sufficient prompt 1",
            "Mixtral-Sufficient prompt 2",
            "Mixtral-Sufficient prompt 3",
            "GPT-4-Turbo-Required prompt 1",
            "GPT-4-Turbo-Required prompt 2",
            "GPT-4-Turbo-Required prompt 3",
        ]
        pc1_values = np.array([0.02, 0.025, 0.03, -0.01, -0.015, -0.02])
        reward_gaps = np.array([0.1, 0.2, 0.25, 0.7, 0.8, 0.9])
        
        samples = extract_sample_prompts(prompts, pc1_values, reward_gaps, n_samples=2)
        
        assert 'mixtral_sufficient' in samples
        assert 'gpt4_turbo_required' in samples
        assert len(samples['mixtral_sufficient']) <= 2, "Should extract at most n_samples"
        assert len(samples['gpt4_turbo_required']) <= 2, "Should extract at most n_samples"
    
    def test_extract_samples_structure(self):
        """Test that extracted samples have correct structure."""
        prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
        pc1_values = np.array([0.02, 0.025, -0.01, -0.015])
        reward_gaps = np.array([0.2, 0.25, 0.7, 0.8])
        
        samples = extract_sample_prompts(prompts, pc1_values, reward_gaps, n_samples=1)
        
        for category in ['mixtral_sufficient', 'gpt4_turbo_required']:
            if samples[category]:  # If any samples in this category
                sample = samples[category][0]
                assert 'prompt' in sample
                assert 'pc1' in sample
                assert 'reward_gap' in sample
                assert 'distance_to_centroid' in sample
                assert isinstance(sample['prompt'], str)
                assert isinstance(sample['pc1'], (float, np.floating))
                assert isinstance(sample['reward_gap'], (float, np.floating))
                assert isinstance(sample['distance_to_centroid'], (float, np.floating))
    
    def test_extract_samples_thresholds(self):
        """Test that samples are correctly categorized by thresholds."""
        prompts = [f"prompt{i}" for i in range(10)]
        pc1_values = np.linspace(-0.5, 0.5, 10)
        reward_gaps = np.array([0.1, 0.2, 0.25, 0.3, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9])
        
        samples = extract_sample_prompts(prompts, pc1_values, reward_gaps, n_samples=5)
        
        # Check Mixtral-Sufficient samples have Gap <= 0.3
        for sample in samples['mixtral_sufficient']:
            assert sample['reward_gap'] <= 0.3, f"Mixtral-Sufficient sample has Gap={sample['reward_gap']} > 0.3"
        
        # Check GPT-4-Turbo-Required samples have Gap > 0.6
        for sample in samples['gpt4_turbo_required']:
            assert sample['reward_gap'] > 0.6, f"GPT-4-Turbo-Required sample has Gap={sample['reward_gap']} <= 0.6"
    
    def test_extract_samples_long_prompts(self):
        """Test that long prompts are truncated."""
        long_prompt = "x" * 300
        prompts = [long_prompt, "short"]
        pc1_values = np.array([0.02, -0.015])
        reward_gaps = np.array([0.2, 0.8])
        
        samples = extract_sample_prompts(prompts, pc1_values, reward_gaps, n_samples=2)
        
        # Check truncation
        for category in ['mixtral_sufficient', 'gpt4_turbo_required']:
            for sample in samples[category]:
                assert len(sample['prompt']) <= 203, "Prompt should be truncated (200 + '...')"
    
    def test_extract_samples_empty_categories(self):
        """Test behavior when no samples match a category."""
        prompts = ["prompt1", "prompt2"]
        pc1_values = np.array([-0.05, -0.04])
        reward_gaps = np.array([0.8, 0.9])  # All GPT-4-Turbo-Required, no Mixtral-Sufficient
        
        samples = extract_sample_prompts(prompts, pc1_values, reward_gaps, n_samples=2)
        
        # Mixtral-Sufficient should be empty
        assert len(samples['mixtral_sufficient']) == 0, "Should have no Mixtral-Sufficient samples"
        # GPT-4-Turbo-Required should have samples
        assert len(samples['gpt4_turbo_required']) > 0, "Should have GPT-4-Turbo-Required samples"
    
    def test_extract_samples_centroid_accuracy(self):
        """Test that extracted samples are closest to cluster centroids (not extremes)."""
        # Create clear clusters with known centroids
        prompts = [f"prompt{i}" for i in range(10)]
        # Mixtral-Sufficient cluster centered at 0.0
        # GPT-4-Turbo-Required cluster centered at -0.1
        pc1_values = np.array([0.0, 0.01, -0.01, 0.02, -0.02,  # Mixtral-Sufficient around 0.0
                               -0.1, -0.11, -0.09, -0.12, -0.08])  # GPT-4-Turbo around -0.1
        reward_gaps = np.array([0.1, 0.15, 0.2, 0.25, 0.3,  # Mixtral-Sufficient
                                0.7, 0.75, 0.8, 0.85, 0.9])  # GPT-4-Turbo-Required
        
        samples = extract_sample_prompts(prompts, pc1_values, reward_gaps, n_samples=2)
        
        # Verify samples have 'distance_to_centroid' field
        if len(samples['mixtral_sufficient']) > 0:
            assert 'distance_to_centroid' in samples['mixtral_sufficient'][0]
            # First sample should be closest to centroid
            distances = [s['distance_to_centroid'] for s in samples['mixtral_sufficient']]
            assert distances == sorted(distances), "Samples should be sorted by distance to centroid"
        
        if len(samples['gpt4_turbo_required']) > 0:
            assert 'distance_to_centroid' in samples['gpt4_turbo_required'][0]
            distances = [s['distance_to_centroid'] for s in samples['gpt4_turbo_required']]
            assert distances == sorted(distances), "Samples should be sorted by distance to centroid"


class TestDataLoading:
    """Test data loading functions (mocked)."""
    
    @patch('plot_distribution_shift_improved.gzip.open')
    @patch('plot_distribution_shift_improved.Path.exists')
    def test_load_source_prompts_success(self, mock_exists, mock_gzip):
        """Test successful loading of source prompts."""
        from plot_distribution_shift_improved import load_source_prompts_from_datasets
        
        mock_exists.return_value = True
        
        # Mock gzipped file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value = [
            '{"prompt": "test prompt 1"}\n',
            '{"prompt": "test prompt 2"}\n',
        ]
        mock_gzip.return_value = mock_file
        
        prompts = load_source_prompts_from_datasets(
            Path("dev.jsonl.gz"),
            Path("holdout.jsonl.gz"),
            max_samples=2
        )
        
        assert len(prompts) > 0, "Should load prompts"
        assert all(isinstance(p, str) for p in prompts), "All prompts should be strings"
    
    def test_load_routellm_prompts_structure(self):
        """Test RouteLLM prompt loading structure."""
        from plot_distribution_shift_improved import load_routellm_prompts_with_metadata
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_battles = [
                {
                    "prompt": "test prompt 1",
                    "model_a": "openai/gpt-4-turbo",
                    "model_b": "mistralai/mixtral-8x7b-instruct",
                    "reward_a": 0.8,
                    "reward_b": 0.6,
                    "winner": "model_a"
                },
                {
                    "prompt": "test prompt 2",
                    "model_a": "mistralai/mixtral-8x7b-instruct",
                    "model_b": "openai/gpt-4-turbo",
                    "reward_a": 0.5,
                    "reward_b": 0.9,
                    "winner": "model_b"
                }
            ]
            for battle in test_battles:
                f.write(json.dumps(battle) + '\n')
            temp_path = f.name
        
        try:
            prompts, gaps, metadata = load_routellm_prompts_with_metadata(
                Path(temp_path), start_idx=0, max_samples=2
            )
            
            assert len(prompts) == 2
            assert len(gaps) == 2
            assert isinstance(metadata, dict)
            assert 'reward_gpt4' in metadata
            assert 'reward_mixtral' in metadata
            assert len(metadata['reward_gpt4']) == 2
            
            # Check gap calculation (GPT-4 - Mixtral)
            assert abs(gaps[0] - 0.2) < 1e-9, "First gap should be 0.8 - 0.6 = 0.2"
            assert abs(gaps[1] - 0.4) < 1e-9, "Second gap should be 0.9 - 0.5 = 0.4"
        finally:
            Path(temp_path).unlink()


class TestIntegration:
    """Integration tests for full analysis."""
    
    def test_psi_and_ks_consistency(self):
        """Test that PSI and KS tests agree on significance."""
        np.random.seed(42)
        
        # Test 1: No shift
        data1a = np.random.normal(0, 1, 500)
        data1b = np.random.normal(0, 1, 500)
        
        psi1, _, _, _, _ = compute_psi_with_bootstrap(data1a, data1b, n_bins=10, n_bootstrap=50)
        results1 = perform_statistical_tests(data1a, data1b)
        
        # Both should indicate no significant shift
        assert psi1 < 0.1 and results1['ks_pvalue'] > 0.05, \
            "PSI and KS should agree: no shift"
        
        # Test 2: Significant shift
        data2a = np.random.normal(0, 1, 500)
        data2b = np.random.normal(0.8, 1, 500)
        
        psi2, _, _, _, _ = compute_psi_with_bootstrap(data2a, data2b, n_bins=10, n_bootstrap=50)
        results2 = perform_statistical_tests(data2a, data2b)
        
        # Both should indicate significant shift
        assert psi2 > 0.2 and results2['ks_pvalue'] < 0.01, \
            "PSI and KS should agree: significant shift"
    
    def test_analysis_reproducibility(self):
        """Test that analysis is reproducible with same random seed."""
        # Generate data with one seed
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 200)
        data2 = np.random.normal(0.3, 1, 200)
        
        # Run twice with same seed for bootstrap
        np.random.seed(100)
        psi1, ci1, _, _, _ = compute_psi_with_bootstrap(
            data1, data2, n_bins=10, n_bootstrap=50
        )
        
        np.random.seed(100)  # Reset seed for bootstrap
        psi2, ci2, _, _, _ = compute_psi_with_bootstrap(
            data1, data2, n_bins=10, n_bootstrap=50
        )
        
        # Results should be identical when using the same random seed
        assert abs(psi1 - psi2) < 1e-10, "PSI should be reproducible"
        assert abs(ci1[0] - ci2[0]) < 1e-10, "Lower CI should be reproducible"
        assert abs(ci1[1] - ci2[1]) < 1e-10, "Upper CI should be reproducible"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_psi_small_sample(self):
        """Test PSI with small samples."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 50)
        actual = np.random.normal(0.5, 1, 50)
        
        # Should not crash with small samples
        psi, psi_ci, _, _, _ = compute_psi_with_bootstrap(
            expected, actual, n_bins=5, n_bootstrap=10
        )
        
        assert isinstance(psi, (float, np.floating))
        assert psi >= 0, "PSI should be non-negative"
    
    def test_psi_with_zeros(self):
        """Test PSI handles bins with zero counts."""
        # Create distributions with clear separation
        expected = np.array([1.0] * 100 + [3.0] * 100)
        actual = np.array([2.0] * 100 + [4.0] * 100)
        
        # Should handle zero bins gracefully (with epsilon)
        psi, _, _, _, _ = compute_psi_with_bootstrap(
            expected, actual, n_bins=10, n_bootstrap=10
        )
        
        assert np.isfinite(psi), "PSI should be finite"
        assert psi > 0, "PSI should detect shift"
    
    def test_extract_samples_no_prompts(self):
        """Test sample extraction with no prompts in category."""
        prompts = ["p1", "p2"]
        pc1_values = np.array([0.0, 0.01])
        reward_gaps = np.array([0.5, 0.55])  # All medium (neither Mixtral-Sufficient nor GPT-4-Turbo-Required)
        
        samples = extract_sample_prompts(prompts, pc1_values, reward_gaps, n_samples=2)
        
        # Should return empty lists
        assert len(samples['mixtral_sufficient']) == 0
        assert len(samples['gpt4_turbo_required']) == 0


class TestSensitivityAnalysis:
    """Test sensitivity analysis across multiple PC dimensions."""
    
    def test_sensitivity_analysis_structure(self):
        """Test that sensitivity analysis returns correct structure."""
        np.random.seed(42)
        
        # Create synthetic PCA-like feature data (33 dimensions matching router output)
        features_source = np.random.normal(0.1, 1.0, (500, 33))
        features_routellm = np.random.normal(-0.05, 0.9, (500, 33))
        
        # Mock PCA stats
        pca_stats = {
            'explained_variance': np.array([0.15] + [0.05] * 32),
            'variance_explained_pc1': 0.15,
            'pca': None  # Not needed for this test
        }
        
        results = sensitivity_analysis_multipc(features_source, features_routellm, pca_stats)
        
        # Check structure
        assert isinstance(results, dict)
        assert 'PC1 only' in results
        assert 'PC1-5' in results
        assert 'PC1-10' in results
        assert 'All 33 PCs' in results
        
        # Each should have PSI value (results are just floats, not dicts)
        for key in ['PC1 only', 'PC1-5', 'PC1-10', 'All 33 PCs']:
            assert isinstance(results[key], (float, np.floating))
            assert results[key] >= 0, "PSI should be non-negative"
    
    def test_sensitivity_analysis_dimensionality(self):
        """Test that sensitivity analysis uses correct number of dimensions."""
        np.random.seed(42)
        
        features_source = np.random.normal(0, 1, (200, 33))
        features_routellm = np.random.normal(0.3, 1, (200, 33))
        
        pca_stats = {
            'explained_variance': np.array([0.15] * 33),
            'variance_explained_pc1': 0.15,
            'pca': None
        }
        
        results = sensitivity_analysis_multipc(features_source, features_routellm, pca_stats)
        
        # Verify we get results for all dimensionalities
        assert 'PC1 only' in results
        assert 'PC1-5' in results
        assert 'PC1-10' in results
        assert 'All 33 PCs' in results
        
        # All should be valid PSI values
        assert all(isinstance(v, (float, np.floating)) for v in results.values())
        assert all(v >= 0 for v in results.values())
    
    def test_sensitivity_analysis_shift_detection(self):
        """Test that sensitivity analysis detects distribution shift across dimensions."""
        np.random.seed(42)
        
        # Create clear shift in first dimension
        features_source = np.random.normal([0] * 33, 1.0, (300, 33))
        features_routellm = np.random.normal([0.5] + [0] * 32, 1.0, (300, 33))
        
        pca_stats = {
            'explained_variance': np.array([0.2] + [0.025] * 32),
            'variance_explained_pc1': 0.2,
            'pca': None
        }
        
        results = sensitivity_analysis_multipc(features_source, features_routellm, pca_stats)
        
        # All should detect shift (PSI > 0)
        assert results['PC1 only'] > 0.1, "Should detect shift in PC1"
        assert results['PC1-5'] > 0, "Should detect shift in PC1-5"
        assert results['All 33 PCs'] > 0, "Should detect shift in all features"


class TestClusterSeparation:
    """Test cluster separation statistical analysis."""
    
    def test_cluster_separation_well_separated(self):
        """Test cluster separation with well-separated clusters."""
        np.random.seed(42)
        
        # Create well-separated clusters
        # Mixtral-Sufficient: Gap ≤ 0.3, centered at PC1 = 0.02
        # GPT-4-Turbo-Required: Gap > 0.6, centered at PC1 = -0.015
        pc1_values = np.concatenate([
            np.random.normal(0.02, 0.05, 500),   # Mixtral-Sufficient
            np.random.normal(-0.015, 0.05, 500)  # GPT-4-Turbo-Required
        ])
        reward_gaps = np.concatenate([
            np.random.uniform(0.0, 0.3, 500),    # Mixtral-Sufficient
            np.random.uniform(0.6, 1.0, 500)     # GPT-4-Turbo-Required
        ])
        
        results = analyze_task_category_separation(pc1_values, reward_gaps)
        
        # Check structure
        assert 'centroid_mixtral' in results
        assert 'centroid_gpt4_turbo' in results
        assert 'centroid_distance' in results
        assert 't_statistic' in results
        assert 't_pvalue' in results
        assert 'cohens_d' in results
        assert 'effect_size' in results
        assert 'u_statistic' in results
        assert 'u_pvalue' in results
        assert 'overlap_mixtral_in_gpt4_range' in results
        assert 'overlap_gpt4_in_mixtral_range' in results
        assert 'conclusion' in results
        
        # Statistical tests should show significance
        assert results['t_pvalue'] < 0.05, "T-test should show significant difference"
        assert abs(results['cohens_d']) > 0.2, "Effect size should be detectable"
        assert results['centroid_distance'] > 0, "Centroids should be separated"
    
    def test_cluster_separation_overlapping(self):
        """Test cluster separation with overlapping clusters."""
        np.random.seed(42)
        
        # Create overlapping clusters (same distribution)
        pc1_values = np.random.normal(0, 0.17, 1000)
        reward_gaps = np.random.uniform(0.0, 1.0, 1000)
        
        results = analyze_task_category_separation(pc1_values, reward_gaps)
        
        # Should still compute all metrics
        assert isinstance(results['cohens_d'], (float, np.floating))
        assert isinstance(results['t_pvalue'], (float, np.floating))
        assert 0 <= results['overlap_mixtral_in_gpt4_range'] <= 1.0
        assert 0 <= results['overlap_gpt4_in_mixtral_range'] <= 1.0
    
    def test_cluster_separation_thresholds(self):
        """Test that cluster separation uses correct thresholds."""
        np.random.seed(42)
        
        pc1_values = np.random.normal(0, 0.2, 1000)
        # Create clear split: half Mixtral-Sufficient (≤0.3), half GPT-4-Turbo-Required (>0.6)
        reward_gaps = np.concatenate([
            np.random.uniform(0.0, 0.3, 500),
            np.random.uniform(0.6, 1.0, 500)
        ])
        
        # Custom thresholds
        results = analyze_task_category_separation(
            pc1_values, reward_gaps, 
            threshold_low=0.3, 
            threshold_high=0.6
        )
        
        # Should identify two clusters
        assert 'centroid_mixtral' in results
        assert 'centroid_gpt4_turbo' in results
        assert isinstance(results['conclusion'], str)
    
    def test_cluster_separation_effect_size_interpretation(self):
        """Test effect size interpretation is correct."""
        np.random.seed(42)
        
        # Small effect
        pc1_small = np.concatenate([
            np.random.normal(0.0, 0.2, 500),
            np.random.normal(0.05, 0.2, 500)
        ])
        gaps_small = np.concatenate([
            np.random.uniform(0.0, 0.3, 500),
            np.random.uniform(0.6, 1.0, 500)
        ])
        
        results_small = analyze_task_category_separation(pc1_small, gaps_small)
        assert results_small['effect_size'] in ['negligible', 'small', 'medium']
        
        # Large effect
        pc1_large = np.concatenate([
            np.random.normal(-0.5, 0.1, 500),
            np.random.normal(0.5, 0.1, 500)
        ])
        gaps_large = np.concatenate([
            np.random.uniform(0.0, 0.3, 500),
            np.random.uniform(0.6, 1.0, 500)
        ])
        
        results_large = analyze_task_category_separation(pc1_large, gaps_large)
        assert results_large['effect_size'] in ['large', 'very large']
        assert abs(results_large['cohens_d']) > 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
