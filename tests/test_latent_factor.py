"""
Unit tests for the Bayesian latent factor model used in CRS and CCS computation.

These tests ensure that:
1. Data preparation functions work correctly
2. Z-score standardization handles edge cases
3. Long format conversion preserves data integrity
4. Weighted z-score method produces valid results
5. 0-100 transformation is correct
6. (Optional) Bayesian model runs and produces reasonable output
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Import the refactored module
from llm_jury.analysis.latent_factor import (
    BenchmarkConfig,
    BenchmarkSuite,
    REASONING_BENCHMARKS,
    CODING_BENCHMARKS,
    extract_benchmark_matrix,
    prepare_long_data,
    fit_latent_factor_model,
    summarize_latent_scores,
    compute_weighted_zscore,
    transform_to_0_100,
    update_models_cache,
    get_benchmark_diagnostics,
    parse_benchmark_args,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_benchmark_configs():
    """Sample benchmark configuration."""
    return {
        'bench_a': {'description': 'Benchmark A', 'invert': False, 'scale': 1},
        'bench_b': {'description': 'Benchmark B', 'invert': False, 'scale': 100},
        'bench_c': {'description': 'Benchmark C', 'invert': False, 'scale': 1},
    }


@pytest.fixture
def sample_models():
    """Sample model data with benchmark scores."""
    return [
        {'name': 'Model A', 'bench_a': 80, 'bench_b': 0.7, 'bench_c': 90},
        {'name': 'Model B', 'bench_a': 70, 'bench_b': 0.6, 'bench_c': 85},
        {'name': 'Model C', 'bench_a': 90, 'bench_b': 0.8, 'bench_c': 95},
        {'name': 'Model D', 'bench_a': 60, 'bench_b': 0.5, 'bench_c': 75},
        {'name': 'Model E', 'bench_a': 85, 'bench_b': None, 'bench_c': 88},  # Missing value
    ]


@pytest.fixture
def sample_models_with_missing():
    """Sample models with more missing values."""
    return [
        {'name': 'Model A', 'bench_a': 80, 'bench_b': 0.7, 'bench_c': None},
        {'name': 'Model B', 'bench_a': None, 'bench_b': 0.6, 'bench_c': 85},
        {'name': 'Model C', 'bench_a': 90, 'bench_b': None, 'bench_c': 95},
        {'name': 'Model D', 'bench_a': 60, 'bench_b': 0.5, 'bench_c': 75},
    ]


# ============================================================================
# Data Extraction Tests
# ============================================================================

class TestExtractBenchmarkMatrix:
    """Tests for extract_benchmark_matrix function."""
    
    def test_basic_extraction(self, sample_models, sample_benchmark_configs):
        """Test basic score extraction."""
        df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        assert len(model_names) == 5  # All models have at least 2 benchmarks
        assert len(benchmark_names) == 3
        assert df_scores.shape == (5, 3)
    
    def test_scaling(self, sample_models, sample_benchmark_configs):
        """Test that scaling is applied correctly."""
        df_scores, _, _, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        # bench_b should be scaled by 100 (0.7 -> 70)
        assert df_scores.loc[0, 'bench_b'] == 70
        assert df_scores.loc[1, 'bench_b'] == 60
    
    def test_min_benchmarks_filter(self, sample_models_with_missing, sample_benchmark_configs):
        """Test that min_benchmarks filter works."""
        df_scores, _, model_names, _ = extract_benchmark_matrix(
            sample_models_with_missing, sample_benchmark_configs, min_benchmarks=2
        )
        
        # All models have at least 2 benchmarks
        assert len(model_names) == 4
        
        # With min_benchmarks=3, only Model D should remain
        df_scores, _, model_names, _ = extract_benchmark_matrix(
            sample_models_with_missing, sample_benchmark_configs, min_benchmarks=3
        )
        assert len(model_names) == 1
        assert model_names[0] == 'Model D'
    
    def test_z_score_standardization(self, sample_models, sample_benchmark_configs):
        """Test z-score standardization."""
        _, df_z, _, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        # Z-scores should have mean ~0 and std ~1 (for non-missing values)
        for col in df_z.columns:
            valid = df_z[col].dropna()
            if len(valid) > 1:
                assert abs(valid.mean()) < 0.01, f"{col} mean should be ~0"
                assert abs(valid.std() - 1.0) < 0.01, f"{col} std should be ~1"
    
    def test_missing_values_preserved(self, sample_models, sample_benchmark_configs):
        """Test that missing values are preserved as NaN."""
        df_scores, df_z, _, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        # Model E (index 4) has missing bench_b
        assert pd.isna(df_scores.loc[4, 'bench_b'])
        assert pd.isna(df_z.loc[4, 'bench_b'])


# ============================================================================
# Long Format Conversion Tests
# ============================================================================

class TestPrepareLongData:
    """Tests for prepare_long_data function."""
    
    def test_basic_conversion(self, sample_models, sample_benchmark_configs):
        """Test basic long format conversion."""
        df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        z_obs, idx_model, idx_bench, n_models, n_benchmarks = prepare_long_data(
            df_z, model_names, benchmark_names
        )
        
        assert n_models == 5
        assert n_benchmarks == 3
        # 5 models * 3 benchmarks - 1 missing = 14 observations
        assert len(z_obs) == 14
        assert len(idx_model) == 14
        assert len(idx_bench) == 14
    
    def test_index_ranges(self, sample_models, sample_benchmark_configs):
        """Test that indices are within valid ranges."""
        df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        z_obs, idx_model, idx_bench, n_models, n_benchmarks = prepare_long_data(
            df_z, model_names, benchmark_names
        )
        
        assert idx_model.min() >= 0
        assert idx_model.max() < n_models
        assert idx_bench.min() >= 0
        assert idx_bench.max() < n_benchmarks
    
    def test_no_nan_in_output(self, sample_models, sample_benchmark_configs):
        """Test that output contains no NaN values."""
        df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        z_obs, _, _, _, _ = prepare_long_data(df_z, model_names, benchmark_names)
        
        assert not np.any(np.isnan(z_obs))
    
    def test_data_integrity(self, sample_models, sample_benchmark_configs):
        """Test that converted data matches original."""
        df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        z_obs, idx_model, idx_bench, _, _ = prepare_long_data(
            df_z, model_names, benchmark_names
        )
        
        # Check that each observation matches the original
        z_values = df_z.to_numpy()
        for i, (z, m_idx, b_idx) in enumerate(zip(z_obs, idx_model, idx_bench)):
            assert z == z_values[m_idx, b_idx], f"Mismatch at observation {i}"


# ============================================================================
# Weighted Z-Score Tests
# ============================================================================

class TestWeightedZScore:
    """Tests for weighted z-score computation."""
    
    def test_basic_computation(self, sample_models, sample_benchmark_configs):
        """Test basic weighted z-score computation."""
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        weights = {'bench_a': 0.4, 'bench_b': 0.3, 'bench_c': 0.3}
        df_result = compute_weighted_zscore(df_z, model_names, weights, score_name='score')
        
        assert len(df_result) == 5
        assert 'score_mean' in df_result.columns
        assert 'score_sd' in df_result.columns
    
    def test_equal_weights(self, sample_models, sample_benchmark_configs):
        """Test with equal weights - should be simple average."""
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        weights = {'bench_a': 1.0, 'bench_b': 1.0, 'bench_c': 1.0}
        df_result = compute_weighted_zscore(df_z, model_names, weights, score_name='score')
        
        # For models with all benchmarks, weighted mean should equal simple mean
        for i, row in df_result.iterrows():
            if row['n_benchmarks'] == 3:
                z_row = df_z.iloc[i]
                expected = z_row.mean()
                assert abs(row['score_mean'] - expected) < 0.01
    
    def test_ordering_preserved(self, sample_models, sample_benchmark_configs):
        """Test that higher raw scores lead to higher composite scores."""
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        weights = {'bench_a': 0.33, 'bench_b': 0.33, 'bench_c': 0.34}
        df_result = compute_weighted_zscore(df_z, model_names, weights, score_name='score')
        
        # Model C should be highest (90, 80, 95)
        # Model D should be lowest (60, 50, 75)
        model_c_idx = model_names.index('Model C')
        model_d_idx = model_names.index('Model D')
        
        assert df_result.iloc[model_c_idx]['score_mean'] > df_result.iloc[model_d_idx]['score_mean']


# ============================================================================
# 0-100 Transformation Tests
# ============================================================================

class TestTransformTo0100:
    """Tests for 0-100 transformation."""
    
    def test_range(self, sample_models, sample_benchmark_configs):
        """Test that transformed scores are in 0-100 range."""
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        weights = {'bench_a': 0.33, 'bench_b': 0.33, 'bench_c': 0.34}
        df_result = compute_weighted_zscore(df_z, model_names, weights, score_name='score')
        df_result = transform_to_0_100(df_result, mean_col='score_mean', output_col='score_100')
        
        valid = df_result['score_100'].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100
    
    def test_min_max(self, sample_models, sample_benchmark_configs):
        """Test that min becomes 0 and max becomes 100."""
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        weights = {'bench_a': 0.33, 'bench_b': 0.33, 'bench_c': 0.34}
        df_result = compute_weighted_zscore(df_z, model_names, weights, score_name='score')
        df_result = transform_to_0_100(df_result, mean_col='score_mean', output_col='score_100')
        
        valid = df_result['score_100'].dropna()
        assert abs(valid.min() - 0) < 0.01
        assert abs(valid.max() - 100) < 0.01
    
    def test_ordering_preserved(self, sample_models, sample_benchmark_configs):
        """Test that ordering is preserved after transformation."""
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            sample_models, sample_benchmark_configs
        )
        
        weights = {'bench_a': 0.33, 'bench_b': 0.33, 'bench_c': 0.34}
        df_result = compute_weighted_zscore(df_z, model_names, weights, score_name='score')
        
        original_order = df_result.sort_values('score_mean', ascending=False)['model'].tolist()
        
        df_result = transform_to_0_100(df_result, mean_col='score_mean', output_col='score_100')
        transformed_order = df_result.sort_values('score_100', ascending=False)['model'].tolist()
        
        assert original_order == transformed_order


# ============================================================================
# Bayesian Model Tests (Optional - slow)
# ============================================================================

@pytest.mark.slow
class TestBayesianModel:
    """
    Tests for the Bayesian latent factor model.
    These are slow and require pymc/arviz.
    Run with: pytest -m slow
    """
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with known latent factor."""
        np.random.seed(42)
        n_models = 20
        n_benchmarks = 3
        
        # True latent factors
        theta_true = np.random.normal(0, 1, n_models)
        
        # True loadings
        lambda_true = np.array([0.8, 0.9, 1.0])
        
        # Generate observations with noise
        z_matrix = np.zeros((n_models, n_benchmarks))
        for i in range(n_models):
            for j in range(n_benchmarks):
                z_matrix[i, j] = lambda_true[j] * theta_true[i] + np.random.normal(0, 0.3)
        
        # Standardize columns
        for j in range(n_benchmarks):
            z_matrix[:, j] = (z_matrix[:, j] - z_matrix[:, j].mean()) / z_matrix[:, j].std()
        
        return z_matrix, theta_true, lambda_true
    
    def test_model_runs(self, synthetic_data):
        """Test that the Bayesian model runs without error."""
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            pytest.skip("pymc/arviz not installed")
        
        z_matrix, theta_true, lambda_true = synthetic_data
        n_models, n_benchmarks = z_matrix.shape
        
        # Convert to long format
        z_obs, idx_model, idx_bench = [], [], []
        for i in range(n_models):
            for j in range(n_benchmarks):
                z_obs.append(z_matrix[i, j])
                idx_model.append(i)
                idx_bench.append(j)
        
        z_obs = np.array(z_obs)
        idx_model = np.array(idx_model)
        idx_bench = np.array(idx_bench)
        
        # Fit model with minimal iterations for speed
        with pm.Model() as model:
            theta = pm.Normal('theta', mu=0, sigma=1, shape=n_models)
            alpha = pm.Normal('alpha', mu=0, sigma=1, shape=n_benchmarks)
            lambda_ = pm.HalfNormal('lambda', sigma=0.7, shape=n_benchmarks)
            sigma = pm.HalfNormal('sigma', sigma=1.0, shape=n_benchmarks)
            
            mu = alpha[idx_bench] + lambda_[idx_bench] * theta[idx_model]
            z = pm.Normal('z', mu=mu, sigma=sigma[idx_bench], observed=z_obs)
            
            idata = pm.sample(100, tune=100, chains=2, random_seed=42, progressbar=False)
        
        # Check that we got results
        assert 'theta' in idata.posterior
        assert 'lambda' in idata.posterior
        assert idata.posterior['theta'].shape[-1] == n_models
    
    def test_recovers_ordering(self, synthetic_data):
        """Test that model recovers approximate ordering of latent factors."""
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            pytest.skip("pymc/arviz not installed")
        
        z_matrix, theta_true, lambda_true = synthetic_data
        n_models, n_benchmarks = z_matrix.shape
        
        # Convert to long format
        z_obs, idx_model, idx_bench = [], [], []
        for i in range(n_models):
            for j in range(n_benchmarks):
                z_obs.append(z_matrix[i, j])
                idx_model.append(i)
                idx_bench.append(j)
        
        z_obs = np.array(z_obs)
        idx_model = np.array(idx_model)
        idx_bench = np.array(idx_bench)
        
        # Fit model
        with pm.Model() as model:
            theta = pm.Normal('theta', mu=0, sigma=1, shape=n_models)
            alpha = pm.Normal('alpha', mu=0, sigma=1, shape=n_benchmarks)
            lambda_ = pm.HalfNormal('lambda', sigma=0.7, shape=n_benchmarks)
            sigma = pm.HalfNormal('sigma', sigma=1.0, shape=n_benchmarks)
            
            mu = alpha[idx_bench] + lambda_[idx_bench] * theta[idx_model]
            z = pm.Normal('z', mu=mu, sigma=sigma[idx_bench], observed=z_obs)
            
            idata = pm.sample(500, tune=500, chains=2, random_seed=42, progressbar=False)
        
        # Get posterior means
        theta_estimated = idata.posterior['theta'].mean(dim=['chain', 'draw']).values
        
        # Check correlation between true and estimated (should be high)
        correlation = np.corrcoef(theta_true, theta_estimated)[0, 1]
        assert abs(correlation) > 0.8, f"Correlation {correlation} too low"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_models(self, sample_benchmark_configs):
        """Test with empty model list."""
        df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
            [], sample_benchmark_configs
        )
        
        assert len(model_names) == 0
        assert df_scores.empty
    
    def test_all_missing_benchmark(self):
        """Test when one benchmark is entirely missing."""
        models = [
            {'name': 'A', 'bench_a': 80, 'bench_b': None},
            {'name': 'B', 'bench_a': 70, 'bench_b': None},
        ]
        configs = {
            'bench_a': {'scale': 1, 'invert': False},
            'bench_b': {'scale': 1, 'invert': False},
        }
        
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            models, configs, min_benchmarks=1
        )
        
        # bench_b should be all NaN
        assert df_scores['bench_b'].isna().all()
        assert df_z['bench_b'].isna().all()
    
    def test_single_model(self):
        """Test with single model."""
        models = [{'name': 'Only', 'bench_a': 80, 'bench_b': 70}]
        configs = {
            'bench_a': {'scale': 1, 'invert': False},
            'bench_b': {'scale': 1, 'invert': False},
        }
        
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            models, configs, min_benchmarks=2
        )
        
        assert len(model_names) == 1
        # Z-scores can't be computed with single model (std=0)
        # Should handle gracefully
    
    def test_identical_scores(self):
        """Test when all models have identical scores."""
        models = [
            {'name': 'A', 'bench_a': 80, 'bench_b': 70},
            {'name': 'B', 'bench_a': 80, 'bench_b': 70},
            {'name': 'C', 'bench_a': 80, 'bench_b': 70},
        ]
        configs = {
            'bench_a': {'scale': 1, 'invert': False},
            'bench_b': {'scale': 1, 'invert': False},
        }
        
        df_scores, df_z, model_names, _ = extract_benchmark_matrix(
            models, configs, min_benchmarks=2
        )
        
        # With identical scores, std=0, z-scores should be NaN
        assert df_z['bench_a'].isna().all() or (df_z['bench_a'] == 0).all()


# ============================================================================
# Tests for update_models_cache
# ============================================================================

class TestUpdateModelsCache:
    """Tests for the cache update function."""
    
    def test_basic_update(self):
        """Test basic cache update."""
        cache_data = {
            'models': [
                {'name': 'Model A', 'some_field': 1},
                {'name': 'Model B', 'some_field': 2},
            ]
        }
        
        df_scores = pd.DataFrame({
            'model': ['Model A', 'Model B'],
            'ccs_mean': [0.5, -0.3],
            'ccs_sd': [0.1, 0.2],
            'ccs_100': [75.0, 25.0],
        })
        
        updated, count = update_models_cache(cache_data, df_scores, 'ccs')
        
        assert count == 2
        assert updated['models'][0]['ccs'] == 0.5
        assert updated['models'][1]['ccs'] == -0.3
        assert updated['models'][0]['ccs_method'] == 'bayesian'
    
    def test_partial_update(self):
        """Test that only matching models are updated."""
        cache_data = {
            'models': [
                {'name': 'Model A'},
                {'name': 'Model B'},
                {'name': 'Model C'},
            ]
        }
        
        df_scores = pd.DataFrame({
            'model': ['Model A'],
            'ccs_mean': [0.5],
            'ccs_sd': [0.1],
        })
        
        updated, count = update_models_cache(cache_data, df_scores, 'ccs')
        
        assert count == 1
        assert 'ccs' in updated['models'][0]
        assert 'ccs' not in updated['models'][1]


# ============================================================================
# Tests for BenchmarkConfig and BenchmarkSuite
# ============================================================================

class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""
    
    def test_creation(self):
        """Test basic creation."""
        cfg = BenchmarkConfig(
            name="test_bench",
            description="Test benchmark",
            scale=100,
            invert=False,
            weight=0.5,
        )
        assert cfg.name == "test_bench"
        assert cfg.scale == 100
        assert cfg.weight == 0.5
    
    def test_to_dict(self):
        """Test serialization to dict."""
        cfg = BenchmarkConfig(
            name="test_bench",
            description="Test benchmark",
            scale=100,
            invert=True,
            weight=0.5,
        )
        d = cfg.to_dict()
        assert d['description'] == "Test benchmark"
        assert d['scale'] == 100
        assert d['invert'] == True
        assert d['weight'] == 0.5
    
    def test_from_dict(self):
        """Test creation from dict."""
        d = {
            'description': 'From dict',
            'scale': 50,
            'invert': True,
            'weight': 0.3,
        }
        cfg = BenchmarkConfig.from_dict("my_bench", d)
        assert cfg.name == "my_bench"
        assert cfg.description == "From dict"
        assert cfg.scale == 50
        assert cfg.weight == 0.3


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""
    
    def test_creation(self):
        """Test suite creation."""
        suite = BenchmarkSuite(
            name="test",
            description="Test suite",
            score_prefix="tst",
        )
        assert suite.name == "test"
        assert suite.score_prefix == "tst"
        assert len(suite.benchmarks) == 0
    
    def test_add_benchmark(self):
        """Test adding benchmarks."""
        suite = BenchmarkSuite(name="test", description="Test")
        suite.add_benchmark("bench_a", "Benchmark A", scale=1, weight=0.5)
        suite.add_benchmark("bench_b", "Benchmark B", scale=100, weight=0.5)
        
        assert len(suite.benchmarks) == 2
        assert "bench_a" in suite.benchmarks
        assert suite.benchmarks["bench_a"].scale == 1
    
    def test_remove_benchmark(self):
        """Test removing benchmarks."""
        suite = BenchmarkSuite(name="test", description="Test")
        suite.add_benchmark("bench_a", "A")
        suite.add_benchmark("bench_b", "B")
        suite.remove_benchmark("bench_a")
        
        assert len(suite.benchmarks) == 1
        assert "bench_a" not in suite.benchmarks
    
    def test_get_weights(self):
        """Test weight normalization."""
        suite = BenchmarkSuite(name="test", description="Test")
        suite.add_benchmark("a", "A", weight=2)
        suite.add_benchmark("b", "B", weight=3)
        
        weights = suite.get_weights()
        assert abs(weights["a"] - 0.4) < 0.01  # 2/5
        assert abs(weights["b"] - 0.6) < 0.01  # 3/5
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_get_configs(self):
        """Test getting configs as dict of dicts."""
        suite = BenchmarkSuite(name="test", description="Test")
        suite.add_benchmark("a", "A", scale=100, invert=True)
        
        configs = suite.get_configs()
        assert "a" in configs
        assert configs["a"]["scale"] == 100
        assert configs["a"]["invert"] == True
    
    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        suite = BenchmarkSuite(
            name="test",
            description="Test suite",
            score_prefix="tst",
        )
        suite.add_benchmark("a", "A", scale=1, weight=0.3)
        suite.add_benchmark("b", "B", scale=100, weight=0.7)
        
        d = suite.to_dict()
        suite2 = BenchmarkSuite.from_dict(d)
        
        assert suite2.name == suite.name
        assert suite2.score_prefix == suite.score_prefix
        assert len(suite2.benchmarks) == 2
        assert suite2.benchmarks["a"].scale == 1
    
    def test_to_json_and_from_json(self, tmp_path):
        """Test JSON file I/O."""
        suite = BenchmarkSuite(
            name="test",
            description="Test suite",
            score_prefix="tst",
        )
        suite.add_benchmark("a", "A", scale=1)
        
        json_path = tmp_path / "test_config.json"
        suite.to_json(json_path)
        
        suite2 = BenchmarkSuite.from_json(json_path)
        assert suite2.name == "test"
        assert "a" in suite2.benchmarks


class TestDefaultSuites:
    """Tests for default benchmark suites."""
    
    def test_reasoning_suite_exists(self):
        """Test that reasoning suite is defined."""
        assert REASONING_BENCHMARKS is not None
        assert REASONING_BENCHMARKS.score_prefix == "crs"
        assert len(REASONING_BENCHMARKS.benchmarks) > 0
    
    def test_coding_suite_exists(self):
        """Test that coding suite is defined."""
        assert CODING_BENCHMARKS is not None
        assert CODING_BENCHMARKS.score_prefix == "ccs"
        assert len(CODING_BENCHMARKS.benchmarks) > 0
    
    def test_reasoning_benchmarks(self):
        """Test reasoning suite has expected benchmarks."""
        names = list(REASONING_BENCHMARKS.benchmarks.keys())
        assert "math_500" in names
        assert "gpqa" in names
    
    def test_coding_benchmarks(self):
        """Test coding suite has expected benchmarks."""
        names = list(CODING_BENCHMARKS.benchmarks.keys())
        assert "humaneval_score" in names
        assert "livecodebench" in names
        assert "scicode" in names


class TestParseBenchmarkArgs:
    """Tests for parse_benchmark_args function."""
    
    def test_default_suite(self):
        """Test using default suite when no args."""
        suite = parse_benchmark_args(
            benchmark_args=None,
            config_path=None,
            default_suite=CODING_BENCHMARKS,
        )
        assert suite.score_prefix == "ccs"
        # CCS has 4 primary + 1 auxiliary benchmark
        assert len(suite.benchmarks) == 5
    
    def test_custom_benchmarks(self):
        """Test parsing custom benchmark specs."""
        suite = parse_benchmark_args(
            benchmark_args=["bench_a:100:0.5", "bench_b:1:0.5"],
            config_path=None,
            default_suite=CODING_BENCHMARKS,
        )
        assert len(suite.benchmarks) == 2
        assert "bench_a" in suite.benchmarks
        assert suite.benchmarks["bench_a"].scale == 100
        assert suite.benchmarks["bench_a"].weight == 0.5
    
    def test_partial_specs(self):
        """Test partial benchmark specs (field only)."""
        suite = parse_benchmark_args(
            benchmark_args=["humaneval_score"],
            config_path=None,
            default_suite=CODING_BENCHMARKS,
        )
        # Should inherit defaults from CODING_BENCHMARKS
        assert "humaneval_score" in suite.benchmarks
        assert suite.benchmarks["humaneval_score"].scale == 1  # Default from suite
    
    def test_from_config_file(self, tmp_path):
        """Test loading from config file."""
        config = {
            "name": "custom",
            "description": "Custom suite",
            "score_prefix": "custom",
            "benchmarks": {
                "my_bench": {"description": "My Bench", "scale": 50, "weight": 1.0}
            }
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f)
        
        suite = parse_benchmark_args(
            benchmark_args=None,
            config_path=str(config_path),
            default_suite=None,
        )
        assert suite.name == "custom"
        assert suite.score_prefix == "custom"
        assert "my_bench" in suite.benchmarks
