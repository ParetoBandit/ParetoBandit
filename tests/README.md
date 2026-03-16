# ParetoBandit Test Suite

Comprehensive unit and integration tests for the ParetoBandit router.

## Running Tests

### Run All Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=paretobandit --cov-report=html
```

### Run Specific Test Files

```bash
# Integration tests (end-to-end)
python -m pytest tests/test_integration.py -v

# Feedback loop tests
python -m pytest tests/test_feedback_loop.py -v

# Prior management tests
python -m pytest tests/test_prior_management.py -v

# Optimization profile tests
python -m pytest tests/test_optimization_profiles.py -v

# Validation methods tests
python -m pytest tests/test_validation_methods.py -v

# Figure 1 validation tests (integration)
python -m pytest tests/test_figure1_validation.py -v
```

---

## Test Coverage

### Integration Tests (`test_integration.py`) — 30 tests

End-to-end tests for the complete routing system:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestCoreImports` | 7 | All core components importable |
| `TestRouterInitialization` | 3 | Cold start, bundled priors, expert priors |
| `TestRoutingDecisions` | 4 | Route, profiles, rankings, candidates |
| `TestFeedbackAndLearning` | 3 | Feedback updates, positive/negative rewards |
| `TestPriorPersistence` | 2 | Save/load state, shippable priors |
| `TestDynamicModelManagement` | 3 | Add/remove models at runtime |
| `TestEndToEndWorkflow` | 3 | Full workflow, specialist discovery, cost-quality |
| `TestEdgeCases` | 5 | Empty/long/unicode prompts, unknown request, single model |

### Feedback Loop Tests (`test_feedback_loop.py`) — 39 tests

Tests for the asynchronous feedback and learning system:

- **Reward Signal Processing**: Reward validation, normalization, clipping
- **Bandit Updates**: Rank-one matrix updates, A_inv recomputation
- **Feedback Types**: Human feedback (thumbs up/down), hard truth (code execution)
- **Integration**: Feedback affects future routing

### Prior Management Tests (`test_prior_management.py`) — 26 tests

Tests for prior loading, saving, and management:

- **PriorManager**: Load/save priors, merge strategies
- **Expert Priors**: Disjoint format (A_stack, b_stack)
- **Shared Priors**: Legacy format (A_shared)
- **Prior Strength**: λ_boost scaling, UCB confidence
- **Dynamic Models**: Add/remove models at runtime

### Optimization Profile Tests (`test_optimization_profiles.py`) — 32 tests

Tests for cost/quality trade-off profiles:

- **Profile Definitions**: balanced, quality_first, cost_saver, low_latency
- **Exploration Rates**: static, safe, balanced, aggressive
- **Utility Calculation**: Quality - λ_cost × Cost - λ_latency × Latency

### Model Manager Tests (`test_model_manager.py`) — 11 tests

Tests for model cache management:

- **Cache Operations**: Load, save, list, remove models
- **TTFT Estimates**: Initialize statistical fields from point estimates
- **Registry Compatibility**: Verify cache works with `build_registry_from_models_cache`
- **API Requirements**: Verify proper error handling for missing API key

### Validation Methods Tests (`test_validation_methods.py`) — 30+ tests

Unit tests for statistical validation functions used in experiments:

- **Statistical Validation**: Mann-Whitney U test, Welch's t-test, Cohen's d, confidence intervals
- **Threshold Evaluation**: Grid search, silhouette scores, Davies-Bouldin index, cluster balance
- **Cluster Quality**: Multi-dimensional cluster quality metrics, separation ratios
- **High-D Analysis**: Separation analysis in original embedding spaces
- **Data Quality**: Duplicate detection (exact and near), diversity scores
- **Reproducibility**: Tests for deterministic results
- **Robustness**: Edge cases (NaN, outliers, imbalance, small samples)

### Figure 1 Validation Tests (`test_figure1_validation.py`) — 25+ tests

Integration tests for Figure 1 validation pipeline with synthetic data:

- **Threshold Validation**: Grid search optimization, sensitivity analysis, multi-metric consistency
- **High-D Structure**: Validation across 2D, 32D, 384D spaces, predictive power tests
- **Data Quality**: Duplicate detection, near-duplicate analysis, diversity metrics
- **Statistical Pipeline**: End-to-end validation workflow, noise robustness
- **Edge Cases**: Small samples, outliers, imbalanced clusters
- **Performance**: Large dataset handling (10K+ samples)
- **Alignment Tax Simulation**: Synthetic data mimicking the real phenomenon

---

## Current Status

```
======================= 190+ tests (estimated) ========================
```

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_integration.py` | 30 | ✅ All passing |
| `test_feedback_loop.py` | 39 | ✅ All passing |
| `test_prior_management.py` | 26 | ✅ All passing |
| `test_optimization_profiles.py` | 32 | ✅ All passing |
| `test_model_manager.py` | 11 | ✅ All passing |
| `test_validation_methods.py` | 30+ | 🆕 **New** - Statistical validation |
| `test_figure1_validation.py` | 25+ | 🆕 **New** - Integration tests |

---

## Test Design Principles

1. **Fast**: Tests run in ~2 minutes total
2. **Isolated**: Tests don't depend on each other
3. **Deterministic**: Same input → same output (seeded random)
4. **Comprehensive**: Core functionality + edge cases
5. **Real Components**: Tests use actual `BanditRouter`, not mocks

---

## Key Fixtures

### `sample_registry`
```python
{
    "openai/gpt-4o": {"cost_per_1m_tokens": 5.0, "latency_ms": 800},
    "openai/gpt-4o-mini": {"cost_per_1m_tokens": 0.15, "latency_ms": 400},
    "anthropic/claude-3.5-sonnet": {"cost_per_1m_tokens": 3.0, "latency_ms": 600},
    "amazon/nova-lite-v1": {"cost_per_1m_tokens": 0.10, "latency_ms": 300},
    "meta-llama/llama-3-70b-instruct": {"cost_per_1m_tokens": 0.88, "latency_ms": 500},
}
```

### `temp_dir`
Temporary directory for test files (auto-cleaned).

---

## Adding New Tests

### Template

```python
import pytest
from paretobandit.core import BanditRouter

class TestNewFeature:
    """Tests for new feature."""

    def test_basic_functionality(self, sample_registry):
        """Test that basic feature works."""
        router = BanditRouter.create(sample_registry, priors="none")
        
        # Act
        result = router.some_method()
        
        # Assert
        assert result is not None
```

### Guidelines

- Use `sample_registry` fixture for model registry
- Use `temp_dir` fixture for file operations
- Test both success and error cases
- Include docstrings describing what's being tested

---

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/paretobandit
pip install -e .
```

### Slow Tests

```bash
# Run only fast tests (skip integration)
python -m pytest tests/test_optimization_profiles.py -v
```

### Coverage Report

```bash
python -m pytest tests/ --cov=paretobandit --cov-report=html
open htmlcov/index.html
```

---

For questions, see the main [README.md](../README.md) or open an issue.
