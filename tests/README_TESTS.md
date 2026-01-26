# BanditRouter Test Suite

## Quick Start

```bash
# Run all router tests
pytest tests/test_router_algorithms.py tests/test_bandit_router.py -v

# Run specific test class
pytest tests/test_router_algorithms.py::TestDisjointLinUCBPolicy -v

# Run with coverage report
pytest tests/test_router_algorithms.py --cov=src/bandit_gpt/router --cov-report=html

# Run only fast tests (exclude slow integration tests)
pytest tests/test_router_algorithms.py -m "not slow"
```

## Test Files

### `test_router_algorithms.py` (NEW)
Comprehensive unit tests for core algorithms in `router.py`:
- **34 tests** covering all major components
- Focus on mathematical correctness and edge cases
- Includes numerical stability tests

### `test_bandit_router.py` (EXISTING)
Integration and resilience tests:
- **10 tests** for end-to-end functionality
- Pessimistic defaults for missing metadata
- Schema corruption handling

## Test Organization

```
tests/
├── test_router_algorithms.py    # Algorithm unit tests (NEW)
│   ├── TestDisjointLinUCBPolicy  # Core bandit (10 tests)
│   ├── TestFeatureExtraction     # Utilities (3 tests)
│   ├── TestCostLatencyPenalties  # Penalty calculation (2 tests)
│   ├── TestParetoFrontier        # Pareto filtering (2 tests)
│   ├── TestSemanticTransfer      # Model admission (2 tests)
│   ├── TestCorrallingRouter      # Expert mixing (4 tests)
│   ├── TestNumericalStability    # Stability (3 tests)
│   ├── TestCostAwareLinUCBRouter # Experimental (4 tests)
│   └── TestRouterIntegration     # Integration (4 tests)
│
├── test_bandit_router.py         # Integration tests (EXISTING)
│   ├── Basic routing tests       # (5 tests)
│   └── Resilience tests          # (5 tests)
│
└── TEST_COVERAGE_SUMMARY.md      # Detailed coverage report
```

## Key Test Categories

### 1. Mathematical Correctness ✅
Tests that verify the mathematical properties of algorithms:
- LinUCB update formulas (A += xx^T, b += rx)
- Sherman-Morrison inverse updates
- UCB score calculation (mean + α × std)
- Exploration-exploitation tradeoffs

**Example:**
```python
def test_update_basic(self):
    """Verify A += xx^T and b += rx"""
    policy.update(model, context, reward)
    expected_A = A_before + np.outer(context, context)
    np.testing.assert_array_almost_equal(A_after, expected_A)
```

### 2. Edge Cases ⚠️
Tests for boundary conditions and unusual inputs:
- Zero vectors, empty contexts
- Missing or malformed metadata
- Dimension mismatches
- Numerical edge cases (very large/small values)

**Example:**
```python
def test_l2_normalize_zero_vector(self):
    """Should return zero vector, not crash"""
    x = np.array([0.0, 0.0, 0.0])
    normalized = l2_normalize(x)
    np.testing.assert_array_almost_equal(normalized, x)
```

### 3. Numerical Stability 🔢
Tests for long-running stability and precision:
- 1000+ update stress tests
- Forgetting factor decay
- Regularization floor maintenance
- Matrix conditioning checks

**Example:**
```python
def test_sherman_morrison_stability(self):
    """Run 1000 updates without numerical issues"""
    for i in range(1000):
        policy.update(model, context, reward=0.5)
        # Check A @ A_inv ≈ I
        identity_error = np.linalg.norm(A @ A_inv - I)
        assert identity_error < 1e-6
```

### 4. Integration Testing 🔗
End-to-end tests for complete workflows:
- Routing pipeline (prompt → model selection → feedback)
- Model registration and admission
- Constraint filtering
- Learning convergence

**Example:**
```python
def test_full_routing_pipeline(self):
    """Complete workflow from prompt to feedback"""
    model, log = router.route("Test prompt")
    assert log.selected_model == model
    router.process_feedback(log.request_id, reward=0.8)
    assert router.model_counts[model] == 1
```

### 5. Resilience Testing 🛡️
Tests for graceful degradation and error handling:
- Missing cost/latency metadata → Pessimistic defaults
- Malformed data types → Type validation
- Schema corruption → Service continuity

**Example:**
```python
def test_estimate_cost_pessimistic_defaults(self):
    """Missing costs should NOT return infinity"""
    cost = router._estimate_cost("model_missing_costs", 1000, 500)
    assert cost != float('inf')  # Service stays up
    assert cost > 0  # Conservative pricing
```

## Test Fixtures

### Common Fixtures

```python
@pytest.fixture
def sample_registry():
    """Minimal registry for testing"""
    return {
        "model_a": {
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 3.0,
            "hle": 0.7
        }
    }

@pytest.fixture
def warmup_priors():
    """Pre-trained matrices for testing"""
    return {
        "context_dim": 5,
        "A": {"model_a": np.eye(5) * 10.0},
        "b": {"model_a": np.random.randn(5) * 2.0}
    }
```

## Debugging Failed Tests

### Common Issues

1. **Floating Point Precision**
   ```python
   # ❌ Bad: Exact equality
   assert alpha_end == 0.1
   
   # ✅ Good: Tolerance
   assert abs(alpha_end - 0.1) < 1e-9
   ```

2. **Random Seed for Reproducibility**
   ```python
   # Set seed for tests with randomness
   np.random.seed(42)
   ```

3. **Dimension Mismatches**
   ```python
   # Check dimensions before operations
   assert context.shape == (dim,)
   assert A.shape == (dim, dim)
   ```

4. **Thread Safety**
   ```python
   # Use locks for concurrent access
   with policy._lock:
       theta = policy.A_inv[model] @ policy.b[model]
   ```

## Performance Benchmarks

### Expected Performance

| Operation | Complexity | Time (d=24) | Time (d=384) |
|-----------|-----------|-------------|--------------|
| select_arm | O(d²) | ~0.1ms | ~2ms |
| update (λ=0) | O(d²) | ~0.2ms | ~5ms |
| update (λ>0) | O(d³) | ~1ms | ~20ms |
| save_state | O(Nd²) | ~10ms | ~100ms |

### Profiling Tests

```bash
# Profile a specific test
python -m cProfile -o profile.stats -m pytest tests/test_router_algorithms.py::test_sherman_morrison_stability

# View results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -e .[test]
      - run: pytest tests/test_router_algorithms.py tests/test_bandit_router.py -v --cov
```

## Contributing New Tests

### Test Naming Convention

```python
# Format: test_<component>_<behavior>_<condition>
def test_linucb_update_with_weight():
    """Test weighted LinUCB updates for importance sampling."""
    pass

def test_pareto_filtering_dominated_models():
    """Test that dominated models are filtered from Pareto frontier."""
    pass
```

### Test Structure

```python
def test_feature_name(self):
    """
    Brief description of what is being tested.
    
    Include:
    - Setup: What initial state is needed
    - Action: What operation is performed
    - Assertion: What should be true afterward
    """
    # Arrange
    router = BanditRouter.create(...)
    
    # Act
    result = router.some_operation()
    
    # Assert
    assert result == expected_value
```

### Documentation

- Add docstrings to all test functions
- Explain non-obvious assertions
- Reference relevant algorithms/papers
- Include example values for clarity

## Known Issues

1. **test_pareto_admission_gate** - Skipped
   - Reason: Uses legacy profile names ("max_quality", etc.)
   - Fix: Update `_is_pareto_dominated()` to use new profile system
   - Tracking: TODO in test file

## Resources

- **LinUCB Paper**: Li et al., 2010 - "A Contextual-Bandit Approach to Personalized News Article Recommendation"
- **Corralling Paper**: Agarwal et al., 2017 - "Corralling a Band of Bandit Algorithms"
- **Sherman-Morrison**: Efficient matrix inverse updates
- **KDD 2026 Paper**: Hyperparameter sensitivity analysis (Appendix D/E)

## Support

For questions or issues with tests:
1. Check `TEST_COVERAGE_SUMMARY.md` for detailed coverage
2. Review algorithm documentation in `router.py`
3. Run tests with `-vv` for verbose output
4. Use `--pdb` to drop into debugger on failure

```bash
# Debug a failing test
pytest tests/test_router_algorithms.py::test_name -vv --pdb
```

