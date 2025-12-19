# Intent Classification Tests

Comprehensive unit tests for the intent classification library.

## Running Tests

### Run All Tests

```bash
# Run all tests with verbose output
python tests/run_all_tests.py

# Or run individually
python tests/test_length_debiasing.py
python tests/test_intent_classifier.py
```

### Using pytest (if installed)

```bash
pytest tests/ -v
```

## Test Coverage

### Length Debiasing Tests (`test_length_debiasing.py`)

**20 tests** covering:

- **Initialization & Configuration**
  - Valid/invalid method selection
  - Parameter passing

- **Orthogonal Projection**
  - Correlation reduction
  - R² reduction
  - Transform on new data
  - Serialization (pickle)
  - Reproducibility

- **Inverse Probability Weighting (IPW)**
  - Weight computation
  - Label requirement
  - Weight validation

- **Iterative Null-space Projection (INLP)**
  - Multiple iterations
  - Convergence criteria
  - Transform on new data

- **Adversarial Training**
  - NotImplementedError (intentional)

- **Edge Cases**
  - Single sample
  - Zero correlation
  - Perfect correlation
  - Constant lengths
  - Variance preservation

- **Utility Functions**
  - `compare_methods()` function
  - Info dictionary validation

### Intent Classifier Tests (`test_intent_classifier.py`)

**15 tests** covering:

- **Classifier API**
  - Initialization
  - Intent label access
  - Prediction without model (error handling)

- **Data Loading & Validation**
  - JSON format validation
  - Required field checking
  - Deduplication

- **Embeddings**
  - Shape validation
  - Normalization

- **Label Encoding**
  - Label-to-index mapping
  - Index-to-label mapping
  - Round-trip encoding/decoding

- **Cross-Validation**
  - Stratified splitting
  - Class distribution preservation
  - Reproducibility with seeds

## Test Results

All tests should pass:

```
Ran 35 tests in ~5 seconds
OK
```

### Current Status

✅ **Length Debiasing:** 20/20 tests passing  
✅ **Intent Classifier:** 15/15 tests passing

## Adding New Tests

### Template for New Test

```python
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

class TestNewFeature(unittest.TestCase):
    """Test description."""
    
    def setUp(self):
        """Setup test fixtures."""
        pass
    
    def test_basic_functionality(self):
        """Test that basic feature works."""
        # Arrange
        expected = ...
        
        # Act
        result = ...
        
        # Assert
        self.assertEqual(result, expected)
```

### Test Organization

- One file per module
- Group related tests in classes
- Use descriptive test names: `test_<what>_<expected_behavior>`
- Include docstrings
- Use setUp/tearDown for fixtures

## Continuous Integration

Tests are designed to be CI-friendly:

- No external dependencies (mocked where needed)
- Deterministic (seeded random numbers)
- Fast execution (~5 seconds total)
- Clear pass/fail status

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure you're running from the project root:

```bash
cd /path/to/banditgpt
python tests/run_all_tests.py
```

### Missing Dependencies

```bash
# Core dependencies
pip install numpy scikit-learn xgboost sentence-transformers

# Optional (for pytest)
pip install pytest pytest-cov
```

### Test Failures

1. Check that you're on the latest code
2. Verify dependencies are installed
3. Run individual test files to isolate issues
4. Check for environment-specific issues (random seeds, paths)

## Coverage Report (Optional)

If you have `pytest-cov` installed:

```bash
pytest tests/ --cov=banditgpt --cov-report=html
open htmlcov/index.html
```

## Test Design Principles

1. **Fast**: Each test runs in milliseconds
2. **Isolated**: Tests don't depend on each other
3. **Deterministic**: Same input → same output
4. **Readable**: Clear test names and structure
5. **Comprehensive**: Edge cases covered
6. **Maintainable**: Easy to update when code changes

---

For questions or issues, see the main project README.
