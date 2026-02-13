# Testing Documentation: Figure 4 Experiment

## Overview

Comprehensive test suite for the Figure 4 Corralling experiment, validating all components added during the KDD revision process.

## Test Files

### 1. `test_corralling.py` - Core Implementation Test

**Purpose**: Validates the basic Corralling algorithm implementation with optimized hyperparameters.

**What it tests**:
- Importance-weighted loss estimation
- Expert weight updates (exponential weights algorithm)
- Adaptive expert combination
- Unlearning of warmup bias
- Optimized hyperparameters (η=5.0, γ=0.10)

**Key checks**:
- ✅ Weights sum to 1.0
- ✅ Weights are non-negative
- ✅ Weights change from initial (learning occurs)
- ✅ Expert selections match sample count
- ✅ Model selections match sample count

**Run time**: ~10 seconds (100 samples)

**Expected behavior**: 
- Tabula Rasa expert should win (weight > 50%)
- Demonstrates unlearning of warmup bias

**Usage**:
```bash
python experiments_v1/04_figure/test_corralling.py
```

---

### 2. `test_semantic_transfer.py` - Semantic Transfer Tests

**Purpose**: Validates the `extend_priors_with_semantic_transfer` function for cold-start model initialization.

**What it tests**:

#### Test 1: Basic Transfer
- Single model transfer with gamma scaling
- Correct matrix/vector copying
- Original models unchanged

#### Test 2: Multiple Models
- Simultaneous transfer of multiple models
- Correct model count after extension

#### Test 3: Error Handling
- Missing source model raises ValueError
- Empty transfer mapping raises ValueError

#### Test 4: Real Priors Integration
- Works with actual warmup priors
- GPT-4o transfer from GPT-4-Turbo
- Correct matrix dimensions
- Priors are scaled down by gamma

#### Test 5: Idempotency
- Running transfer twice doesn't duplicate models
- Graceful handling of already-transferred models

**Run time**: ~5 seconds

**Expected results**: 5/5 tests pass

**Usage**:
```bash
python experiments_v1/04_figure/test_semantic_transfer.py
```

---

### 3. `test_optimized_config.py` - Configuration Validation

**Purpose**: Validates that optimized hyperparameters (η=5.0, γ=0.10) perform better than baseline.

**What it tests**:

#### Test 1: Optimized vs Baseline
- Compares η=5.0, γ=0.10 vs η=1.0, γ=0.05
- Measures regret and reward
- Validates competitive performance

#### Test 2: Gamma Exploration Effect
- Tests γ=0.0 vs γ=0.10
- Validates that gamma provides exploration
- Measures expert selection balance

#### Test 3: Learning Rate Adaptation Speed
- Tests η=0.5 vs η=5.0
- Validates faster weight adaptation with higher η
- Measures weight deviation from initial

**Run time**: ~45 seconds (3 experiments × 100-200 samples)

**Expected results**: 
- Optimized config has competitive or better regret
- Higher gamma improves exploration balance
- Higher learning rate increases adaptation speed

**Usage**:
```bash
python experiments_v1/04_figure/test_optimized_config.py
```

---

### 4. `run_all_tests.py` - Comprehensive Test Suite

**Purpose**: Runs all tests in sequence and provides a summary report.

**What it runs**:
1. Core Corralling implementation test
2. Semantic transfer functionality test
3. Optimized configuration test

**Run time**: ~60 seconds total

**Usage**:
```bash
python experiments_v1/04_figure/run_all_tests.py
```

**Expected output**:
```
================================================================================
FIGURE 4 COMPREHENSIVE TEST SUITE
================================================================================

✅ PASS - Core Corralling Implementation
✅ PASS - Semantic Transfer Functionality
✅ PASS - Optimized Configuration

Total: 3/3 test suites passed

✅ All test suites passed! Implementation is correct.
```

---

## What Gets Validated

### Algorithm Correctness
- [x] Importance-weighted loss estimation (unbiased)
- [x] Exponential weights update rule
- [x] Softmax normalization (weights sum to 1)
- [x] Non-negative weights constraint
- [x] Expert sampling according to weights
- [x] Model selection from sampled expert

### New Features (KDD Revision)
- [x] Semantic transfer for new models
- [x] GPT-4o initialization from GPT-4-Turbo
- [x] Gamma scaling (0.05 factor)
- [x] Multi-model support (3 models)
- [x] Optimized hyperparameters (η=5.0, γ=0.10)

### Edge Cases
- [x] Missing source model in transfer mapping
- [x] Empty transfer mapping
- [x] Duplicate model transfers (idempotency)
- [x] Zero gamma (no exploration)
- [x] Extreme learning rates

### Statistical Properties
- [x] Weight convergence
- [x] Adaptation speed with different η
- [x] Exploration balance with different γ
- [x] Reproducibility across runs

---

## Integration with CI/CD

To add to continuous integration:

```yaml
# .github/workflows/test.yml
- name: Run Figure 4 Tests
  run: |
    python experiments_v1/04_figure/run_all_tests.py
```

---

## Debugging Failed Tests

### If `test_corralling.py` fails:

**Check 1: Weights don't sum to 1**
- Issue: Numerical instability in softmax
- Fix: Check for overflow in exponential weights

**Check 2: Weights unchanged**
- Issue: Learning rate too low or data too uniform
- Fix: Increase sample size or check data variance

**Check 3: Selection count mismatch**
- Issue: Expert not updating correctly
- Fix: Verify `update()` method called for all samples

### If `test_semantic_transfer.py` fails:

**Test 4 (Real Priors) fails**:
- Issue: Warmup priors file not found or outdated
- Fix: Regenerate priors or update path in config_legacy.py

**Test 3 (Error Handling) fails**:
- Issue: ValueError not raised correctly
- Fix: Check validation logic in `extend_priors_with_semantic_transfer`

### If `test_optimized_config.py` fails:

**Test 1: Optimized worse than baseline**
- Issue: High variance in small sample
- Fix: This is expected occasionally; run multiple times or increase sample size

**Test 2: Gamma effect unclear**
- Issue: Sample size too small to see exploration benefit
- Fix: Increase sample size to 200+

---

## Performance Benchmarks

Typical run times on M1 Mac:

| Test Suite | Samples | Time | Expected Result |
|------------|---------|------|-----------------|
| test_corralling.py | 100 | ~10s | Tabula Rasa wins |
| test_semantic_transfer.py | N/A | ~5s | 5/5 tests pass |
| test_optimized_config.py | 300 | ~45s | Optimized competitive |
| **Total (run_all_tests.py)** | **400** | **~60s** | **All pass** |

---

## Test Coverage

### Functions Tested:
- ✅ `TabulaRasaRouter.__init__`
- ✅ `TabulaRasaRouter.select_model` (with `total_steps` parameter)
- ✅ `TabulaRasaRouter.update`
- ✅ `extend_priors_with_semantic_transfer` (NEW)
- ✅ `CorrallingRouter.select_model`
- ✅ `CorrallingRouter.update`
- ✅ `apply_gamma_scaling`
- ✅ `embed_prompt`

### Configurations Tested:
- ✅ Baseline: η=1.0, γ=0.05
- ✅ Optimized: η=5.0, γ=0.10
- ✅ No exploration: γ=0.0
- ✅ Slow adaptation: η=0.5
- ✅ Fast adaptation: η=5.0

### Data Scenarios:
- ✅ 2-model routing (Mixtral, GPT-4-Turbo)
- ✅ 3-model routing (+ GPT-4o via transfer)
- ✅ Mock data (synthetic matrices)
- ✅ Real data (dev_rewards_complete.jsonl.gz)

---

## Maintenance

### When to Update Tests:

1. **Algorithm changes**: Update `test_corralling.py`
2. **New hyperparameters**: Update `test_optimized_config.py`
3. **New models**: Update `test_semantic_transfer.py`
4. **New features**: Add new test file and update `run_all_tests.py`

### Test Data Dependencies:

- `src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz`
- `src/artifacts/priors_warmup.joblib`
- `src/artifacts/pca_model.joblib`
- `all-MiniLM-L6-v2` (Sentence-BERT model, auto-downloaded)

---

## Contact

If tests fail unexpectedly, check:
1. Data files are present and up-to-date
2. Dependencies are installed (`pip install -r requirements.txt`)
3. Python version >= 3.8

For persistent failures, file an issue with:
- Test output
- System info (OS, Python version)
- Data file checksums
