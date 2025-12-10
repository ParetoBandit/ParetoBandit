# Synthetic Code Removal Log

**Date**: December 10, 2025  
**Action**: Removed all synthetic data generation code from project  
**Reason**: Ensure 100% real data usage, no synthetic fallbacks

## Summary

Removed all synthetic data generation functions from the codebase. The project now uses ONLY real data from established benchmarks and datasets.

## Functions Removed

### 1. `create_synthetic_ground_truth()` 
**File**: `research/kdd/prepare_fair_dataset.py`  
**Lines**: 312-372 (61 lines)  
**Purpose**: Generated synthetic winner labels for samples without ground truth  
**Usage**: Called in main() for WildBench samples (already removed)

**Removed code**:
- Function definition (61 lines)
- Function call in main() (replaced with filter for real labels only)

**Replacement**:
```python
# REMOVED: Synthetic ground truth generation (December 10, 2025)
# All samples now require real ground truth labels
# Only samples with actual winner labels are used
all_samples = [s for s in all_samples if s.winner is not None]
```

### 2. `generate_synthetic_agentic()`
**File**: `scripts/intent_classification/collect_real_intent_data.py`  
**Lines**: 276-310 (35 lines)  
**Purpose**: Fallback synthetic agentic prompts when Glaive dataset unavailable  
**Usage**: Called when Glaive dataset loading fails

**Removed code**:
- Function definition (35 lines)
- Fallback call replaced with empty list return

**Replacement**:
```python
except Exception as e:
    print(f"  ✗ Error loading Glaive: {e}")
    print(f"  ⚠️  SYNTHETIC FALLBACK REMOVED - Returning empty list")
    print(f"  Please ensure Glaive dataset is available or use real data")
    return []  # Return empty list instead of synthetic data
```

## What Was Kept (Acceptable Synthetic Data)

### Test Fixtures (NOT Production Code)

**File**: `tests/test_latent_factor.py`  
**Function**: `synthetic_data()` pytest fixture  
**Purpose**: Unit testing BLF statistical model  
**Status**: ✅ KEPT - Standard testing practice

**Why this is acceptable**:
1. **Testing only**: Never used in production or paper
2. **Standard practice**: All statistical software uses synthetic test data
3. **Controlled validation**: Known ground truth to verify model correctness
4. **Isolated**: Cannot contaminate real data pipeline

**Example**:
```python
@pytest.fixture
def synthetic_data(self):
    """Generate synthetic data with known latent factor."""
    np.random.seed(42)
    n_models = 20
    n_benchmarks = 5
    theta_true = np.random.randn(n_models)
    lambda_true = np.random.uniform(0.5, 2.0, n_benchmarks)
    # ... generate test data with known parameters
    return z_matrix, theta_true, lambda_true
```

This is equivalent to:
- Testing a sorting algorithm with random arrays
- Testing a ML model with toy datasets
- Testing statistical inference with simulated data

## Impact Assessment

### Files Modified

| File | Lines Removed | Lines Added | Net Change |
|------|---------------|-------------|------------|
| `research/kdd/prepare_fair_dataset.py` | 61 | 7 | -54 |
| `scripts/intent_classification/collect_real_intent_data.py` | 35 | 4 | -31 |
| **Total** | **96** | **11** | **-85** |

### Functional Changes

**`prepare_fair_dataset.py`**:
- **Before**: Generated synthetic labels for samples without ground truth
- **After**: Filters to only samples with real ground truth
- **Impact**: Smaller dataset, but 100% real data

**`collect_real_intent_data.py`**:
- **Before**: Fell back to synthetic agentic prompts if Glaive fails
- **After**: Returns empty list if real data unavailable
- **Impact**: Forces use of real data or explicit failure

## Verification

### No Synthetic Data Generation Remains

Searched for all synthetic data patterns:

```bash
# Search for synthetic data generation
grep -r "synthetic" --include="*.py" research/ scripts/ llm_jury/
grep -r "generate.*data" --include="*.py" research/ scripts/ llm_jury/
grep -r "create.*labels" --include="*.py" research/ scripts/ llm_jury/
```

**Results**:
- ❌ No production synthetic data generation
- ✅ Only test fixtures remain (acceptable)
- ✅ Documentation mentions "synthetic" but doesn't generate it

### All Data Sources Verified Real

✅ **Benchmarks**: HumanEval, MBPP, SummEdits, MixEval (all real)  
✅ **Human Preferences**: Chatbot Arena (>500K real comparisons)  
✅ **Operational**: TTFT, throughput, pricing (real measurements)  
✅ **Safety**: Vectara Hallucination (real expert annotations)

## Documentation Updated

### Paper Statements

**Added to §3.7 (Reproducibility)**:
> "All data presented in this paper is derived from real sources... **We use NO synthetic, simulated, or generated data** for benchmark scores, quality assessments, or model evaluations."

### Verification Documents

Created comprehensive verification:
- `DATA_AUTHENTICITY_VERIFICATION.md` (236 lines)
- `SYNTHETIC_CODE_REMOVAL.md` (this file)

## Code Comments Added

All removal sites now have clear comments:

```python
# REMOVED: create_synthetic_ground_truth() function
# Synthetic data generation removed from project (December 10, 2025)
# All data used in LLM Jury is real data from established benchmarks
# See KDD/data/DATA_AUTHENTICITY_VERIFICATION.md for details
```

## Testing Impact

### Unit Tests: NO IMPACT

The `synthetic_data()` pytest fixture in `test_latent_factor.py` is kept because:
1. It's standard practice for testing statistical models
2. It's isolated from production code
3. It helps verify BLF implementation correctness
4. Cannot contaminate real data pipeline

### Integration Tests: IMPROVED

- Routing experiments now use ONLY real LMSYS Arena data
- Intent classification requires real datasets (Glaive, etc.)
- No synthetic fallbacks mask missing real data

## Before vs. After

### Before (With Synthetic)
```python
# prepare_fair_dataset.py
all_samples = load_lmsys_data(...)
wildbench_samples = load_wildbench_data(...)  # Some have no labels
all_samples.extend(wildbench_samples)

# Fill in missing labels with synthetic heuristics
all_samples = create_synthetic_ground_truth(all_samples)  # ❌ Synthetic

# Now all samples have labels (real or synthetic)
train, val, test = split_dataset(all_samples)
```

### After (Real Data Only)
```python
# prepare_fair_dataset.py
all_samples = load_lmsys_data(...)
# WildBench removed entirely (no synthetic labels needed)

# Filter to only real labels
all_samples = [s for s in all_samples if s.winner is not None]  # ✅ Real only

# Now all samples have REAL labels
train, val, test = split_dataset(all_samples)
```

## Rollback Procedure

If synthetic data generation needs to be restored (NOT RECOMMENDED):

```bash
# 1. Check out previous version
git diff HEAD~1 research/kdd/prepare_fair_dataset.py
git diff HEAD~1 scripts/intent_classification/collect_real_intent_data.py

# 2. Restore functions
git checkout HEAD~1 -- research/kdd/prepare_fair_dataset.py

# 3. Re-run with synthetic fallback
python research/kdd/prepare_fair_dataset.py
```

**However**: This would violate the data authenticity guarantee in the paper.

## Future Guidelines

### Adding New Data Sources

When adding new data sources, ensure:
1. ✅ Data from established benchmarks or real evaluations
2. ✅ Ground truth from actual measurements, not heuristics
3. ✅ Human judgments from real users, not simulations
4. ❌ NO synthetic fallbacks "just in case"
5. ❌ NO generated labels from heuristics

### Acceptable Use of Synthetic Data

Synthetic data is ONLY acceptable for:
1. ✅ Unit testing (pytest fixtures)
2. ✅ Algorithm development (toy examples)
3. ✅ Visualization examples (demonstrations)
4. ❌ **NEVER** for production benchmarks
5. ❌ **NEVER** for model quality assessment
6. ❌ **NEVER** for paper results

## Summary

### What Was Removed
- ❌ `create_synthetic_ground_truth()` (61 lines)
- ❌ `generate_synthetic_agentic()` (35 lines)
- ❌ Synthetic fallback calls (2 locations)
- **Total**: 96 lines of synthetic data generation

### What Remains (Real Data Only)
- ✅ HumanEval, MBPP evaluations
- ✅ SummEdits factual consistency
- ✅ MixEval multi-domain
- ✅ Chatbot Arena human preferences
- ✅ Operational measurements (TTFT, pricing)
- ✅ Safety metrics (Vectara)

### What Remains (Testing Only)
- ✅ `synthetic_data()` pytest fixture (acceptable)

### Impact
- **Codebase**: Cleaner, no synthetic code smell
- **Data**: 100% real, no contamination risk
- **Paper**: Stronger authenticity guarantee
- **Reproducibility**: Forces use of real data sources

---

**Removal completed**: December 10, 2025  
**Status**: ✅ **Zero synthetic data in production code**  
**Verification**: All data sources confirmed real
