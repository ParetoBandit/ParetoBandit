# Duplicate Method Definition Fix

## Issue
The `DisjointLinUCBPolicy` class had a critical duplicate method definition for `_check_numerical_stability`:

1. **First definition (Line 519)** - "Hard Reset" approach:
   - Wiped the precision matrix `A` to `λI`
   - Reset the preference vector `b` to zeros
   - **Problem**: Erased all learned knowledge, defeating the purpose of the bandit algorithm

2. **Second definition (Line 817)** - "Soft Reset" approach:
   - Injected fresh regularization: `A ← A + λI`
   - Preserved learned preferences in both `A` and `b`
   - **Correct**: Maintains learned knowledge while stabilizing numerics

## Resolution
**Removed the first (incorrect) definition at Line 519.**

### Why the Second Definition is Correct

The soft reset approach is mathematically sound because:

1. **Preserves Information**: The learned covariance structure and preferences remain intact
2. **Adds Regularization**: Injecting `λI` improves the condition number without destroying signal
3. **Maintains Bandit Properties**: The algorithm continues to exploit learned knowledge while exploring
4. **Numerical Stability**: Increases eigenvalues uniformly, preventing singular matrices

The hard reset would have caused:
- Loss of all historical learning
- Inefficient re-exploration of already-tested models
- Poor performance in ablation studies

## Verification

✅ All tests pass, including `test_lock_contention.py` which specifically tests `bandit_is_stable()`
✅ Method is now defined exactly once in the class
✅ Correct signature: `_check_numerical_stability(self, model: str, config: RouterConfig = None) -> None`
✅ Both call sites (lines 3039 and 3058) pass the required `config` parameter

## Impact on Paper

Since Python's method resolution means the second definition was always active, the ablation studies in the paper were conducted with the **correct** soft reset implementation. However, the presence of the duplicate suggests code quality issues that needed addressing.

## Files Modified
- `/Users/annette/repostitories/banditGPT/src/bandit_gpt/router.py`: Removed lines 519-525 (duplicate method definition)

## Related Methods
- `bandit_is_stable()`: Kept intact (used in tests, provides O(d) stability check via trace)
- `safe_inv()`: Used by the correct implementation to recompute `A_inv` after regularization

