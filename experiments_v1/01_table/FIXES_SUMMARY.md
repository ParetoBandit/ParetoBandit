# Table 1 Fixes Summary

**Date**: February 13, 2026  
**Status**: ✅ Both issues resolved

---

## Issue 1: Double-Counting in Table Presentation (FIXED)

### Problem
The table listed two separate rows that used the **same** 80,000 prompts:
- **PCA Training**: 80,000 prompts
- **Warmup Priors**: 80,000 prompts

This created a misleading sum:
- Implied total: 80,000 + 80,000 + 1,121 + 750 = **161,871** prompts ❌
- Actual total stated: **81,871** prompts
- This discrepancy confused readers about the actual dataset size

### Solution
Combined into a single **Warmup** row that clearly shows both purposes:

```latex
Warmup & RouteLLM Battles & 80,000 & PCA training (384→32) + LinUCB priors (A, b)
```

### Result
- Accurate count: 80,000 + 1,121 + 750 = **81,871** ✓
- No confusion about duplicate counting
- Clear that the same data serves both purposes

---

## Issue 2: Conflating Training and Evaluation Set Sizes (FIXED)

### Problem
The original footnote claimed:

> "Evaluation set (1,871 prompts total) exceeds prior work on LLM routing (RouteLLM: ~1,000 prompts)"

This was **misleading** because:
- The 1,871 included the **dev set** (1,121 prompts) used for **training** (online learning & hyperparameter tuning)
- Only the **holdout set** (750 prompts) is truly held-out for evaluation
- Counting training data as "evaluation data" inflates the apparent rigor
- **Actual comparison**: 750 evaluation prompts vs. RouteLLM's ~1,000 = **smaller**, not larger ❌

### Solution
Removed the misleading comparison and separated training vs. evaluation data clearly:

**Before**:
```latex
Evaluation set (1,871 prompts total) exceeds prior work on LLM routing (RouteLLM: ~1,000 prompts). 
Holdout set (750) provides sufficient statistical power...
```

**After**:
```latex
Development set (1,121 prompts) enables online learning with sufficient data for bandit convergence. 
Holdout set (750 prompts) provides rigorous held-out evaluation with stratified sampling and 
sufficient statistical power for detecting meaningful performance differences.
```

### Result
- Clear distinction between training (dev) and evaluation (holdout)
- No inflated claims about dataset size
- Honest presentation: 750 held-out prompts for evaluation
- Removed misleading comparison to RouteLLM

---

## Issue 3: Under-specified Chi-Square Test (FIXED)

### Problem
The original footnote stated:

> "Chi-square test confirms similar distributions (χ²=0.78, p=0.94)"

This was **scientifically unsound** because:
- The test compared distributions across 5 semantic categories (Coding, Conversational, Creative, Knowledge, Math/Logic)
- **These categories are hidden from readers** - they were deliberately removed from the table
- The categorization heuristic achieved only **49% accuracy** against LLM consensus
- Testing distributions of labels with ~50% accuracy = testing distributions of **noisy/random labels**
- A reviewer cannot evaluate the claim without knowing what variable was tested

**Key contradiction**: The table removed categories because "categories were not used in any downstream experiment," yet kept a statistical test result that **only makes sense** in the context of those categories.

### Why This Matters
1. **Unreproducible**: Readers don't know what was tested
2. **Unscientific**: Testing noisy labels (49% accuracy) doesn't validate representativeness
3. **Circular logic**: Removed categories from table but kept their statistical test
4. **Misleading confidence**: Presents a precise statistical result (p=0.94) based on unreliable measurements

### Solution
**Removed the chi-square claim entirely** and replaced with a clear methodological statement:

**Before**:
```latex
Dev and holdout sets use stratified sampling to ensure representative coverage. 
Chi-square test confirms similar distributions (χ²=0.78, p=0.94).
```

**After**:
```latex
Dev and holdout sets created using stratified sampling by task complexity to ensure 
representative coverage across prompt types.
```

### Result
- No reference to hidden categorization scheme
- No statistical test based on unreliable labels
- Clear statement of methodology (stratified sampling)
- Honest about what was done without unsupported statistical claims

---

## Files Changed

1. **`table1_dataset.tex`**
   - Combined PCA Training + Warmup Priors into single Warmup row
   - Fixed Sample Size footnote to separate dev (training) from holdout (evaluation)

2. **`generate_table1.py`**
   - Updated generation logic to match corrected table format
   - Fixed footnote generation to avoid conflating training/evaluation

3. **`README.md`**
   - Clarified "unique prompts" terminology
   - Added "[shared for PCA + priors]" notation

---

## Verification

### Math Check ✓
```
Warmup:      80,000
Development:  1,121
Holdout:        750
―――――――――――――――――
Total:       81,871 ✓
```

### Conceptual Check ✓
```
Training Data:
├─ Warmup:      80,000 (PCA + priors)
└─ Development:  1,121 (online learning)
   Total:       81,121

Evaluation Data:
└─ Holdout:        750 (held-out)
   Total:          750
```

---

## Impact

### Before (Problematic)
- Readers confused about double-counting (80k + 80k = 160k?)
- Inflated claims about "evaluation set" size (1,871 vs. 1,000)
- Training data (dev) misrepresented as evaluation data
- Table total (81,871) didn't match sum of rows (161,871)
- Chi-square test referenced hidden categorization with 49% accuracy
- Statistical claims unreproducible (readers don't know what was tested)

### After (Corrected)
- Clear that 80,000 prompts serve dual purpose (PCA + priors)
- Honest about evaluation set size (750 held-out prompts)
- Clear separation between training (dev) and evaluation (holdout)
- Table math adds up correctly (80,000 + 1,121 + 750 = 81,871) ✓
- Removed misleading chi-square test based on noisy labels
- Clear methodological statement (stratified sampling) without unsupported statistical claims

---

## Key Principles Applied

1. **Transparency**: Be honest about dataset sizes and purposes
2. **Accuracy**: Don't double-count data or inflate claims
3. **Clarity**: Distinguish training data from evaluation data
4. **Integrity**: Remove misleading comparisons that don't hold up

---

## Status

✅ Issue 1 (Double-Counting) - **RESOLVED**  
✅ Issue 2 (Training vs. Evaluation) - **RESOLVED**  
✅ Issue 3 (Chi-Square Test) - **RESOLVED**  
✅ Files updated and verified  
✅ Generation script matches table  
✅ Ready for publication

---

**Last Updated**: February 13, 2026
