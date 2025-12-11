# Narrative Consistency Fix Summary

## The Problem (BEFORE)

**Inconsistency**: The abstract claimed 88.1% accuracy, but all tables in Section 4 showed 94.5% baseline results. This created confusion for readers.

### What Was Confusing:

1. **Abstract**: "Our method achieves 88.1% accuracy..."
2. **Table 3**: Shows 94.47% accuracy
3. **Table 4**: Shows baseline per-class results
4. **Reader reaction**: "Wait, is this a typo? Are they cherry-picking numbers?"

---

## The Solution (AFTER)

### ✅ **Abstract - Now Explains BOTH Models**

**Before**: 
> "Our method achieves 88.1% accuracy..."

**After**:
> "**Our baseline model achieves 94.5% accuracy** using gradient boosting on pre-trained sentence embeddings. However, through adversarial testing, we discover a critical length bias: the baseline fails on 100% of long non-summarization prompts. **To address this, we apply orthogonal projection, yielding our robust model with 88.1% accuracy and 75% bias reduction.**"

**Impact**: Readers now understand there are TWO models with different goals.

---

### ✅ **Section 4.1 - Renamed to "Baseline Model Performance"**

**Before**: "Overall Performance" (ambiguous)

**After**: "**Baseline Model Performance**" + "We first present results for the baseline model to establish the performance ceiling before addressing bias."

**Impact**: Clear that these are baseline results, not the final system.

---

### ✅ **Tables 3 & 4 - Clearly Labeled as "Baseline"**

**Before**:
- Table 3: "Overall Performance"
- Table 4: "Per-Class Performance" (decorrelated model data)

**After**:
- Table 3: "**Baseline Model Performance** (5-Fold Stratified CV)"
- Table 4: "**Baseline Per-Class Performance**"

**Impact**: No ambiguity about which model these tables describe.

---

### ✅ **NEW Section 4.7 - Proposed Solution**

Clearly explains the orthogonal projection method **before** showing results.

---

### ✅ **NEW Section 4.8 - Robust Model Performance**

**Dedicated section** for the 88.1% accuracy model with complete tables:

**Table 5: Robust Model Performance**
- Accuracy: **88.08% ± 2.71%**
- F1-Score: **88.19%**

**Table 6: Robust Model Per-Class Performance**
- All classes with decorrelated results
- Clearly labeled as "Robust Model"

---

### ✅ **NEW Section 4.9 - Baseline vs. Robust Comparison**

**Side-by-side comparison tables** so readers can see the trade-off:

**Table 7: Comprehensive Comparison**

| Metric | Baseline | Robust (Decorrelated) | Change |
|--------|----------|----------------------|--------|
| Overall Accuracy | 94.47% | 88.08% | -6.39% |
| Length Correlation | -0.10 | 0.00 | Removed ✅ |
| Length Artifact | 100% failure | 25% failure | -75% ✅ |

**Table 8: Stratified Performance by Length**

| Length Bucket | Baseline | Robust | Issue |
|---------------|----------|--------|-------|
| Short | 92.1% | 87.3% | Slight drop |
| Medium | 93.2% | 88.9% | Slight drop |
| Long | **98.2%** | 88.5% | ⚠️ Baseline suspiciously high |

**Impact**: Readers clearly see why we choose the robust model despite lower accuracy.

---

### ✅ **Table 9 - Comparison to Prior Work Updated**

Now includes BOTH our models:

| Method | Accuracy | Length Bias Test |
|--------|----------|-----------------|
| Our Baseline | 94.5% | ❌ 100% failure |
| **Our Robust Model** | **88.1%** | ✅ **25% failure** |

---

## Paper Flow (AFTER FIX)

### Clear Narrative Arc:

1. **Introduction**: Define the problem (intent classification for routing)

2. **Section 4.1**: Present baseline results (94.5%) - **establish upper bound**

3. **Section 4.6**: Discover critical flaw through adversarial testing - **the plot twist**

4. **Section 4.7**: Propose solution (orthogonal projection) - **the fix**

5. **Section 4.8**: Present robust model results (88.1%) - **the hero**

6. **Section 4.9**: Compare baseline vs. robust - **justify the trade-off**

7. **Discussion**: Why we choose fairness over raw accuracy

---

## Key Changes Summary

| Element | Before | After |
|---------|--------|-------|
| **Abstract** | Only mentions 88.1% | Explains both 94.5% baseline and 88.1% robust |
| **Section 4.1** | "Overall Performance" | "**Baseline** Model Performance" |
| **Table 3** | Ambiguous | "**Baseline** Model Performance" |
| **Table 4** | Mixed data | "**Baseline** Per-Class Performance" |
| **Section 4.7** | (didn't exist) | **NEW**: Proposed Solution |
| **Section 4.8** | (didn't exist) | **NEW**: **Robust** Model Performance |
| **Section 4.9** | (didn't exist) | **NEW**: Baseline vs. Robust Comparison |
| **Table 5** | (didn't exist) | **NEW**: Robust Model Performance |
| **Table 6** | (didn't exist) | **NEW**: Robust Per-Class Performance |
| **Table 7** | (didn't exist) | **NEW**: Comprehensive Comparison |
| **Table 8** | (didn't exist) | **NEW**: Stratified Performance Comparison |

---

## Why This Matters

### Before Fix:
- ❌ Reader sees 88.1% in abstract, 94.5% in tables → **Confusion**
- ❌ Unclear which model is recommended
- ❌ Looks like cherry-picking numbers
- ❌ Trade-off not explicitly justified

### After Fix:
- ✅ Clear two-model narrative: baseline (high accuracy, biased) vs. robust (lower accuracy, fair)
- ✅ Explicit justification for choosing 88.1% model
- ✅ Side-by-side comparisons show trade-offs transparently
- ✅ Demonstrates scientific integrity (tested both, reported both, chose fairness)

---

## For KDD Reviewers

**The fixed narrative shows**:

1. **Scientific Method**: We built a strong baseline (94.5%), then **discovered its flaw** through rigorous testing
2. **Honesty**: We don't hide the baseline's higher accuracy - we report both models transparently
3. **Principled Trade-offs**: We explicitly choose fairness (88.1%) over benchmark gaming (94.5%)
4. **Reproducibility**: Both models fully documented with clear tables

**Reviewers will appreciate**: The paper tells a complete story, not just reporting the "best" number.

---

## Bottom Line

**Fixed**: The paper now has a clear, honest narrative that explains why we recommend the 88.1% robust model despite having a 94.5% baseline. No more confusion between abstract and results tables.

**Message**: "We could report 94.5% and look better, but that model fails catastrophically in production. We choose to report the robust 88.1% model that actually works."

**This is good science.** ✅
