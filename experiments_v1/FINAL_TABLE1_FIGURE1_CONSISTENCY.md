# Table 1 ↔ Figure 1 Final Consistency Check

**Date**: February 13, 2026 (After Cleanup)  
**Status**: ✅ **CONSISTENT** - All major inconsistencies resolved

---

## Summary

After archiving the old script and updating documentation:
- ✅ **Figure 1 uses N=750** (holdout only)
- ✅ **Matches Table 1** (holdout for "final evaluation")
- ✅ **No data contamination** (dev set reserved for training)
- ⚠️ **Minor issue**: 4 validation scripts still use N=1,871 but not referenced in paper

---

## Table 1: Dataset Splits

| Split | Source | Size | Purpose |
|-------|--------|------|---------|
| Warmup | RouteLLM Battles | 80,000 | PCA training + LinUCB priors |
| Development | LMSYS Arena | 1,121 | **Online learning & calibration** |
| Holdout | LMSYS Arena | 750 | **Final evaluation** |

**Key**: Dev is training data, Holdout is evaluation data

---

## Figure 1: Actual Configuration

### Active Script ✅
**`plot_figure1_revised.py`**

```python
def load_holdout_only(holdout_file: Path):
    """Load holdout data ONLY (no dev contamination)."""
    # Only loads CANONICAL_HOLDOUT_DATA_PATH
    # N = 750
```

### LaTeX Caption ✅
```latex
\caption{\textbf{Model Preference Heterogeneity in LMSYS Held-Out Prompts ($N=750$).}
```

### Data Sources ✅
- **PCA training**: 80K routing prompts (matches Table 1 warmup)
- **Figure 1 analysis**: 750 holdout prompts (matches Table 1 holdout)
- **Dev set**: NOT used in Figure 1 (reserved for training, as Table 1 states)

---

## Point-by-Point Consistency

### 1. Sample Size ✅
**Table 1**: Holdout = 750 for "final evaluation"  
**Figure 1**: N = 750 (caption and script)  
**Status**: ✅ **CONSISTENT**

### 2. Data Purpose ✅
**Table 1**: Dev (1,121) for "online learning & calibration" (training)  
**Figure 1**: Does NOT use dev set  
**Status**: ✅ **CONSISTENT** - Dev properly excluded

### 3. PCA Training ✅
**Table 1**: Warmup (80K) for "PCA training (384→32)"  
**Figure 1**: Uses "domain-adapted PCA, trained on 80K routing prompts"  
**Status**: ✅ **CONSISTENT**

### 4. Models ✅
**Table 1**: "mixtral-8x7b-instruct and gpt-4-turbo evaluations"  
**Figure 1**: "Reward gap (R_GPT-4-Turbo - R_Mixtral)"  
**Status**: ✅ **CONSISTENT**

### 5. Evaluation Data ✅
**Table 1**: Holdout (750) for "final evaluation"  
**Figure 1**: Uses holdout for heterogeneity analysis  
**Status**: ✅ **CONSISTENT**

### 6. Documentation ✅
**README**: States N=750 (after cleanup)  
**Caption**: States N=750  
**Script**: Loads 750 prompts  
**Status**: ✅ **CONSISTENT**

---

## Remaining Minor Issues

### Validation Scripts (Not Critical)

These 4 scripts still use dev+holdout (N=1,871):
1. `check_cluster_stats.py`
2. `analyze_cluster_diversity.py`
3. `validate_high_dimensional.py`
4. `validate_threshold.py`

**Impact**: ⚠️ **MINIMAL**
- These scripts are NOT referenced in paper LaTeX files
- They appear to be internal validation/exploration
- Main Figure 1 generation is correct

**Recommendation**: 
- Either update them to use N=750 (holdout only)
- Or document them as "exploratory analysis using full available data"
- Or archive them if obsolete

---

## What Changed vs Original Report

### Original Issues (Before Cleanup)
1. ❌ Main script `plot_lmsys_holdout_pca.py` used generic PCA
2. ❌ Documentation inconsistent (mixed N=750 and N=1,871)
3. ❌ Multiple scripts generating same figure with different methods

### Fixed (After Cleanup)
1. ✅ Archived old generic PCA script
2. ✅ Active script uses routing PCA, N=750
3. ✅ Updated README to reference correct script
4. ✅ Fixed N=1,871 → N=750 in documentation
5. ✅ Verified figure caption matches implementation

---

## Verification Checklist

- [x] **Figure caption**: States N=750 ✅
- [x] **Active script**: Uses holdout only (N=750) ✅
- [x] **PCA training**: Uses 80K warmup (matches Table 1) ✅
- [x] **Models**: GPT-4-Turbo and Mixtral (matches Table 1) ✅
- [x] **Dev set**: NOT used in Figure 1 ✅
- [x] **README**: References correct script and N=750 ✅
- [x] **Old script**: Archived ✅
- [ ] **Validation scripts**: Still use N=1,871 (minor, not in paper) ⚠️

---

## Bottom Line

### Main Question: "Are there inconsistencies between Table 1 and Figure 1?"

**Answer**: ✅ **NO - All major inconsistencies resolved**

### What's Consistent Now
1. ✅ Figure 1 uses **N=750** (holdout only)
2. ✅ Dev set (1,121) **not used** in Figure 1
3. ✅ Holdout used for "final evaluation" as Table 1 states
4. ✅ PCA trained on **80K** warmup as Table 1 states
5. ✅ Same **models** (GPT-4-Turbo, Mixtral)
6. ✅ **Documentation** matches implementation

### Remaining Minor Issue
⚠️ 4 validation scripts use N=1,871, but:
- Not referenced in paper
- Appear to be internal/exploratory
- Don't affect main Figure 1

### Recommendation for Validation Scripts

**Option 1 (Recommended)**: Update to use N=750
```python
# Change from:
prompts, gaps = load_lmsys_holdout_with_gaps(dev_file, holdout_file)

# To:
prompts, gaps = load_holdout_only(holdout_file)
```

**Option 2**: Document as exploratory
```python
"""
NOTE: This script uses dev+holdout (N=1,871) for exploratory analysis.
Main Figure 1 uses holdout only (N=750) to avoid contamination.
"""
```

**Option 3**: Archive if obsolete
```bash
mv check_cluster_stats.py archived/
mv analyze_cluster_diversity.py archived/
mv validate_high_dimensional.py archived/
mv validate_threshold.py archived/
```

---

## Comparison: Before vs After

| Aspect | Before Cleanup | After Cleanup | Status |
|--------|----------------|---------------|--------|
| Active script | `plot_lmsys_holdout_pca.py` (generic PCA) | `plot_figure1_revised.py` (routing PCA) | ✅ Fixed |
| Sample size | Mixed (750 and 1,871) | Consistent (750) | ✅ Fixed |
| Documentation | Inconsistent references | Consistent references | ✅ Fixed |
| Data usage | Unclear which N | Clear: N=750 holdout only | ✅ Fixed |
| Dev contamination | Potential risk | Eliminated | ✅ Fixed |
| Validation scripts | Use N=1,871 | Still use N=1,871 | ⚠️ Minor |

---

## Confidence Level

**Table 1 ↔ Figure 1 Consistency**: ✅ **HIGH**

**Evidence**:
1. Script explicitly loads holdout only
2. Caption explicitly states N=750
3. PCA source matches Table 1 (80K warmup)
4. Models match Table 1 (GPT-4-Turbo, Mixtral)
5. Dev set properly excluded
6. Documentation updated and consistent

**Only caveat**: 4 validation scripts use N=1,871, but they're not used in the paper, so this doesn't affect paper consistency.

---

**Status**: ✅ **RESOLVED** - Table 1 and Figure 1 are now consistent  
**Remaining work**: Optional cleanup of validation scripts (low priority)  
**Paper ready**: Yes, main figure and documentation are consistent
