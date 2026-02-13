# Table 1 ↔ Figure 1 Consistency Analysis

**Date**: February 13, 2026  
**Status**: ⚠️ **MAJOR INCONSISTENCY FOUND** - Data contamination issue

---

## Executive Summary

**CRITICAL FINDING**: Figure 1 has **inconsistent data usage** across different scripts:
- Some scripts use **N=1,871** (dev + holdout) ❌ **CONTAMINATED**
- Other scripts use **N=750** (holdout only) ✅ **CLEAN**

**This contradicts Table 1**, which states dev set is for "online learning & calibration" (training data), not for Figure 1 analysis.

---

## Table 1: Dataset Description

From `experiments_v1/01_table/table1_dataset.tex`:

| Split | Source | Size | Purpose |
|-------|--------|------|---------|
| Warmup | RouteLLM Battles | 80,000 | PCA training + LinUCB priors |
| **Development** | LMSYS Arena | **1,121** | **Online learning & calibration** |
| **Holdout** | LMSYS Arena | **750** | **Final evaluation** |
| **Total** | | **81,871** | |

**Key Point**: Development set is explicitly for **"Online learning & calibration"** (i.e., training data).

---

## Figure 1: Actual Data Usage

### Version 1: CONTAMINATED (validation scripts)

**Scripts that load dev + holdout (N=1,871)**:
1. `check_cluster_stats.py`
2. `analyze_cluster_diversity.py`
3. `validate_high_dimensional.py`
4. `validate_threshold.py`

**Evidence from `check_cluster_stats.py` (line 26)**:
```python
def load_lmsys_holdout_with_gaps(dev_file, holdout_file):
    prompt_rewards = {}
    for file_path in [dev_file, holdout_file]:  # ❌ LOADS BOTH
        with gzip.open(file_path, 'rt') as f:
            # ... processes both files
```

**Result**: N=1,871 prompts (dev 1,121 + holdout 750)

### Version 2: CLEAN (main script)

**Script that loads holdout only (N=750)**:
1. `plot_lmsys_holdout_pca.py` (main Figure 1 generator)

**Evidence from `plot_lmsys_holdout_pca.py` (line 73-96)**:
```python
def load_lmsys_holdout_with_gaps(holdout_file: Path):
    """
    Load LMSYS HOLDOUT-ONLY prompts with reward gaps.
    
    FIXES APPLIED:
    - Issue #2: Use holdout ONLY (no dev contamination)
    - Dev set excluded (reserved for training in Table 2)
    """
    # ... only loads holdout_file
```

**Result**: N=750 prompts (holdout only)

---

## Documentation Inconsistency

### README Claims

From `experiments_v1/01_figure/README.md`:

**Line 11**: ✅ CORRECT
> "Holdout Analysis (N=750): Discovery on clean holdout data"

**Line 32**: ❌ **INCORRECT**
> "Projects 1,871 LMSYS prompts onto PCA space"

**Line 88**: ✅ CORRECT
> "Holdout Analysis (N=750)"

**Multiple other references**: Mix of N=750 and N=1,871

---

## Why This Matters

### Data Contamination Risk

**If dev set is used in Figure 1 (N=1,871)**:
1. ❌ **Contamination**: Dev set is for "online learning & calibration" (Table 1)
2. ❌ **Circular analysis**: Testing on data used for training/calibration
3. ❌ **Inflated statistics**: Larger N → more power, but invalid

**If holdout only is used (N=750)**:
1. ✅ **Clean evaluation**: True held-out data
2. ✅ **Matches Table 1**: Holdout is for "final evaluation"
3. ✅ **Valid inference**: No contamination

### Which Version is "Official"?

**Main Figure 1 script** (`plot_lmsys_holdout_pca.py`):
- Uses **N=750** (holdout only) ✅
- Explicitly states "no dev contamination"
- This is likely the **correct version**

**Validation scripts**:
- Use **N=1,871** (dev + holdout) ❌
- These appear to be **older versions** before contamination was fixed
- Should be updated or removed

---

## Specific Inconsistencies

### 1. Sample Size

**Table 1**: Dev = 1,121, Holdout = 750  
**Figure 1 README (line 32)**: "1,871 prompts" (dev + holdout)  
**Figure 1 main script**: 750 prompts (holdout only)  
**Figure 1 validation scripts**: 1,871 prompts (dev + holdout)

**Verdict**: ⚠️ **INCONSISTENT** - Documentation says 1,871, but main script uses 750

### 2. Data Purpose

**Table 1**: Dev for "online learning & calibration"  
**Figure 1 validation scripts**: Include dev in analysis  
**Figure 1 main script**: Exclude dev ("reserved for training")

**Verdict**: ⚠️ **INCONSISTENT** - Validation scripts violate Table 1's stated purpose

### 3. Evaluation Claims

**Table 1**: Holdout (750) for "final evaluation"  
**Figure 1**: Claims vary between N=750 and N=1,871

**Verdict**: ⚠️ **INCONSISTENT** - If N=1,871, then dev is part of "evaluation" despite Table 1 saying it's for "training"

---

## Impact on Paper Claims

### If Figure 1 uses N=1,871 (contaminated)

**Problems**:
1. ❌ **Contradicts Table 1**: Dev is supposed to be for training
2. ❌ **Data leakage**: Testing on calibration data
3. ❌ **Invalid statistics**: Inflated power from larger N
4. ❌ **Misleading comparison**: "Exceeds RouteLLM's ~1,000 prompts" (actually 750 < 1,000)

### If Figure 1 uses N=750 (clean)

**Correct approach**:
1. ✅ **Matches Table 1**: Holdout for evaluation
2. ✅ **No leakage**: True held-out data
3. ✅ **Valid statistics**: Proper sample size
4. ⚠️ **Honest comparison**: 750 < RouteLLM's ~1,000 (smaller, not larger)

---

## Recommendations

### 1. Clarify Which N is Used ✅ URGENT

**Action**: Verify which version of Figure 1 is in the paper
- If paper shows N=1,871 → **Major problem**, needs fixing
- If paper shows N=750 → **Correct**, update documentation

### 2. Update Documentation

**Action**: Fix README to consistently state N=750

**Changes needed**:
- Line 32: "Projects 1,871 LMSYS prompts" → "Projects 750 held-out LMSYS prompts"
- All references to "1,871" should be "750" (unless discussing dev+holdout total)

**Rationale**: Main script uses N=750, so documentation should match

### 3. Update or Remove Validation Scripts

**Option A (Recommended)**: Update to use holdout only
```python
# OLD (contaminated)
prompts, gaps = load_lmsys_holdout_with_gaps(dev_file, holdout_file)

# NEW (clean)
prompts, gaps = load_lmsys_holdout_only(holdout_file)
```

**Option B**: Remove validation scripts if outdated
- Archive or delete contaminated versions
- Prevent future confusion

### 4. Verify Figure 1 Caption

**Check paper Figure 1 caption**:
- Does it say N=750 or N=1,871?
- Does it clarify "holdout only" vs "dev + holdout"?

**Recommended caption note**:
> "Analysis uses holdout set only (N=750). Development set (N=1,121) is reserved for online learning calibration (not shown)."

---

## Root Cause Analysis

### Timeline (Inferred)

1. **Original version**: Used dev + holdout (N=1,871)
   - Validation scripts written with this approach
   - README documented N=1,871

2. **Issue discovered**: Dev contamination problem identified
   - "Issue #2: Dev contamination" mentioned in script comments
   - Recognized that dev is training data, not evaluation data

3. **Main script fixed**: `plot_lmsys_holdout_pca.py` updated to holdout only (N=750)
   - Explicit "no dev contamination" comment
   - "Dev set excluded (reserved for training)"

4. **Validation scripts NOT updated**: Still use N=1,871
   - Likely oversight
   - Creates inconsistency

5. **README partially updated**: Mixed N=750 and N=1,871 references
   - Incomplete cleanup

---

## Verification Checklist

To resolve this inconsistency, verify:

- [ ] **Which Figure 1 is in the paper?**
  - Generated by `plot_lmsys_holdout_pca.py` (N=750) ✅ ?
  - Or by validation scripts (N=1,871) ❌ ?

- [ ] **What does the figure caption say?**
  - N=750 (correct) ?
  - N=1,871 (contaminated) ?

- [ ] **Are validation scripts used anywhere?**
  - For supplementary analysis?
  - Should they be updated or removed?

- [ ] **Is dev set actually used for training/calibration?**
  - If yes → Figure 1 must use holdout only
  - If no → Table 1 description is wrong

---

## Connection to Issue 5 & 6

Interestingly, this data usage inconsistency is **separate from** the Issues 5 & 6 we just resolved:

- **Issue 5 (Model confound)**: ✅ RESOLVED - Using gpt-4-turbo throughout
- **Issue 6 (PCA confound)**: ✅ RESOLVED - Validated excellent transfer
- **Data contamination issue**: ⚠️ **NEW FINDING** - Inconsistent N across scripts

**All three issues are independent**:
- Issue 5/6: About model/PCA consistency
- This issue: About which data subset is used

---

## Bottom Line

### The Core Problem

**Table 1 says**: Dev (1,121) for training, Holdout (750) for evaluation  
**Figure 1 reality**: Some scripts use dev+holdout (1,871), others use holdout only (750)  
**Documentation**: Inconsistent - mixes N=750 and N=1,871

### The Correct Approach

**Figure 1 should use**: Holdout only (N=750)  
**Reason**: Matches Table 1's stated purpose for each split  
**Evidence**: Main script already does this correctly  
**Problem**: Validation scripts and documentation don't match

### The Fix

1. ✅ **Main analysis is correct** (`plot_lmsys_holdout_pca.py` uses N=750)
2. ⚠️ **Update validation scripts** to use holdout only (N=750)
3. ⚠️ **Fix documentation** to consistently state N=750
4. ✅ **Verify paper figure** uses the clean N=750 version

---

## Questions for User

1. **Which Figure 1 is in the paper?**
   - Generated by `plot_lmsys_holdout_pca.py`? (hope so)
   - Or by validation scripts?

2. **Are validation scripts results reported anywhere?**
   - In supplementary materials?
   - Just for internal checks?

3. **Is dev set actually used for training?**
   - Table 1 says "online learning & calibration"
   - Is this implemented? Where?

4. **Should we update validation scripts or just remove them?**

---

**Status**: ⚠️ **INCONSISTENCY IDENTIFIED** - Needs resolution  
**Severity**: **HIGH** - Could be data contamination  
**Action Required**: Verify paper uses clean N=750 version, update documentation  
**Estimated Fix Time**: ~1 hour (if paper is already correct, just docs to fix)
