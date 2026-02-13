# Figure 1: Critical Issues Summary

## Status: NOT REPRODUCIBLE

After methodological review, **Figure 1 findings do not replicate** with proper methods.

---

## 8 Major Issues Identified

| # | Issue | Fixable? | Status |
|---|-------|----------|--------|
| 1 | Circular PCA (trained on routing data) | Yes | ✅ Fixed |
| 2 | Dev contamination (train/test leak) | Yes | ✅ Fixed |
| 3 | Circular threshold (chosen on target) | Yes | ⚠️ Not fixed |
| 4 | Speculative mechanism (causal claims) | No | ❌ Cannot fix |
| 5 | Weak high-D structure (silhouette=0.057) | No | ❌ Cannot fix |
| 6 | Overstated correlation (ρ²=0.16, 16%) | No | ❌ Cannot fix |
| 7 | Misleading scale (no 1M rewards) | No | ❌ Cannot fix |
| 8 | Low cluster diversity (homogeneous) | No | ❌ Cannot fix |

---

## Results with Clean Methodology

**After fixing Issues #1-2:**

```
Data: N=750 (holdout only, no dev contamination)
PCA: Generic C4 corpus (10K samples, no routing bias)
Threshold: PC1 = 0.3

Distribution:
- Low PC1: 749/750 (99.9%)
- High PC1: 1/750 (0.1%)

Statistics:
- Mann-Whitney p = 0.983 (NOT significant)
- Cohen's d = NaN (only 1 sample)
- NO bimodal structure found
```

**Original vs Clean:**
- Original (circular): N=1,871, p<10^-143, bimodal structure
- Clean (fixed): N=750, p=0.983, NO structure
- **Structure was methodological artifact**

---

## Evidence Against Original Claims

Multiple independent lines show problems:

1. **Clean methodology:** p=0.983 (not significant)
2. **High-D validation:** Silhouette=0.057 (random)
3. **Separation ratio:** 0.81<1.0 (clusters overlap)
4. **Effect size:** ρ²=0.16 (moderate, only 16% variance)
5. **Distribution:** 749/750 in one cluster
6. **Projection artifact:** Only visible in 2D, not 384D
7. **No scale validation:** 1M dataset has no rewards
8. **Low diversity:** High PC1 diversity=0.355 vs Low PC1=0.953 (homogeneous templates)

**None of these can be explained away.**

---

## Recommendation

### REMOVE FIGURE 1

**Why:**
- Findings don't replicate (p=0.983)
- Multiple unfixable issues (#4-7)
- Trying to salvage = motivated reasoning
- Paper stronger without it

**What remains valid:**
- ✅ Table 2 (routing performance)
- ✅ Figure 2 (distribution shift)
- ✅ Practical cost savings

**These justify publication.**

---

## What To Remove

### From Paper
- [ ] Figure 1 (bimodal visualization)
- [ ] "Alignment Tax" from abstract/intro
- [ ] "RLHF failure mode" claims
- [ ] "Forensic Agility" framing
- [ ] "$2.3M savings" projection (unsupported)

### From Code/Docs
- [ ] `plot_lmsys_holdout_pca.py` references in paper
- [ ] `plot_lmsys_1M_pca.py` scale validation claims
- [ ] README economic projections

---

## If Reviewers Ask

**Q: "What about the Alignment Tax?"**

**A:** 
> "Initial exploratory analysis suggested bimodal structure. However, methodological review revealed eight concerns: circular PCA, dev contamination, circular threshold, weak high-D clustering (silhouette=0.057), unvalidated mechanism, overstated correlation (ρ²=0.16), misleading scale validation (no rewards at 1M), and low cluster diversity (0.355 vs 0.953, suggesting narrow template category). After corrections, structure doesn't replicate (p=0.983). We removed these claims and focus on validated contributions: learned routing under distribution shift."

**Shows:** Integrity, thoroughness, honesty.

---

## Files for Reference

**Analysis:**
- `ALL_ISSUES_SUMMARY.md` - Complete technical analysis (detailed)
- `FINAL_RECOMMENDATION.md` - Action plan
- `ISSUES_CHECKLIST.md` - Status tracker
- `README_ISSUES.md` - This file (quick reference)

**Code:**
- `scripts/train_pca_generic.py` - Generic PCA (fixes #1)
- `plot_lmsys_holdout_pca.py` - Holdout-only (fixes #2)
- `compare_pca_models.py` - Validation script

---

## Timeline

**Fixed Issues:** ~6 hours work (done)
**Paper Updates:** ~4-5 hours editing
**Total:** ~10 hours to clean methodology

**Result:** Honest paper with validated claims.

---

## Key Takeaway

**The "Alignment Tax" was a methodological artifact.**

Created by: Circular PCA + Dev contamination + Circular threshold + Projection effects + Causal overreach + Statistical misrepresentation + Misleading scale claims.

**Disappears with proper methods.**

**Remove Figure 1. Focus on validated routing performance.**
