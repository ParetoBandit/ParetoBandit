# Figure 1: One-Page Summary

## Bottom Line
**10 issues identified (8 major, 2 minor). After fixes: NO structure (p=0.983). Remove Figure 1.**

---

## The 9 Issues

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | **Circular PCA** - Trained on routing data | Major | ✅ Fixed |
| 2 | **Dev contamination** - Used training data | Critical | ✅ Fixed |
| 3 | **Circular threshold** - Chosen on target | Major | ⚠️ Not fixed |
| 4 | **Speculative mechanism** - No validation | Major | ❌ Can't fix |
| 5 | **Weak high-D** - Silhouette=0.057 (random) | Critical | ❌ Can't fix |
| 6 | **Overstated correlation** - ρ²=0.16 (16%) | Moderate | ❌ Can't fix |
| 7 | **Misleading scale** - No 1M rewards | Moderate | ❌ Can't fix |
| 8 | **Low diversity** - 0.355 (homogeneous) | Moderate | ❌ Can't fix |
| 9 | **Single observations** - No variance, unclear source | Minor | ❌ Can't fix |
| 10 | **Near-duplicate reporting** - Pair rate vs prompt rate | Minor | ⚠️ Reporting |

**Result:** 2/10 fixed (20%), but structure still disappears

---

## Results After Fixing #1-2

```
Clean Methodology:
- Data: N=750 (holdout only)
- PCA: Generic C4 (not routing data)
- Result: 749/750 in one cluster
- Statistics: p=0.983 (NOT significant)
- Conclusion: NO bimodal structure
```

---

## Evidence Stack (All Point to Problems)

1. Clean methods → p=0.983 (not significant)
2. High-D validation → silhouette=0.057 (random)
3. Separation → 0.81<1.0 (clusters overlap)
4. Effect size → ρ²=0.16 (only 16% variance)
5. Distribution → 749/750 in one cluster
6. Projection → Only visible in 2D, not 384D
7. Scale → 1M has no reward labels
8. Diversity → 0.355 vs 0.953 (homogeneous)
9. Rewards → Single observations, source unclear
10. Duplicates → 201 pairs could involve 60% of prompts

**All 10 lines converge: Structure is artifact + weak + narrow + poorly measured.**

---

## What Was Wrong

**Original (Circular):**
- PCA: Trained on routing data (tautological)
- Data: N=1,871 (includes dev = training data)
- Threshold: Chosen to maximize gaps (circular)
- Claims: "RLHF causes failures" (unvalidated)
- Validation: "High-D confirms" (actually silhouette=0.057)
- Stats: "Strongly predictive" (actually ρ²=0.16, moderate)
- Scale: "$2.3M savings" (1M has no rewards)
- Diversity: "Good" (actually 0.355 = homogeneous)

**Result:** p<10^-143, bimodal structure

**Clean (Fixed):**
- PCA: Trained on C4 (neutral)
- Data: N=750 (holdout only)
- Everything else: Same problems remain

**Result:** p=0.983, NO structure

---

## Recommendation

### ✅ KEEP (Validated)
- Table 2: Routing performance
- Figure 2: Distribution shift analysis
- Practical cost savings (without extrapolation)

### ❌ REMOVE
- Figure 1 and all related claims
- "Alignment Tax" narrative
- "RLHF failure mode" claims
- "$2.3M savings" projection
- "Forensic Agility" framing

---

## For Reviewers

**Q: "What happened to the Alignment Tax?"**

**A: (30 seconds)**
> "Methodological review found 8 issues: circular PCA, dev contamination, circular threshold, weak high-D clustering (silhouette=0.057), unvalidated mechanism, overstated correlation (16% variance), no rewards at scale, and low cluster diversity. After fixes, no structure (p=0.983). Removed to focus on validated contributions."

---

## Files Created

**Read these for details:**
1. `ONE_PAGE_SUMMARY.md` ← You are here
2. `README_ISSUES.md` - Quick reference (2 pages)
3. `ALL_ISSUES_SUMMARY.md` - Complete analysis (detailed)
4. `FINAL_RECOMMENDATION.md` - Action plan

**All in:** `experiments_v1/01_figure/`

---

## Action Items

- [ ] Delete Figure 1 from paper (15 min)
- [ ] Remove "Alignment Tax" from abstract (5 min)
- [ ] Rewrite introduction (1-2 hours)
- [ ] Update results section (1 hour)
- [ ] Remove scale/diversity claims (30 min)

**Total:** ~4-5 hours → Honest paper

---

## Key Message

**The "Alignment Tax" was a methodological artifact.**

**Remove Figure 1. Keep validated results.**

**Still a good paper - just more honest.**
