# Table 1 Experiment Review: Executive Summary

**Date**: February 13, 2026  
**Reviewer**: KDD Statistical Review  
**Experiment**: Dataset Composition and Provenance (experiments_v1/01_table/)

---

## TL;DR

### **Good News** ✅
- Core experimental design is **excellent**
- Data provenance is **exemplary**
- Statistical methodology is **sound**
- Stratification validation is **strong**

### **Bad News** ❌
- Categorization validation claims were **misleading** (now fixed)
- Model substitution is **not validated**
- Distribution shift connection to robustness is **unclear**

### **Verdict**: ⚠️ **Major Revision → Minor Revision** (after Tier 1 fixes ✅)

---

## What I Reviewed

I examined the complete experiment in `experiments_v1/01_table/` as a KDD reviewer, asking:

1. **Is the experiment statistically sound?**
2. **Is it scientifically rigorous?**
3. **Is it fair?**
4. **Do the authors make correct interpretations?**

**Context discovered**: After reviewing ALL experiments (Tables 2, Figures 3-8), I found that **semantic categories are used ONLY descriptively in Table 1**. They play **no mechanistic role** in any experiment. This is critical context that significantly affects the review.

---

## Key Findings

### ✅ **EXCELLENT: Data Quality & Statistical Rigor**

**What's done well**:

1. **Complete provenance**: Every data point traced to source (LMSYS Arena, RouteLLM)
2. **Stratification**: Dev vs Holdout χ²=0.78, p=0.94 (perfect match)
3. **Leakage prevention**: 243 overlaps removed, automated verification
4. **Proper statistics**: Wilson score confidence intervals, chi-square tests
5. **Sample size**: 1,871 evaluation prompts exceeds prior work
6. **Transparency**: "Categories used descriptively; findings independent of accuracy"

**This is publication-quality work in experimental design.**

---

### ❌ **CRITICAL ISSUE: Misleading Validation Claims** ✅ **FIXED**

**What was wrong**:

Original claim in LaTeX table:
> "Validated using 3 LLM annotators... with substantial inter-annotator agreement (Fleiss' κ=0.75, n=100), confirming categories are reliable and meaningful."

**Problems**:
1. "Validated" implies comparison to ground truth (human labels) ❌
2. κ=0.75 measures **LLM-to-LLM agreement**, not accuracy ❌
3. Actual heuristic accuracy vs LLM consensus: **49%** ❌
4. Math/Logic category: 0% precision, 0% recall ❌
5. Largest category (Conversational): 29% recall ❌

**This is circular reasoning**: Using AI to validate AI.

**What was fixed** ✅:

New text in LaTeX table:
> "Inter-LLM agreement assessed using 3 LLM annotators... with Fleiss' κ=0.75 (n=100), indicating substantial agreement among LLMs. Note: This measures LLM consensus, not ground truth accuracy; heuristic accuracy vs. LLM majority vote is 49%. Categories used descriptively to characterize dataset composition; main experimental findings are independent of category accuracy."

**Impact**:
- ✅ Honest and transparent
- ✅ No overclaiming
- ✅ Clear about limitations
- ✅ Defensible for publication

**Files updated**:
- `table1_dataset_composition.tex`
- `analyze_dataset_composition.py`
- `README.md`
- Table regenerated

---

### ⚠️ **MAJOR CONCERN: Model Substitution Not Validated**

**The problem**:

- **Warmup data**: mixtral-8x7b vs **gpt-4-turbo** (80k battles)
- **Evaluation data**: mixtral-8x7b vs **gpt-4o** (1,871 prompts)

The paper claims:
> "Routing principles generalize across same-capability models"

**But no evidence is provided!**

**Why this matters**:

LinUCB priors encode **which model users preferred for which prompts**. If GPT-4o has different strengths/weaknesses than GPT-4-Turbo:
- Priors may be misleading
- Performance may suffer
- Results may not generalize

**What's needed**:

1. Correlation analysis: Do gpt-4-turbo and gpt-4o have correlated rewards?
2. Win rate comparison: Similar performance vs mixtral?
3. Transfer effectiveness: Do priors help or hurt? (This is in Table 2!)

**Action required**: Model substitution validation (2-3 days)
- Load both model rewards
- Compute correlation (expect r > 0.7)
- Add appendix section + table footnote

**Priority**: ⭐⭐⭐ **HIGH** (reviewers will definitely ask)

---

### ⚠️ **MODERATE CONCERN: Distribution Shift Connection Unclear**

**The acknowledged shift**:

- **Warmup**: 49.8% Conversational, 19.9% Coding
- **Evaluation**: 38% Conversational, 39% Coding
- **Difference**: 19% shift in Coding prompts (χ²=238.5, p<0.001)

**The claim**:
> "Demonstrates BanditGPT's robustness to distribution variation"

**Reviewer question**: "Where's the evidence for robustness?"

**The evidence exists** (in Table 2):
- Warmup-only: 79 regret (harmful due to mismatch)
- Tabula rasa: 40 regret (optimal, ignores mismatch)
- Corralling: 44 regret (near-optimal, adapts to mismatch)

**The problem**: This connection is not explicit in Table 1.

**Solution**: Add cross-references (1 day)
- Table 1: Forward reference to Table 2 robustness analysis
- Table 2: Backward reference to Table 1 mismatch severity

**Priority**: ⭐⭐ **MEDIUM-HIGH** (easy fix, high impact)

---

## Context: Why This Matters (or Doesn't)

### **Discovery: Categories Are Descriptive Only**

After reviewing all experiments, I found:
- **Table 1**: Uses categories for dataset composition description
- **Table 2**: No categories used
- **Figures 3-8**: No categories used
- **Main experiments**: Run on 384-D embeddings, not categories

**Conclusion**: The 49% accuracy is **not a problem** for the main results because:
1. Categories don't affect the bandit algorithm (uses embeddings)
2. Categories don't stratify results
3. Categories are explicitly labeled "descriptive"

**The only problem was**: Claiming "validated categories" when they're 49% accurate.

**Now fixed** ✅: Claims updated to be honest.

---

## Comparison: Before vs After Fixes

### **BEFORE** ❌

**Categorization claims**:
- "Validated with κ=0.75"
- "Confirms categories are reliable and meaningful"
- Gives impression of rigorous validation

**Problem**:
- Misleading (validation is LLM-only)
- Overclaims (49% accuracy)
- Not defensible to reviewers

**Model substitution**:
- Claims generalization
- No evidence provided

**Distribution shift**:
- Acknowledged
- Connection to robustness unclear

---

### **AFTER** ✅

**Categorization claims**:
- "Inter-LLM agreement assessed (κ=0.75)"
- "Measures LLM consensus, not ground truth"
- "Heuristic accuracy: 49%"
- "Descriptive purpose; main findings independent"

**Improvement**:
- ✅ Honest and transparent
- ✅ Limitations clear
- ✅ Defensible to reviewers

**Model substitution**:
- ⚠️ Still not validated (next action item)

**Distribution shift**:
- ⚠️ Connection still needs clarification (next action item)

---

## Scientific Assessment

### **Statistical Soundness**: ⚠️ **MOSTLY SOUND**

| Aspect | Rating | Notes |
|--------|--------|-------|
| Data provenance | ✅ Excellent | Complete, transparent |
| Stratification | ✅ Excellent | χ²=0.78, p=0.94 |
| Confidence intervals | ✅ Excellent | Wilson score, appropriate |
| Leakage prevention | ✅ Excellent | Automated checks |
| Sample size | ✅ Good | 1,871 prompts sufficient |
| Categorization | ✅ Fixed | Now honest about limitations |
| Model substitution | ⚠️ Unclear | Not validated |

### **Fairness**: ✅ **FAIR**

- Data from public dataset (LMSYS Arena)
- Stratified sampling ensures representativeness
- No obvious selection biases
- **Minor**: Language distribution not analyzed

### **Interpretation**: ✅ **NOW CORRECT** (after fixes)

- ✅ Categories are descriptive (correctly stated)
- ✅ Stratification is effective (validated)
- ✅ Distribution shift acknowledged (quantified)
- ✅ Validation limitations transparent (after fix)
- ⚠️ Model substitution generalization (still needs evidence)

---

## Publication Readiness

### **Current Status**: ⚠️ **MINOR REVISION NEEDED**

After Tier 1 fixes ✅, the main blocker is:
- Model substitution validation (2-3 days)
- Distribution shift cross-references (1 day)

**Timeline**: ~1 week to "ready for acceptance"

### **Recommendation Trajectory**

| Stage | Status | Recommendation |
|-------|--------|----------------|
| **Original** | Misleading claims | Major Revision |
| **After Tier 1** ✅ | Honest, transparent | Minor Revision |
| **After Tier 2** | Validated, connected | Accept |
| **After Tier 3** (optional) | Gold standard | Strong Accept |

---

## Next Steps (Prioritized)

### **MUST DO** (1 week)

1. **Model substitution validation** ⭐⭐⭐ (2-3 days)
   - Compute correlation between gpt-4-turbo and gpt-4o
   - Add appendix section with results
   - Update table footnotes

2. **Distribution shift cross-references** ⭐⭐ (1 day)
   - Add forward reference: Table 1 → Table 2
   - Add backward reference: Table 2 → Table 1
   - Make robustness evidence explicit

### **NICE TO HAVE** (2+ weeks)

3. **Human validation** ⭐ (1-2 weeks)
   - Recruit 2-3 expert annotators
   - Validate 200-300 prompts
   - Report inter-rater reliability + accuracy

4. **Stratified analysis** ⭐ (2-3 days)
   - Analyze performance by category
   - Show category-specific effects
   - Validate clustering hypothesis

---

## Key Documents Created

1. **`REVIEWER_ASSESSMENT.md`**: Complete technical review (15 pages)
2. **`ACTION_PLAN.md`**: Prioritized fix list with timelines
3. **`REVIEW_SUMMARY.md`**: This executive summary

**All documents located in**: `experiments_v1/01_table/`

---

## Conclusion

### **The Good** ✅

The experiment is **fundamentally sound**:
- Excellent experimental design
- Rigorous statistical methodology
- Transparent documentation
- Publication-quality data provenance

### **The Issues** ⚠️

Three issues need attention:
1. ✅ **Fixed**: Misleading validation claims → now honest
2. ⚠️ **Needs work**: Model substitution → add validation
3. ⚠️ **Needs work**: Distribution shift → add cross-references

### **The Path Forward** 🎯

**Immediate** (Tier 1): ✅ **DONE**
- Categorization claims fixed
- Table regenerated
- Transparency added

**Short-term** (Tier 2, ~1 week):
- Validate model substitution
- Clarify distribution shift robustness
- **Result**: Ready for acceptance

**Long-term** (Tier 3, optional):
- Human validation
- Stratified analysis
- **Result**: Gold standard submission

### **Bottom Line**

With Tier 1 fixes ✅ complete and Tier 2 (~1 week) underway, this will be a **strong, defensible submission** ready for publication at a top venue.

**Current recommendation**: **Minor Revision** → **Accept** (after model substitution validation)

---

**Review Date**: February 13, 2026  
**Reviewer**: Statistical methodology expert  
**Next Action**: Model substitution validation (start immediately)  
**Expected resolution**: 1 week to publication-ready
