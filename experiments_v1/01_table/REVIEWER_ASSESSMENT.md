# Statistical Review: Table 1 Dataset Composition Experiment
**Reviewer Role**: KDD Reviewer  
**Date**: February 13, 2026  
**Experiment**: experiments_v1/01_table/

---

## EXECUTIVE SUMMARY

**Overall Assessment**: ⚠️ **MAJOR REVISION REQUIRED**

The dataset composition analysis is **well-executed** in terms of data provenance, stratification validation, and statistical rigor. However, the **categorization validation claims are misleading** and must be corrected before publication.

**Key Issues**:
1. ❌ **Critical**: Validation uses LLMs, not humans (circular reasoning)
2. ❌ **Critical**: Claims "validated categories" when heuristic accuracy is 49%
3. ⚠️ **Major**: Model substitution (gpt-4-turbo → gpt-4o) not validated
4. ⚠️ **Moderate**: Distribution shift acknowledged but impact on results unclear

**Good Practices**:
- ✅ Excellent data provenance documentation
- ✅ Strong stratification validation (χ²=0.78, p=0.94 for dev vs holdout)
- ✅ Proper confidence intervals (Wilson score)
- ✅ Transparent about category purpose ("descriptive only")

---

## DETAILED ASSESSMENT

### 1. CATEGORIZATION VALIDATION ❌ **CRITICAL ISSUE**

#### **Problem**: Circular Validation

**What the paper claims**:
> "Validated using 3 LLM annotators... with substantial inter-annotator agreement (Fleiss' κ=0.75, n=100), confirming categories are reliable and meaningful."

**What's actually measured**:
- κ=0.75 measures **LLM-to-LLM agreement**, NOT ground truth accuracy
- Heuristic accuracy vs. LLM consensus: **49%** (see `validation_results_100.json`)
- Math/Logic category: **0% precision, 0% recall**
- Conversational category: **29% recall** (20 out of 45 correct)

**Why this is problematic**:
1. Using AI to validate AI is circular reasoning
2. "Validation" typically means comparison to ground truth (human labels)
3. κ=0.75 only means the 3 LLMs agree with each other, not that they're correct
4. The claim "confirming categories are reliable and meaningful" is not supported by the data

#### **Confusion Matrix Analysis** (from validation_results_100.json)

```
Conversational (n=45 prompts):
  - Correctly classified: 13 (29%)
  - Misclassified as Knowledge: 20 (44%)
  - Misclassified as Creative: 7 (16%)
  - Misclassified as Math/Logic: 4 (9%)
```

**This is problematic because**:
- The largest category (Conversational: 37.5% of dataset) has only 29% accuracy
- More prompts are misclassified as "Knowledge" than correctly identified

#### **Solution** ✅ **IMPLEMENTED**

Updated claims to be accurate:
- Changed "validated" → "inter-LLM agreement assessed"
- Added explicit note: "measures LLM consensus, not ground truth accuracy"
- Added accuracy: "heuristic accuracy vs. LLM majority vote is 49%"
- Emphasized: "categories used descriptively; main findings independent of accuracy"

**Files updated**:
- ✅ `table1_dataset_composition.tex` (LaTeX table)
- ✅ `analyze_dataset_composition.py` (generation script)
- ✅ `README.md` (documentation)

---

### 2. MODEL SUBSTITUTION ⚠️ **MAJOR CONCERN**

#### **Problem**: Warmup vs Evaluation Use Different Models

**Warmup data**: mixtral-8x7b vs **gpt-4-turbo** (80k battles)  
**Evaluation data**: mixtral-8x7b vs **gpt-4o** (1,871 prompts)

**The paper claims**:
> "Model substitution (gpt-4-turbo→gpt-4o) reflects current flagship tier; routing principles generalize across same-capability models."

**Problem**: No evidence provided for this claim.

#### **Why this matters**:

GPT-4o has different characteristics:
- **Speed**: 2× faster than GPT-4-Turbo
- **Cost**: Originally lower ($2.50 vs $10 per 1M tokens)
- **Capabilities**: Different training data, different behavior
- **User preferences**: May differ from GPT-4-Turbo

The LinUCB warmup priors encode **which model users preferred for which prompts**. If GPT-4o has different strengths/weaknesses than GPT-4-Turbo, the priors may be misleading.

#### **Evidence needed**:

1. Correlation analysis: Do gpt-4-turbo and gpt-4o have similar win rates on a held-out set?
2. Performance analysis: Do the learned priors transfer effectively?
3. Ablation study: Compare performance with/without warmup priors when substituting models

#### **Current status**: ⚠️ **ACKNOWLEDGED BUT NOT VALIDATED**

The paper acknowledges the substitution but provides no empirical validation. Reviewers may question whether results generalize.

**Recommended action**:
- Add empirical validation (correlation analysis on held-out data)
- OR soften claims about generalization
- OR acknowledge as a limitation

---

### 3. DISTRIBUTION SHIFT ⚠️ **MODERATE CONCERN**

#### **Acknowledged in paper**:

Warmup vs Evaluation distributions differ significantly:
- **Warmup**: 49.8% Conversational, 19.9% Coding
- **Evaluation**: 38% Conversational, 39% Coding
- **Statistical test**: χ²=238.5, p<0.001, Cramér's V=0.05

#### **Authors' interpretation**:
> "Distribution differs... due to different model pair and time period. [This] demonstrates BanditGPT's robustness to distribution variation."

#### **Reviewer concern**:

The authors correctly identify this as **statistically significant** but claim "negligible effect size" (V=0.05). However:
- **19% shift in Coding prompts** (19.9% → 39%) is substantial
- For a bandit algorithm that learns from warmup, this could introduce bias
- The claim of "robustness" is not directly tested—we only see final performance

#### **What's missing**:

Evidence that the shift doesn't harm performance:
1. Ablation: Does tabula rasa (no warmup) outperform warmup in some experiments?
2. Analysis: Does the algorithm detect and correct for the mismatch?
3. Breakdown: Is performance worse on the shifted categories (Coding)?

#### **Current status**: ✅ **PARTIALLY ADDRESSED**

Looking at Table 2 results, we see:
- Warmup-only: 79 regret (harmful)
- Tabula rasa: 40 regret (optimal)
- Corralling η=1.0: 44 regret (near-optimal)

**This validates the robustness claim**—Corralling detects the mismatch and corrects for it. However, this connection should be made more explicit in Table 1.

**Recommended action**:
- Add forward reference: "See Table 2 for validation of robustness to this mismatch"
- OR add a sentence explaining why this shift is not problematic (Corralling adapts)

---

### 4. VALIDATION SAMPLE SIZE 📊 **MODERATE CONCERN**

#### **Current validation**: n=100 prompts

For per-category metrics with 5 categories:
- Coding: n=37 samples
- Conversational: n=45 samples
- Creative: n=4 samples (too small!)
- Knowledge: n=12 samples
- Math/Logic: n=2 samples (too small!)

#### **Problem**:
- **Creative** (10% of dataset): Only 4 validation samples
- **Math/Logic** (5.9% of dataset): Only 2 validation samples

With such small samples, precision/recall estimates are unreliable:
- Math/Logic: 0% precision (but only 2 samples!)
- Creative: 75% recall (but only 4 samples!)

#### **Recommended sample size**:

For reliable per-category metrics (95% CI width ≤ 20%):
- Need ≥30 samples per category minimum
- Total needed: ~150-200 prompts with stratified sampling
- OR: Report aggregate accuracy only (not per-category)

#### **Current status**: ⚠️ **ACKNOWLEDGED**

The paper does acknowledge "categories used descriptively" which mitigates this concern. If categories are truly descriptive only, the validation sample size is less critical.

---

## POSITIVE ASPECTS ✅

### 1. **Excellent Data Provenance**

The documentation is exemplary:
- Clear source attribution (LMSYS Chat Arena, RouteLLM)
- HuggingFace dataset links provided
- Processing steps documented
- Artifacts listed with paths

**This is publication-quality work.**

### 2. **Strong Stratification Validation**

Dev vs Holdout comparison:
- χ²=0.78, p=0.94 (excellent—no significant difference)
- Cramér's V=0.02 (negligible effect size)
- **Conclusion**: Stratification is effective

### 3. **Proper Confidence Intervals**

- Wilson score intervals (appropriate for proportions)
- 95% confidence level
- Narrow intervals (dataset is large enough)

Example: Coding 20.3% [19.8%, 20.8%]

### 4. **Automated Leakage Detection**

- 243 overlapping prompts removed (0.24%)
- Verification that warmup and evaluation sets are disjoint
- **Critical for preventing data leakage**

### 5. **Transparent About Limitations**

The paper already states:
> "Categories used descriptively to characterize the dataset; main experimental findings are independent of category accuracy."

**This is the right framing.** The issue is that earlier claims about "validated categories" contradict this transparency.

---

## STATISTICAL SOUNDNESS ASSESSMENT

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Data Provenance** | ✅ Excellent | Complete, transparent, reproducible |
| **Stratification** | ✅ Excellent | χ²=0.78, p=0.94 validates effectiveness |
| **Confidence Intervals** | ✅ Excellent | Wilson score, appropriate |
| **Leakage Prevention** | ✅ Excellent | Automated checks, verified disjoint |
| **Sample Size** | ✅ Good | 1,871 evaluation prompts sufficient |
| **Categorization Claims** | ❌ Poor | Misleading validation claims (49% accuracy) |
| **Model Substitution** | ⚠️ Unclear | Not validated empirically |
| **Distribution Shift** | ✅ Good | Acknowledged; impact tested in Table 2 |

---

## RECOMMENDATIONS FOR PUBLICATION

### **TIER 1: MUST FIX** ✅ **DONE**

1. ✅ **Revise categorization validation claims**
   - Change "validated" → "inter-LLM agreement assessed"
   - Add explicit note about lack of human ground truth
   - Report 49% accuracy honestly
   - Emphasize descriptive purpose

**Status**: Implemented in all files.

### **TIER 2: STRONGLY RECOMMENDED**

2. **Add model substitution validation**
   - Correlate gpt-4-turbo and gpt-4o win rates
   - Show that priors transfer effectively
   - OR acknowledge as limitation

3. **Strengthen distribution shift discussion**
   - Add forward reference to Table 2
   - Explain why 19% shift in Coding prompts is not problematic
   - Cite Corralling's adaptation as evidence

4. **Increase validation sample size (if feasible)**
   - Target: 200+ prompts with stratified sampling
   - Ensure ≥30 samples per category
   - OR: Remove per-category metrics, report aggregate only

### **TIER 3: OPTIONAL IMPROVEMENTS**

5. **Add human validation (gold standard)**
   - 2-3 expert annotators
   - 200-300 prompts
   - Compute inter-rater reliability AND accuracy
   - Compare heuristic to human consensus

6. **Perform stratified analysis by category**
   - Does performance differ for Coding vs Conversational?
   - Are there category-specific effects?
   - This would justify the categorization effort

---

## VERDICT

### **Scientific Soundness**: ⚠️ **MOSTLY SOUND**
- Core experimental design is excellent
- Statistical methods are appropriate
- Data quality assurance is thorough
- **BUT**: Categorization validation claims are misleading

### **Fairness**: ✅ **FAIR**
- Data sourced from public dataset (LMSYS Arena)
- No obvious biases in selection
- Stratification ensures representativeness
- **Minor concern**: Language distribution not analyzed

### **Interpretation**: ⚠️ **PARTIALLY CORRECT**
- ✅ Correct: Categories are descriptive only
- ✅ Correct: Stratification is effective
- ✅ Correct: Distribution shift is acknowledged
- ❌ Incorrect: "Validated categories" claim (49% accuracy)
- ⚠️ Unclear: Model substitution generalization

### **Publication Recommendation**: **MAJOR REVISION**

**Rationale**:
The categorization validation issue is significant enough to warrant major revision. However, the fix is straightforward (revise claims, report accurate numbers). The rest of the experimental design is publication-quality.

**After revision**: **ACCEPT**
- Core work is strong
- Fix addresses the main concern
- Transparency about descriptive purpose is commendable

---

## ACTION ITEMS

### **Completed** ✅

1. ✅ Updated LaTeX table with honest validation claims
2. ✅ Updated Python script to generate corrected table
3. ✅ Updated README with clear limitations
4. ✅ Regenerated table with corrected text

### **Recommended Next Steps**

1. **Add model substitution analysis** (2-3 days)
   - Load gpt-4-turbo and gpt-4o rewards
   - Compute correlation on held-out set
   - Add 1 paragraph + 1 table to appendix

2. **Strengthen distribution shift discussion** (1 day)
   - Add forward reference to Table 2 in Table 1 notes
   - Explain that Corralling adapts (cite Table 2 results)

3. **Consider human validation** (1-2 weeks)
   - If time permits, add human ground truth
   - Would elevate paper from "good" to "excellent"

---

## FINAL ASSESSMENT

### **Before Fixes**:
- ❌ Misleading validation claims
- ⚠️ Categorization presented as more rigorous than it is
- ⚠️ Model substitution not validated
- **Recommendation**: Major Revision

### **After Fixes** ✅:
- ✅ Honest, transparent validation claims
- ✅ Clear about descriptive purpose
- ✅ Limitations acknowledged
- ⚠️ Model substitution still not validated (but now acknowledged)
- **Recommendation**: Minor Revision → Accept (pending validation analysis)

---

**Reviewer Confidence**: High  
**Expertise**: Experimental design, statistical methodology, bandit algorithms  
**Recommendation**: **Accept with Minor Revisions** (after model substitution analysis)

---

**Date**: February 13, 2026  
**Review Completed**: experiments_v1/01_table/  
**Status**: Tier 1 fixes implemented ✅
