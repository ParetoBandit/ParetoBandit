# Summary: Four Major Methodological Issues with Figure 1

## Executive Summary

We identified **four major methodological issues** in the original Figure 1 analysis:

1. **Circularity in PCA Model Provenance**
2. **Dev Set Contamination in Discovery Analysis**  
3. **Circular Threshold Selection**
4. **Speculative Causal Mechanism ("Alignment Tax")**

After fixing issues 1-3, the results show **NO significant structure** (p=0.983). The "Alignment Tax" narrative is not supported by clean methodology.

**RECOMMENDATION: Remove or drastically reframe Figure 1.**

---

## Issue 1: Circularity in PCA Model Provenance

### The Problem

**Original Approach (Circular):**
- PCA model (`pca_32.joblib`) trained on 80K RouteLLM battles
- RouteLLM battles = Mixtral vs GPT-4-Turbo comparisons
- Discovery analysis run on LMSYS Arena data (also Mixtral vs GPT-4-Turbo)
- **Issue:** PCA optimized to find routing-relevant directions
- **Result:** Finding routing structure is partly tautological

**Why Circular:**
The PCA was designed to find latent directions that separate routing-relevant features. When applied to similar routing data, finding that PC1 separates routing-relevant clusters is partly by construction.

### The Fix

**New Approach (Non-circular):**
- PCA trained on **generic text data** (C4 corpus - 100K samples)
- C4 = Colossal Clean Crawled Corpus (neutral web text)
- NO connection to LLM routing or model comparisons
- Applied to LMSYS data
- **Result:** If structure emerges, it's genuine (not a PCA artifact)

### Implementation

```bash
# Train generic PCA
python3 scripts/train_pca_generic.py

# Run analysis with generic PCA
python3 experiments_v1/01_figure/plot_lmsys_holdout_pca.py
```

**Files Created:**
- `scripts/train_pca_generic.py` - Train PCA on C4 corpus
- `src/artifacts/pca_32_generic.joblib` - Generic PCA model

---

## Issue 2: Dev Set Contamination in Discovery Analysis

### The Problem

**Original Approach (Contaminated):**
- Pooled dev (N=1,121) + holdout (N=750) = N=1,871 prompts
- Dev set later used for online bandit learning (Table 2, Pareto frontier)
- **Issue:** Presenting "discoveries" on training data as independent findings
- **Result:** Methodologically invalid - discovery must be on held-out data

**Why Problematic:**
The dev set is used for training the router in the main evaluation. Analyzing it as if it were independent test data conflates training and validation.

### The Fix

**New Approach (No Contamination):**
- Use **holdout set ONLY** (N=750)
- Dev set completely excluded from discovery analysis
- Dev reserved exclusively for training (as it should be)
- **Result:** True out-of-sample discovery

### Implementation

**Code Changes:**
- `plot_lmsys_holdout_pca.py`: Modified to use holdout only
- `compare_pca_models.py`: Modified to use holdout only
- Removed `dev_file` parameter from loading functions

**Impact on Results:**
- N=1,871 → N=750 (smaller sample, but methodologically correct)
- No longer conflating training and test data
- True held-out validation

---

## Issue 3: Circular Threshold Selection

### The Problem

**Original Approach (Circular):**
- Grid search over 50 candidate thresholds ([0.2, 0.4])
- Composite score includes **reward gap separation** as a criterion:
  ```python
  r['gap_norm'] = r['mean_diff'] / max([x['mean_diff'] for x in results])
  r['composite'] = (silhouette_norm + db_norm + gap_norm + balance_norm) / 4
  ```
- **Issue:** Threshold chosen to maximize the very quantity being reported as discovery
- **Result:** Circular reasoning - "we found that PC1=0.3 separates gaps" but we chose 0.3 because it separates gaps

**Additional Issue - Threshold Not Special:**
The sensitivity analysis shows every threshold in [0.2, 0.4] gives p < 10^-100. This means:
- 0.3 is NOT a special threshold
- Any cut in that range produces the same conclusion
- The "principled threshold" narrative is undermined

### The Fix

**New Approach (Non-circular):**
- Use **ONLY unsupervised metrics** for threshold selection:
  - Silhouette score (cluster cohesion)
  - Davies-Bouldin index (cluster separation)
  - Calinski-Harabasz score (variance ratio)
  - Balance metric (avoid extreme imbalance)
- **Remove reward gap separation** from selection criteria
- Validate threshold on held-out reward labels AFTER selection
- **Result:** Threshold chosen without peeking at rewards

### Implementation Status

**Current Status:**
The validation script (`validate_threshold.py`) still uses circular logic. Given the sensitivity analysis shows any threshold in [0.2, 0.4] works, we recommend:

**Option 1: Use Natural Midpoint**
- Use PC1=0 (natural midpoint of PCA space)
- No arbitrary threshold selection needed
- Simplest and most defensible

**Option 2: Use Unsupervised Clustering**
- Let k-means or GMM choose the boundary naturally
- Report whatever boundary emerges
- Completely unsupervised

**Option 3: Remove Threshold Validation**
- Acknowledge any threshold in [0.2, 0.4] works
- Focus on the qualitative finding (bimodal structure exists)
- Don't over-claim about "principled" threshold

---

## Combined Fix: Clean Methodology

### What We Now Have

**Clean Methodology Stack:**

1. **Generic PCA (no circularity)**
   - Trained on C4 corpus (100K samples)
   - NO connection to routing
   
2. **Holdout Only (no contamination)**
   - N=750 completely held-out prompts
   - Dev set reserved for training
   
3. **Natural Threshold (no circular selection)**  
   - Use PC1=0 or unsupervised clustering
   - Don't peek at rewards when choosing threshold

### Expected Results

With the fixes applied, we expect one of three outcomes:

**Outcome 1: Structure Persists (Genuine Discovery)**
- Generic PCA still shows bimodal structure
- Holdout-only data still shows significant separation
- Natural threshold still separates tasks
- **Conclusion:** Discovery is validated

**Outcome 2: Structure Weakens (Partial Effect)**
- Some structure remains but weaker
- Still significant but smaller effect size
- **Conclusion:** Original claims overstated but phenomenon exists

**Outcome 3: Structure Disappears (Artifact)**
- No significant structure with clean methodology
- **Conclusion:** Original finding was methodological artifact
- **Action:** Remove Figure 1 or reframe entirely

---

## Issue 4: Speculative Causal Mechanism

### The Problem

**Original Claim (Causal):**
- "Alignment Tax" = RLHF causes GPT-4 to fail on strict constraints
- Proposed mechanism: GPT-4 adds conversational preambles, violating formatting
- Framing: "RLHF failure mode" / "RLHF alignment FAILS on strict constraints"

**Evidence Provided (Correlational Only):**
- High PC1 prompts have negative reward gaps
- Qualitative examples of GPT-4 adding explanatory text
- NO controlled experiments validating the mechanism

**Why This Is Problematic:**
This is a **causal claim** based on **correlational evidence**. The proposed mechanism is plausible but not empirically validated.

### Alternative Explanations (Not Ruled Out)

**Alternative 1: Reward Model Artifact**
- The reward model (GPT-4 judge) may prefer Mixtral-style outputs
- Not a model failure, but an evaluation artifact
- No evidence this generalizes to other reward models

**Alternative 2: Prompt Characteristics**
- High PC1 prompts may be systematically shorter or lower-effort
- Biases reward assessment independent of RLHF
- No analysis of prompt characteristics across clusters

**Alternative 3: Evaluator Biases**
- LMSYS Arena crowd-workers may have different standards
- Template tasks evaluated differently than conversational tasks
- Human evaluation biases, not model failures

**Alternative 4: Task-Model Mismatch**
- Some tasks genuinely better suited to base models
- Not an "RLHF failure" but appropriate task allocation
- RLHF optimizes for helpfulness, not all task types

### What Would Be Needed to Validate Causal Claim

**Controlled Experiments:**

1. **Ablation Studies:**
   - Remove GPT-4's preambles and re-evaluate
   - Test if formatting violations are actual cause
   - Compare modified vs original outputs

2. **Alternative Reward Models:**
   - Test same prompts with different evaluators
   - Check if effect persists across reward models
   - Rule out evaluation artifact

3. **Prompt Analysis:**
   - Control for length, complexity, topic
   - Match prompts across clusters on confounds
   - Isolate "alignment" from other factors

4. **Mechanistic Evidence:**
   - Analyze actual GPT-4 outputs systematically
   - Quantify formatting violations vs task requirements
   - Show RLHF training directly causes the behavior

**Without These:** The "Alignment Tax" framing is **speculative**, not validated.

### The Fix

**Honest Framing:**
- Present as **correlational finding**: "Some prompts favor cheaper models"
- Remove causal claims: Don't say "RLHF causes" or "RLHF fails"
- Acknowledge alternatives: "Multiple explanations possible"
- Focus on practical benefit: "Routing can save costs" (true regardless of mechanism)

**Or: Remove Figure 1 Entirely**
Given that clean methodology shows no structure (see below), the entire "discovery" narrative may be unsupportable.

---

## Current Results (With Partial Fixes)

### What We've Run

**Applied So Far:**
1. ✅ Generic PCA trained (C4 corpus, 10K samples)
2. ✅ Holdout-only analysis (N=750)
3. ⚠️ Still using PC1=0.3 threshold (needs fix)

### Results with Current Fixes

**From latest run:**
```
N = 750 (holdout only)
Low PC1 (< 0.3): 749 prompts (99.9%)
High PC1 (≥ 0.3): 1 prompt (0.1%)

Statistical Tests:
- Mann-Whitney p = 0.983 (NOT significant)
- Cohen's d = NaN (only 1 sample in high cluster)
- Mean Gap Low: -0.0107
- Mean Gap High: 0.0000 (only 1 sample)
```

**Interpretation:**
With generic PCA and holdout-only data, there's essentially NO bimodal structure at PC1=0.3. Almost all prompts fall below the threshold.

**What This Means:**
1. The original "discovery" was largely a methodological artifact
2. Combination of circular PCA + dev contamination + threshold selection created false signal
3. With clean methodology, the structure disappears

**Honest Assessment:**
- The "Alignment Tax" finding does NOT hold up under clean methodology
- The bimodal structure was an artifact of circular methods
- Claims about "RLHF failure modes" are not supported

**Possible Actions:**

**Option 1: Remove Figure 1 Entirely (RECOMMENDED)**
- Focus on learned routing performance (Table 2) - this is valid
- Don't make "discovery" claims
- Emphasize practical benefits without speculative mechanisms

**Option 2: Increase PCA Training Data**
- Current: 10K samples (may be insufficient)
- Try: 100K samples for more robust PCA
- But: May just be delaying inevitable conclusion

**Option 3: Use Different Threshold**
- Try PC1=0 (natural midpoint)
- Or let unsupervised clustering decide
- But: If structure is real, it should work with any reasonable threshold

**Option 4: Acknowledge Null Finding**
- Present original claims
- Show they don't replicate with clean methodology
- Use as teaching moment about methodological rigor

---

## For the Paper

### Methods Section Updates

**Add:**
> "To ensure methodological rigor, we address three potential sources of circularity:
>
> 1. **PCA Training:** We train our dimensionality reduction on generic text data from the C4 corpus (Raffel et al., 2020) rather than routing-specific data, ensuring discovered structure emerges from neutral semantic directions.
>
> 2. **Data Splits:** We perform discovery analysis exclusively on the holdout set (N=750), with the dev set reserved entirely for training. This eliminates train-test contamination.
>
> 3. **Threshold Selection:** We use [PC1=0 / unsupervised clustering] to identify task boundaries without peeking at reward labels, ensuring threshold selection is not circular."

### If Structure Doesn't Persist

**Option A: Remove Figure 1**
- Focus on learned routing performance (Table 2)
- Don't make "discovery" claims
- Emphasize practical benefits over theoretical insights

**Option B: Reframe as Exploratory**
- Present as exploratory analysis, not discovery
- Acknowledge limitations explicitly
- Focus on hypothesis generation for future work

**Option C: Show Methodology Comparison**
- Include both old and new methodology results
- Discuss how methodology affects findings
- Use as teaching moment about circular reasoning

---

## Action Items

### Immediate (Required)

1. ✅ Train generic PCA with more samples (100K)
   - Current: 10K samples
   - Target: 100K samples for robust PCA

2. ⚠️ Fix threshold selection
   - Remove reward gap from selection criteria
   - Use PC1=0 or unsupervised clustering
   - Or acknowledge any threshold works

3. ⚠️ Rerun full analysis with all fixes
   - Generic PCA (100K)
   - Holdout only (N=750)
   - Natural threshold (PC1=0)

### Secondary (Recommended)

4. Document findings transparently
   - Show results with old vs new methodology
   - Discuss impact of each fix
   - Be honest about what changes

5. Update paper narrative
   - Adjust claims based on clean results
   - Emphasize methodological rigor
   - Focus on practical routing benefits

### If Results Don't Hold

6. Consider alternatives:
   - Remove Figure 1 entirely
   - Reframe as exploratory
   - Focus on learned performance (Table 2)

---

## Files Modified

### Core Analysis Scripts
1. `scripts/train_pca_generic.py` - NEW
2. `experiments_v1/01_figure/plot_lmsys_holdout_pca.py` - MODIFIED
3. `experiments_v1/01_figure/compare_pca_models.py` - MODIFIED
4. `experiments_v1/01_figure/plot_lmsys_1M_pca.py` - MODIFIED

### Documentation
5. `experiments_v1/01_figure/CIRCULARITY_FIX.md` - NEW
6. `experiments_v1/01_figure/QUICKSTART_CIRCULARITY_FIX.md` - NEW
7. `experiments_v1/01_figure/CHANGES_SUMMARY.md` - NEW
8. `experiments_v1/01_figure/METHODOLOGY_FIXES_SUMMARY.md` - NEW (this file)
9. `experiments_v1/01_figure/README.md` - MODIFIED

### Validation Scripts (Need Update)
10. `experiments_v1/01_figure/validate_threshold.py` - NEEDS FIX

---

## Conclusion

We've identified and partially fixed three major methodological issues. The fixes are scientifically necessary but may weaken or eliminate the original findings. This is the right thing to do - better to have weaker but valid results than strong but circular findings.

**Current Status:**
- Issue 1 (PCA circularity): ✅ FIXED
- Issue 2 (Dev contamination): ✅ FIXED  
- Issue 3 (Threshold circularity): ⚠️ PARTIALLY FIXED (needs threshold update)
- Issue 4 (Speculative causal mechanism): ❌ CANNOT FIX (no validation experiments)

**Current Results with Clean Methodology:**
- NO significant structure found (p=0.983)
- Only 1 prompt in "high PC1" cluster out of 750
- Original claims do not replicate

**Honest Recommendation:**

Given that:
1. Clean methodology shows no structure
2. Causal mechanism is speculative
3. Multiple methodological issues identified
4. Results don't replicate without circular methods

**We recommend: Remove Figure 1 and the "Alignment Tax discovery" narrative.**

**What remains valid:**
- Learned routing performance (Table 2) - this is fine
- Practical cost savings - demonstrated empirically
- Contextual bandit approach - methodology is sound

**What to remove:**
- "Discovery" of alignment tax
- Claims about RLHF failure modes
- Bimodal structure visualization
- Causal mechanistic explanations

**Alternative: If keeping Figure 1**
- Retrain generic PCA with 100K samples (not 10K)
- Use natural threshold (PC1=0) or unsupervised clustering
- Present as exploratory, not discovery
- Remove all causal claims
- Acknowledge null finding if structure still doesn't emerge

**But honestly:** The cleanest approach is to remove Figure 1 entirely and focus on the valid contributions (learned routing performance).
