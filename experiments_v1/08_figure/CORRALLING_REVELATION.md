# The Corralling Revelation: Why Seed 42 Was Misleading

**Date**: February 13, 2026  
**Status**: 🔥 **Root cause identified**

---

## Executive Summary

The original experiment claimed to test "sensitivity to prior strength (n_eff)" but was **actually** testing "whether Corralling chooses to use semantic transfer at all."

**Key finding**: Corralling abandons the semantic transfer expert in 67% of seeds (2/3), making n_eff completely irrelevant in those cases.

---

## The Smoking Gun

### Corralling Expert Weights by Seed

| Seed | Warmup Expert | Tabula Rasa Expert | Interpretation |
|------|---------------|-------------------|----------------|
| **42** | **100%** | 0% | Semantic transfer is USED → n_eff matters |
| **43** | **0%** | **100%** | Semantic transfer IGNORED → n_eff irrelevant |
| **44** | **0%** | **100%** | Semantic transfer IGNORED → n_eff irrelevant |

### Proof: Reward Comparison

**Seed 42** (Warmup expert active):
- n_eff=1.0: 4.477
- n_eff=20.0: 4.280  
- **Gap: +4.6%** ← n_eff matters!

**Seed 43** (Tabula rasa active):
- n_eff=1.0: 4.254
- n_eff=20.0: 4.267
- **Gap: -0.3%** ← n_eff irrelevant!

**Seed 44** (Tabula rasa active):
- n_eff=1.0: 4.228
- n_eff=20.0: 4.228
- **Gap: 0.0%** ← n_eff COMPLETELY ignored!

---

## Why This Happened

### Corralling Decision Logic

Corralling meta-learning chooses between:
1. **Warmup Expert** (CostAwareLinUCBRouter) - Uses semantic transfer with n_eff
2. **Tabula Rasa Expert** (CostAwareTabulaRasaRouter) - Ignores all priors, learns from scratch

**Decision is made based on early performance** (t=0-300):
- If warmup expert wins early → trust it (seed 42)
- If tabula rasa wins early → trust it (seeds 43-44)

### Why Early Performance Varies

**Data ordering affects which expert looks better**:
- **Seed 42 ordering**: Early prompts match warmup prior distribution → warmup succeeds
- **Seed 43-44 ordering**: Early prompts mismatch priors → warmup fails, tabula rasa explores better

---

## Implications

### 1. Original Claims Are Invalidated

❌ **"n_eff=1.0 is empirically optimal"**
- **Reality**: Only matters in 33% of seeds where warmup expert is used

❌ **"Weak priors outperform strong priors"**  
- **Reality**: In 67% of seeds, ALL priors are ignored (no difference)

❌ **"Narrow robustness band confirms production-readiness"**
- **Reality**: Robustness comes from Corralling switching to tabula rasa, not from n_eff insensitivity

### 2. The Real Story

✅ **Corralling is working as designed**:
- Seed 42: Priors are helpful → uses them → benefits from optimal n_eff
- Seeds 43-44: Priors are harmful → ignores them → n_eff irrelevant

✅ **System is robust, but not for the claimed reason**:
- Not because "all n_eff values work equally well"
- But because "Corralling falls back to cold start when priors fail"

✅ **The wrong question was asked**:
- **Original question**: "What is the optimal n_eff?"
- **Better question**: "When should we use semantic transfer vs cold start?"

---

## Why Seed 42 Gave Misleading Results

**Seed 42 is a special case** where:
1. Early data ordering happens to favor warmup priors
2. Corralling gives 100% weight to warmup expert
3. n_eff differences are fully expressed
4. Results look like "semantic transfer is working great"

**But this is only 1 of 3 seeds!**

In the majority case (seeds 43-44):
1. Early data ordering disfavors warmup priors
2. Corralling abandons warmup expert
3. n_eff parameter is completely ignored
4. Performance is identical regardless of n_eff

**Cherry-picking seed 42 made semantic transfer look more important than it is.**

---

## What This Means for the Paper

### Section Must Be Completely Rewritten

**Old narrative**: "We optimize n_eff for semantic transfer"

**New narrative**: "Corralling automatically switches between semantic transfer and cold start"

### Honest Findings to Report

1. **Corralling switches strategies based on data**:
   - 33% of seeds: Uses semantic transfer (warmup expert wins)
   - 67% of seeds: Uses cold start (tabula rasa expert wins)

2. **n_eff only matters when semantic transfer is used**:
   - When warmup expert active: n_eff=1.0 beats n_eff=20.0 by 4.6%
   - When tabula rasa active: All n_eff values are identical (ignored)

3. **System robustness comes from meta-learning, not parameter tuning**:
   - Corralling's ability to switch experts is the real robustness mechanism
   - n_eff calibration is secondary

### Recommended Revisions

**Option A: Reframe as Meta-Learning Success**
- Title: "Adaptive Expert Selection in Semantic Transfer"
- Claim: "Corralling automatically chooses between transfer and cold start"
- Deployment: "Trust Corralling; n_eff=5.0 is fine when warmup expert is used"

**Option B: Remove Section**
- Delete n_eff sensitivity analysis
- Focus on Corralling's expert switching behavior (more interesting!)

**Option C: Honest Null Result**
- Report: "n_eff effect is seed-dependent (33% of cases)"
- Report: "Corralling often prefers cold start over semantic transfer (67%)"
- Deployment: "System works regardless of n_eff due to expert switching"

---

## Production Implications

### Should We Revert n_eff Change (1.0 → 5.0)?

**No longer matters!**

Why:
1. In 67% of traffic patterns, Corralling will use tabula rasa → n_eff ignored
2. In 33% of traffic patterns, Corralling will use warmup → n_eff matters
3. Production A/B test would see **33% of users** with n_eff effect, 67% with none
4. Result: Tiny overall effect size (~2% = 0.33 × 6%)

**Better production question**: 
- Should we tune Corralling learning rate to prefer warmup vs tabula rasa?
- Currently Corralling favors tabula rasa (67% weight) - is that optimal?

---

## Scientific Lessons

### What We Learned About Experiment Design

1. **Corralling confounds causal interpretation**:
   - We thought we were testing n_eff effect
   - Actually testing expert selection + n_eff effect (entangled)

2. **Single-seed experiments can be doubly misleading**:
   - Cherry-picked seed (seed 42)
   - In a cherry-picked expert regime (warmup-dominant)

3. **Meta-learning requires meta-analysis**:
   - Must track expert weights over time
   - Must check if conclusions depend on which expert is active

### How to Fix Future Experiments

**When using Corralling**:
1. Always report expert weights alongside performance
2. Stratify analysis by dominant expert (warmup vs tabula rasa regimes)
3. Test with AND without Corralling to isolate effects

**Alternative**: 
- Turn off Corralling (`use_corralling=False`)
- Test semantic transfer in isolation
- Then separately test Corralling's switching behavior

---

## Revised Experimental Protocol

### Experiment 8A: n_eff Sensitivity (Pure Semantic Transfer)

**Setup**: `use_corralling=False`, only use CostAwareLinUCBRouter

**Purpose**: Isolate n_eff effect without meta-learning confound

**Hypothesis**: n_eff=1.0 beats n_eff=20.0 when forced to use semantic transfer

### Experiment 8B: Meta-Learning Analysis (Corralling Behavior)

**Setup**: `use_corralling=True`, track expert weights

**Purpose**: Understand when Corralling chooses semantic transfer vs cold start

**Hypothesis**: Corralling prefers tabula rasa when priors mismatch data

### Experiment 8C: Combined Performance

**Setup**: Current experiment, but report:
- Overall performance (blended)
- Performance stratified by dominant expert
- Expert weights over time

**Purpose**: Show that robustness comes from adaptive expert selection

---

## Conclusion

The original experiment had a **confound** (Corralling expert selection) that went unnoticed because seed 42 happened to favor the warmup expert.

**The real story**:
- Seed 42: Warmup expert active → n_eff matters → strong effect
- Seeds 43-44: Tabula rasa active → n_eff ignored → no effect

**Multi-seed validation saved the paper** by revealing:
1. Original claims were seed-specific, not general
2. n_eff effect depends on Corralling's expert choice
3. System robustness comes from meta-learning, not parameter tuning

**This is actually MORE interesting scientifically** - Corralling's adaptive behavior is the real innovation, not n_eff optimization!

---

**Recommendation**: Rewrite Section 8 to focus on Corralling's meta-learning behavior rather than n_eff calibration. The adaptive expert selection is the real robustness mechanism.
