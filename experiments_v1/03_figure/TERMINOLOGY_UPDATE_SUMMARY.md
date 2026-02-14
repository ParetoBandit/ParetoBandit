# ✅ Terminology Update Complete: "Better Expert" Clarified

**Date:** February 14, 2026  
**Issue:** Ambiguous term "better expert" used throughout documentation  
**Solution:** Replaced with precise, mechanism-based language

---

## What Changed

### Before (Ambiguous)
> "γ=0.05 commits 90% to better expert"

❓ **Questions this raises:**
- Which expert is "better"?
- Better at what?
- How is "better" determined?

### After (Precise)
> "γ=0.05 allocates 90% weight to the empirically higher-reward expert (based on observed losses)"

✅ **Now clear:**
- "Higher-reward" = lower cumulative loss
- Determined empirically through observed performance
- Explicit mechanism: weight tracks cumulative loss

---

## Files Updated (8 total)

### 1. **Core Code Documentation**
**File:** `src/bandit_gpt/router.py`

Added clarification:
```python
- Decisiveness: Achieves lowest minimum weights (~10^-4), indicating strong adaptation
  (allocates 80-90%+ weight to the higher-reward expert based on empirical performance)
```

### 2. **LaTeX Figure Caption**
**File:** `experiments_v1/03_figure/figure_gamma_ablation_caption.tex`

Updated Panel (C) description:
```latex
the meta-learner allocates 80--90\% weight to the empirically higher-reward expert 
(lower observed loss), not failure.
```

### 3. **Paper Appendix**
**File:** `paper/sections/appendix_d.tex`

Clarified in Appendix D.3:
```latex
The meta-learner confidently allocates 80--90\% weight to the empirically 
higher-reward expert (based on observed losses), minimizing allocation to 
the lower-reward expert
```

### 4-8. **Supporting Documentation**
- `GAMMA_ONE_PAGE_SUMMARY.md`
- `GAMMA_ABLATION_STORY.md`
- `GAMMA_ABLATION_REVIEWER_USER_GUIDE.md` *(+ added "Key Concept" section)*
- `GAMMA_ABLATION_COMPLETE_PACKAGE.md`
- `GAMMA_DEFAULT_UPDATE.md`

All instances of "better expert" replaced with:
- "higher-reward expert"
- "lower-loss expert"
- "empirically superior expert"

---

## New Terminology Guide

| Context | Use This |
|---------|----------|
| **Short form** | "higher-reward expert" |
| **Technical** | "lower-loss expert" (cumulative loss tracking) |
| **Full precision** | "empirically higher-reward expert based on observed losses" |
| **Avoid** | ❌ "better expert", "winning expert", "superior expert" (without qualifier) |

---

## The Mechanism Explained

### How Corralling Determines Expert Quality

```python
# For each request cycle:
1. Corralling selects Expert i with probability p_i
2. Expert i selects a model
3. Observe reward: r_t (e.g., +1 for thumbs up, -1 for thumbs down)
4. Compute loss: loss_t = -r_t
5. Update cumulative loss: L_i += loss_t

# Compute weights:
weight_i ∝ exp(-learning_rate × L_i)

# Result:
# - Expert with LOWER cumulative loss → HIGHER weight
# - Expert with HIGHER cumulative loss → LOWER weight
```

### In Our Experiments

**Experts:**
- Warmup: Uses pre-trained priors
- Tabula Rasa: Learns from scratch

**On LMSYS Holdout:**
- Warmup: Lower cumulative loss → Weight: 94%
- Tabula Rasa: Higher cumulative loss → Weight: 6%

**Therefore:** Warmup is the "higher-reward expert" on this dataset.

---

## Key Addition: "Key Concept" Section

Added to `GAMMA_ABLATION_REVIEWER_USER_GUIDE.md`:

```markdown
## 🔍 Key Concept: What Makes One Expert "Better"?

### How Corralling Evaluates Experts

Each routing cycle:
1. Corralling selects an expert (e.g., Warmup) with probability p
2. That expert selects a model (e.g., GPT-4)
3. User feedback arrives: reward = +1 or -1
4. Corralling computes loss: loss = -reward

Over many cycles:
- Expert with lower cumulative loss → "higher-reward expert"
- Expert with higher cumulative loss → "lower-reward expert"

**Key Point:** "Better" is determined empirically by observed performance,
not assumed a priori.
```

---

## Impact

### For Reviewers
✅ Clear mechanism linking weights to performance  
✅ No ambiguity about what "better" means  
✅ Explicit loss-tracking explanation  

### For Users
✅ Understand how system evaluates experts  
✅ Know that determination is data-driven  
✅ Can interpret weight evolution meaningfully  

### For Developers
✅ Consistent terminology across codebase  
✅ Clear documentation trail  
✅ Future docs have terminology guide  

---

## Quick Reference

**When describing expert selection, always:**

1. ✅ Use "higher-reward expert" or "lower-loss expert"
2. ✅ Mention it's based on "empirical performance" or "observed losses"
3. ✅ Explain mechanism if space allows: `weight ∝ exp(-η × cumulative_loss)`
4. ❌ Avoid bare terms like "better", "winning", "superior" without context

---

## Example Transformations

### Panel (C) Description

**Before:**
> "Commits to better expert"

**After:**
> "Allocates 90% to the higher-reward expert based on empirical performance"

### Docstring

**Before:**
> "Decisiveness: Strong adaptation"

**After:**
> "Decisiveness: Strong adaptation (allocates 80-90%+ weight to the higher-reward expert based on empirical performance)"

### User Guide

**Before:**
> "System identifies which expert is performing better"

**After:**
> "System tracks cumulative loss per expert and allocates higher weight to the expert with lower observed losses (higher rewards)"

---

## Verification

Run this grep to confirm no bare "better expert" remains:

```bash
cd experiments_v1/03_figure
grep -r "better expert" *.md *.tex | grep -v "higher-reward expert" | grep -v "lower-loss expert"
# Should return no results (or only historical references)
```

---

**Status:** ✅ Complete  
**Files Updated:** 8  
**New Documentation:** 2 (EXPERT_TERMINOLOGY_CLARIFICATION.md, this file)  
**Terminology:** Now precise and mechanism-based
