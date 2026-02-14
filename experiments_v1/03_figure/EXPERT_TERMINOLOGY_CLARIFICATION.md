# Expert Terminology Clarification

**Date:** February 14, 2026  
**Purpose:** Clarify what "better expert" means in Corralling documentation

---

## The Problem

Documentation uses phrases like:
- "commits to better expert"
- "90%+ to better expert"
- "identifies superior expert"

**This is ambiguous!** What makes one expert "better"?

---

## The Solution

### Precise Terminology

**Replace vague terms with:**
- ✅ **"higher-reward expert"** (based on observed rewards)
- ✅ **"lower-loss expert"** (based on cumulative losses)
- ✅ **"empirically superior expert"** (based on actual performance)

### How Corralling Determines Which Expert is "Better"

```python
# Corralling Meta-Learning Process:

# Step 1: Track cumulative loss per expert
cumulative_loss[expert_i] += observed_loss_t

# Step 2: Compute exponential weights
raw_weights[i] = exp(-learning_rate * cumulative_loss[i])

# Step 3: Apply mixing (gamma floor)
probability[i] = (1 - gamma) * normalized(raw_weights[i]) + gamma/K

# Result:
# - Expert with LOWEST cumulative loss → HIGHEST weight
# - Expert with HIGHEST cumulative loss → LOWEST weight
```

### In Our Experiments

**Setup:**
- Expert 1: Warmup (uses pre-trained priors from RouteLLM)
- Expert 2: Tabula Rasa (learns from scratch)

**On LMSYS Holdout Data:**
```
After 750 prompts:
  Warmup cumulative loss:     Lower  → Weight: 0.94 (94%)
  Tabula Rasa cumulative loss: Higher → Weight: 0.06 (6%)

Conclusion: Warmup is the "higher-reward expert" on this dataset
```

### Why This Matters

**Before (Ambiguous):**
> "γ=0.05 commits 90% to better expert"

❓ Which expert? Better how? Better at what?

**After (Precise):**
> "γ=0.05 allocates 90% weight to the empirically higher-reward expert (based on observed losses)"

✅ Clear mechanism: weights track empirical performance

---

## Updated Terminology Guide

| ❌ Avoid | ✅ Use Instead |
|---------|---------------|
| "better expert" | "higher-reward expert" or "lower-loss expert" |
| "superior expert" | "empirically superior expert (based on observed losses)" |
| "winning expert" | "expert with lower cumulative loss" |
| "best expert" | "highest-weighted expert" |

### Longer Explanations (When Space Allows)

**Full clarity:**
> "The meta-learner allocates 80-90% weight to the expert with empirically lower cumulative loss (higher observed rewards), based on routing performance tracked over hundreds of requests."

**Medium clarity:**
> "Allocates 90% to the higher-reward expert (based on observed performance)"

**Short form (with context):**
> "Commits to higher-reward expert" (after explaining loss tracking)

---

## Example: Panel (C) Description

### Before
> "γ=0.05 achieves lowest minimum weights, indicating strong adaptation to the superior expert (90%+)"

### After (LaTeX - Full Precision)
> "γ=0.05 achieves lowest minimum weights ($\sim$10\textsuperscript{-4}), indicating strong adaptation---the meta-learner allocates 80--90\% weight to the empirically higher-reward expert (lower observed loss)"

### After (Markdown - Balanced)
> "γ=0.05 achieves lowest minimum weights (~10^-4), indicating strong adaptation: allocates 90%+ weight to the higher-reward expert based on empirical performance"

---

## Files Updated (Feb 14, 2026)

1. ✅ `src/bandit_gpt/router.py` - Added explanation in docstring
2. ✅ `figure_gamma_ablation_caption.tex` - Updated Panel (C) description
3. ✅ `paper/sections/appendix_d.tex` - Added "based on observed losses"
4. ✅ `GAMMA_ONE_PAGE_SUMMARY.md` - Clarified terminology
5. ✅ `GAMMA_ABLATION_STORY.md` - Updated examples
6. ✅ `GAMMA_ABLATION_REVIEWER_USER_GUIDE.md` - Added "Key Concept" section
7. ✅ `GAMMA_ABLATION_COMPLETE_PACKAGE.md` - Multiple updates
8. ✅ `GAMMA_DEFAULT_UPDATE.md` - Updated terminology

---

## For Future Documentation

**When introducing Corralling:**

1. **Always explain the mechanism first:**
   ```
   Corralling tracks cumulative loss per expert:
   - Lower loss → Higher weight
   - Higher loss → Lower weight
   ```

2. **Use precise language:**
   - "higher-reward expert" not "better expert"
   - "based on empirical performance" not just "superior"

3. **Show the math (when appropriate):**
   ```
   Expert weight ∝ exp(-η × cumulative_loss)
   ```

---

## Verification Checklist

For any new documentation mentioning expert selection:

- [ ] Defines what makes one expert "better" (loss-based)
- [ ] Uses precise terminology ("higher-reward", "lower-loss")
- [ ] Explains that determination is empirical, not a priori
- [ ] Links weights explicitly to observed performance
- [ ] Avoids ambiguous terms like "winning" or "best"

---

**Status:** Terminology clarified across all gamma ablation documentation  
**Key Principle:** "Better" = empirically determined via cumulative loss tracking
