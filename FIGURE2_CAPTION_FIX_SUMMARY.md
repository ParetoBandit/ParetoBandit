# Figure 2 Caption Correction - Complete Summary

**Date**: January 25, 2026  
**Status**: ✅ Fixed and Verified

---

## Overview

You correctly identified that the caption for Figure 2 (Feature Distribution Shift) had an inaccurate description of the bottom subplot. The issue has been corrected.

---

## The Problem

### Original Caption (INCORRECT)
```latex
Feature Distribution Shift. (Top) The deployment distribution (Red) is 
significantly left-shifted compared to the training data (Blue), with PSI=0.275. 
(Bottom) This shift is driven by a higher prevalence of "Easy Tasks" in production 
than anticipated.
```

### What Was Wrong

The caption stated that the **bottom subplot** shows "a higher prevalence of 'Easy Tasks' **in production**" - but the bottom subplot actually displays the **training/source data** decomposition, NOT the production/deployment data.

**Bottom subplot actually shows:**
- Title: "Source/Prior Data: Easy vs Hard Task Distribution"
- Green: Source Easy tasks (45.4%, PC1 < 0.0)
- Purple: Source Hard tasks (22.4%, PC1 > 0.2)
- Purpose: Explains the bimodal structure of the **training data**

---

## The Solution

### Corrected Caption (CORRECT)
```latex
Feature Distribution Shift. (Top) The deployment distribution (Red) is 
significantly left-shifted compared to the training data (Blue), with PSI=0.275, 
indicating more easy tasks in production. (Bottom) The training data exhibits a 
bimodal structure with two distinct task clusters: Easy tasks (45.4%, PC1 < 0.0) 
and Hard tasks (22.4%, PC1 > 0.2), explaining why the distribution shift impacts 
prior calibration.
```

### What Changed

1. **Top subplot**: Added "indicating more easy tasks in production" to clarify the interpretation
2. **Bottom subplot**: Now correctly states "The training data exhibits a bimodal structure..."
3. **Connection**: Added "explaining why the distribution shift impacts prior calibration" to link the two subplots

---

## Verification

### From the Figure
- **Top subplot**: 
  - Blue (Source/Prior): mean = 0.060
  - Red (RouteLLM/Deployment): mean = -0.004
  - Left-shift confirmed ✅
  
- **Bottom subplot**: 
  - Shows Source/Prior data decomposition ✅
  - Green: Easy tasks (45.4%)
  - Purple: Hard tasks (22.4%)

### From the Code (`plot_distribution_shift.py`)

Line 401: `# Plot SOURCE difficulty-based densities`

Lines 428-433:
```python
ax2.set_title(
    'Source/Prior Data: Easy vs Hard Task Distribution\n'
    f'Bimodal Structure Explained by Two Distinct Task Clusters',
    ...
)
```

Lines 554-558:
```python
shift_dir = "Easy" if mean_shift < 0 else "Hard"
if mean_shift < 0:
    print(f"      → More easy prompts in RouteLLM vs Source")
```

Mean shift = -0.064 < 0 → confirms RouteLLM has more easy tasks ✅

---

## Why This Matters

The corrected caption now:

1. **Accurately describes visual content**: Each subplot description matches what's actually shown
2. **Clarifies the narrative**: 
   - Top: Shows the distribution shift (deployment vs training)
   - Bottom: Explains why this shift matters (training has bimodal structure)
3. **Maintains scientific accuracy**: The interpretation is now consistent with the data and code
4. **Improves reader understanding**: Readers can now correctly interpret what each subplot represents

---

## Files Modified

- ✅ `paper/sections/empirical_motivation.tex` - Updated Figure 2 caption (line 61)

## Documentation Created

- ✅ `experiments_v1/01.5_figure/CAPTION_CORRECTION_SUMMARY.md` - Detailed technical explanation
- ✅ `FIGURE2_CAPTION_FIX_SUMMARY.md` - This summary document

---

## Status

✅ **Caption corrected**  
✅ **No LaTeX linting errors**  
✅ **Documentation complete**  
✅ **Ready for paper compilation**

