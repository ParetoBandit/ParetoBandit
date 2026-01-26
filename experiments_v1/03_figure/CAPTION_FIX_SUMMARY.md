# Figure 2 Caption Correction Summary

**Date**: January 25, 2026  
**Issue**: Mismatch between caption description and actual figure content

---

## Problem Identified

The original caption's **(Bottom)** section described a "Communication Protocol" with a three-phase cycle, but the actual bottom subplot of Figure 2 shows the **Feedback Phase** with specific mathematical formulas.

### Original Caption (Bottom Section)
```
(Bottom) The Communication Protocol follows a three-phase cycle:
(i) Recommendation: Each expert proposes an action and confidence score.
(ii) Selection: Coordinator samples an expert based on πₜ, executes its recommended action.
(iii) Feedback: Observed reward updates both the selected expert's parameters and the coordinator's trust weights via multiplicative updates.
```

### Actual Figure Content (Bottom Section)
The bottom of the figure displays:
- **Title**: "Feedback Phase"
- **Loss calculation**: ℓ = (1-r)/πᵢ = (1-0.92)/0.72 = 0.111
- **Coordinator update**: L[i] ← L[i] + ℓ, π ← normalize(exp(-ηL))
- **Expert update**: A ← A + φ(x,a)φ(x,a)ᵀ, b ← b + rφ(x,a)

---

## Solution

### Corrected Caption (Bottom Section)
```
(Bottom) The Feedback Phase computes importance-weighted losses and updates both layers:
The system calculates the loss ℓ = (1-r)/πᵢ for the selected expert, updates the coordinator's cumulative losses L[i] ← L[i] + ℓ, renormalizes the trust distribution π ← normalize(exp(-ηL)), and updates the expert's LinUCB parameters (A, b) using the observed context-action-reward tuple.
```

### Additional Clarification
The communication flow information (which was valuable but misplaced) was moved to the **(Middle)** section:
```
The coordinator samples an expert based on πₜ, which then recommends an action that is executed.
```

---

## Figure Structure Verification

**Actual layers in Figure 2** (from code in `generate_figure2.py`):
1. **Top**: Coordinator Layer (lines 68-79)
2. **Middle**: Expert Layer - Warmup Expert (lines 86-107) and Tabula Rasa Expert (lines 109-131)
3. **Middle-Bottom**: Selected Action / Execution Layer (lines 136-142)
4. **Bottom**: Feedback Phase (lines 147-155)

**Phase labels shown in figure**:
- Phase 1: Selection (left side)
- Phase 2: Recommend (left side)
- Phase 3: Execution (left side)
- Phase 4: Feedback (left side)

---

## Impact

✅ **Caption now accurately describes what's shown in each subplot**  
✅ **Mathematical formulas in caption match those displayed in the figure**  
✅ **Communication flow information preserved but relocated appropriately**  
✅ **Maintains technical accuracy while improving clarity**

---

## Files Modified

- `experiments_v1/02_figure/figure_2_caption.tex` - Updated caption text

## Files Referenced

- `experiments_v1/02_figure/generate_figure2.py` - Figure generation code
- `paper/figures/figure2_corralled_architecture.png` - Actual figure image

