# CRITICAL BUG FIX: Alpha Decay Not Actually Tested

**Date:** February 13, 2026  
**Severity:** CRITICAL - Invalidates all alpha ablation results  
**Status:** FIXED - Re-run required

---

## The Problem

All experiments in `03_figure/` were **NOT actually testing alpha decay strategies**. They were comparing **constant alpha values at the END of decay schedules**.

### Root Cause

The experiments called `router.select_model(context)` WITHOUT passing the `total_steps` parameter:

```python
# BROKEN CODE (all experiments had this)
selected_model = router.select_model(context)  # ❌ total_steps defaults to 0
```

When `total_steps=0` (the default), `get_current_alpha()` immediately returns `alpha_end`:

```python
# From router.py lines 3447-3449
def get_current_alpha(self, total_steps: int) -> float:
    if total_steps == 0:
        return self.alpha_end  # ❌ Jumps to END immediately!
    fraction = min(self.t / total_steps, 1.0)
    return self.alpha_start + fraction * (self.alpha_end - self.alpha_start)
```

### What the Experiments Actually Tested

| Configuration | Intended Behavior | Actual Behavior |
|--------------|------------------|-----------------|
| **Homogeneous Constant** | Both experts: α=2.0 (constant) | ✅ Both at α=2.0 (CORRECT) |
| **Homogeneous Decay** | Both experts: α decays 1.0→0.01 | ❌ Both at α=**0.01** (NO DECAY!) |
| **Current Heterogeneous** | E1 decays 1.0→0.01, E2 constant 2.0 | ❌ E1 at **0.01**, E2 at 2.0 (NO DECAY!) |
| **Reversed Heterogeneous** | E1 constant 2.0, E2 decays 1.0→0.01 | ❌ E1 at 2.0, E2 at **0.01** (NO DECAY!) |

### Invalid Results

The published results comparing:
- **Homogeneous Constant (α=2.0)**: 60.6 ± 1.4 regret
- **Homogeneous Decay (α=1.0→0.01)**: 90.2 ± 7.8 regret
- **"48% improvement"** claim

**These are NOT comparing decay strategies!** They're comparing:
- **High exploration (α=2.0)** vs **No exploration (α=0.01)**
- This is comparing two **constant** alpha values (2.0 vs 0.01)
- The 90.2 regret comes from having **200× less exploration** (0.01 vs 2.0)

---

## The Fix

### Code Changes

Added `total_steps` parameter to all `select_model()` calls in 4 experiments:

```python
# FIXED CODE (applied to all experiments)
total_steps = len(data)  # Total training steps for alpha decay
selected_model = router.select_model(context, total_steps=total_steps)  # ✅ Proper decay!
```

### Files Modified

1. ✅ `experiment_3_heterogeneous_alpha_ablation.py` - Line 177
2. ✅ `experiment_5_gamma_ablation.py` - Line 140  
3. ✅ `experiment_2a_weight_evolution.py` - Line 146
4. ✅ `experiment_2bc_convergence_dynamics.py` - Line 174

### Verification

Production code in `src/bandit_gpt/router.py` was **already correct**:
- Line 2557: `best_model = self.corralling_router.select_model(x, total_steps=total_steps)` ✅
- Line 3084: `model = self.experts[expert_idx].select_model(context, total_steps=total_steps)` ✅

**Only the experiments were broken.**

---

## Impact on Paper Claims

### Claims That Are Now Invalid

1. ❌ **"Constant α=2.0 outperforms adaptive decay by 48%"**
   - This was comparing α=2.0 vs α=0.01 (both constant)
   - NOT comparing constant vs decay strategies
   
2. ❌ **"Heterogeneous alpha strategy is the core innovation"**
   - Results showed heterogeneous (64.4) was WORSE than homogeneous constant (60.6)
   - But this was comparing α=0.01/2.0 vs α=2.0/2.0 (both without decay)
   
3. ❌ **All ablation statistics in Appendix D**
   - Table comparing configurations (lines 173-177 in appendix_d.tex)
   - Figures showing alpha ablation results
   - README claims about 48% improvement

### Claims That May Still Be Valid

1. ✅ **Semantic transfer mechanism** - Not affected by this bug
2. ✅ **Corralling meta-learning** - Not affected by this bug  
3. ✅ **Distribution shift analysis** - Not affected by this bug
4. ❓ **Actual alpha strategy comparison** - UNKNOWN until re-run

---

## Action Items

### Required (Before Publication)

1. **Re-run all experiments in `03_figure/`** with fixed code
   - [ ] experiment_3_heterogeneous_alpha_ablation.py
   - [ ] experiment_5_gamma_ablation.py  
   - [ ] experiment_2a_weight_evolution.py
   - [ ] experiment_2bc_convergence_dynamics.py

2. **Compare new vs old results**
   - Document what changed
   - Identify which claims remain valid
   - Identify new findings

3. **Update paper sections**
   - [ ] Abstract - Remove/revise 48% claim if invalid
   - [ ] Introduction - Update contribution claims
   - [ ] Methodology (Section 3) - Lines 68-69 about constant α
   - [ ] Results - Remove invalid comparisons
   - [ ] Appendix D - Update Table and all text
   - [ ] Discussion/Conclusion - Revise based on new findings

4. **Update supporting materials**
   - [ ] README.md - Lines 25, 69, 182
   - [ ] experiments_v1/03_figure/README.md - All alpha claims
   - [ ] HETEROGENEOUS_EXPERTS_STRATEGY.md - Validation claims
   - [ ] All figure captions mentioning alpha strategies

### Recommended

5. **Add regression test** to prevent future bugs
   ```python
   def test_alpha_decay_actually_works():
       """Ensure alpha actually decays during experiments."""
       router = CostAwareLinUCBRouter(...)
       
       # Without total_steps, should return alpha_end
       alpha_no_steps = router.get_current_alpha(total_steps=0)
       assert alpha_no_steps == router.alpha_end
       
       # With total_steps, should decay properly
       router.t = 500
       alpha_midway = router.get_current_alpha(total_steps=1000)
       expected = router.alpha_start + 0.5 * (router.alpha_end - router.alpha_start)
       assert abs(alpha_midway - expected) < 0.01
   ```

6. **Document experiment execution protocol**
   - Always pass `total_steps` to `select_model()`
   - Verify alpha is actually changing during trials
   - Log alpha values at key steps for validation

---

## Lessons Learned

### Design Flaws

1. **Silent failures**: `total_steps=0` should have raised a warning or error
2. **Default behavior**: Should default to proper decay, not evaluation mode
3. **Testing gaps**: No validation that alpha actually decays during experiments

### Process Improvements

1. **Log alpha values**: Add logging to track actual alpha during experiments
2. **Validation assertions**: Check that alpha changes as expected
3. **Integration tests**: Test experiments end-to-end, not just production code
4. **Code review**: Experiments need same rigor as production code

---

## Expected Outcomes After Re-Run

### Scenario A: Constant Still Wins (But Less Dramatically)

If constant α=2.0 still outperforms decay:
- Regret difference might be smaller (e.g., 10-20% instead of 48%)
- Claim remains valid but needs updated numbers
- Need to explain WHY constant is better (domain mismatch, etc.)

### Scenario B: Decay Actually Helps

If decay now performs better:
- Paper narrative needs complete rewrite
- Heterogeneous might now make sense
- Need to understand why (sample efficiency, convergence, etc.)

### Scenario C: Heterogeneous Emerges as Winner

If current heterogeneous (decay + constant) performs best:
- Original hypothesis validated!
- Paper claims become valid
- Need to explain the mechanism (regime switching, etc.)

---

## Timeline

- **Fix applied**: February 13, 2026
- **Re-run target**: Within 24 hours
- **Analysis complete**: Within 48 hours
- **Paper updates**: Within 72 hours
- **Verification**: Before any submission/publication

---

## Contact

For questions about this fix, contact the research team.

**CRITICAL**: Do NOT submit paper or make public claims until experiments are re-run with corrected code.
