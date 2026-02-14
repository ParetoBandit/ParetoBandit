# Re-Run Plan: Experiments After Config Change

**Date:** February 13, 2026  
**Status:** ✅ Router config updated to reversed heterogeneous  
**Next:** Re-run all affected experiments

---

## Phase 1: Router Configuration ✅ COMPLETE

- [x] Update router.py (lines 2070-2138)
- [x] Create validation test
- [x] Run validation test (all passed)
- [x] Configuration confirmed: Warmup constant α=2.0, Tabula decay α=1.0→0.01

---

## Phase 2: Re-Run Experiments

### Priority Matrix

| Experiment | Uses Corralling? | Priority | Est. Time | Status |
|-----------|-----------------|----------|-----------|--------|
| **03_figure/experiment_3** (Alpha ablation) | ✅ Yes | P0 | Done | ✅ COMPLETE |
| **03_figure/experiment_5** (Gamma ablation) | ✅ Yes | P0 | 2h | 🔄 READY |
| **03_figure/experiment_2a** (Weight evolution) | ✅ Yes | P1 | 1h | 🔄 READY |
| **03_figure/experiment_2bc** (Convergence) | ✅ Yes | P1 | 2h | 🔄 READY |
| **04_figure** (Corralling weight evolution) | ✅ Yes | P0 | 3h | 📋 TODO |
| **07_figure** (Zero-shot readiness) | ✅ Yes | P0 | 4h | 📋 TODO |
| **08_figure** (Sensitivity analysis) | ✅ Yes | P1 | 3h | 📋 TODO |
| **02_table** (Performance comparison) | ❓ Maybe | P0 | 2h | 📋 TODO |

---

## Detailed Re-Run Plan

### Group A: Already Fixed (03_figure experiments)

These were fixed in the bug fix phase. Re-run with new router config:

#### A1. experiment_3_heterogeneous_alpha_ablation.py ✅
- **Status:** COMPLETE (already ran with fixed code)
- **Result:** Reversed heterogeneous won (43.4 vs 49.6)
- **Action:** Results valid, no re-run needed

#### A2. experiment_5_gamma_ablation.py
- **Command:** `cd experiments_v1/03_figure && python experiment_5_gamma_ablation.py`
- **Expected:** Gamma=0.05 performance with reversed config
- **Watch for:** Whether gamma=0.001 still performs better
- **Log:** Save to `experiment_5_rerun_reversed_config.log`

#### A3. experiment_2a_weight_evolution.py
- **Command:** `cd experiments_v1/03_figure && python experiment_2a_weight_evolution.py`
- **Expected:** Weight evolution patterns with reversed config
- **Watch for:** How quickly Corralling adapts
- **Log:** Save to `experiment_2a_rerun_reversed_config.log`

#### A4. experiment_2bc_convergence_dynamics.py
- **Command:** `cd experiments_v1/03_figure && python experiment_2bc_convergence_dynamics.py`
- **Expected:** Convergence comparison across strategies
- **Watch for:** Whether Corralling still ranks 2nd
- **Log:** Save to `experiment_2bc_rerun_reversed_config.log`

---

### Group B: Main Paper Figures (Critical)

#### B1. Figure 4: Corralling Weight Evolution

**Location:** `experiments_v1/04_figure/`

**Files to check:**
```bash
ls experiments_v1/04_figure/*.py
```

**Likely files:**
- `plot_corralling_semantic.py` or similar
- Any script generating Figure 4 showing expert weight evolution

**What to verify:**
1. Does it use CorrallingRouter?
2. What are the current results?
3. Will reversed config change the weights?

**Expected changes:**
- Expert weights might flip (warmup now gets higher weight?)
- Regret should decrease ~14%
- Weight variance might change

**Action:**
```bash
cd experiments_v1/04_figure
# Find the main experiment script
grep -l "CorrallingRouter\|corralling" *.py
# Run it
python <main_script>.py
```

#### B2. Figure 7: Zero-Shot Readiness  

**Location:** `experiments_v1/07_figure/`

**Critical:** This tests semantic transfer with heterogeneous experts

**Files to check:**
```bash
ls experiments_v1/07_figure/*.py | grep -E "plot|experiment"
```

**What to verify:**
1. Uses heterogeneous experts strategy
2. Tests GPT-5.1 semantic transfer
3. Shows expert weight evolution

**Expected changes:**
- Semantic transfer benefit might increase (better baseline)
- Expert selection dynamics might change
- Final performance should improve

**Action:**
```bash
cd experiments_v1/07_figure
# Find main scripts
ls *.py | head -10
# Run the key experiments
```

#### B3. Figure 8: Sensitivity Analysis

**Location:** `experiments_v1/08_figure/`

**What it tests:** Hyperparameter sensitivity (η, N_eff, etc.)

**Expected changes:**
- Baseline performance changes (43.4 instead of 49.6)
- Sensitivity curves might shift
- Optimal hyperparameters might change

**Action:**
```bash
cd experiments_v1/08_figure
ls *.py
# Run sensitivity sweeps
```

---

### Group C: Tables and Comparisons

#### C1. Table 2: Performance Comparison

**Location:** `experiments_v1/02_table/`

**What it compares:**
- Warmup vs Tabula Rasa vs Corralling
- Performance under distribution shift

**Critical check:**
- Does it use Corralling?
- If yes, re-run to get new Corralling numbers
- If no, results might still be valid

**Action:**
```bash
cd experiments_v1/02_table
ls *.py
grep -l "Corralling" *.py
# Run if needed
```

---

## Execution Strategy

### Option 1: Systematic (Recommended)

Run experiments in order, validate each before proceeding:

```bash
# Day 1: Group A (03_figure experiments)
cd experiments_v1/03_figure
python experiment_5_gamma_ablation.py | tee logs/exp5_reversed.log
python experiment_2a_weight_evolution.py | tee logs/exp2a_reversed.log
python experiment_2bc_convergence_dynamics.py | tee logs/exp2bc_reversed.log

# Validate: Check if results match expectations
# Document: Create comparison tables

# Day 2: Group B (Main figures)
cd experiments_v1/04_figure
# Identify main script
python <main_script>.py | tee logs/figure4_reversed.log

cd experiments_v1/07_figure
# Run zero-shot experiments
python <main_script>.py | tee logs/figure7_reversed.log

cd experiments_v1/08_figure
# Run sensitivity analysis
python <main_script>.py | tee logs/figure8_reversed.log

# Day 3: Group C + Validation
cd experiments_v1/02_table
# Run if needed
python <main_script>.py | tee logs/table2_reversed.log

# Create comparison reports for all experiments
```

### Option 2: Parallel (Faster but riskier)

Run multiple experiments in parallel if you have compute:

```bash
# Terminal 1
cd experiments_v1/03_figure && python experiment_5_gamma_ablation.py

# Terminal 2  
cd experiments_v1/04_figure && python <main_script>.py

# Terminal 3
cd experiments_v1/07_figure && python <main_script>.py

# etc.
```

**Pros:** Faster (6-8 hours instead of 2-3 days)  
**Cons:** Harder to debug if something fails

---

## Validation Checklist

After each experiment:

### Sanity Checks
- [ ] Experiment completed without errors
- [ ] Regret is in expected range (40-50, not 60-90)
- [ ] Logs show "Reversed Heterogeneous" config
- [ ] Alpha values logged correctly (warmup constant, tabula decay)
- [ ] Results are reproducible (re-run with same seed gives same result)

### Comparison Checks
- [ ] New regret < Old regret (should improve ~14%)
- [ ] Variance is reasonable (not too high)
- [ ] Expert weights make sense (no single expert dominates 100%)
- [ ] Learning curves are smooth (not erratic)

### Documentation
- [ ] Save log file with timestamp
- [ ] Create comparison table (old vs new)
- [ ] Note any unexpected findings
- [ ] Take screenshots of key figures

---

## Expected Results Summary

| Experiment | Old Regret | Expected New | Change |
|-----------|-----------|--------------|--------|
| experiment_3 (alpha) | 64.4 | **43.4** | **-33%** ✅ |
| experiment_5 (gamma) | 60.6 | ~43-48 | -20% to -30% |
| experiment_2a (weights) | N/A | More stable? | TBD |
| experiment_2bc (convergence) | 59.2 | ~45-50 | -15% to -25% |
| Figure 4 (Corralling) | TBD | Better | TBD |
| Figure 7 (Zero-shot) | TBD | Better | TBD |
| Figure 8 (Sensitivity) | TBD | Shifted | TBD |

**Key metric:** All Corralling experiments should show ~14% improvement

---

## Troubleshooting

### If regret increases instead of decreasing:

1. **Check configuration:**
   ```python
   python tests/test_reversed_heterogeneous_config.py
   ```

2. **Verify total_steps is passed:**
   ```bash
   grep "select_model.*total_steps" experiments_v1/*_figure/*.py
   ```

3. **Check log for alpha values:**
   ```bash
   grep "Alpha\|alpha" <experiment_log>.log
   ```

### If results are unstable (high variance):

1. **Increase seeds:** Change N_SEEDS from 5 to 10
2. **Check data shuffling:** Ensure RNG is seeded correctly
3. **Validate convergence:** Plot learning curves

### If experiment crashes:

1. **Check dependencies:** Ensure all packages installed
2. **Check data files:** Verify holdout data exists
3. **Check memory:** Large experiments might need more RAM
4. **Check GPU:** If using MPS/CUDA, verify availability

---

## Deliverables

After completing all re-runs:

### 1. Comparison Reports
- `RESULTS_GROUP_A.md` - 03_figure experiments
- `RESULTS_GROUP_B.md` - Main paper figures  
- `RESULTS_GROUP_C.md` - Tables

### 2. Updated Figures
- New versions of all figures in `experiments_v1/*/results/`
- Side-by-side comparisons (old vs new)

### 3. Summary Statistics
- Master table with all old vs new results
- Statistical significance tests
- Performance improvement summary

### 4. Paper Update Guide
- List of all claims that changed
- Suggested replacement text
- Figure/table update checklist

---

## Timeline

### Conservative (Systematic)
- Day 1: Group A (4 experiments, 6 hours)
- Day 2: Group B (3 figures, 10 hours)
- Day 3: Group C + Validation (4 hours)
- **Total: 3 days**

### Optimistic (Parallel)
- Day 1: All experiments (6-8 hours compute)
- Day 2: Validation + comparison (8 hours)
- **Total: 2 days**

### Realistic
- Day 1: Group A + Figure 4 (8 hours)
- Day 2: Figures 7-8 + Table 2 (10 hours)
- Day 3: Validation + documentation (6 hours)
- **Total: 2.5 days**

---

## Next Steps

1. **Choose execution strategy** (systematic vs parallel)
2. **Start with Group A** (03_figure experiments)
3. **Validate each result** before proceeding
4. **Document everything** as you go
5. **Create comparison reports** after each group

---

## Progress Tracking

Update this section as experiments complete:

- [x] Router configuration updated
- [x] Validation test created and passed
- [ ] experiment_5_gamma_ablation.py
- [ ] experiment_2a_weight_evolution.py
- [ ] experiment_2bc_convergence_dynamics.py
- [ ] Figure 4 (Corralling evolution)
- [ ] Figure 7 (Zero-shot readiness)
- [ ] Figure 8 (Sensitivity)
- [ ] Table 2 (Performance comparison)
- [ ] Comparison reports created
- [ ] Paper updates drafted

**Current status:** Router config validated, ready to start Group A re-runs
