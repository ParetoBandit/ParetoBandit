# Quick Start: Day 2 - Group B Experiments

**Date:** February 14, 2026  
**Goal:** Complete Group B experiments (main paper figures)  
**Est. Time:** 8-10 hours

---

## Where We Left Off ✅

Yesterday (Day 1) we:
- ✅ Fixed critical bug in experiments
- ✅ Updated router to reversed heterogeneous config
- ✅ Re-ran all Group A experiments (03_figure)
- ✅ Got consistent 20-30% improvement
- ✅ Created comprehensive documentation

**Status:** 50% complete, ready for Group B

---

## Today's Goal 🎯

Re-run main paper figures with the new optimal configuration:
1. Figure 4: Corralling weight evolution
2. Figure 7: Zero-shot readiness  
3. Figure 8: Sensitivity analysis
4. Table 2: Performance comparison (if needed)

---

## Quick Commands

### Check what's been done:
```bash
cd /Users/annette/repostitories/banditGPT

# Verify router config
python tests/test_reversed_heterogeneous_config.py

# See Group A results
cat GROUP_A_COMPLETE_SUMMARY.md

# Check progress
cat PROGRESS_SUMMARY.md
```

### Start Group B experiments:

#### Step 1: Figure 4 (Corralling Evolution)
```bash
cd experiments_v1/04_figure

# Find the main script
ls *.py | grep -E "plot|run|main"

# Identify which one generates Figure 4
grep -l "Corralling\|corralling" *.py

# Run it (replace <script> with actual filename)
python <script>.py 2>&1 | tee logs/figure4_reversed_$(date +%Y%m%d).log
```

#### Step 2: Figure 7 (Zero-Shot)
```bash
cd ../07_figure

# Find main scripts
ls *.py | grep -E "plot|run|main"

# Run the main experiment
python <script>.py 2>&1 | tee logs/figure7_reversed_$(date +%Y%m%d).log
```

#### Step 3: Figure 8 (Sensitivity)
```bash
cd ../08_figure

# Find main scripts
ls *.py

# Run sensitivity analysis
python <script>.py 2>&1 | tee logs/figure8_reversed_$(date +%Y%m%d).log
```

#### Step 4: Table 2 (If Needed)
```bash
cd ../02_table

# Check if uses Corralling
grep -l "Corralling\|corralling" *.py

# If yes, run it
python <script>.py 2>&1 | tee logs/table2_reversed_$(date +%Y%m%d).log
```

---

## Expected Results

### Figure 4 (Corralling Evolution)
- **Old:** Expert weights show warmup expert decaying
- **New:** Expert weights show warmup constant, tabula decaying
- **Expect:** Different weight evolution pattern
- **Regret:** Should be ~43-45 (consistent with Group A)

### Figure 7 (Zero-Shot)
- **Old:** Semantic transfer with original config
- **New:** Semantic transfer with reversed config
- **Expect:** Similar or better semantic transfer benefit
- **Regret:** Should improve ~20-30%

### Figure 8 (Sensitivity)
- **Old:** Sensitivity curves with original baseline
- **New:** Sensitivity curves with better baseline
- **Expect:** All curves shift down (better performance)
- **Baseline:** Should be ~43-45 instead of ~60

---

## Validation Checklist

After each experiment:
- [ ] No errors in log file
- [ ] Regret is reasonable (~43-45 for Corralling)
- [ ] Figure looks sensible
- [ ] Results consistent with Group A
- [ ] Create comparison (old vs new)

---

## If Something Goes Wrong

### Experiment crashes:
```bash
# Check the log file
tail -100 <log_file>

# Verify router config
python tests/test_reversed_heterogeneous_config.py

# Check if data files exist
ls -lh data/
```

### Results look wrong:
```bash
# Check router configuration in the code
grep -A20 "Heterogeneous Experts Strategy" src/bandit_gpt/router.py

# Verify total_steps is passed
grep "select_model.*total_steps" experiments_v1/*/figure/*.py

# Compare to Group A results
cat GROUP_A_COMPLETE_SUMMARY.md
```

### Need help:
```bash
# Read the guides
cat EXECUTIVE_SUMMARY.md        # Overview
cat RERUN_PLAN.md               # Detailed plan  
cat PAPER_UPDATE_GUIDE.md       # Paper updates
```

---

## Progress Tracking

Update as you complete each:
- [ ] Figure 4 complete
- [ ] Figure 7 complete
- [ ] Figure 8 complete
- [ ] Table 2 complete (if needed)
- [ ] Comparison reports created
- [ ] Results validated

---

## Time Management

### Morning (4 hours)
- **Hour 1-2:** Figure 4 (most critical)
- **Hour 3-4:** Figure 7 (key contribution)

### Afternoon (4-6 hours)
- **Hour 5-7:** Figure 8 (supplementary)
- **Hour 8:** Table 2 + validation
- **Hour 9-10:** Create comparison reports

### Evening (if time)
- Start paper updates (Appendix D)
- Draft new discussion section
- Remove "48%" from abstract

---

## Documentation

### As experiments complete:

Create summaries like:
```markdown
# Figure 4 Results: Reversed Config

**Old:** X regret, weights show Y pattern
**New:** A regret, weights show B pattern
**Change:** C% improvement
**Validation:** ✅/❌
**Implications:** ...
```

Save to:
- `experiments_v1/04_figure/RESULTS_REVERSED_CONFIG.md`
- `experiments_v1/07_figure/RESULTS_REVERSED_CONFIG.md`
- etc.

---

## End of Day Goals

By end of Day 2, you should have:
- ✅ All Group B experiments complete
- ✅ Comparison tables (old vs new)
- ✅ Validation complete
- ✅ Issues documented
- ✅ Ready to start paper updates (Day 3)

**Progress target:** 75% complete

---

## Day 3 Preview

Tomorrow you'll:
1. Update all paper sections
2. Regenerate figures
3. Update tables
4. Verify consistency

**Estimated time:** 8 hours

---

## Questions?

- **What's the priority?** → Start with Figure 4 (most critical)
- **How long will it take?** → 8-10 hours total
- **What if I'm stuck?** → Read `RERUN_PLAN.md` and `ACTION_PLAN.md`
- **Where are the old results?** → In the paper and various READMEs

---

## Key Files to Have Open

1. `RERUN_PLAN.md` - Detailed experiment guide
2. `GROUP_A_COMPLETE_SUMMARY.md` - For comparison
3. `PROGRESS_SUMMARY.md` - To track status

---

**Ready? Start with Figure 4!**

```bash
cd experiments_v1/04_figure
ls *.py | head -10
```

Good luck! 🚀
