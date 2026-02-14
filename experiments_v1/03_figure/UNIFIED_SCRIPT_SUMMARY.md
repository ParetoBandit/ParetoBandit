# ✨ Unified Experiment Script - Complete

**Created:** February 14, 2026  
**Purpose:** Consolidate all 03_figure experiments into a single, bug-fixed runner  
**Status:** ✅ Ready to use

---

## 🎯 What We Created

### 1. **Unified Runner Script** (`run_all_experiments.py`)

**Features:**
- ✅ All 4 experiments in one file (645 lines)
- ✅ Bug-fixed: `selection_token` properly captured and passed
- ✅ Shared resource loading (efficient)
- ✅ Flexible: Run all or subset of experiments
- ✅ Command-line arguments for customization
- ✅ Progress tracking with tqdm
- ✅ Comprehensive JSON output

**What it includes:**

| Experiment | Lines | What It Does |
|-----------|-------|--------------|
| **Shared Setup** | 1-110 | Resource loading, data loading |
| **2A: Weight Evolution** | 111-225 | Track expert weight adaptation |
| **2BC: Convergence** | 227-364 | Compare strategy convergence rates |
| **3: Alpha Ablation** | 366-475 | Test constant vs adaptive exploration |
| **5: Gamma Ablation** | 477-574 | Test mixing parameter values |
| **Main Runner** | 576-645 | Orchestrate all experiments |

### 2. **Quick Start Guide** (`QUICK_START.md`)

User-friendly guide with:
- One-line command to run everything
- Command options and examples
- Expected runtime estimates
- Troubleshooting section
- Pro tips for overnight runs

---

## 📊 Comparison: Before vs After

### Before (4 Separate Files)

```
experiment_2a_weight_evolution.py       (381 lines)
experiment_2bc_convergence_dynamics.py  (495 lines)
experiment_3_heterogeneous_alpha_ablation.py (407 lines)
experiment_5_gamma_ablation.py          (426 lines)
---
Total: 1,709 lines in 4 files
❌ Bug in all 4 files (missing selection_token)
❌ Need to fix bug 4 times
❌ Resources loaded 4 times (inefficient)
```

### After (1 Unified File)

```
run_all_experiments.py                  (645 lines)
---
Total: 645 lines in 1 file
✅ Bug fixed once, applies to all
✅ Resources loaded once (efficient)
✅ Consistent implementation
✅ Easier to maintain
```

**Reduction:** 62% fewer lines with better organization!

---

## 🚀 How to Use

### Quick Test (Verify Bug Fix)

```bash
cd experiments_v1/03_figure
python run_all_experiments.py --experiments 2a --seeds 2
```

**Expected output:**
```
Seed 42: Final weights = [0.XXX, 0.YYY], Regret = XX.X
```

✅ **Weights should be different** (not 0.500/0.500)  
✅ **Should complete in ~5 minutes**

---

### Full Run (All Experiments)

```bash
python run_all_experiments.py
```

**This runs:**
- Experiment 2A: 10 seeds × 750 prompts = 7,500 trials
- Experiment 2BC: 3 strategies × 10 seeds × 750 prompts = 22,500 trials
- Experiment 3: 3 configs × 5 seeds × 750 prompts = 11,250 trials
- Experiment 5: 4 gamma values × 5 seeds × 750 prompts = 15,000 trials

**Total:** ~56,250 trials  
**Time:** 2-3 hours

---

### Custom Run

```bash
# Just ablation studies (faster)
python run_all_experiments.py --experiments 3,5 --seeds-ablation 3

# High-precision validation
python run_all_experiments.py --seeds 20 --seeds-ablation 10
```

---

## 📁 Output Files

```
results/
├── weight_evolution/
│   └── statistics.json              # ✅ REGENERATED
│
├── convergence/
│   └── convergence_statistics.json  # 🔄 Will be regenerated
│
├── ablation/
│   └── ablation_statistics.json     # 🔄 Will be regenerated
│
├── gamma_ablation/
│   └── gamma_statistics.json        # 🔄 Will be regenerated
│
└── all_experiments_summary.json     # NEW: Combined results
```

---

## 🔧 The Bug Fix (Applied Throughout)

### What Was Wrong

```python
# ❌ OLD CODE (in all 4 scripts):
selected_model, _ = router.select_model(context)
router.update(context, selected_model, reward)

# Result: Weights frozen at 0.5/0.5, no meta-learning
```

### What's Fixed

```python
# ✅ NEW CODE (in unified script):
selected_model, selection_token = router.select_model(context)
router.update(context, selected_model, reward, selection_token)

# Result: Weights adapt correctly, meta-learning works
```

**Impact:** 21% performance improvement (regret: 50.2 → 39.5)

---

## 📋 Status of Original Files

| File | Status | Action |
|------|--------|--------|
| `experiment_2a_weight_evolution.py` | ✅ Fixed manually | Can keep or archive |
| `experiment_2bc_convergence_dynamics.py` | ❌ Still has bug | Use unified script instead |
| `experiment_3_heterogeneous_alpha_ablation.py` | ❌ Still has bug | Use unified script instead |
| `experiment_5_gamma_ablation.py` | ❌ Still has bug | Use unified script instead |
| **`run_all_experiments.py`** | ✅ **NEW: Bug-fixed** | **Use this!** |

---

## 🎯 Recommended Next Steps

### Option 1: Quick Validation (5 minutes)

```bash
# Verify bug fix works
python run_all_experiments.py --experiments 2a --seeds 2
```

✅ **Best for:** Sanity check before full run

---

### Option 2: Run All Experiments (2-3 hours)

```bash
# Full validation with default settings
python run_all_experiments.py
```

✅ **Best for:** Complete results, ready for paper  
⏰ **Tip:** Run overnight or during lunch

---

### Option 3: Continue Down Original List

Fix other experiment folders (04_figure, 05_figure, etc.) while this runs in background:

```bash
# Terminal 1: Run 03_figure experiments
python run_all_experiments.py > run_03.log 2>&1 &

# Terminal 2: Continue to next experiment folder
cd ../04_figure
# ... check for bugs there ...
```

✅ **Best for:** Systematic coverage of all experiments

---

## 📊 Expected Results (After Bug Fix)

Based on our corrected Experiment 2A run:

| Metric | Expected Value |
|--------|----------------|
| **Final Warmup Weight** | 0.879 ± 0.183 |
| **Final Tabula Weight** | 0.121 ± 0.183 |
| **Weight Change** | +90% (adaptive) |
| **Average Regret** | 39.5 ± 5.6 |
| **Convergence** | Within 100-200 requests |

If you see weights at exactly 0.500 ± 0.000, the bug is back!

---

## 🆘 Troubleshooting

### "ModuleNotFoundError"

```bash
# Install dependencies
pip install sentence-transformers numpy matplotlib tqdm joblib
```

### "File not found: CANONICAL_HOLDOUT_DATA_PATH"

Check that data files exist:
```bash
ls -lh ../../../data/
```

### Weights Still Frozen

Double-check the script has this pattern:
```python
selected_model, selection_token = router.select_model(...)
router.update(..., selection_token)
```

---

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| **`QUICK_START.md`** | How to run the unified script |
| **`run_all_experiments.py`** | The actual unified script (645 lines) |
| **`CRITICAL_BUG_FIX_2026-02-14.md`** | Technical bug details |
| **`PRODUCTION_USER_GUIDE.md`** | For deploying to production |
| **`SUMMARY_2026-02-14.md`** | Research findings summary |
| **`UPDATE_SUMMARY_FOR_USER.md`** | Quick overview |
| **This file** | Unified script documentation |

---

## ✅ Checklist

Before running the unified script:

- [ ] Installed dependencies
- [ ] Data files accessible
- [ ] Reviewed `QUICK_START.md`
- [ ] Decided on seed count (2 for quick, 10 for default, 20 for thorough)
- [ ] Have 2-3 hours available (or run overnight)

After running:

- [ ] Check `all_experiments_summary.json`
- [ ] Verify weights adapted (not frozen at 0.5)
- [ ] Compare to old results (should see improvement)
- [ ] Update paper with corrected statistics

---

## 🎉 Benefits of Unified Approach

1. **Single Source of Truth** - One place to fix bugs
2. **Consistent Implementation** - Same bug fix everywhere
3. **Efficient Resource Use** - Load encoder/PCA once
4. **Easier Maintenance** - Update once, applies to all
5. **Better Testing** - Can run subset for quick validation
6. **Complete Results** - Combined JSON output
7. **Progress Tracking** - See all experiments in one run

---

## 💡 Pro Tip

Run this before committing results:

```bash
# Quick validation
python run_all_experiments.py --experiments 2a --seeds 2

# Check weights adapted
cat results/weight_evolution/statistics.json | grep warmup

# Should show something like:
#   "warmup": 0.879,
#   "warmup_std": 0.183,
# NOT:
#   "warmup": 0.500,
#   "warmup_std": 0.000,  # <- Bug still present!
```

---

**Ready to run?**

```bash
cd experiments_v1/03_figure
python run_all_experiments.py
```

🚀 Let's go!
