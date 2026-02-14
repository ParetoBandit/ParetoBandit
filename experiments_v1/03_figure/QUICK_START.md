# Quick Start: Run All Experiments

**Created:** February 14, 2026  
**Status:** ✅ Bug-fixed and production-ready

---

## 🚀 One Command to Run Everything

```bash
cd experiments_v1/03_figure
python run_all_experiments.py
```

This runs all 4 experiments with the corrected `selection_token` implementation.

---

## ⚙️ Command Options

### Run Specific Experiments

```bash
# Run only weight evolution
python run_all_experiments.py --experiments 2a

# Run weight evolution and convergence
python run_all_experiments.py --experiments 2a,2bc

# Run ablation studies only
python run_all_experiments.py --experiments 3,5
```

### Adjust Number of Seeds

```bash
# Quick test (fewer seeds)
python run_all_experiments.py --seeds 3 --seeds-ablation 2

# Full validation (more seeds)
python run_all_experiments.py --seeds 20 --seeds-ablation 10

# Default (balanced)
python run_all_experiments.py --seeds 10 --seeds-ablation 5
```

---

## 📊 What Gets Generated

After running, check these directories:

```
results/
├── weight_evolution/
│   └── statistics.json          # Exp 2A: Weight adaptation
├── convergence/
│   └── convergence_statistics.json  # Exp 2BC: Learning speed
├── ablation/
│   └── ablation_statistics.json     # Exp 3: Alpha configs
├── gamma_ablation/
│   └── gamma_statistics.json        # Exp 5: Gamma values
└── all_experiments_summary.json     # Combined results
```

---

## ⏱️ Expected Runtime

| Configuration | Time Estimate |
|--------------|---------------|
| **Quick test** (3 seeds) | ~45 minutes |
| **Default** (10 seeds) | ~2-3 hours |
| **Full validation** (20 seeds) | ~5-6 hours |

*Times are approximate and depend on hardware (M1/M2 Macs are faster)*

---

## ✅ What's Fixed

All experiments now include the **critical bug fix**:

```python
# ✅ CORRECT (what the unified script does):
selected_model, selection_token = router.select_model(context)
router.update(context, selected_model, reward, selection_token)

# ❌ WRONG (what the old scripts did):
selected_model, _ = router.select_model(context)  # Token discarded
router.update(context, selected_model, reward)     # No meta-learning!
```

---

## 📋 Experiments Included

| Experiment | What It Tests | Expected Result (Corrected) |
|-----------|---------------|----------------------------|
| **2A** | Weight evolution | Warmup: 0.46 → 0.88 (+90%) |
| **2BC** | Convergence speed | Corralling vs baselines |
| **3** | Alpha ablation | Constant α=2.0 is optimal |
| **5** | Gamma ablation | γ=0.05 is optimal |

---

## 🔍 Monitoring Progress

The script provides real-time progress:

```
================================================================================
🚀 UNIFIED EXPERIMENT RUNNER - FIGURE 3
================================================================================
Experiments: ['2a', '2bc', '3', '5']
Seeds (2a,2bc): 10
Seeds (3,5): 5
================================================================================

📦 Loading shared resources...
   ✅ Models: 2
   ✅ Context Dim: 33

📊 Loading holdout data...
   ✅ Loaded 750 unique prompts

================================================================================
EXPERIMENT 2A: EXPERT WEIGHT EVOLUTION
================================================================================
🔬 Running 10 trials...
Seed 42: 100%|██████████| 750/750 [00:07<00:00, 98.5it/s]
   Seed 42: Final weights = [0.879, 0.121], Regret = 39.5
...
```

---

## 🚨 If Something Goes Wrong

### Frozen Weights

If you see weights staying at exactly 0.500:

```python
# Check the script for this pattern:
selected_model, selection_token = router.select_model(...)  # ✅
router.update(..., selection_token)  # ✅

# NOT this:
selected_model, _ = router.select_model(...)  # ❌ Token discarded
```

### Out of Memory

If you run out of memory:
- Reduce `--seeds` (try 3 instead of 10)
- Run experiments separately (use `--experiments` flag)
- Close other applications

### Slow Performance

- Ensure using MPS/CUDA if available
- Check if other processes are using GPU
- Consider running overnight for full validation

---

## 📚 Documentation

- **Bug details**: `CRITICAL_BUG_FIX_2026-02-14.md`
- **Production guide**: `PRODUCTION_USER_GUIDE.md`
- **Summary**: `SUMMARY_2026-02-14.md`
- **For users**: `UPDATE_SUMMARY_FOR_USER.md`

---

## 🎯 Next Steps After Running

1. **Check results**: Review `all_experiments_summary.json`
2. **Validate weights**: Ensure adaptation occurred (variance > 0.05)
3. **Compare to broken**: Old results had frozen weights
4. **Update paper**: Use corrected statistics in LaTeX

---

## 💡 Pro Tips

### Run Overnight

```bash
# Run with full validation, save output
nohup python run_all_experiments.py --seeds 20 --seeds-ablation 10 > run.log 2>&1 &

# Check progress
tail -f run.log
```

### Quick Sanity Check

```bash
# Fast test to verify everything works
python run_all_experiments.py --experiments 2a --seeds 2
```

### Parallel Execution

The script runs experiments **sequentially** to share resources efficiently. If you want parallel execution:

```bash
# Run in separate terminals
python run_all_experiments.py --experiments 2a &
python run_all_experiments.py --experiments 2bc &
python run_all_experiments.py --experiments 3 &
python run_all_experiments.py --experiments 5 &
```

---

## ✨ Key Features

✅ **Single source of truth** - All experiments in one file  
✅ **Bug-fixed** - selection_token properly handled  
✅ **Shared resources** - Loads encoder/PCA once  
✅ **Flexible** - Run all or subset of experiments  
✅ **Progress tracking** - tqdm progress bars  
✅ **Comprehensive output** - JSON statistics for all experiments  

---

**Ready to run?**

```bash
python run_all_experiments.py
```

That's it! 🚀
