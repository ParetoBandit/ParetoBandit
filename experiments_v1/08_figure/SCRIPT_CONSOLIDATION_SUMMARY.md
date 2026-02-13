# Script Consolidation Summary

**Date**: February 13, 2026  
**Purpose**: Consolidate redundant experiment scripts into unified workflow

---

## What Changed

### ✅ Created Unified Script

**New file**: `run_figure8_analysis.py`

**Features**:
- Runs experiments ONCE and caches results to `figure8_unified_results.pkl`
- Generates BOTH figure AND table from cached data
- Smart caching: instant re-runs if data exists
- Supports `--force-rerun` flag to ignore cache

**Outputs**:
1. **Figure**: `results/figure8_regime_stratified_CORRECTED.png` (2×2 regime-stratified)
2. **Table**: `results/appendixC_neff_sensitivity.tex` (Appendix C: Sensitivity Analysis)
3. **Console**: Regime classification + performance statistics
4. **Cache**: `results/figure8_unified_results.pkl` (for fast re-runs)

---

### ❌ Deleted Redundant Scripts

Removed 4 scripts that re-ran the same experiments:

1. **`plot_sensitivity.py`** ❌
   - Generated misleading single-seed hybrid figure
   - Suggested n_eff=1.0 was universally optimal (false)

2. **`plot_regime_stratified_analysis.py`** ❌  
   - Redundant: functionality now in unified script

3. **`plot_expert_selection_analysis.py`** ❌  
   - Redundant: functionality now in unified script

4. **`plot_sensitivity_multiseed.py`** ❌  
   - Diagnostic tool, but results now in unified script

---

### ✅ Kept Essential Scripts

**Core**:
- `run_figure8_analysis.py` - Unified experiment runner ⭐

**Ablations**:
- `plot_ablation_no_corralling.py` - Separate ablation (Corralling OFF)

**Diagnostics**:
- `diagnose_corralling_weights.py` - Expert weight analysis tool
- `check_figure7_weights.py` - Cross-experiment validation

---

## Performance Improvements

### Before:
```
Time to generate figure: ~3 minutes
Time to generate table: N/A (manual)
Total re-runs needed: 3+ (separate scripts)
Total time: ~10+ minutes
```

### After:
```
Time for first run: ~5 minutes (experiments + figure + table)
Time for subsequent runs: ~instant (uses cache)
Total re-runs needed: 1 (unified script)
Total time: ~5 minutes (first run) → instant (cached)
```

**Speedup**: ~50% faster first run, >99% faster subsequent runs

---

## Usage

### Quick Run (Uses Cache)
```bash
python experiments_v1/08_figure/run_figure8_analysis.py
```

### Force Re-Run (Ignore Cache)
```bash
python experiments_v1/08_figure/run_figure8_analysis.py --force-rerun
```

### Clear Cache and Re-Run
```bash
rm experiments_v1/08_figure/results/figure8_unified_results.pkl
python experiments_v1/08_figure/run_figure8_analysis.py
```

---

## Key Results (From Unified Script)

### Regime Classification
- **Warmup-dominant**: 33% (seed 42 only)
- **Tabula rasa-dominant**: 67% (seeds 43-44)

### Performance by Regime

| Regime | n_eff=1.0 | n_eff=20.0 | Effect |
|--------|-----------|------------|--------|
| **Warmup** | 4.477 | 4.280 | **+4.60%** ✓ |
| **Tabula Rasa** | 4.241 | 4.247 | **-0.15%** (null) |
| **Overall** | 4.319 | 4.258 | **+1.44%** (n.s.) |

### Key Insight
> n_eff only matters when Corralling uses the warmup expert (~33% of time). In the majority of cases (67%), Corralling switches to cold-start exploration, making n_eff irrelevant. System robustness comes from adaptive meta-learning, not parameter tuning.

---

## Files Generated

### Primary Outputs
```
results/
├── figure8_regime_stratified_CORRECTED.png   # Main figure (2×2 layout)
├── appendixC_neff_sensitivity.tex            # LaTeX table (Appendix C)
└── figure8_unified_results.pkl               # Cached data (for speed)
```

### Legacy Files (Can be deleted)
```
results/
├── figure8_expert_selection_revised.png      # Old format
├── multiseed_results.pkl                     # Old cache
└── regime_stratified_results.pkl             # Old cache
```

---

## Migration Guide

### For Users of Old Scripts

**If you were running**:
```bash
python experiments_v1/08_figure/plot_sensitivity.py
```

**Now run**:
```bash
python experiments_v1/08_figure/run_figure8_analysis.py
```

**Benefits**:
- ✅ Correct figure (regime-stratified, not single-seed)
- ✅ Automatic table generation (LaTeX format)
- ✅ Cached results (instant re-runs)
- ✅ Regime classification (shows 33%/67% split)

---

## Documentation Updates

**Updated files**:
- `README.md` - Updated Quick Start, File listing, Reproducibility section
- `SCRIPT_CONSOLIDATION_SUMMARY.md` (this file) - Migration guide

**TODO**:
- [ ] Update paper figures (use regime-stratified version)
- [ ] Update paper table (use generated LaTeX table)
- [ ] Update experimental protocol description (mention regime-dependence)

---

## Testing

Verified that unified script:
- ✅ Runs experiments and caches results
- ✅ Generates correct figure (2×2 regime-stratified)
- ✅ Generates correct table (LaTeX format)
- ✅ Prints console summary (regime classification)
- ✅ Respects cache (instant re-run)
- ✅ Supports --force-rerun flag
- ✅ Produces identical results to old scripts (when same config)

---

## Summary

**Problem**: Multiple scripts re-ran same expensive experiments, generating redundant/misleading figures.

**Solution**: Unified script that runs experiments once, caches results, and generates both figure and table.

**Result**: 
- 50-99% time savings
- Single source of truth for Figure 8
- Automatic table generation
- Correct regime-stratified analysis

---

**Author**: Consolidation completed February 13, 2026  
**Status**: ✅ Complete - Ready for use
