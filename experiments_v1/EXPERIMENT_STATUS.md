# Experiment Re-run Status

**Date:** February 13, 2026  
**Purpose:** Increase trial counts for statistical rigor as per conference review feedback

## Summary

All three experiments have been successfully re-run with increased trial counts to address statistical power concerns raised in the paper review.

## Experiments Completed

### 1. Figure 8: Sensitivity Analysis (N_eff Parameter)
- **Previous:** 3 random seeds
- **Updated:** 30 random seeds (seeds 42-71)
- **Runtime:** ~23 minutes
- **Status:** ✅ Complete
- **Output Files:**
  - `/experiments_v1/08_figure/results/figure8_unified_results.pkl` (2.7 MB)
  - `/experiments_v1/08_figure/results/figure8_regime_stratified.png` (1.2 MB)
  - `/paper/figures/figure8_regime_stratified.png` (updated)

### 2. Table 2: Mismatch Robustness (Learning Rate Comparison)
- **Previous:** 10 random seeds
- **Updated:** 30 random seeds
- **Runtime:** ~21 minutes
- **Status:** ✅ Complete
- **Output Files:**
  - `/data/eta_0.1_holdout_multiseed/results_multiseed.json` (7.8 KB)
  - `/data/eta_0.1_holdout_multiseed/multiseed_comparison.png` (117 KB)
  - `/data/eta_1.0_holdout_multiseed/results_multiseed.json` (7.8 KB)
  - `/data/eta_1.0_holdout_multiseed/multiseed_comparison.png` (115 KB)
  - `/data/statistical_comparison/comparison_results.json` (11 KB)

### 3. Figure 5: Pareto Frontier (Cost-Quality Trade-off)
- **Previous:** 5 trials per λ value
- **Updated:** 20 trials per λ value (seeds 42-61)
- **Runtime:** ~59 minutes
- **Status:** ✅ Complete
- **Output Files:**
  - `/experiments_v1/05_figure/results/intermediate_pareto_results.json` (4.7 KB)
  - `/experiments_v1/05_figure/results/figure5_pareto_frontier.png` (422 KB)
  - `/experiments_v1/05_figure/results/figure5_pareto_frontier_hires.png` (1.0 MB)
  - `/paper/figures/figure5_pareto_frontier.png` (updated)

## Statistical Improvements

### Figure 8 (Sensitivity Analysis)
- **Old:** N=3 seeds → insufficient for proportion estimates
- **New:** N=30 seeds → adequate power for detecting effects >2%
- **Impact:** Can now confidently report regime frequencies and aggregate impacts

### Table 2 (Mismatch Robustness)
- **Old:** N=10 seeds → 8-10% power for d=0.22
- **New:** N=30 seeds → 22-25% power for d=0.22
- **Impact:** Improved (though still limited) ability to detect small effect sizes

### Figure 5 (Pareto Frontier)
- **Old:** n=5 trials → thin coverage, wide error bars
- **New:** n=20 trials → 4x more data, tighter confidence intervals
- **Impact:** Error bars reduced by ~50% (1.96/√20 vs 1.96/√5)

## Code Changes

### Modified Files
1. `/experiments_v1/08_figure/run_figure8_analysis.py`
   - Changed: `SEEDS = [42, 43, 44]` → `SEEDS = list(range(42, 72))`

2. `/experiments_v1/02_table/run_holdout_evaluation_multiseed.py`
   - Changed: Default `--num-seeds` from 10 to 30

3. `/experiments_v1/02_table/run_statistical_validation.sh`
   - Changed: `NUM_SEEDS=10` → `NUM_SEEDS=30`
   - Fixed: Added proper paths to Python scripts

4. `/experiments_v1/05_figure/generate_pareto_frontier.py`
   - Changed: `range(5)` → `range(20)` for trials
   - Changed: CI multiplier from `1.96/√5` to `1.96/√20`

## Paper Updates

All paper text has been updated to reflect the new trial counts:
- References to "N=3 seeds" → "N=30 seeds"
- References to "N=10 seeds" → "N=30 seeds"
- References to "5 trials" → "20 trials"
- References to "seeds 42-46" → "seeds 42-61"
- Power analysis sections updated with new statistical power estimates
- Confidence interval descriptions updated

## Next Steps

1. ✅ All experiments complete
2. ✅ All figures updated
3. ✅ All paper text updated
4. ⏭️ Ready for paper compilation and final review

## Notes

- All experiments ran successfully with exit code 0
- No errors or warnings encountered
- Output files are properly formatted and ready for analysis
- Paper figures have been updated in `/paper/figures/`
