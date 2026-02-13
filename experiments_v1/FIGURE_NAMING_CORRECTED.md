# Figure Naming Correction - Complete

**Date**: February 13, 2026  
**Status**: ✅ **COMPLETE**

---

## Problem

Figure files in the experiments didn't always match their figure numbers in the paper, causing confusion during paper preparation.

---

## Solution Applied

Updated all plotting scripts and renamed existing figures to match the correct figure numbering scheme.

---

## Main Paper Figures (Correct Naming)

| Folder | Paper Figure | File Name | Status |
|--------|--------------|-----------|--------|
| **01_figure** | Figure 1 | `figure1_lmsys_holdout_pca.png` | ✅ Correct |
| **02_figure** | Figure 2 | `figure2_distribution_shift.png` | ✅ **FIXED** (was `distribution_shift_pc1.png`) |
| **03_figure** | Figure 3 | `figure3_corralled_architecture_corrected.png` | ✅ Correct |
| **04_figure** | Figure 4 | `figure4_corralling_semantic_analysis.png` | ✅ **FIXED** (was `figure3_*`) |
| **05_figure** | Figure 5 | `figure5_pareto_frontier.png` | ✅ Correct |
| **07_figure** | Figure 6 | `figure6_adaptive_efficiency.png` | ✅ Correct |
| **08_figure** | Figure 7/8 | `figure8_regime_stratified_CORRECTED.png` | ✅ Correct |

---

## Changes Made

### 1. ✅ Updated 02_figure (Figure 2)

**Script**: `plot_distribution_shift_improved.py`

**Before**:
```python
output_file = output_dir / "distribution_shift_pc1.png"
output_file_hires = output_dir / "distribution_shift_pc1_hires.png"
```

**After**:
```python
output_file = output_dir / "figure2_distribution_shift.png"
output_file_hires = output_dir / "figure2_distribution_shift_hires.png"
```

**Files Renamed**:
- `distribution_shift_pc1.png` → `figure2_distribution_shift.png` ✅
- `distribution_shift_pc1_hires.png` → `figure2_distribution_shift_hires.png` ✅

---

### 2. ✅ Updated 04_figure (Figure 4)

**Script**: `corralled_semantic_analysis.py`

**Before**:
```python
output_file = output_dir / 'figure3_corralling_semantic_analysis.png'
output_file_hires = output_dir / 'figure3_corralling_semantic_analysis_hires.png'
```

**After**:
```python
output_file = output_dir / 'figure4_corralling_semantic_analysis.png'
output_file_hires = output_dir / 'figure4_corralling_semantic_analysis_hires.png'
```

**Files Renamed**:
- `figure3_corralling_semantic_analysis.png` → `figure4_corralling_semantic_analysis.png` ✅
- `figure3_corralling_semantic_analysis_hires.png` → `figure4_corralling_semantic_analysis_hires.png` ✅

---

## Appendix Figures (06_figure) ✅ FIXED

The **06_figure** folder contains supplementary material for **Appendix D** (ablations) and **Appendix E** (catastrophic failure detection). All figures have been renamed from `figure5_*` and `figure6_*` to proper appendix naming.

### Scripts Updated (9 files)

| Script | Old Output | New Output | Appendix |
|--------|-----------|-----------|----------|
| `generate_figure5_catastrophic_failure.py` | `figure5_catastrophic_failure.*` | `appendixE_catastrophic_failure.*` | E |
| `generate_figure6_main.py` | `figure5_catastrophic_failure.*` | `appendixE_catastrophic_failure.*` | E |
| `supplementary/generate_figure5_real_linucb.py` | `figure5_real_linucb.*` | `appendixE_real_linucb.*` | E |
| `supplementary/generate_figure5_realistic.py` | `figure5_realistic_scenario.*` | `appendixE_realistic_scenario.*` | E |
| `supplementary/generate_figure5_multiseed.py` | `figure5_multiseed_statistics.*` | `appendixE_multiseed_statistics.*` | E |
| `supplementary/real_linucb_semantic_transfer_simplified.py` | `figure6_real_linucb_semantic_transfer.*` | `appendixE_semantic_transfer.*` | E |
| `supplementary/ablation_learning_rate_catastrophic.py` | `figure6_learning_rate_ablation.*` | `appendixD_learning_rate_ablation.*` | D |
| `archive/generate_figure5_synthetic_fixed.py` | `figure5_corralling_weights.*` | `appendixE_corralling_weights.*` | E |
| `archive/generate_figure5_synthetic.py` | `figure5_corralling_weights.*` | `appendixE_corralling_weights.*` | E |

### Files Renamed (14 files)

**Appendix E (Extended Results)**:
- `figure5_catastrophic_failure.*` → `appendixE_catastrophic_failure.*` ✅
- `figure5_corralling_weights.*` → `appendixE_corralling_weights.*` ✅
- `figure5_multiseed_statistics.*` → `appendixE_multiseed_statistics.*` ✅
- `figure5_real_linucb.*` → `appendixE_real_linucb.*` ✅
- `figure5_realistic_scenario.*` → `appendixE_realistic_scenario.*` ✅
- `figure6_real_linucb_semantic_transfer.*` → `appendixE_semantic_transfer.*` ✅

**Appendix D (Ablation Studies)**:
- `figure6_learning_rate_ablation.*` → `appendixD_learning_rate_ablation.*` ✅

**See**: `06_figure/FIGURE_NAMING_UPDATED.md` for complete details.

---

## Figure Numbering Convention

### Main Paper (figures 1-8)

```
Figure 1: Alignment Tax Discovery (01_figure)
Figure 2: Distribution Shift (02_figure)
Figure 3: Corralling Architecture (03_figure)
Figure 4: Semantic Projection (04_figure)
Figure 5: Pareto Frontier (05_figure)
Figure 6: Zero-Shot Readiness (07_figure)
Figures 7/8: Sensitivity Analysis (08_figure)
```

**Note**: 06_figure is skipped in main paper numbering because it contains appendix material.

### Appendix Figures

```
Appendix A: Mathematical proofs (no figures yet)
Appendix B: Dataset details (1M scale analysis)
Appendix C: Sensitivity analysis (from 08_figure)
Appendix D: Ablation studies (from 03_figure, 04_figure, 06_figure)
Appendix E: Extended results (from 06_figure) ← Catastrophic failure
Appendix F: Implementation details (diagrams/tables)
Appendix G: Discussion (no figures)
```

---

## Verification

### All Main Paper Figures Now Match

```bash
01_figure/results/figure1_*.png     ✅ Figure 1
02_figure/results/figure2_*.png     ✅ Figure 2
03_figure/results/figure3_*.png     ✅ Figure 3
04_figure/results/figure4_*.png     ✅ Figure 4
05_figure/results/figure5_*.png     ✅ Figure 5
07_figure/results/figure6_*.png     ✅ Figure 6
08_figure/results/figure8_*.png     ✅ Figures 7/8
```

### Supplementary Figures

```bash
06_figure/results/figure5_*.png     📦 Appendix E (supplementary)
06_figure/results/figure6_*.png     📦 Appendix D/E (supplementary)
```

---

## Scripts Updated

| Script | Location | Change |
|--------|----------|--------|
| `plot_distribution_shift_improved.py` | `02_figure/` | Updated filename from `distribution_shift_pc1` to `figure2_distribution_shift` |
| `corralled_semantic_analysis.py` | `04_figure/` | Updated filename from `figure3_*` to `figure4_*` |

---

## Next Steps (Optional)

### ✅ Appendix Figures Renamed

All 06_figure supplementary files have been renamed:

```bash
# Catastrophic failure figures (Appendix E) - ✅ COMPLETE
figure5_catastrophic_failure.png      → appendixE_catastrophic_failure.png ✅
figure5_corralling_weights.png        → appendixE_corralling_weights.png ✅
figure5_multiseed_statistics.png      → appendixE_multiseed_statistics.png ✅
figure5_real_linucb.png               → appendixE_real_linucb.png ✅
figure5_realistic_scenario.png        → appendixE_realistic_scenario.png ✅
figure6_real_linucb_semantic_transfer.png → appendixE_semantic_transfer.png ✅

# Learning rate ablation (Appendix D) - ✅ COMPLETE
figure6_learning_rate_ablation.png    → appendixD_learning_rate_ablation.png ✅
```

**Status**: ✅ Complete - all scripts and files updated.

---

## Summary

✅ **Fixed 2 main paper figures** (Figure 2 and Figure 4)  
✅ **Fixed 06_figure appendix figures** (9 scripts, 14 files renamed to appendix naming)  
✅ **All main paper figures now match their numbers** (1-8)  
✅ **All appendix figures clearly labeled** (appendixD_*, appendixE_*)  
✅ **11 scripts updated** to generate correct filenames  
✅ **18 existing files renamed** to match new convention  

**Status**: **ALL** figure naming is now **consistent and correct**.

---

## Quick Reference

### Figure → Folder → File Mapping

```
Figure 1 → 01_figure → figure1_lmsys_holdout_pca.png
Figure 2 → 02_figure → figure2_distribution_shift.png         ✅ FIXED
Figure 3 → 03_figure → figure3_corralled_architecture_corrected.png
Figure 4 → 04_figure → figure4_corralling_semantic_analysis.png  ✅ FIXED
Figure 5 → 05_figure → figure5_pareto_frontier.png
Figure 6 → 07_figure → figure6_adaptive_efficiency.png
Fig 7/8  → 08_figure → figure8_regime_stratified_CORRECTED.png

Appendix → 06_figure → figure5_*/figure6_* (supplementary)
```

---

**Last Updated**: February 13, 2026  
**Status**: ✅ COMPLETE  
**Changes**: 
- Main paper: 2 scripts updated, 4 files renamed
- Appendix: 9 scripts updated, 14 files renamed  
- **Total**: 11 scripts, 18 files renamed
