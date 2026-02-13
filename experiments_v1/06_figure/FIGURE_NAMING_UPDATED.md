# 06_figure Figure Naming - Updated to Appendix Convention

**Date**: February 13, 2026  
**Status**: ✅ **COMPLETE**

---

## Overview

All figures in `06_figure` have been renamed from `figure5_*` and `figure6_*` to proper appendix naming (`appendixE_*` and `appendixD_*`) to clarify that these are supplementary materials, not main paper figures.

---

## Changes Made

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
| `archive/generate_figure5_synthetic.py` | `figure5_corralling_weights.*` | `appendixE_corralling_weights.*` | E |
| `archive/generate_figure5_synthetic.py` | `figure5_corralling_weights.*` | `appendixE_corralling_weights.*` | E |

---

### Files Renamed (14 files)

#### Appendix E: Extended Results (Catastrophic Failure Detection)

| Old Name | New Name |
|----------|----------|
| `figure5_catastrophic_failure.png` | `appendixE_catastrophic_failure.png` |
| `figure5_catastrophic_failure.pdf` | `appendixE_catastrophic_failure.pdf` |
| `figure5_corralling_weights.png` | `appendixE_corralling_weights.png` |
| `figure5_corralling_weights.pdf` | `appendixE_corralling_weights.pdf` |
| `figure5_multiseed_statistics.png` | `appendixE_multiseed_statistics.png` |
| `figure5_multiseed_statistics.pdf` | `appendixE_multiseed_statistics.pdf` |
| `figure5_real_linucb.png` | `appendixE_real_linucb.png` |
| `figure5_real_linucb.pdf` | `appendixE_real_linucb.pdf` |
| `figure5_realistic_scenario.png` | `appendixE_realistic_scenario.png` |
| `figure5_realistic_scenario.pdf` | `appendixE_realistic_scenario.pdf` |
| `figure6_real_linucb_semantic_transfer.png` | `appendixE_semantic_transfer.png` |
| `figure6_real_linucb_semantic_transfer.pdf` | `appendixE_semantic_transfer.pdf` |

#### Appendix D: Ablation Studies

| Old Name | New Name |
|----------|----------|
| `figure6_learning_rate_ablation.png` | `appendixD_learning_rate_ablation.png` |
| `figure6_learning_rate_ablation.pdf` | `appendixD_learning_rate_ablation.pdf` |

---

## Rationale

### Why This Change Was Needed

1. **Confusion**: Files named `figure5_*` and `figure6_*` in the `06_figure` folder suggested they were main paper figures 5 and 6, but:
   - Main paper Figure 5 is in `05_figure` (Pareto Frontier)
   - Main paper Figure 6 is in `07_figure` (Zero-Shot Readiness)
   
2. **Clarity**: The `06_figure` folder contains **supplementary materials** for the appendix, not main paper figures.

3. **Organization**: New naming clearly indicates destination appendix section.

---

## Figure Destinations

### Appendix D: Ablation Studies
```
appendixD_learning_rate_ablation.png/pdf
→ Appendix D, Section D.X: Learning Rate Sensitivity for Catastrophic Failure Detection
```

**Content**: Learning rate ablation study showing how different η values affect catastrophic failure detection performance.

---

### Appendix E: Extended Results (Catastrophic Failure Detection)
```
appendixE_catastrophic_failure.png/pdf
→ Appendix E, Section E.1: Catastrophic Failure Detection (Main Figure)

appendixE_real_linucb.png/pdf
→ Appendix E, Section E.1: Real LinUCB Performance

appendixE_realistic_scenario.png/pdf
→ Appendix E, Section E.1: Realistic Failure Scenario

appendixE_multiseed_statistics.png/pdf
→ Appendix E, Section E.1: Multi-Seed Statistical Validation

appendixE_semantic_transfer.png/pdf
→ Appendix E, Section E.1: Semantic Transfer Analysis

appendixE_corralling_weights.png/pdf
→ Appendix E, Section E.1: Corralling Weight Evolution
```

**Content**: Extended analysis of when Corralling helps vs. doesn't help with catastrophic model failures. Shows that Corralling is effective for large distribution shifts (d > 1.0) but not necessary for subtle optimizations (d < 0.2).

---

## Current Figure Naming Convention

### Main Paper (01-08_figure)
```
Figure 1: 01_figure → figure1_lmsys_holdout_pca.png
Figure 2: 02_figure → figure2_distribution_shift.png
Figure 3: 03_figure → figure3_corralled_architecture.png
Figure 4: 04_figure → figure4_corralling_semantic_analysis.png
Figure 5: 05_figure → figure5_pareto_frontier.png
Figure 6: 07_figure → figure6_adaptive_efficiency.png
Figures 7/8: 08_figure → figure8_regime_stratified.png
```

### Appendix Figures (Supplementary)
```
Appendix D: 03_figure, 04_figure, 06_figure → appendixD_*.png
Appendix E: 06_figure → appendixE_*.png
```

**Note**: `06_figure` is skipped in main paper numbering because it contains only appendix material.

---

## Verification

### Before
```bash
$ ls 06_figure/results/figure*.png
figure5_catastrophic_failure.png     # ❌ Confusing
figure5_corralling_weights.png       # ❌ Confusing
figure5_multiseed_statistics.png     # ❌ Confusing
figure5_real_linucb.png              # ❌ Confusing
figure5_realistic_scenario.png       # ❌ Confusing
figure6_learning_rate_ablation.png   # ❌ Confusing
figure6_real_linucb_semantic_transfer.png # ❌ Confusing
```

### After
```bash
$ ls 06_figure/results/appendix*.png
appendixD_learning_rate_ablation.png      # ✅ Clear (Appendix D)
appendixE_catastrophic_failure.png        # ✅ Clear (Appendix E)
appendixE_corralling_weights.png          # ✅ Clear (Appendix E)
appendixE_multiseed_statistics.png        # ✅ Clear (Appendix E)
appendixE_real_linucb.png                 # ✅ Clear (Appendix E)
appendixE_realistic_scenario.png          # ✅ Clear (Appendix E)
appendixE_semantic_transfer.png           # ✅ Clear (Appendix E)
```

---

## Integration with Appendix Structure

These figures are referenced in the organized appendix:

### Appendix D (Ablation Studies)
```
experiments_v1/appendix/D_ablation_studies/figures/
├── appendixD_learning_rate_ablation.png  ← Copied from 06_figure/results/
├── appendixD_learning_rate_ablation.pdf
└── ... (other ablation figures)
```

### Appendix E (Extended Results)
```
experiments_v1/appendix/E_extended_results/
├── E1_catastrophic_failure.tex           ← References appendixE_*.png
├── E1_catastrophic_failure_extended.tex
└── figures/                              ← Can copy appendixE_*.png here
```

---

## Benefits

### 1. Clarity ✅
- File names now clearly indicate they're appendix figures
- No confusion with main paper figures 5 and 6

### 2. Organization ✅
- Easy to identify which appendix section each figure belongs to
- `appendixD_*` = Ablation Studies
- `appendixE_*` = Extended Results (Catastrophic Failure)

### 3. Consistency ✅
- Matches the appendix structure in `experiments_v1/appendix/`
- Follows conference naming conventions

### 4. Maintainability ✅
- Scripts updated to generate correct filenames
- Future runs will use new naming automatically

---

## Usage

### Generating Figures

All scripts now output with appendix naming:

```bash
# Catastrophic failure (main figure)
python experiments_v1/06_figure/generate_figure5_catastrophic_failure.py
# → Creates appendixE_catastrophic_failure.png/pdf

# Learning rate ablation
python experiments_v1/06_figure/supplementary/ablation_learning_rate_catastrophic.py
# → Creates appendixD_learning_rate_ablation.png/pdf

# Other supplementary figures
python experiments_v1/06_figure/supplementary/generate_figure5_real_linucb.py
# → Creates appendixE_real_linucb.png/pdf
```

### Referencing in LaTeX

Update appendix LaTeX files to use new names:

```latex
% Appendix D
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{experiments_v1/06_figure/results/appendixD_learning_rate_ablation.png}
  \caption{Learning rate sensitivity for catastrophic failure detection.}
  \label{fig:appendix_d_learning_rate}
\end{figure}

% Appendix E
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{experiments_v1/06_figure/results/appendixE_catastrophic_failure.png}
  \caption{Catastrophic failure detection with Corralling.}
  \label{fig:appendix_e_catastrophic}
\end{figure}
```

---

## Summary

✅ **9 scripts updated** to generate appendix-named figures  
✅ **14 existing files renamed** to match new convention  
✅ **Appendix D** figures clearly marked (`appendixD_*`)  
✅ **Appendix E** figures clearly marked (`appendixE_*`)  
✅ **No confusion** with main paper figures  

**Status**: All supplementary figures now properly named for appendix use.

---

**See also**:
- `FIGURE_NAMING_CORRECTED.md` - Main paper figure naming fixes (02_figure, 04_figure)
- `experiments_v1/appendix/README.md` - Appendix structure documentation

---

**Last Updated**: February 13, 2026  
**Changes**: 9 scripts, 14 files renamed  
**Status**: ✅ COMPLETE
