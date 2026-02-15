# Appendix B: Dataset Details and Scale Validation

## Overview
This section provides comprehensive dataset documentation, statistical validation methodology, and large-scale validation experiments.

## Contents

### B.1: Dataset Composition and Provenance
**Source**: Extended from `02_table/table_dataset_composition.tex`

**Content**:
- Complete data provenance (LMSYS Chat Arena)
- Semantic category distributions
- Statistical validation (chi-square tests, confidence intervals)
- Quality assurance (leakage detection, stratification verification)

**Key Statistics**:
- Total: 81,871 prompts
- Warmup Set: 80,000
- Dev Set: 1,121
- Holdout Set: 750

---

### B.2: Validation Methodology
**File**: `B2_validation_methodology.tex`  
**Source**: `01_figure/validation_methodology.tex`

**Content**:
- Statistical testing procedures
- Confidence interval calculations
- LLM validation approach
- Cross-validation methodology

---

### B.3: Global Manifold Stability (1M Scale Analysis)
**File**: `B3_1M_scale_analysis.tex`  
**Source**: `appendix_d/figure_1M_analysis.tex`

**Content**:
- Analysis of 594,199 prompts (317× scale increase)
- Production-scale semantic discontinuity
- Spectral invariance demonstration
- Economic implications at scale

**Key Results**:
- PC1 variance ratio identical: 3.10% (holdout) vs 3.10% (1M)
- PC2 variance ratio identical: 2.29% vs 2.29%
- Distribution shift: 82.4% → 94.1% routine tasks
- Economic impact: $2.3M/year in cost savings potential

---

## Related Sections
- **Main Paper Table 1**: Summarized dataset composition
- **Main Paper Figure 1**: Uses holdout set (N=1,871)
- **Appendix A**: Theoretical justification for spectral stability

---

## Files
```
B_dataset_details/
├── README.md                          (this file)
├── B1_dataset_composition.tex         (to be created - extended table)
├── B2_validation_methodology.tex      (statistical methods)
├── B3_1M_scale_analysis.tex          (large-scale validation)
└── figures/
    └── (1M scale comparison figures)
```
