# Appendix B: Dataset Details

## Overview
Dataset documentation and statistical validation supporting Table 2 (data provenance) and Figure 1 (cross-domain PCA generalization).

## Contents

### B.1: Validation Methodology
**File**: `B1_validation_methodology.tex`  
**Source**: `01_figure/validation_methodology.tex`

**Content**:
- Statistical testing procedures (Spearman rank correlation)
- Confidence interval calculations
- LLM validation approach
- Cross-validation methodology

**Supports**: Figure 1's Spearman ρ significance claim (p < 0.0001)

---

### B.2: Distribution Shift Details
**File**: `B2_distribution_shift_details.tex`  
**Source**: Distribution shift analysis

**Content**:
- Cross-domain generalization evidence (RouteLLM → LMSYS)
- Why PCA trained on 80K RouteLLM battles transfers to 750 LMSYS holdout prompts
- Same model pair (Mixtral vs GPT-4-Turbo) ensures capability-space alignment

**Supports**: Figure 1's core claim that PCA features predict preference on unseen data

---

## Removed Content

| Item | Reason |
|------|--------|
| ~~B1: Dataset Composition~~ | Table 2 in main paper already covers this; never created |
| ~~B3: 1M Scale Analysis~~ | Impressive but tangential — no main figure relies on 1M-scale validation |

---

## Related Sections
- **Main Paper Table 2**: Dataset composition and splits
- **Main Paper Figure 1**: Uses holdout set for PCA validation
- **Appendix A**: Theoretical justification for spectral stability

---

## Files
```
B_dataset_details/
├── README.md                          (this file)
├── B1_validation_methodology.tex      (statistical methods)
└── B2_distribution_shift_details.tex  (cross-domain transfer evidence)
```

---

**Last Updated**: February 15, 2026
