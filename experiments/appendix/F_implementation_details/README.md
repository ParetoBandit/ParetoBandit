# Appendix F: Implementation Details

## Overview
Configuration parameters and experimental setup for reproducibility. Covers all settings needed to replicate Figures 1, 3, 4, and 6.

## Contents

### F.1: Configuration Details
**File**: `F1_configuration_details.tex`  
**Source**: `03_figure/latex_appendix_config.tex`

**Content**:
- System architecture configuration
- Hyperparameter settings and rationale
- Default values and ranges

**Key Parameters**:
- `prior_n_effective`: Effective prior sample size (default: 5.0)
- `eta`: Meta-algorithm learning rate (default: 1.0)
- `gamma`: Exploration floor / mixing parameter (default: 0.05-0.10)
- `alpha`: UCB exploration bonus (default: adaptive)

---

### F.2: Experimental Setup
**File**: `F2_experimental_setup.tex`  
**Source**: `08_figure/experiments_setup_compact.tex`

**Content**:
- Hardware specifications
- Software dependencies
- Dataset preparation procedures
- Evaluation protocols
- Reproducibility guidelines

**Computational Requirements**:
- CPU: Standard multi-core processor (no GPU required)
- RAM: 8-16 GB for typical datasets
- Storage: ~5 GB for full LMSYS dataset
- Runtime: 2-3 minutes per experiment (typical)

---

## Removed Content

| Item | Reason |
|------|--------|
| ~~F3: Strategy Selection Guide~~ | Duplicates Figure 3 findings in a different format; better suited for GitHub README than scientific appendix |
| ~~F4: Hyperparameter Selection Guide~~ | Never created; Appendix C covers sensitivity comprehensively |

---

## Related Sections
- **Appendix C**: Hyperparameter sensitivity validates parameter choices listed here
- **Appendix D**: Ablation studies justify configuration decisions

---

## Files
```
F_implementation_details/
├── README.md                          (this file)
├── F1_configuration_details.tex       (config parameters)
└── F2_experimental_setup.tex          (setup procedures)
```

---

**Last Updated**: February 15, 2026
