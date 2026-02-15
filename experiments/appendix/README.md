# BanditGPT Paper Appendix

**Supplementary Material** — trimmed to directly support Figures 1, 3, 4, Table 2, and Figure 6.

---

## Overview

**Total Appendix Sections**: 7 (A-G)  
**Active LaTeX Files**: 10  
**Master File**: `APPENDIX_MASTER.tex`

---

## Quick Navigation

| Section | Topic | Supports | Key Content |
|---------|-------|----------|-------------|
| **A** | Mathematical Foundations | Fig 1, Fig 3 | Spectral separation proof, regret bounds |
| **B** | Dataset Details | Fig 1, Table 2 | Validation methodology, distribution shift |
| **C** | Hyperparameter Sensitivity | Fig 4 | 20x robustness range ($n_{\text{eff}}$) |
| **D** | Ablation Studies | Fig 3, Fig 4, Fig 6 | 45-experiment grid, η/α/γ ablations |
| **E** | Extended Results | Fig 6 | Catastrophic failure detection (K=5) |
| **F** | Implementation Details | All | Configuration, experimental setup |
| **G** | Limitations | All | System constraints, future work |

---

## Structure

```
appendix/
├── README.md                           # This file
├── APPENDIX_MASTER.tex                 # Master LaTeX file
├── APPENDIX_CONTENT_MAP.md             # Detailed keep/cut rationale
│
├── A_mathematical_foundations/
│   ├── README.md
│   └── A1_spectral_separation_proof.tex
│
├── B_dataset_details/
│   ├── README.md
│   ├── B1_validation_methodology.tex
│   └── B2_distribution_shift_details.tex
│
├── C_hyperparameter_sensitivity/
│   ├── README.md
│   ├── C1_comprehensive_sensitivity.tex
│   ├── C2_robustness_summary.tex        (optional)
│   └── figures/
│
├── D_ablation_studies/
│   ├── README.md
│   ├── D1_corralling_ablation.tex
│   └── figures/
│       ├── figure6_learning_rate_ablation.pdf
│       ├── figure_alpha_ablation.png
│       └── figure_gamma_ablation.png
│
├── E_extended_results/
│   ├── README.md
│   ├── E1_catastrophic_failure.tex
│   └── E1_catastrophic_failure_extended.tex  (optional)
│
├── F_implementation_details/
│   ├── README.md
│   ├── F1_configuration_details.tex
│   └── F2_experimental_setup.tex
│
├── G_additional_discussion/
│   ├── README.md
│   ├── G1_limitations.tex
│   └── G1_limitations_addendum.tex
│
└── E_catastrophic_failure_experiment/    # Canonical experiment code for Fig 6
    ├── README.md
    ├── generate_figure6_5model.py
    ├── figure6_corralling_kdd.tex
    └── results/
```

---

## Compilation

### Compile Entire Appendix (Standalone)

```bash
cd experiments/appendix
pdflatex APPENDIX_MASTER.tex
pdflatex APPENDIX_MASTER.tex  # Run twice for references
```

### Include in Main Paper

```latex
% At the end of your main content
\appendix
\input{experiments/appendix/APPENDIX_MASTER.tex}
```

---

## What Was Cut (and Why)

| Cut Item | Reason |
|----------|--------|
| B1: Dataset composition table | Table 2 in main paper is sufficient |
| B3: 1M scale analysis | No main figure depends on it |
| C2-C4: Individual parameter sweeps | C1 covers all comprehensively |
| E2: Three-model routing | Experiment removed from scope |
| E3: Cost profiles | Never created; Fig 4 Pareto sweep covers this |
| E4: Distribution shift | Never created; B.2 covers cross-domain transfer |
| F3: Strategy selection guide | Practitioner content; belongs in GitHub README |
| G1: Practical recommendations | Redundant with Fig 3 + F3 |
| G3: Broader impact | Not required by target venue |
| G4: Corralling vs offline | Not required by main figures |

See `APPENDIX_CONTENT_MAP.md` for the full traceability matrix.

---

**Last Updated**: February 15, 2026  
**Status**: Trimmed, aligned to main paper figures
