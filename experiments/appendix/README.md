# BanditGPT Paper Appendix

**Supplementary Material** — trimmed to directly support Figures 1, 3, 4, Table 2, and Figure 6.

---

## Overview

**Total Appendix Sections**: 5 (A-E)  
**Active LaTeX Files**: 9 (+ APPENDIX\_MASTER.tex)  
**Master File**: `APPENDIX_MASTER.tex`

---

## Quick Navigation

| Section | Topic | Supports | Key Content |
|---------|-------|----------|-------------|
| **A** | Mathematical Foundations | Fig 1, 3, 4, 6 | Regret bounds (LinUCB + Corralling), safety guarantee (γ-mixing) + ablation table (45 experiments), prior transfer (n_eff) |
| **B** | Dataset Details | Fig 1, Table 2 | Spearman validation, cross-domain transfer, feature pipeline (384D → 33D) |
| **C** | Extended Results | Fig 6 | Catastrophic failure detection (K=5 portfolio, 20 seeds) |
| **D** | Implementation Details | All | Configuration parameters, experimental setup |
| **E** | Limitations & Future Work | All | System constraints, honest limitations, future directions |

---

## Structure

```
appendix/
├── README.md                           # This file
├── APPENDIX_MASTER.tex                 # Master LaTeX file
├── APPENDIX_CONTENT_MAP.md             # Detailed content mapping
├── QUICK_START.md                      # Quick navigation guide
│
├── A_mathematical_foundations/
│   ├── README.md
│   ├── A1_regret_decomposition.tex     # Composite regret bound
│   ├── A2_safety_guarantee.tex         # γ-mixing safety proof + ablation table
│   └── A3_warmup_transfer.tex          # Prior transfer + n_eff + limitations
│
├── B_dataset_details/
│   ├── README.md
│   ├── B1_validation_methodology.tex   # Spearman correlation design
│   └── B2_cross_domain_transfer.tex    # Data provenance + feature pipeline
│
├── C_extended_results/
│   ├── README.md
│   └── C1_catastrophic_failure.tex     # K=5 portfolio experiment
│
├── D_implementation_details/
│   ├── README.md
│   ├── D1_configuration_details.tex    # Router parameters + experiment configs
│   └── D2_experimental_setup.tex       # Hardware, software, protocol
│
├── E_limitations_and_future_work/
│   ├── README.md
│   └── E1_limitations.tex              # Limitations and applicability
│
└── E_catastrophic_failure_experiment/  # Canonical experiment code for Fig 6
    ├── README.md
    ├── generate_figure6_5model.py
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
| Former Appendix C (Ablation Studies) | 45-experiment ablation table consolidated into A.2 (safety guarantee) |
| Former Appendix C (Hyperparameter Sensitivity) | n_eff analysis consolidated into A.3; Corralling params now in A.2 |
| Old A1 (Spectral separation proof) | Thompson Sampling theory mismatched LinUCB + Corralling implementation |
| Old B1 (Clustering-based validation) | Replaced with correct Spearman methodology matching Figure 1 |

See `APPENDIX_CONTENT_MAP.md` for the full traceability matrix.

---

**Last Updated**: February 15, 2026  
**Status**: Trimmed, aligned to main paper figures, folder names match appendix letters
