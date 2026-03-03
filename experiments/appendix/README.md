# BanditGPT Paper Appendix

Supplementary material supporting Figures 1, 3, 4, 5, Table 2, and Figure 6.

---

## Quick Navigation

| Section | Topic | Supports | Key Content |
|---------|-------|----------|-------------|
| **A** | Mathematical Foundations | Fig 1, 3, 4, 6 | Regret bounds (LinUCB + Corralling), safety guarantee (γ-mixing) + ablation table (45 experiments), prior transfer (n_eff) |
| **B** | Dataset Details | Fig 1, Table 2 | Spearman validation, cross-domain transfer, feature pipeline (384D → 33D) |
| **C** | Extended Results | Fig 6 | Catastrophic failure detection (K=5 portfolio, 20 seeds) |
| **D** | Implementation Details | All | Configuration parameters, experimental setup |
| **E** | Limitations & Future Work | All | System constraints, honest limitations, bandit router positioning (PILOT/BaRP taxonomy) |
| **F** | Hard Constraint Enforcement | Fig 8 | Per-request budget/latency constraint validation |
| **H** | Prior Strength, Exploration & Forgetting Ablation | Fig 5 | 3D alpha x n_eff x gamma grid, optimal hyperparameters |

**Master File**: `APPENDIX_MASTER.tex`

---

## Structure

```
appendix/
├── README.md                           # This file
├── APPENDIX_MASTER.tex                 # Master LaTeX file
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
│   ├── E1_limitations.tex              # Limitations and applicability
│   └── E2_positioning.tex             # Bandit router taxonomy + differentiators
│
├── E_catastrophic_failure_experiment/  # Experiment code for Figure 6
│   ├── README.md
│   ├── generate_figure9_5model.py
│   └── results/
│
├── F_constraint_impact/               # Hard constraint enforcement
│   ├── README.md
│   ├── run_constraint_experiment.py
│   └── section_constraint_impact.tex
│
└── H_alpha_neff_ablation/             # Prior strength, exploration & forgetting ablation
    ├── README.md
    ├── run_3d_grid_ablation.py
    ├── section_alpha_neff_ablation.tex
    ├── figure_alpha_neff_caption.tex
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
