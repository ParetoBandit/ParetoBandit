# BanditGPT Experiments

This directory contains all experimental code for the banditGPT paper.

**Organization Principle**: Each numbered folder corresponds to a specific figure or table in the paper, enabling 1:1 traceability from "Figure X" in the PDF to the exact script that generated it.

---

## Paper-to-Code Mapping

| Paper Object | Content | Experiment Folder | Key Output |
|---|---|---|---|
| **Figure 1** | Contextual Sensitivity across K=10 Models | [`01_figure/`](01_figure/) | `figure1_k10_contextual.png` |
| **Figure 2** | Router Architecture Diagram | [`02_figure/`](02_figure/) | `figure2_architecture.pdf` |
| **Table 1** | Dataset Description and Experimental Splits | [`02_table/`](02_table/) | `table1_dataset.tex` |
| **Figure 3** | K=2 BanditGPT vs RouteLLM (Pareto + learning curve) | [`03_figure/`](03_figure/) | `figure3_k2.png` |
| **Figure 4** | K=10 Multi-Model Pareto Frontier | [`04_figure/`](04_figure/) | `figure4_k10.png` |
| **Figure 5** | K-Scaling: Hybrid vs Disjoint LinUCB | [`05_figure/`](05_figure/) | `k_scaling_figure.png` |
| **Figure 6** | Distribution Shift Factorial | [`06_distribution_shift/`](06_distribution_shift/) | `figure_distribution_shift.png` |
| **Figure 7** | banditGPT vs Linear Thompson Sampling | [`07_figure/`](07_figure/) | `figure7_lints_comparison.pdf` |
| **Figure 8** | Cumulative Regret During Online Learning | [`08_figure/`](08_figure/) | `figure8_cumulative_regret.pdf` |

### Appendix Experiments

| Paper Object | Content | Experiment Folder | Key Output |
|---|---|---|---|
| **Figure 9** | Catastrophic Failure Detection (K=5 Portfolio) | [`appendix/E_catastrophic_failure_experiment/`](appendix/E_catastrophic_failure_experiment/) | `figure9_5model.png` |
| **Figure 11** | Corralling Prior Quality Degradation Sweep | [`appendix/E_prior_degradation/`](appendix/E_prior_degradation/) | `figure3_prior_degradation.png` |
| **Figure 12** | Hard Constraint Enforcement | [`appendix/F_constraint_impact/`](appendix/F_constraint_impact/) | `figure_constraint_impact.png` |

Additional appendix content is organized in [`appendix/`](appendix/).

---

## Directory Structure

```
experiments/
├── README.md                     # This file
├── 01_figure/                    # Figure 1: Model Preference Heterogeneity
├── 02_figure/                    # Figure 2: Router Architecture Diagram
├── 02_table/                     # Table 1: Dataset Description
├── 03_figure/                    # Figure 3: K=2 BanditGPT vs RouteLLM
├── 04_figure/                    # Figure 4: K=10 Multi-Model Pareto Frontier
├── 05_figure/                    # Figure 5: K-Scaling (Hybrid vs Disjoint)
├── 06_distribution_shift/        # Figure 6: Distribution Shift Factorial
├── 07_figure/                    # Figure 7: banditGPT vs LinTS
├── 08_figure/                    # Figure 8: Cumulative Regret
├── appendix/                     # Appendix experiments and extended results
│   ├── D_semantic_transfer_ablation/  # Semantic Transfer Ablation (Appendix D)
│   ├── E_catastrophic_failure_experiment/  # Catastrophic Failure Detection
│   ├── E_prior_degradation/      # Prior Quality Degradation Sweep (Appendix E)
│   └── F_constraint_impact/      # Hard Constraint Enforcement (Appendix F)
└── utils/                        # Shared utilities (router_factory, plotting, etc.)
```

---

**Last Updated**: February 2026
