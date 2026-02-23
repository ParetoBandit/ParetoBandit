# BanditGPT Experiments

This directory contains all experimental code for the banditGPT paper.

**Organization Principle**: Each numbered folder corresponds to a specific figure or table in the paper, enabling 1:1 traceability from "Figure X" in the PDF to the exact script that generated it.

---

## Paper-to-Code Mapping

| Paper Object | Content | Experiment Folder | Key Output |
|---|---|---|---|
| **Figure 1** | Model Preference Heterogeneity (PCA analysis) | [`01_figure/`](01_figure/) | `figure1_lmsys_holdout_pca.png` |
| **Figure 2** | Router Architecture Diagram | [`02_figure/`](02_figure/) | `figure2_architecture.pdf` |
| **Table 2** | Dataset Description and Experimental Splits | [`02_table/`](02_table/) | `table1_dataset.tex` |
| **Figure 3** | Corralling Prior Quality Degradation Sweep | [`03_figure/`](03_figure/) | `figure3_prior_degradation.png` |
| **Figure 4** | Pareto Frontier (banditGPT vs baselines) | [`04_figure/`](04_figure/) | `figure4.png` |
| **Figure 5** | K-Scaling: Hybrid vs Disjoint LinUCB | [`05_figure/`](05_figure/) | `k_scaling_figure.png` |
| **Figure 6** | Multi-Model Pareto Frontier (K=5, K=10) | [`06_figure/`](06_figure/) | `figure6_multimodel_pareto.png` |
| **Figure 7** | banditGPT vs Linear Thompson Sampling | [`07_figure/`](07_figure/) | `figure7_lints_comparison.pdf` |
| **Figure 8** | Cumulative Regret During Online Learning | [`08_figure/`](08_figure/) | `figure8_cumulative_regret.pdf` |

### Appendix Experiments

| Paper Object | Content | Experiment Folder | Key Output |
|---|---|---|---|
| **Figure 6 (Appendix)** | Catastrophic Failure Detection (K=5 Portfolio) | [`appendix/E_catastrophic_failure_experiment/`](appendix/E_catastrophic_failure_experiment/) | `figure6_5model.png` |

Additional appendix content is organized in [`appendix/`](appendix/).

---

## Directory Structure

```
experiments/
├── README.md                     # This file
├── 01_figure/                    # Figure 1: Model Preference Heterogeneity
├── 02_figure/                    # Figure 2: Router Architecture Diagram
├── 02_table/                     # Table 2: Dataset Description
├── 03_figure/                    # Figure 3: Prior Quality Degradation Sweep
├── 04_figure/                    # Figure 4: Pareto Frontier
├── 05_figure/                    # Figure 5: K-Scaling (Hybrid vs Disjoint)
├── 06_figure/                    # Figure 6: Multi-Model Pareto (K=5, K=10)
├── 07_figure/                    # Figure 7: banditGPT vs LinTS
├── 08_figure/                    # Figure 8: Cumulative Regret
├── appendix/                     # Appendix experiments and extended results
│   └── E_catastrophic_failure_experiment/  # Catastrophic Failure Detection
└── utils/                        # Shared utilities (router_factory, plotting, etc.)
```

---

**Last Updated**: February 2026
