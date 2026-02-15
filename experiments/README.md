# BanditGPT Experiments

This directory contains all experimental code for the banditGPT paper.

**Organization Principle**: Each numbered folder corresponds to a specific figure or table in the paper, enabling 1:1 traceability from "Figure X" in the PDF to the exact script that generated it.

---

## Paper-to-Code Mapping

| Paper Object | Content | Experiment Folder | Key Output |
|---|---|---|---|
| **Figure 1** | Model Preference Heterogeneity (PCA analysis) | [`01_figure/`](01_figure/) | `figure1_lmsys_holdout_pca.png` |
| **Table 2** | Dataset Description and Experimental Splits | [`02_table/`](02_table/) | `table1_dataset.tex` |
| **Figure 3** | Corralling as Insurance (Prior Quality Degradation) | [`03_figure/`](03_figure/) | `figure3_prior_degradation.png` |
| **Figure 4** | Pareto Frontier (banditGPT vs RouteLLM) | [`04_figure/`](04_figure/) | `figure5_pareto_frontier.png` |
| **Figure 6** | Catastrophic Failure Detection (K=5 Portfolio) | [`06_figure/`](06_figure/) | `figure6_5model.png` |

**Note:** Figure 2 (architecture diagram) and Figure 5 (sensitivity zoomed, appendix) do not have dedicated experiment directories — they are static diagrams or produced from appendix-level analysis.

### Appendix Experiments

Additional appendix content is organized in [`appendix/`](appendix/).

---

## Directory Structure

```
experiments/
├── README.md                     # This file
├── 01_figure/                    # Figure 1: Model Preference Heterogeneity
├── 02_table/                     # Table 2: Dataset Description
├── 03_figure/                    # Figure 3: Corralling Insurance Analysis
├── 04_figure/                    # Figure 4: Pareto Frontier
├── 06_figure/                    # Figure 6: Catastrophic Failure Detection
├── appendix/                     # Appendix experiments and extended results
└── utils/                        # Shared utilities
```

---

**Last Updated**: February 2026
