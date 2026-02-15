# Appendix Content Map

**Date**: February 15, 2026  
**Purpose**: Definitive map of appendix contents — aligned to Figures 1, 3, 4, 6, and Table 2.

---

## Design Principle

Every appendix item must trace to a specific claim in the main paper. If it doesn't support Figures 1, 3, 4, 6, or Table 2, it's cut. No content should appear in two appendix sections.

---

## Section-by-Section Map

### Appendix A: Mathematical Foundations

| File | Supports | Content |
|------|----------|---------|
| `A1_regret_decomposition.tex` | Fig 3, Fig 4 | LinUCB + Corralling regret bounds, composite decomposition |
| `A2_safety_guarantee.tex` | Fig 3, Fig 4, Fig 6 | γ-mixing safety proof, recovery bound, ablation table (45 experiments over η/γ, sublinear regret validation β=0.669) |
| `A3_warmup_transfer.tex` | Fig 4 | Prior transfer theory, n_eff analysis, naive-vs-correct injection, practical recommendation, limitations |

**Note**: A.2 now contains the ablation table formerly in Appendix C (Ablation Studies). A.3 contains the prior transfer sensitivity content formerly in Appendix C (Hyperparameter Sensitivity).

---

### Appendix B: Dataset Details

| File | Supports | Content |
|------|----------|---------|
| `B1_validation_methodology.tex` | Fig 1, Table 2 | Spearman correlation design, null baseline, statistical tests |
| `B2_cross_domain_transfer.tex` | Fig 1, Table 2 | Data provenance, feature pipeline (384D → 33D), PCA explained variance |

---

### Appendix C: Extended Results

| File | Supports | Content |
|------|----------|---------|
| `C1_catastrophic_failure.tex` | Fig 6 | K=5 portfolio, production router, 20 seeds, portfolio scaling |

**Experiment code**: `E_catastrophic_failure_experiment/generate_figure6_5model.py`

---

### Appendix D: Implementation Details

| File | Supports | Content |
|------|----------|---------|
| `D1_configuration_details.tex` | All | Part 1: Library router parameters (all classes in `router.py`); Part 2: Experiment configs; Implementation notes (init\_lambda, two-level cost, loss\_decay) |
| `D2_experimental_setup.tex` | All | Hardware specs, software versions, runtimes, evaluation protocol, zero-leakage design |

**Scope**: D.1 is the authoritative bridge between `router.py` and the paper.

---

### Appendix E: Limitations and Future Work

| File | Supports | Content |
|------|----------|---------|
| `E1_limitations.tex` | All | Prior quality dependency, strategy trade-offs, ablation mechanism validation, variance/reproducibility, regime-dependent behavior, computational overhead, generalizability |
| `E2_positioning.tex` | Section 2 | Bandit router taxonomy (4 families), 5 key differentiators vs. PILOT/BaRP/LLM Bandit, honest acknowledgment of concurrent work advantages |

**Scope**: E.1 covers honest system constraints. E.2 provides detailed architectural positioning against concurrent bandit-based LLM routers, supporting the brief comparison in the main paper's Related Work section.

---

## Consolidated Content

Former **Appendix C (Ablation Studies)** was removed as a standalone section.
Its useful content was folded into:
- **A.2**: Ablation table (45 experiments over η/γ), sublinear regret validation, exploration floor analysis

Former **Appendix C (Hyperparameter Sensitivity)** was previously removed as a standalone section.
Its useful content was folded into:
- **A.3**: Naive vs. correct prior injection comparison, practical n_eff recommendation [2, 10]

The original `C1_corralling_ablation.tex` and `C1_comprehensive_sensitivity.tex` are retained in git history.

---

## Deduplication Boundaries

| Topic | Owner | Others must NOT cover |
|-------|-------|-----------------------|
| $n_{\text{eff}}$ theory + sensitivity | **A.3** | A.2 references A.3, D.1 lists values only |
| $\eta$, $\gamma$ ablation | **A.2** | D.1 lists values only, E.1 discusses limitations |
| Strategy selection guidance | **E.1** | Removed from D.1 (was duplicated) |
| Monitoring / deployment code | **None** | Removed from appendix (belongs in GitHub README) |
| Hyperparameter values (tables) | **D.1** | Other sections reference D.1 for values |
| Hardware/software/protocol | **D.2** | Not repeated elsewhere |
| PILOT/BaRP/LLM Bandit comparison | **E.2** | Section 2 has brief summary; E.2 has full taxonomy + differentiators |

---

## Traceability Matrix

| Main Paper Element | Appendix Support |
|-------------------|-----------------|
| **Figure 1** (PCA validation) | A.1 (regret bounds), B.1 (methodology), B.2 (feature pipeline) |
| **Table 2** (data provenance) | B.1 (validation methodology), B.2 (cross-domain transfer) |
| **Figure 3** (corralling insurance) | A.1 (regret bounds), A.2 (safety + ablation) |
| **Figure 4** (Pareto frontier) | A.3 ($n_{\text{eff}}$ theory), A.2 ($\eta$ validation), D.1/D.2 (reproducibility) |
| **Figure 6** (catastrophic failure) | C.1 (K=5 experiment), A.2 (safety guarantee) |
| **Section 2** (Related Work) | E.2 (bandit router positioning, taxonomy) |
| **All** | D.1/D.2 (reproducibility), E.1 (limitations) |

---

## Active File Count

| Metric | Count |
|--------|-------|
| Appendix sections | 5 (A-E) |
| Active LaTeX files | 10 |
| Experiment scripts | 1 (`generate_figure6_5model.py`) |

---

## Figures Pending Generation

| Figure | Script | Output Path |
|--------|--------|-------------|
| Figure 1 | `experiments/01_figure/plot_figure1.py` | `01_figure/results/` |
| Figure 3 | `experiments/03_figure/run_all_experiments.py` | `03_figure/results/` |
| Figure 4 | `experiments/04_figure/generate_pareto_frontier.py` | `04_figure/results/` |
| Figure 6 | `appendix/E_catastrophic_failure_experiment/generate_figure6_5model.py` | `E_catastrophic_failure_experiment/results/` |

---

**Status**: Clean — former Appendix C (Ablation Studies) consolidated into A.2, sections re-lettered D→C, E→D, F→E  
**Last Updated**: February 15, 2026
