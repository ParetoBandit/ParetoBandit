# Appendix Content Map (Final)

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
| `A1_spectral_separation_proof.tex` | Fig 1, Fig 3 | Spectral separation proof, regret bounds |

---

### Appendix B: Dataset Details

| File | Supports | Content |
|------|----------|---------|
| `B1_validation_methodology.tex` | Fig 1, Table 2 | Statistical significance tests, threshold validation, dimensionality robustness, data quality |

---

### Appendix C: Hyperparameter Sensitivity (Semantic Transfer)

| File | Supports | Content |
|------|----------|---------|
| `C1_comprehensive_sensitivity.tex` | Fig 4 | $n_{\text{eff}}$ sensitivity: Bayesian formulation, 20x robustness, imperfect neighbors |

**Scope**: Validates the semantic transfer prior strength ($n_{\text{eff}}$). Does NOT cover Corralling parameters ($\eta$, $\gamma$) — those are in Appendix D.

---

### Appendix D: Ablation Studies (Corralling)

| File | Supports | Content |
|------|----------|---------|
| `D1_corralling_ablation.tex` | Fig 3, Fig 4, Fig 6 | 45-experiment grid search over $\eta$ and $\gamma$, sublinear regret validation ($\beta$=0.669) |

**Scope**: Validates Corralling meta-learner parameters ($\eta$, $\gamma$). Cross-references Appendix C for $n_{\text{eff}}$ validation.

---

### Appendix E: Extended Results

| File | Supports | Content |
|------|----------|---------|
| `E1_catastrophic_failure.tex` | Fig 6 | K=5 portfolio, production router, 20 seeds, portfolio scaling |

**Experiment code**: `E_catastrophic_failure_experiment/generate_figure6_5model.py`

---

### Appendix F: Implementation Details

| File | Supports | Content |
|------|----------|---------|
| `F1_configuration_details.tex` | All | Part 1: Library router parameters (all classes in `router.py`); Part 2: Experiment configs; Implementation notes (init\_lambda, two-level cost, loss\_decay) |
| `F2_experimental_setup.tex` | All | Hardware specs, software versions, runtimes, evaluation protocol, zero-leakage design |

**Scope**: F1 is the authoritative bridge between `router.py` and the paper. Library defaults alongside experiment values. No analysis, no recommendations, no code examples.

---

### Appendix G: Limitations and Future Work

| File | Supports | Content |
|------|----------|---------|
| `G1_limitations.tex` | All | Prior quality dependency, strategy trade-offs, variance, computational overhead, generalizability |
| `G1_limitations_addendum.tex` | All | Regime-dependent effects, generalizability of regime frequencies |

**Scope**: Honest discussion of system constraints. Merge addendum into G1 for camera-ready.

---

## Deduplication Boundaries

To prevent content overlap, each section has a clear ownership boundary:

| Topic | Owner | Others must NOT cover |
|-------|-------|-----------------------|
| $n_{\text{eff}}$ sensitivity | **C1** | D1 references C1, F1 lists values only |
| $\eta$, $\gamma$ ablation | **D1** | F1 lists values only, G1 discusses limitations |
| Strategy selection guidance | **G1** | Removed from F1 (was duplicated) |
| Monitoring / deployment code | **None** | Removed from appendix (belongs in GitHub README) |
| Regret numbers (49.5, 59.2, 74.7) | **G1** | F1 does not include performance tables |
| Hyperparameter values (tables) | **F1** | Other sections reference F1 for values |
| Hardware/software/protocol | **F2** | Not repeated elsewhere |

---

## Traceability Matrix

| Main Paper Element | Appendix Support |
|-------------------|-----------------|
| **Figure 1** (PCA validation) | A1 (spectral proof), B1 (methodology) |
| **Table 2** (data provenance) | B1 (validation methodology) |
| **Figure 3** (corralling insurance) | A1 (regret bounds), D1 (ablation) |
| **Figure 4** (Pareto frontier) | C1 ($n_{\text{eff}}$ sensitivity), D1 ($\eta$ validation), F1/F2 (reproducibility) |
| **Figure 6** (catastrophic failure) | E1 (K=5 experiment), A1 (meta-algorithm bounds) |
| **All** | F1/F2 (reproducibility), G1 (limitations) |

---

## Active File Count

| Metric | Count |
|--------|-------|
| Appendix sections | 7 (A-G) |
| Active LaTeX files | 8 |
| Experiment scripts | 1 (`generate_figure6_5model.py`) |

---

## Figures Pending Generation

| Figure | Script | Output Path |
|--------|--------|-------------|
| Figure 1 | `experiments/01_figure/plot_figure1.py` | `01_figure/results/` |
| Figure 3 | `experiments/03_figure/run_all_experiments.py` | `03_figure/results/` |
| Figure 4 | `experiments/04_figure/generate_pareto_frontier.py` | `04_figure/results/` |
| Figure 6 | `appendix/E_catastrophic_failure_experiment/generate_figure6_5model.py` | `E_catastrophic_failure_experiment/results/` |

**Note**: The E1 `\includegraphics` is currently commented out pending figure generation.

---

**Status**: Clean — F1 aligned to router.py, outdated references (Figure 8, 08\_figure) removed, C1 formula corrected, D1 loss\_decay documented  
**Last Updated**: February 15, 2026
