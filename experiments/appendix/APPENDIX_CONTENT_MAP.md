# Appendix Content Map (Trimmed)

**Date**: February 15, 2026  
**Purpose**: Definitive map of what stays, what's cut, and why — aligned to Figures 1, 3, 4, 6, and Table 2.

---

## Design Principle

Every appendix item must trace to a specific claim in the main paper. If it doesn't support Figures 1, 3, 4, 6, or Table 2, it's cut.

---

## Section-by-Section Map

### Appendix A: Mathematical Foundations — KEEP (unchanged)

| File | Supports | Status |
|------|----------|--------|
| `A1_spectral_separation_proof.tex` | Fig 1 (spectral separation), Fig 3 (regret bounds) | KEEP |

---

### Appendix B: Dataset Details — KEEP (trimmed)

| File | Supports | Status |
|------|----------|--------|
| `B1_validation_methodology.tex` | Fig 1 Spearman test methodology | KEEP |
| `B2_distribution_shift_details.tex` | Fig 1 cross-domain PCA generalization | KEEP |
| ~~`B1_dataset_composition.tex`~~ | Table 2 already covers this | CUT (never created) |
| ~~`B3_1M_scale_analysis.tex`~~ | No main figure relies on 1M scale | CUT |

---

### Appendix C: Hyperparameter Sensitivity — KEEP (trimmed)

| File | Supports | Status |
|------|----------|--------|
| `C1_comprehensive_sensitivity.tex` | Fig 4 robustness (20x range) | KEEP |
| `C2_robustness_summary.tex` | Space-constrained alternative to C1 | KEEP (optional) |
| ~~C2: Learning Rate~~ | Covered within C1 | CUT (stub only) |
| ~~C3: Mixing Parameter~~ | Covered within C1 | CUT (stub only) |
| ~~C4: Imperfect Neighbors~~ | Covered within C1 | CUT (stub only) |

---

### Appendix D: Ablation Studies — KEEP (expanded)

| File | Supports | Status |
|------|----------|--------|
| `D1_corralling_ablation.tex` | Fig 3 config, Fig 4 eta=1.0 | KEEP |
| `figures/figure6_learning_rate_ablation.pdf` | Figure 6 eta=0.3 choice | KEEP (new addition) |
| `figures/figure_alpha_ablation.png` | Fig 3 alpha=2.0 choice | KEEP |
| `figures/figure_gamma_ablation.png` | Fig 3 gamma=0.05 choice | KEEP |
| ~~D.2: Feature Engineering Ablation~~ | Never created | CUT |
| ~~D.4: Multi-Seed Statistical Validation~~ | Covered in D1 and STATISTICAL_NOTES.md | CUT |

---

### Appendix E: Extended Results — KEEP (slimmed to E1 only)

| File | Supports | Status |
|------|----------|--------|
| `E1_catastrophic_failure.tex` | Figure 6 (K=5 catastrophic failure) | KEEP |
| `E1_catastrophic_failure_extended.tex` | Optional deeper analysis | KEEP (optional) |
| ~~E2: Three-Model Routing~~ | Experiment removed | CUT |
| ~~E3: Alternative Cost Profiles~~ | Never created; Fig 4 covers this | CUT |
| ~~E4: Distribution Shift~~ | Never created; B.2 covers this | CUT |

**Note**: Canonical experiment code lives in `E_catastrophic_failure_experiment/`. E1 tex files are the LaTeX write-up included in APPENDIX_MASTER.

---

### Appendix F: Implementation Details — KEEP (trimmed)

| File | Supports | Status |
|------|----------|--------|
| `F1_configuration_details.tex` | Reproducibility (all experiments) | KEEP |
| `F2_experimental_setup.tex` | Reproducibility (hardware/software) | KEEP |
| ~~`F3_strategy_selection_guide.tex`~~ | Duplicates Fig 3 findings; better as GitHub README | CUT |
| ~~F4: Hyperparameter Guide~~ | Never created; Appendix C covers this | CUT |

---

### Appendix G: Limitations and Future Work — KEEP (trimmed)

| File | Supports | Status |
|------|----------|--------|
| `G1_limitations.tex` | Required for venue submission | KEEP |
| `G1_limitations_addendum.tex` | Merge into G1 for camera-ready | KEEP |
| ~~`G1_practical_recommendations.tex`~~ | Duplicates Fig 3 + F3 content | CUT |
| ~~G3: Broader Impact~~ | Never created; add only if venue requires | CUT |
| ~~G4: Corralling vs Offline~~ | Never created; not required | CUT |

---

## Traceability Matrix

Every kept appendix item maps to a main paper element:

| Main Paper Element | Appendix Support |
|-------------------|-----------------|
| **Figure 1** (PCA validation) | A1 (spectral proof), B1 (methodology), B2 (distribution shift) |
| **Table 2** (data provenance) | B1 (validation methodology) |
| **Figure 3** (corralling insurance) | A1 (regret bounds), D1 (ablation), D.3 (alpha/gamma plots) |
| **Figure 4** (Pareto frontier) | C1 (sensitivity), D1 (eta=1.0 validation), F1/F2 (reproducibility) |
| **Figure 6** (catastrophic failure) | D.2 (eta ablation), E1 (full experiment), A1 (meta-algorithm bounds) |
| **All** | G2 (limitations) |

---

## Files to Delete (candidates)

These files are kept on disk but excluded from APPENDIX_MASTER.tex compilation:

| File | Reason Excluded |
|------|----------------|
| `F3_strategy_selection_guide.tex` | Practitioner guide, not scientific appendix |
| `G1_practical_recommendations.tex` | Redundant with Fig 3 + F3 |

These were never created and their stubs are removed from READMEs:

| Planned Item | Never Created Because |
|-------------|----------------------|
| B1_dataset_composition.tex | Table 2 sufficient |
| B3_1M_scale_analysis.tex | Tangential to main figures |
| C2-C4 sensitivity subsections | C1 is comprehensive |
| E2-E4 extended results | Experiments removed or never run |
| F4_hyperparameter_guide.tex | Appendix C covers this |
| G3_broader_impact.tex | Not required by current venues |
| G4_corralling_vs_offline.tex | Not required |

---

## Active File Count

| Metric | Count |
|--------|-------|
| Appendix sections | 7 (A-G) |
| Active LaTeX files | 10 |
| Active figures | 4+ |
| Excluded but on-disk | 2 |
| Never created (stubs removed) | 7+ |

---

**Status**: Trimmed and aligned to main paper figures  
**Last Updated**: February 15, 2026
