# Appendix Organization Plan: Conference-Compliant Structure

**Date**: February 13, 2026  
**Purpose**: Organize all supplementary experiments and analyses into a well-structured appendix

---

## Current State Analysis

### Main Paper Content (Should NOT be in Appendix)
- **Figure 1** (01_figure): Alignment Tax Discovery & Validation - Core contribution
- **Table 1** (01_table): Dataset Composition - Essential context
- **Figure 2** (02_figure): Feature Distribution Shift - Core motivation
- **Table 2** (02_table): Performance Gap Analysis - Core results
- **Figure 3** (03_figure): Corralling Architecture - Core system design
- **Figure 4** (04_figure): Corralled Bandit with Semantic Projection - Core algorithm
- **Figure 5** (05_figure): Pareto Frontier - Core economic value proposition
- **Figure 6** (07_figure): Zero-Shot Readiness - Core innovation
- **Figure 7/8** (08_figure): Sensitivity Analysis - Core validation

### Content for Appendix Organization

#### Category 1: Mathematical Proofs and Theory
- ✅ **03_appendix/spectral_separation_proof.tex** - Mathematical proof of error bounding
- ✅ **appendix_c/spectral_separation_proof.tex** - Duplicate (need to consolidate)

#### Category 2: Extended Dataset Analysis
- ✅ **01_figure/figure_1M_analysis.tex** - 1M scale validation
- ✅ **appendix_d/figure_1M_analysis.tex** - Global manifold stability (duplicate)
- ✅ **01_figure/validation_methodology.tex** - Statistical validation details

#### Category 3: Hyperparameter Sensitivity & Robustness
- ✅ **appendix_d/hyperparameter_sensitivity.tex** - Comprehensive sensitivity analysis
- ✅ **appendix_e/hyperparameter_robustness.tex** - Concise robustness summary
- ✅ **08_figure/figure8_sensitivity_compact.tex** - Compact sensitivity results
- ✅ **08_figure/figure8_sensitivity_update.tex** - Updated sensitivity analysis

#### Category 4: Ablation Studies
- ✅ **04_figure/appendix_ablation_study.tex** - Corralling hyperparameter ablation
- ✅ **04_figure/results_ablation/** - Ablation study results
- ✅ **03_figure/results/ablation/** - Alpha ablation results
- ✅ **03_figure/results/gamma_ablation/** - Gamma ablation results
- ✅ **06_figure/results/figure6_learning_rate_ablation.pdf** - Learning rate ablation

#### Category 5: Additional Experimental Results
- ✅ **06_figure/** - Corralling for catastrophic failure detection (supplementary results)
- ✅ **06_figure/supplementary/** - Additional catastrophic failure experiments
- ✅ **04_figure/results_3models/** - 3-model routing results

#### Category 6: Implementation & Configuration Details
- ✅ **03_figure/latex_appendix_config.tex** - Configuration details
- ✅ **03_figure/latex_table_strategy_guide.tex** - Strategy selection guide
- ✅ **08_figure/experiments_setup_compact.tex** - Experimental setup details
- ✅ **08_figure/experiments_table.tex** - Experimental configuration table

#### Category 7: Extended Discussion & Practical Guidance
- ✅ **03_figure/latex_section_5.3_practical_recommendations.tex** - Practical recommendations
- ✅ **03_figure/latex_section_6_limitations.tex** - Limitations discussion
- ✅ **08_figure/limitations_addendum.tex** - Additional limitations

---

## Conference-Compliant Appendix Structure

### Recommended Organization

```
Appendix A: Mathematical Foundations
├── A.1: Spectral Separation and Error Bounding
├── A.2: Regret Bounds for Meta-Algorithms
└── A.3: Bayesian Formulation of Semantic Transfer

Appendix B: Dataset Details and Scale Validation
├── B.1: Dataset Composition and Provenance (extended from Table 1)
├── B.2: Statistical Validation Methodology
├── B.3: Global Manifold Stability (1M Scale Analysis)
└── B.4: Production-Scale Semantic Discontinuity

Appendix C: Hyperparameter Sensitivity Analysis
├── C.1: Prior Strength Sensitivity (n_eff)
├── C.2: Learning Rate Sensitivity (η)
├── C.3: Mixing Parameter Sensitivity (γ)
├── C.4: Robustness to Imperfect Neighbors
└── C.5: Summary and Practical Guidelines

Appendix D: Ablation Studies
├── D.1: Corralling Algorithm Components
├── D.2: Feature Engineering Ablation
├── D.3: Exploration Strategy Ablation (α, γ)
└── D.4: Multi-Seed Statistical Validation

Appendix E: Extended Experimental Results
├── E.1: Catastrophic Failure Detection
├── E.2: Three-Model Routing Results
├── E.3: Alternative Cost Profiles
└── E.4: Distribution Shift Robustness

Appendix F: Implementation Details
├── F.1: System Architecture and Configuration
├── F.2: Computational Requirements
├── F.3: Hyperparameter Selection Guide
└── F.4: Strategy Selection Table

Appendix G: Additional Discussion
├── G.1: Practical Deployment Recommendations
├── G.2: Limitations and Future Work
├── G.3: Broader Impact and Ethical Considerations
└── G.4: When to Use Corralling vs. Offline Optimization
```

---

## File Mapping to New Structure

### Appendix A: Mathematical Foundations
- `03_appendix/spectral_separation_proof.tex` (primary source)
- Content: Proofs, bounds, theoretical analysis

### Appendix B: Dataset Details
- `01_figure/validation_methodology.tex`
- `appendix_d/figure_1M_analysis.tex`
- `01_table/table_dataset_composition.tex` (extended version)

### Appendix C: Hyperparameter Sensitivity
- `appendix_d/hyperparameter_sensitivity.tex` (primary - most comprehensive)
- `08_figure/figure8_sensitivity_compact.tex` (can be consolidated)
- Figures from `08_figure/results/`

### Appendix D: Ablation Studies
- `04_figure/appendix_ablation_study.tex`
- Figures from `04_figure/results_ablation/`
- Figures from `03_figure/results/ablation/`
- `06_figure/results/figure6_learning_rate_ablation.pdf`

### Appendix E: Extended Results
- `06_figure/` content (catastrophic failure)
- `04_figure/results_3models/` content
- `02_table/` extended robustness results

### Appendix F: Implementation Details
- `03_figure/latex_appendix_config.tex`
- `03_figure/latex_table_strategy_guide.tex`
- `08_figure/experiments_setup_compact.tex`
- `08_figure/experiments_table.tex`

### Appendix G: Discussion
- `03_figure/latex_section_5.3_practical_recommendations.tex`
- `03_figure/latex_section_6_limitations.tex`
- `08_figure/limitations_addendum.tex`

---

## Duplicate Files to Consolidate

1. **Spectral Separation Proof**:
   - `03_appendix/spectral_separation_proof.tex`
   - `appendix_c/spectral_separation_proof.tex`
   - → Keep in new `appendix/A_mathematical_foundations/`

2. **1M Scale Analysis**:
   - `01_figure/figure_1M_analysis.tex`
   - `appendix_d/figure_1M_analysis.tex`
   - → Keep in new `appendix/B_dataset_details/`

3. **Hyperparameter Sensitivity**:
   - `appendix_d/hyperparameter_sensitivity.tex` (comprehensive)
   - `appendix_e/hyperparameter_robustness.tex` (concise)
   - `08_figure/figure8_sensitivity_compact.tex`
   - → Consolidate into `appendix/C_hyperparameter_sensitivity/`

4. **Sensitivity Figures**:
   - Multiple versions of sensitivity plots in `08_figure/results/`
   - → Consolidate best versions

---

## Action Items

### Phase 1: Create New Appendix Structure
1. Create `experiments_v1/appendix/` folder with A-G subfolders
2. Map existing content to new structure
3. Consolidate duplicate content
4. Create master appendix LaTeX file

### Phase 2: Content Organization
1. Copy/move relevant .tex files to appropriate appendix sections
2. Copy/move relevant figures to appendix folders
3. Update cross-references
4. Create section-level README files

### Phase 3: Quality Control
1. Verify all figures referenced in appendix exist
2. Check for broken references
3. Ensure consistent formatting
4. Validate LaTeX compilation

### Phase 4: Documentation
1. Create APPENDIX_MASTER.tex with \input commands
2. Create APPENDIX_FIGURES.md listing all appendix figures
3. Update main README.md with appendix structure
4. Add citation guidelines

---

## Conference-Specific Requirements

### Format Guidelines
- **Font**: Times Roman, 10pt
- **Margins**: 1 inch all around
- **Figures**: High-resolution (300+ DPI), color or grayscale
- **Tables**: Simple, clear formatting with booktabs
- **References**: Numbered, consistent citation style
- **Sections**: Clearly numbered (A, B, C...)
- **Page Numbers**: Continuous from main paper

### Content Guidelines
- Focus on reproducibility
- Include sufficient detail for replication
- Provide statistical validation
- Document all hyperparameters
- Include negative results if informative
- Keep appendix modular (each section self-contained)

### Length Considerations
- conference typically allows unlimited appendix length (online)
- But recommend 15-20 pages for readability
- Prioritize most important supplementary material
- Consider online-only supplementary materials for extensive details

---

## Benefits of This Organization

1. **Modular**: Each appendix section is self-contained
2. **Logical Flow**: Theory → Data → Hyperparameters → Ablations → Results → Implementation → Discussion
3. **Easy Navigation**: Clear section labels (A-G)
4. **Reproducibility**: All details needed to replicate experiments
5. **Publication-Ready**: Follows conference conventions
6. **Flexible**: Can easily add/remove sections
7. **Maintainable**: Clear file structure and mapping

---

## Next Steps

1. Review this plan
2. Create appendix folder structure
3. Begin content migration
4. Consolidate duplicates
5. Create master LaTeX file
6. Compile and verify

**Estimated Time**: 2-3 hours for complete reorganization
