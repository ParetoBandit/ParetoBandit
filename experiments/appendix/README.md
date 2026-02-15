# BanditGPT Paper Appendix

**Supplementary Material**

---

## Overview

This directory contains the complete appendix for the banditGPT paper, organized into seven main sections (A-G) following conference conference standards.

**Total Appendix Sections**: 7  
**Organization**: Modular, self-contained sections  
**Format**: LaTeX source files + figures  
**Master File**: `APPENDIX_MASTER.tex`

---

## Quick Navigation

| Section | Topic | Key Content |
|---------|-------|-------------|
| **[A](#appendix-a-mathematical-foundations)** | Mathematical Foundations | Proofs, bounds, theoretical analysis |
| **[B](#appendix-b-dataset-details)** | Dataset Details | Data provenance, 1M scale validation |
| **[C](#appendix-c-hyperparameter-sensitivity)** | Hyperparameter Sensitivity | Robustness across 20× parameter range |
| **[D](#appendix-d-ablation-studies)** | Ablation Studies | Component validation, 45 experiments |
| **[E](#appendix-e-extended-results)** | Extended Results | Catastrophic failure, multi-model routing |
| **[F](#appendix-f-implementation-details)** | Implementation Details | Config, deployment, best practices |
| **[G](#appendix-g-additional-discussion)** | Additional Discussion | Limitations, ethics, future work |

---

## Structure

```
appendix/
├── README.md                           # This file
├── APPENDIX_MASTER.tex                 # Master LaTeX file (includes all sections)
│
├── A_mathematical_foundations/
│   ├── README.md                       # Section overview
│   ├── A1_spectral_separation_proof.tex
│   └── figures/
│
├── B_dataset_details/
│   ├── README.md
│   ├── B2_validation_methodology.tex
│   ├── B3_1M_scale_analysis.tex
│   └── figures/
│
├── C_hyperparameter_sensitivity/
│   ├── README.md
│   ├── C1_comprehensive_sensitivity.tex
│   ├── C5_robustness_summary.tex
│   └── figures/
│       └── (sensitivity plots)
│
├── D_ablation_studies/
│   ├── README.md
│   ├── D1_corralling_ablation.tex
│   └── figures/
│       ├── ablation_study.png
│       ├── figure6_learning_rate_ablation.pdf
│       ├── figure_alpha_ablation.png
│       └── figure_gamma_ablation.png
│
├── E_extended_results/
│   ├── README.md
│   ├── E1_catastrophic_failure.tex
│   ├── E1_catastrophic_failure_extended.tex
│   └── figures/
│
├── F_implementation_details/
│   ├── README.md
│   ├── F1_configuration_details.tex
│   ├── F2_experimental_setup.tex
│   ├── F3_strategy_selection_guide.tex
│   └── figures/
│
└── G_additional_discussion/
    ├── README.md
    ├── G1_practical_recommendations.tex
    ├── G2_limitations.tex
    ├── G2_limitations_addendum.tex
    └── figures/
```

---

## Appendix A: Mathematical Foundations

**Purpose**: Theoretical foundations and formal proofs

**Contents**:
- Spectral separation and error bounding proof
- Regret bounds for meta-algorithms
- Decision margin hypothesis
- Bayesian formulation

**Key Results**:
- Routing error: $P_{\text{error}} \le \exp\left(-\frac{\Delta_{\text{gap}}^2}{2\sigma^2}\right)$
- Regret bound: $R(T) \le \mathcal{O}\left( \frac{\ln(N)}{\eta} + \sqrt{T \ln(T)} \right)$

**Files**: 1 main LaTeX file, mathematical derivations

---

## Appendix B: Dataset Details

**Purpose**: Comprehensive data documentation and scale validation

**Contents**:
- Dataset composition and provenance (81,871 prompts)
- Statistical validation methodology
- 1M scale analysis (594,199 prompts, 317× scale increase)
- Production-scale semantic discontinuity

**Key Results**:
- PC1 variance invariant: 3.10% (holdout) vs 3.10% (1M)
- Distribution shift: 82.4% → 94.1% routine tasks
- Economic impact: $2.3M/year savings potential

**Files**: 2 main LaTeX files + validation tables

---

## Appendix C: Hyperparameter Sensitivity

**Purpose**: Demonstrate robustness to hyperparameter choices

**Contents**:
- Comprehensive sensitivity analysis ($n_{\text{eff}} \in [1.0, 20.0]$)
- Learning rate and mixing parameter sensitivity
- Robustness to imperfect semantic neighbors
- Practical guidelines

**Key Results**:
- Perfect robustness: All $n_{\text{eff}}$ yield identical performance
- Mean reward: 4.48 across all values (+39.2% vs Cold Start)
- Weak prior ($n_{\text{eff}}=1.0$) sufficient for major improvement

**Files**: 2 LaTeX files + sensitivity figures

---

## Appendix D: Ablation Studies

**Purpose**: Validate component contributions through controlled experiments

**Contents**:
- Corralling hyperparameter ablation (45 experiments)
- 5 learning rates × 3 mixing parameters × 3 seeds
- Feature engineering ablation
- Exploration strategy ablation

**Key Results**:
- Optimal: $\eta=5.0$, $\gamma=0.10$ (Regret: 59.33 ± 3.40)
- Sublinear regret confirmed: $\beta=0.669$ (R²=0.9903)
- Exploration floor reduces variance 14× (std: 54.46 → 3.40)

**Files**: 1 LaTeX file + 4 ablation figures

---

## Appendix E: Extended Results

**Purpose**: Additional experiments and supplementary evaluations

**Contents**:
- Catastrophic failure detection (3-phase scenario)
- Three-model routing results
- Alternative cost profiles
- Distribution shift robustness

**Key Insights**:
- Use Corralling for catastrophic failures (d > 1.0)
- NOT for subtle optimization (d < 0.2)
- System scales to 3+ models effectively

**Files**: 2+ LaTeX files + extended result figures

---

## Appendix F: Implementation Details

**Purpose**: Enable reproduction and practical deployment

**Contents**:
- Configuration parameters and defaults
- Experimental setup and protocols
- Strategy selection guide
- Hyperparameter selection guidelines
- Production deployment checklist

**Key Resources**:
- Quick reference configs (conservative, aggressive, default)
- Troubleshooting guide
- Performance optimization tips
- Installation instructions

**Files**: 3 LaTeX files + configuration tables

---

## Appendix G: Additional Discussion

**Purpose**: Extended discussion of limitations, ethics, and future work

**Contents**:
- Practical deployment recommendations
- System limitations and mitigation strategies
- Broader impact and ethical considerations
- When to use Corralling vs. offline optimization
- Future research directions

**Key Topics**:
- Environmental and economic impact
- Fairness and bias considerations
- Safety and transparency
- Alternative approaches

**Files**: 3 LaTeX files + discussion materials

---

## Compilation Instructions

### Compile Entire Appendix (Standalone)

```bash
cd experiments/appendix
pdflatex APPENDIX_MASTER.tex
pdflatex APPENDIX_MASTER.tex  # Run twice for references
```

### Include in Main Paper

In your main paper LaTeX file:

```latex
% At the end of your main content
\appendix
\input{experiments/appendix/APPENDIX_MASTER.tex}
```

### Compile Individual Sections

Each section can be compiled independently:

```bash
cd experiments/appendix/C_hyperparameter_sensitivity
pdflatex C1_comprehensive_sensitivity.tex
```

---

## conference Format Compliance

### Requirements Met ✅

- **Font**: Times Roman, 10pt
- **Margins**: 1 inch all sides
- **Figures**: High-resolution (300+ DPI)
- **Tables**: Simple formatting with booktabs
- **References**: Numbered, consistent citations
- **Sections**: Clearly numbered (A, B, C...)
- **Structure**: Modular, self-contained sections

### Format Guidelines

1. **Figures**: Place in respective `figures/` subdirectories
2. **Tables**: Use booktabs package (toprule, midrule, bottomrule)
3. **Math**: Use amsmath for equations, number important ones
4. **References**: Use `\label{}` and `\ref{}` for cross-references
5. **Citations**: Use numbered bibliography style

---

## Content Statistics

| Metric | Count |
|--------|-------|
| Total Sections | 7 (A-G) |
| LaTeX Files | 12+ |
| Figures | 15+ |
| Tables | 8+ |
| Experiments Documented | 45+ |
| Pages (estimated) | 20-25 |

---

## Source File Mapping

### Original Location → Appendix Location

**Mathematical Foundations**:
- `03_appendix/spectral_separation_proof.tex` → `A/A1_spectral_separation_proof.tex`

**Dataset Details**:
- `01_figure/validation_methodology.tex` → `B/B2_validation_methodology.tex`
- `appendix_d/figure_1M_analysis.tex` → `B/B3_1M_scale_analysis.tex`

**Hyperparameter Sensitivity**:
- `appendix_d/hyperparameter_sensitivity.tex` → `C/C1_comprehensive_sensitivity.tex`
- `appendix_e/hyperparameter_robustness.tex` → `C/C5_robustness_summary.tex`

**Ablation Studies**:
- `04_figure/appendix_ablation_study.tex` → `D/D1_corralling_ablation.tex`

**Extended Results**:
- `06_figure/figure5_corralling_kdd.tex` → `E/E1_catastrophic_failure.tex`

**Implementation Details**:
- `03_figure/latex_appendix_config.tex` → `F/F1_configuration_details.tex`
- `03_figure/latex_table_strategy_guide.tex` → `F/F3_strategy_selection_guide.tex`
- `08_figure/experiments_setup_compact.tex` → `F/F2_experimental_setup.tex`

**Discussion**:
- `03_figure/latex_section_5.3_practical_recommendations.tex` → `G/G1_practical_recommendations.tex`
- `03_figure/latex_section_6_limitations.tex` → `G/G2_limitations.tex`
- `08_figure/limitations_addendum.tex` → `G/G2_limitations_addendum.tex`

---

## Maintenance Notes

### Adding New Content

1. Identify appropriate appendix section (A-G)
2. Create numbered subsection file (e.g., `C2_new_analysis.tex`)
3. Add figure to `figures/` subdirectory
4. Update section README.md
5. Add `\input{}` command to `APPENDIX_MASTER.tex`
6. Update this main README

### Consolidating Duplicates

Some content exists in multiple locations. Priority order:
1. `appendix/` (new organized structure) - PRIMARY
2. `appendix_d/`, `appendix_e/` (old structure) - DEPRECATED
3. Individual experiment folders - KEEP FOR REFERENCE

---

## Related Documentation

- **Main experiments README**: `../README.md`
- **Organization plan**: `../APPENDIX_ORGANIZATION_PLAN.md`
- **Cleanup summary**: `../CLEANUP_SUMMARY.md`
- **Cross-experiment validation**: `../CROSS_EXPERIMENT_VALIDATION.md`

---

## Contact & Contribution

For questions about appendix organization or to suggest improvements:

1. Read section-specific README in each subdirectory
2. Check `APPENDIX_ORGANIZATION_PLAN.md` for design rationale
3. Follow conference format guidelines for any additions
4. Maintain modularity - each section should be self-contained

---

**Last Updated**: February 13, 2026  
**Status**: ✅ Structure complete, content organized  
**Ready for**: Compilation and integration with main paper

---

## Quick Commands

```bash
# Navigate to appendix
cd experiments/appendix

# List all sections
ls -d */

# Compile master appendix
pdflatex APPENDIX_MASTER.tex

# Check all LaTeX files
find . -name "*.tex" -type f

# Check all figures
find . -name "*.png" -o -name "*.pdf" | grep figures

# Word count (approximate)
texcount APPENDIX_MASTER.tex
```
