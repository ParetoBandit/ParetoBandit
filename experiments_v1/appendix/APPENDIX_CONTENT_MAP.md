# Appendix Content Mapping & Verification

**Date**: February 13, 2026  
**Purpose**: Visual map of what content is in each appendix section and verification checklist

---

## Content Distribution

### Main Paper vs. Appendix Decision Tree

```
┌─────────────────────────────────────────────────────────┐
│ Is this content ESSENTIAL for understanding the work?   │
└────────────────┬────────────────────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
        YES              NO
         │                │
         ▼                ▼
   MAIN PAPER      ┌─────────────────────────────────┐
   - Core figures  │ Does it provide PROOF/THEORY?   │
   - Key tables    └──────────┬──────────────────────┘
   - Main results            YES → Appendix A
                              NO  → ┐
                                    │
   ┌─────────────────────────────────┴────────────────┐
   │ Does it provide EXTENDED DATA/VALIDATION?        │
   └──────────┬───────────────────────────────────────┘
             YES → Appendix B
              NO  → ┐
                    │
   ┌────────────────┴─────────────────────────────────┐
   │ Does it show PARAMETER ROBUSTNESS?               │
   └──────────┬─────────────────────────────────────┘
             YES → Appendix C
              NO  → ┐
                    │
   ┌────────────────┴─────────────────────────────────┐
   │ Does it show COMPONENT CONTRIBUTION?             │
   └──────────┬─────────────────────────────────────┘
             YES → Appendix D (Ablations)
              NO  → ┐
                    │
   ┌────────────────┴─────────────────────────────────┐
   │ Does it show ADDITIONAL EXPERIMENTS?             │
   └──────────┬─────────────────────────────────────┘
             YES → Appendix E
              NO  → ┐
                    │
   ┌────────────────┴─────────────────────────────────┐
   │ Does it provide IMPLEMENTATION DETAILS?          │
   └──────────┬─────────────────────────────────────┘
             YES → Appendix F
              NO  → Appendix G (Discussion)
```

---

## Section-by-Section Content Map

### Appendix A: Mathematical Foundations (THEORY)

**Purpose**: Prove theoretical claims and derive bounds

| Content | Original Location | New Location | Status |
|---------|------------------|--------------|--------|
| Spectral separation proof | `03_appendix/spectral_separation_proof.tex` | `A/A1_spectral_separation_proof.tex` | ✅ Copied |
| Error bounding derivation | ↑ Same file | ↑ Same file | ✅ Included |
| Regret bounds | ↑ Same file | ↑ Same file | ✅ Included |
| Meta-algorithm analysis | ↑ Same file | ↑ Same file | ✅ Included |

**What belongs here**:
- ✅ Mathematical proofs
- ✅ Theoretical bounds and derivations
- ✅ Formal lemmas and theorems
- ❌ Empirical validation (goes to C/D)
- ❌ Implementation details (goes to F)

---

### Appendix B: Dataset Details (DATA)

**Purpose**: Document data sources, validation, and scale

| Content | Original Location | New Location | Status |
|---------|------------------|--------------|--------|
| Dataset composition table | `01_table/table_dataset_composition.tex` | `B/B1_*` | 📝 To extend |
| Validation methodology | `01_figure/validation_methodology.tex` | `B/B2_validation_methodology.tex` | ✅ Copied |
| 1M scale analysis | `appendix_d/figure_1M_analysis.tex` | `B/B3_1M_scale_analysis.tex` | ✅ Copied |
| Spectral invariance results | ↑ Same file | ↑ Same file | ✅ Included |

**What belongs here**:
- ✅ Data provenance and sources
- ✅ Statistical validation procedures
- ✅ Scale validation experiments
- ✅ Dataset composition details
- ❌ Experimental results using data (goes to E)
- ❌ Algorithm validation (goes to C/D)

---

### Appendix C: Hyperparameter Sensitivity (ROBUSTNESS)

**Purpose**: Demonstrate system robustness to parameter choices

| Content | Original Location | New Location | Status |
|---------|------------------|--------------|--------|
| Comprehensive sensitivity | `appendix_d/hyperparameter_sensitivity.tex` | `C/C1_comprehensive_sensitivity.tex` | ✅ Copied |
| Concise summary | `appendix_e/hyperparameter_robustness.tex` | `C/C5_robustness_summary.tex` | ✅ Copied |
| Sensitivity figures | `08_figure/results/*sensitivity*.png` | `C/figures/` | ✅ Copied |
| Learning rate sensitivity | Various experiment folders | `C/C2_*` | 📝 To consolidate |
| Mixing parameter sensitivity | Various experiment folders | `C/C3_*` | 📝 To consolidate |

**What belongs here**:
- ✅ Hyperparameter sweeps ($n_{\text{eff}}$, $\eta$, $\gamma$)
- ✅ Robustness demonstrations (20× range)
- ✅ Sensitivity curves and plots
- ✅ Practical guidelines for parameter selection
- ❌ Component ablations (goes to D)
- ❌ Feature engineering analysis (goes to D)

---

### Appendix D: Ablation Studies (COMPONENTS)

**Purpose**: Validate contribution of each system component

| Content | Original Location | New Location | Status |
|---------|------------------|--------------|--------|
| Corralling ablation | `04_figure/appendix_ablation_study.tex` | `D/D1_corralling_ablation.tex` | ✅ Copied |
| Ablation results table | ↑ Same file | ↑ Same file | ✅ Included |
| Ablation study figure | `04_figure/results_ablation/ablation_study.png` | `D/figures/` | ✅ Copied |
| Learning rate ablation fig | `06_figure/results/figure6_learning_rate_ablation.pdf` | `D/figures/` | ✅ Copied |
| Alpha ablation | `03_figure/results/ablation/figure_alpha_ablation.png` | `D/figures/` | ✅ Copied |
| Gamma ablation | `03_figure/results/gamma_ablation/figure_gamma_ablation.png` | `D/figures/` | ✅ Copied |

**What belongs here**:
- ✅ Component removal experiments
- ✅ Feature engineering ablations
- ✅ Exploration strategy variations
- ✅ Multi-seed statistical validation
- ❌ Hyperparameter sweeps (goes to C)
- ❌ New experiments (goes to E)

---

### Appendix E: Extended Results (EXPERIMENTS)

**Purpose**: Show additional experiments and supplementary results

| Content | Original Location | New Location | Status |
|---------|------------------|--------------|--------|
| Catastrophic failure | `06_figure/figure5_corralling_kdd.tex` | `E/E1_catastrophic_failure.tex` | ✅ Copied |
| Failure extended | `06_figure/figure6_corralling_kdd.tex` | `E/E1_catastrophic_failure_extended.tex` | ✅ Copied |
| 3-model results | `04_figure/results_3models/` | `E/E2_*` | 📝 To document |
| Cost profile analysis | `05_figure/` extended | `E/E3_*` | 📝 To document |
| Distribution shift | `02_figure/` and `02_table/` | `E/E4_*` | 📝 To document |

**What belongs here**:
- ✅ Catastrophic failure detection experiments
- ✅ Multi-model routing results (3+ models)
- ✅ Alternative cost profiles
- ✅ Distribution shift robustness
- ✅ Real-world scenario testing
- ❌ Core results (goes to main paper)
- ❌ Ablations (goes to D)

---

### Appendix F: Implementation Details (PRACTICE)

**Purpose**: Enable reproduction and practical deployment

| Content | Original Location | New Location | Status |
|---------|------------------|--------------|--------|
| Configuration details | `03_figure/latex_appendix_config.tex` | `F/F1_configuration_details.tex` | ✅ Copied |
| Experimental setup | `08_figure/experiments_setup_compact.tex` | `F/F2_experimental_setup.tex` | ✅ Copied |
| Strategy guide | `03_figure/latex_table_strategy_guide.tex` | `F/F3_strategy_selection_guide.tex` | ✅ Copied |
| Config table | `08_figure/experiments_table.tex` | `F/F1_*` | 📝 To merge |

**What belongs here**:
- ✅ Configuration parameters and defaults
- ✅ Hardware/software requirements
- ✅ Hyperparameter selection guidelines
- ✅ Strategy selection decision trees
- ✅ Deployment checklists
- ✅ Troubleshooting guides
- ❌ Theoretical analysis (goes to A)
- ❌ Experimental results (goes to E)

---

### Appendix G: Additional Discussion (CONTEXT)

**Purpose**: Provide broader context, limitations, and future directions

| Content | Original Location | New Location | Status |
|---------|------------------|--------------|--------|
| Practical recommendations | `03_figure/latex_section_5.3_practical_recommendations.tex` | `G/G1_practical_recommendations.tex` | ✅ Copied |
| Limitations | `03_figure/latex_section_6_limitations.tex` | `G/G2_limitations.tex` | ✅ Copied |
| Limitations addendum | `08_figure/limitations_addendum.tex` | `G/G2_limitations_addendum.tex` | ✅ Copied |
| Broader impact | To be created | `G/G3_broader_impact.tex` | 📝 To create |
| Corralling vs offline | To be created | `G/G4_corralling_vs_offline.tex` | 📝 To create |

**What belongs here**:
- ✅ Deployment recommendations and best practices
- ✅ System limitations and constraints
- ✅ Ethical considerations and broader impact
- ✅ Future research directions
- ✅ Alternative approaches discussion
- ❌ Implementation details (goes to F)
- ❌ Experimental validation (goes to C/D/E)

---

## Content Consolidation Tasks

### Duplicates to Resolve

1. **Spectral Separation Proof**:
   - `03_appendix/spectral_separation_proof.tex` ✅ Used
   - `appendix_c/spectral_separation_proof.tex` ⚠️ Duplicate (can archive)
   - **Action**: Keep `03_appendix/` version, archive `appendix_c/`

2. **1M Scale Analysis**:
   - `01_figure/figure_1M_analysis.tex` ⚠️ Check if different
   - `appendix_d/figure_1M_analysis.tex` ✅ Used
   - **Action**: Compare versions, consolidate if identical

3. **Hyperparameter Sensitivity**:
   - `appendix_d/hyperparameter_sensitivity.tex` ✅ Used (comprehensive)
   - `appendix_e/hyperparameter_robustness.tex` ✅ Used (concise)
   - `08_figure/figure8_sensitivity_compact.tex` ⚠️ Duplicate
   - `08_figure/figure8_sensitivity_update.tex` ⚠️ Duplicate
   - **Action**: Use `appendix_d/` as C1, `appendix_e/` as C5, archive others

4. **Limitations**:
   - `03_figure/latex_section_6_limitations.tex` ✅ Used
   - `08_figure/limitations_addendum.tex` ✅ Used
   - **Action**: Keep both, consolidate into single section if needed

---

## Verification Checklist

### Structure ✅
- [x] Appendix A-G folders created
- [x] Figures subfolders in each section
- [x] README.md in each section
- [x] APPENDIX_MASTER.tex created
- [x] Main appendix README.md created

### Content Migration ✅
- [x] Mathematical proofs → Appendix A
- [x] Dataset validation → Appendix B
- [x] Hyperparameter sensitivity → Appendix C
- [x] Ablation studies → Appendix D
- [x] Extended results → Appendix E (partial)
- [x] Implementation details → Appendix F
- [x] Discussion → Appendix G

### Figures ✅
- [x] Ablation figures → D/figures/
- [x] Sensitivity figures → C/figures/
- [ ] Extended results figures → E/figures/ (ongoing)

### LaTeX Compilation 📝
- [ ] Test compile APPENDIX_MASTER.tex
- [ ] Verify all \input{} paths
- [ ] Check figure references
- [ ] Verify cross-references
- [ ] Check table formatting

---

## To-Do Items

### High Priority
1. **Create B1**: Extended dataset composition table (extend from 01_table)
2. **Document E2**: Three-model routing results from `04_figure/results_3models/`
3. **Test compile**: Run `pdflatex APPENDIX_MASTER.tex` to verify structure
4. **Copy missing figures**: Any remaining figures from experiment folders

### Medium Priority
5. **Create G3**: Broader impact section
6. **Create G4**: Corralling vs. offline optimization discussion
7. **Consolidate C2-C4**: Learning rate and mixing parameter sensitivity from various sources
8. **Create E3**: Cost profile analysis from `05_figure/` extended data
9. **Create E4**: Distribution shift analysis from `02_figure/` and `02_table/`

### Low Priority
10. Archive old appendix folders (`03_appendix/`, `appendix_c/`, `appendix_d/`, `appendix_e/`)
11. Create appendix-only compilation script
12. Generate appendix figure index
13. Create citation reference guide

---

## Statistics

### Current State
- **Sections created**: 7 (A-G) ✅
- **LaTeX files copied**: 12 ✅
- **README files created**: 8 ✅
- **Figures copied**: ~10 ✅
- **Master LaTeX file**: 1 ✅

### Estimated Completion
- **Core structure**: 100% ✅
- **Content migration**: 75% ⚠️
- **Documentation**: 95% ✅
- **Figure organization**: 60% ⚠️
- **Verification**: 30% ⚠️

---

## Quick Commands for Verification

```bash
# Navigate to appendix
cd /Users/annette/repostitories/banditGPT/experiments_v1/appendix

# Count LaTeX files
find . -name "*.tex" | wc -l

# Count README files
find . -name "README.md" | wc -l

# List all figures
find . -type f \( -name "*.png" -o -name "*.pdf" \) -path "*/figures/*"

# Check for broken \input references (requires compilation attempt)
pdflatex -interaction=nonstopmode APPENDIX_MASTER.tex 2>&1 | grep -i "file not found"

# Verify all sections have README
for dir in A_* B_* C_* D_* E_* F_* G_*; do 
  [ -f "$dir/README.md" ] && echo "✅ $dir" || echo "❌ $dir"
done
```

---

**Status**: Structure complete, core content migrated, verification in progress  
**Next Step**: Test LaTeX compilation and complete remaining documentation  
**Last Updated**: February 13, 2026
