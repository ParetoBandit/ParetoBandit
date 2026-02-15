# Appendix C: Hyperparameter Sensitivity Analysis

## Overview
Comprehensive sensitivity analysis demonstrating robustness of the system to hyperparameter choices, particularly the effective prior sample size ($n_{\text{eff}}$) and other key parameters.

## Contents

### C.1: Comprehensive Sensitivity Analysis
**File**: `C1_comprehensive_sensitivity.tex`  
**Source**: `appendix_d/hyperparameter_sensitivity.tex`

**Content**:
- Experimental setup and methodology
- Bayesian formulation of prior strength
- Results across 20× range ($n_{\text{eff}} \in \{1.0, 2.0, 5.0, 10.0, 20.0\}$)
- Comparison to cold start baseline
- Interpretation and practical guidelines
- Robustness to imperfect neighbors
- Comparison to alternative approaches
- Deprecated parameters discussion

**Key Results**:
- Perfect robustness: All $n_{\text{eff}}$ values yield identical performance
- Mean reward: 4.48 across all values (+39.2% vs Cold Start)
- Weak prior ($n_{\text{eff}}=1.0$) sufficient for major improvement
- Performance driven by variance reduction, not reward inflation

---

### C.2: Learning Rate Sensitivity (η)
**Content**:
- Impact of meta-algorithm learning rate
- Optimal $\eta=1.0$ validation
- Trade-offs between adaptation speed and stability

---

### C.3: Mixing Parameter Sensitivity (γ)
**Content**:
- Exploration floor parameter analysis
- Impact on expert starvation prevention
- Optimal $\gamma \in [0.05, 0.10]$ range

---

### C.4: Robustness to Imperfect Neighbors
**Content**:
- Performance when semantic neighbor is mismatched
- Adaptation speed under different prior strengths
- Conservative range recommendations

---

### C.5: Robustness Summary
**File**: `C5_robustness_summary.tex`  
**Source**: `appendix_e/hyperparameter_robustness.tex`

**Content**:
- Concise 1-page summary
- Sensitivity figure (overlapping trajectories)
- Quantitative results table
- Key interpretation

---

## Figures

### Main Sensitivity Figure
**Location**: `figures/neff_sensitivity.png`  
**Description**: Performance trajectories for all $n_{\text{eff}}$ values overlapping, all outperforming Cold Start baseline

**Key Visual Insight**: All blue lines (transfer methods) overlap and stay above red line (Cold Start), demonstrating perfect robustness across 20× hyperparameter range.

### Additional Ablation Figures
- Learning rate sensitivity curves
- Mixing parameter impact plots
- Multi-seed variance visualization

---

## Related Sections
- **Main Paper Figure 8**: Compact sensitivity results in main text
- **Appendix A**: Theoretical justification for Bayesian formulation
- **Appendix D**: Ablation studies complement sensitivity analysis

---

## Key Takeaways for Practitioners

1. **No Magic Numbers**: Any $n_{\text{eff}} \in [1, 20]$ works well
2. **Default Recommendation**: $n_{\text{eff}}=5.0$ as balanced default
3. **Conservative Range**: $n_{\text{eff}} \in [2, 10]$ for uncertain semantic similarity
4. **Weak Priors Work**: Even $n_{\text{eff}}=1.0$ provides +39.2% improvement
5. **Robust to Misspecification**: Performance stable across wide hyperparameter ranges

---

## Files
```
C_hyperparameter_sensitivity/
├── README.md                          (this file)
├── C1_comprehensive_sensitivity.tex   (detailed analysis)
├── C5_robustness_summary.tex         (concise summary)
└── figures/
    ├── neff_sensitivity.png        (main sensitivity plot)
    ├── (additional sensitivity figures)
    └── (learning rate and mixing parameter plots)
```
