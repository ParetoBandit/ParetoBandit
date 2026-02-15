# Appendix D: Ablation Studies

## Overview
Systematic ablation studies validating component contributions for Figure 3 (corralling mechanism) and Figure 4 (Pareto configuration). Now also includes the learning rate ablation under catastrophic failure, supporting Figure 6.

## Contents

### D.1: Corralling Algorithm Ablation
**File**: `D1_corralling_ablation.tex`  
**Source**: `04_figure/appendix_ablation_study.tex`

**Content**:
- Comprehensive grid search over 15 configurations
- 5 learning rates ($\eta \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$)
- 3 mixing parameters ($\gamma \in \{0.0, 0.05, 0.10\}$)
- 3 random seeds per configuration (45 total experiments)

**Key Results**:
- **Optimal Configuration**: $\eta=5.0$, $\gamma=0.10$ (Regret: 59.33 +/- 3.40)
- **Theoretical Validation**: Growth exponent $\beta=0.669$ confirms sublinear regret
- **Exploration Floor**: $\gamma > 0$ reduces variance 14x (std=54.46 -> 3.40)

**Supports**: Figure 3 configuration choices (why $\gamma=0.05$) and Figure 4's use of $\eta=1.0$

---

### D.2: Learning Rate Ablation Under Catastrophic Failure
**Figure**: `figures/figure6_learning_rate_ablation.pdf`  
**Source**: `E_catastrophic_failure_experiment/supplementary/ablation_learning_rate_catastrophic.py`

**Content**:
- Impact of $\eta$ on catastrophic failure detection speed
- Justifies $\eta=0.3$ for the safety regime (Figure 6)
- Connects to the three-regime learning rate framework:

| Regime | $\eta$ | Experiment | Use Case |
|--------|--------|------------|----------|
| Safety | 0.3 | Figure 6 | Fast catastrophic detection |
| Moderate | 1.0 | Figure 4 | Pareto sweep |
| Convergence | 5.0 | D.1 ablation | Full prior unlearning |

**Supports**: Figure 6's choice of $\eta=0.3$ for safety-critical detection

---

### D.3: Exploration Strategy Ablation
**Figures**:
- `figures/figure_alpha_ablation.png`: Alpha ($\alpha$) parameter sweep
- `figures/figure_gamma_ablation.png`: Gamma ($\gamma$) parameter sweep

**Supports**: Figure 3's configuration ($\alpha=2.0$ constant, $\gamma=0.05$ floor)

---

## Removed Content

| Item | Reason |
|------|--------|
| ~~D.2: Feature Engineering Ablation~~ | Planned but never created; structural features are part of the PCA pipeline, not separately ablated |
| ~~D.4: Multi-Seed Statistical Validation~~ | Covered within D.1 (3 seeds per config) and Figure 4's STATISTICAL_NOTES.md |
| ~~Cold-Start Initialization via Semantic Transfer~~ | Moved to Figure 6 context where it's directly relevant |

---

## Key Findings

### 1. Learning Rate Impact
- Higher $\eta \geq 1.0$ enables faster adaptation
- Optimal $\eta=5.0$ for convergence regime; $\eta=0.3$ for safety regime
- The right $\eta$ depends on the deployment scenario (see regime table above)

### 2. Exploration Floor Necessity
- $\gamma=0$: 14x higher variance (std=54.46)
- $\gamma=0.10$: Optimal stability (std=3.40)
- Prevents expert starvation

### 3. Sublinear Regret Confirmation
- Log-log regression: $\beta=0.669 \pm 0.002$
- R^2 = 0.9903 (excellent fit)
- Confirms $O(T^{0.669})$ regret growth

---

## Related Sections
- **Main Paper Figure 3**: Corralling insurance — uses configs validated here
- **Main Paper Figure 4**: Pareto frontier — uses $\eta=1.0$ validated here
- **Figure 6**: Catastrophic failure — uses $\eta=0.3$ validated here
- **Appendix C**: Hyperparameter sensitivity complements ablation studies

---

## Files
```
D_ablation_studies/
├── README.md                          (this file)
├── D1_corralling_ablation.tex         (comprehensive ablation study)
└── figures/
    ├── figure6_learning_rate_ablation.pdf  (η ablation under catastrophic failure)
    ├── figure_alpha_ablation.png           (α parameter sweep)
    └── figure_gamma_ablation.png           (γ parameter sweep)
```

---

**Last Updated**: February 15, 2026
