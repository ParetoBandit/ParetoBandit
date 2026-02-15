# Appendix D: Ablation Studies

## Overview
Systematic ablation studies validating the contribution of each component and hyperparameter choice through controlled experiments.

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
- **Optimal Configuration**: $\eta=5.0$, $\gamma=0.10$ (Regret: 59.33 ± 3.40)
- **Statistical Reproducibility**: Low variance confirms robust results
- **Theoretical Validation**: Growth exponent $\beta=0.669$ confirms sublinear regret
- **Exploration Floor**: $\gamma > 0$ reduces variance 14× (std=54.46 → 3.40)

**Table**: Complete ablation results ranked by cumulative regret

---

### D.2: Feature Engineering Ablation
**Content**:
- Impact of structural features (code blocks, LaTeX, etc.)
- Complexity projection contribution
- Comparison to vanilla LinUCB

**Key Results**:
- Full system vs. no structural features
- Full system vs. no complexity projection
- Both components necessary for optimal performance

---

### D.3: Exploration Strategy Ablation
**Content**:
- Alpha (α) parameter sensitivity
- Gamma (γ) parameter sensitivity
- UCB exploration bonus impact

**Figures**:
- `figures/figure_alpha_ablation.png`: Alpha parameter sweep
- `figures/figure_gamma_ablation.png`: Gamma parameter sweep
- `figures/figure6_learning_rate_ablation.pdf`: Learning rate ablation

---

### D.4: Multi-Seed Statistical Validation
**Content**:
- Variance analysis across random seeds
- Confidence interval calculations
- Statistical significance testing
- Reproducibility confirmation

---

## Figures

### Main Ablation Study Figure
**Location**: `figures/ablation_study.png`  
**Description**: Comprehensive ablation study results showing performance across all configurations

### Learning Rate Ablation
**Location**: `figures/figure6_learning_rate_ablation.pdf`  
**Description**: Impact of learning rate on catastrophic failure detection

### Feature Ablation
**Locations**:
- `figures/figure_alpha_ablation.png`: Alpha parameter impact
- `figures/figure_gamma_ablation.png`: Gamma parameter impact

---

## Key Findings

### 1. Learning Rate Impact
- Higher $\eta \geq 1.0$ enables faster adaptation
- Optimal $\eta=5.0$ achieves aggressive updates while maintaining stability
- Low $\eta < 0.5$ shows excessive variance

### 2. Exploration Floor Necessity
- $\gamma=0$: 14× higher variance (std=54.46)
- $\gamma=0.10$: Optimal stability (std=3.40)
- Prevents expert starvation
- Stabilizes learning dynamics

### 3. Robustness Range
- Performance competitive across $\eta \in [0.5, 5.0]$ when $\gamma \geq 0.05$
- Reasonable tolerance to hyperparameter misspecification
- Not dependent on "lucky" initialization

### 4. Theoretical Confirmation
- Log-log regression: $\beta=0.669 \pm 0.002$
- R² = 0.9903 (excellent fit)
- Confirms sublinear regret growth: $O(T^{0.669})$
- Satisfies PAC-learnability criterion ($\beta < 1$)

---

## Cold-Start Initialization via Semantic Transfer

GPT-4o initialization strategy:
- No direct warmup priors available (only Mixtral + GPT-4-Turbo)
- Used semantic transfer from GPT-4-Turbo
- Conservative scaling: $\gamma=0.05$ to reflect uncertainty
- Result: Discovered as dominant choice (70.8% usage)
- Correctly overrode warmup expert's GPT-4-Turbo bias (79.9%)

---

## Related Sections
- **Main Paper Figure 4**: Core algorithm visualization
- **Appendix C**: Hyperparameter sensitivity complements ablation studies
- **Appendix E**: Extended results show real-world impact

---

## Files
```
D_ablation_studies/
├── README.md                          (this file)
├── D1_corralling_ablation.tex        (comprehensive ablation study)
└── figures/
    ├── ablation_study.png             (main ablation results)
    ├── figure6_learning_rate_ablation.pdf
    ├── figure_alpha_ablation.png
    └── figure_gamma_ablation.png
```
