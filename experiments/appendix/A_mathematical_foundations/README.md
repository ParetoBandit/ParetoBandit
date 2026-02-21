# Appendix A: Mathematical Foundations

## Overview
Regret analysis, safety guarantees, and formal derivations for the two-level
LinUCB + Corralling architecture used in banditGPT.  All theoretical results
reference the *actual* algorithm implemented in `src/bandit_gpt/router.py`.

## Contents

### A.1: Composite Regret Bound for LinUCB under Corralling
**File**: `A1_regret_decomposition.tex`

**Content**:
- Per-expert regret bound for Disjoint LinUCB (Theorem 1, citing Li et al. 2010; Abbasi-Yadkani et al. 2011)
- Meta-regret bound for Exp4/Corralling (Theorem 2, citing Agarwal et al. 2017)
- Composite decomposition: R_total ≤ O(√(T ln 2)) + min{R_warmup, R_tabula_rasa}
- Discussion of gap between theoretical O(√T) and empirical O(T^0.669)

**Cross-references**: Appendix A.2 (empirical regret validation via ablation table), Appendix C.1 (meta-regret gap)

### A.2: Safety Guarantee via γ-Mixing
**File**: `A2_safety_guarantee.tex`

**Content**:
- Formal proof of expert probability floor: p_i ≥ γ/N for all i, t
- Bounded importance-weighted estimator (max loss ≤ N/γ)
- Recovery time bound after distribution shift
- Exploration–exploitation trade-off at meta level
- Includes ablation table (45 experiments): 15 configurations × 3 seeds validating η/γ choices

**Cross-references**: Appendix C.1 (95% failure detection via K=5 experiment)

### A.3: Warmup Prior Transfer — Acceleration, Risk, and Limitations
**File**: `A3_warmup_transfer.tex`

**Content**:
- Prior initialization via semantic neighbor transfer (n_eff formulation)
- Acceleration bound under accurate priors
- Risk analysis under misspecified priors
- How Corralling bounds negative transfer to O(√T) overhead
- Correct vs. naive prior injection comparison (why n_eff controls variance, not direction)
- Practical n_eff recommendation ([2, 10] for uncertain neighbors)
- Honest discussion of five limitations:
  1. Isotropic regularization in PCA space
  2. n_eff conflates prior strength with accuracy
  3. Stationary expert-level updates
  4. Linear realizability assumption
  5. Heuristic semantic neighbor selection

**Cross-references**: A.2 (ablation table validates warmup vs tabula rasa)

---

## Related Sections
- **Appendix C**: Extended results (validates safety guarantee via catastrophic failure detection)
- **Appendix D**: Implementation details (production configuration)
- **Appendix E**: Limitations and future work (extends A.3 limitations)

---

## Files
```
A_mathematical_foundations/
├── README.md                           (this file)
├── A0_algorithm_pseudocode.tex         (complete routing system pseudocode)
├── A1_regret_decomposition.tex         (composite regret bound)
├── A2_safety_guarantee.tex             (γ-mixing safety proof + ablation table)
└── A3_warmup_transfer.tex              (prior transfer analysis + limitations)
```
