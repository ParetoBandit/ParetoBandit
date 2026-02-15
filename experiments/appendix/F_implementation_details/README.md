# Appendix F: Implementation Details

## Overview
Configuration parameters and experimental setup for reproducibility. Covers all settings needed to replicate Figures 1, 3, 4, and 6.

## Contents

### F.1: Configuration Details
**File**: `F1_configuration_details.tex`  
**Source**: Derived from `src/bandit_gpt/router.py` (authoritative reference)

**Content**:
- Part 1: Library router parameters (all configurable knobs in `router.py`)
- Part 2: Experiment-specific configurations (values used in Figures 3, 4, 6)
- Implementation notes (init_lambda/update_lambda pattern, two-level cost mechanism, loss_decay)

**Key Classes Documented**:
- `DisjointLinUCBPolicy` (dim, alpha, init_lambda, update_lambda, forgetting_factor)
- `CorrallingRouter` (learning_rate, gamma, loss_decay, meta_lr_halflife, cost_weight)
- `CostAwareLinUCBRouter` (alpha_start/end, cost_penalty, warmup_priors)
- `CostAwareTabulaRasaRouter` (alpha_start/end, cost_penalty, ridge_lambda)
- `RegistrationConfig` (n_effective tiers, slow_bias, fast_bias)

---

### F.2: Experimental Setup
**File**: `F2_experimental_setup.tex`  
**Source**: Rewritten to document hardware, software, and evaluation protocol

**Content**:
- Hardware specifications
- Software dependencies
- Dataset preparation procedures
- Evaluation protocols
- Reproducibility guidelines

**Computational Requirements**:
- CPU: Standard multi-core processor (no GPU required)
- RAM: 8-16 GB for typical datasets
- Storage: ~5 GB for full LMSYS dataset
- Runtime: 2-3 minutes per experiment (typical)

---

## Removed Content

| Item | Reason |
|------|--------|
| ~~F3: Strategy Selection Guide~~ | Duplicates Figure 3 findings in a different format; better suited for GitHub README than scientific appendix |
| ~~F4: Hyperparameter Selection Guide~~ | Never created; Appendix C covers sensitivity comprehensively |

---

## Related Sections
- **Appendix C**: Hyperparameter sensitivity validates parameter choices listed here
- **Appendix D**: Ablation studies justify configuration decisions

---

## Files
```
F_implementation_details/
├── README.md                          (this file)
├── F1_configuration_details.tex       (config parameters)
└── F2_experimental_setup.tex          (setup procedures)
```

---

**Last Updated**: February 15, 2026
