# Appendix D: Implementation Details

## Overview
Configuration parameters and experimental setup for reproducibility. Covers all settings needed to replicate Figures 1, 3, 4, 5, and 6.

## Contents

### D.1: Configuration Details
**File**: `D1_configuration_details.tex`  
**Source**: Derived from `src/bandit_gpt/router.py` (authoritative reference)

**Content**:
- Part 1: Library router parameters (all configurable knobs in `router.py`)
- Part 2: Experiment-specific configurations (values used in Figures 3, 4, 5, 6)
- Implementation notes (init_lambda/update_lambda pattern, two-level cost mechanism, loss_decay)

**Key Classes Documented**:
- `DisjointLinUCBPolicy` (dim, alpha, init_lambda, update_lambda, forgetting_factor)
- `CorrallingRouter` (learning_rate, gamma, loss_decay, meta_lr_halflife, cost_weight)
- `CostAwareLinUCBRouter` (alpha_start/end, cost_penalty, warmup_priors)
- `CostAwareTabulaRasaRouter` (alpha_start/end, cost_penalty, ridge_lambda)
- `RegistrationConfig` (n_effective tiers, slow_bias, fast_bias)

---

### D.2: Experimental Setup
**File**: `D2_experimental_setup.tex`  
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

## Related Sections
- **Appendix A.3**: Prior transfer theory validates n_eff parameter choices
- **Appendix A.2**: Ablation table justifies Corralling configuration decisions (45 experiments)

---

## Files
```
D_implementation_details/
├── README.md                          (this file)
├── D1_configuration_details.tex       (config parameters)
└── D2_experimental_setup.tex          (setup procedures)
```

---

**Last Updated**: February 22, 2026
