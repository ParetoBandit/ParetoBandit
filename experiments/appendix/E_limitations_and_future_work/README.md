# Appendix E: Limitations and Future Work

## Overview
System limitations, assumptions, future research directions, and detailed positioning against concurrent bandit-based LLM routers (PILOT, BaRP, LLM Bandit).

## Contents

### E.1: Limitations and Applicability
**File**: `E1_limitations.tex`

**Content**:
- Prior quality dependency (alpha=2.0 finding specific to Figure 3 mismatch scenario)
- Strategy selection trade-offs (Corralling overhead vs. safety guarantee)
- Mechanism validation through ablation (homogeneous vs. heterogeneous exploration)
- Outcome variance and reproducibility (20 seeds main, 3 seeds ablation)
- Regime-dependent hyperparameter effects (warmup-dominant vs. tabula rasa-dominant)
- Computational overhead (latency, memory, logging)
- Generalizability (single dataset, two models)

### E.2: Positioning Among Bandit-Type LLM Routers
**File**: `E2_positioning.tex`

**Content**:
- Taxonomy table: Simple MAB | Plain LinUCB | PILOT/BaRP-style | banditGPT (8 aspects)
- 5 key differentiators: prior machinery, production engineering, Corralling meta-learner, constraint handling, observability
- Honest acknowledgment of PILOT/BaRP advantages (learned embeddings, preference-conditioned inference, knapsack cost)
- Referenced from main paper Section 2 (Related Work)

---

## Consolidation History

| Item | Disposition |
|------|-------------|
| ~~E1 Addendum (regime-dependent effects)~~ | Best content merged into E1 "Regime-Dependent Behavior" paragraph; addendum removed (broken cross-refs, fictional model, draft quality) |
| ~~E2: Practical Deployment Recommendations~~ | Superseded by E2 positioning section |
| ~~E3: Broader Impact~~ | Never created; only required if venue mandates it (e.g., NeurIPS) |

---

## Related Sections
- **Appendix A.2**: Ablation table addresses the "specific configuration" limitation (15 configs x 3 seeds)
- **Appendix A.3**: Prior transfer analysis addresses the "brittle parameters" limitation (n_eff robustness)
- **Figure 6 / Appendix C**: Catastrophic failure addresses the "stationarity assumption" limitation
- **Main paper Section 2**: Related work subsection on cost-aware bandit routers (brief; E.2 provides full detail)
- **Main paper Section 4.2**: RouteLLM baseline justification (open-source + peer-reviewed rationale)
- **Main paper conclusion**: Future work (algorithmic extensions, orchestration) covered there

---

## Files
```
E_limitations_and_future_work/
├── README.md                          (this file)
├── E1_limitations.tex                 (limitations and applicability)
└── E2_positioning.tex                 (bandit router taxonomy and differentiators)
```

---

**Last Updated**: February 15, 2026
