# Figure 6: Catastrophic Failure Detection (K=5 Portfolio)

**Experiment Goal**: Evaluate Corralling's secondary safety benefit -- automatic failover under catastrophic model failure -- on a 5-model portfolio using production router components with semantic transfer.

**Key Result**: Corralling detects failure in 95% of seeds (median 34 steps), and the gap vs. EMA narrows monotonically with portfolio size: Delta = -0.286 (K=2), -0.154 (K=3), **-0.090 (K=5)**.

---

## Overview

This experiment evaluates **Corralling's response to catastrophic model failure** on a **5-model portfolio** using the production router: `CostAwareLinUCBRouter` (warmup expert), `CostAwareTabulaRasaRouter` (cold-start expert), and `CorrallingRouter` with importance-weighted meta-learning.

**Why K=5?** With K=2, failure detection is a trivial binary tracking problem where EMA dominates. With K=5, post-failure routing across 4 remaining models is a *contextual* decision -- each model excels in a different context region. EMA's epsilon/K = 2% per-model exploration budget is insufficient to evaluate all alternatives, causing it to over-concentrate on a single model.

**Warmup priors**: All 5 models have informed priors:
- Mixtral, GPT-4-Turbo: direct RouteLLM battle priors (scaled to N_eff=10)
- GPT-3.5, Haiku, GPT-4o: semantic transfer from nearest known neighbor (First-Child Bias Correction)

---

## Files

```
E_catastrophic_failure_experiment/
├── generate_figure6_5model.py     # Main experiment (K=5, production router)
└── README.md                      # This file
```

**LaTeX write-up**: `../C_extended_results/C1_catastrophic_failure.tex` (included in APPENDIX_MASTER)

---

## Key Results (K=5, N=20 Seeds)

### Phase-by-Phase Performance

| Method | Healthy | Failure | Recovery |
|--------|---------|---------|----------|
| Oracle | 0.824 | 0.800 | 0.822 |
| **banditGPT** | **0.763** | **0.646** | **0.762** |
| EMA Tracker | 0.736 | 0.735 | 0.783 |
| Static GPT-4-Turbo | 0.812 | 0.153 | 0.820 |

### Portfolio Size Scaling

| K | banditGPT | EMA | Delta | Detection |
|---|-----------|-----|-------|-----------|
| 2 | 0.478 | 0.764 | -0.286 | 65% |
| 3 | 0.596 | 0.750 | -0.154 | 85% |
| **5** | **0.646** | **0.735** | **-0.090** | **95%** |

Gap narrows 69% from K=2 to K=5.

---

## Reproduction

```bash
cd experiments/appendix/E_catastrophic_failure_experiment
python generate_figure6_5model.py
# Output: results/figure6_5model.png
# Runtime: ~7 seconds
```

---

**Last Updated**: February 15, 2026
