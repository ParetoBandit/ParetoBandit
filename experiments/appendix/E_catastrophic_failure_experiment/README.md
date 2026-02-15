# Figure 6: Catastrophic Failure Detection (K=5 Portfolio)

**Experiment Goal**: Evaluate Corralling's secondary safety benefit — automatic failover under catastrophic model failure — on a 5-model portfolio using production router components with semantic transfer.

**Key Result**: Corralling detects failure in 95% of seeds (median 34 steps), and the gap vs. EMA narrows monotonically with portfolio size: Δ = -0.286 (K=2), -0.154 (K=3), **-0.090 (K=5)**.

---

## Overview

This experiment evaluates **Corralling's response to catastrophic model failure** on a **5-model portfolio** using the production router: `CostAwareLinUCBRouter` (warmup expert), `CostAwareTabulaRasaRouter` (cold-start expert), and `CorrallingRouter` with importance-weighted meta-learning.

**Why K=5?** With K=2, failure detection is a trivial binary tracking problem where EMA dominates. With K=5, post-failure routing across 4 remaining models is a *contextual* decision — each model excels in a different context region. EMA's ε/K = 2% per-model exploration budget is insufficient to evaluate all alternatives, causing it to over-concentrate on a single model.

**Warmup priors**: All 5 models have informed priors:
- Mixtral, GPT-4-Turbo: direct RouteLLM battle priors (scaled to N_eff=10)
- GPT-3.5, Haiku, GPT-4o: semantic transfer from nearest known neighbor (First-Child Bias Correction)

---

## Core Files

```
experiments/appendix/E_catastrophic_failure_experiment/
├── generate_figure6_5model.py               # Main experiment (K=5, production router)
├── figure6_corralling_kdd.tex               # Standalone LaTeX figure + methodology
├── README.md                                # This file
├── results/
│   ├── figure6_5model.png                   # Main figure (used in paper)
│   ├── figure6_5model.pdf
│   ├── appendixD_learning_rate_ablation.png # Learning rate ablation
│   └── appendixD_learning_rate_ablation.pdf
├── supplementary/
│   └── ablation_learning_rate_catastrophic.py
└── archive/                                 # Superseded experiments
    ├── generate_figure6_main.py             # Original (mock experts, K=2)
    ├── generate_figure6_corrected.py        # Corrected K=2
    ├── generate_figure6_3model.py           # K=3
    └── ...
```

---

## Key Results (K=5, N=20 Seeds)

### Phase-by-Phase Performance

| Method | Healthy | Failure | Recovery |
|--------|---------|---------|----------|
| Oracle | 0.824 | 0.800 | 0.822 |
| **banditGPT** | **0.763** | **0.646** | **0.762** |
| EMA Tracker | 0.736 | 0.735 | 0.783 |
| Static GPT-4-Turbo | 0.812 | 0.153 | 0.820 |

### Detection Metrics

| Metric | Result |
|--------|--------|
| Detection rate | 95% (19/20 seeds) |
| Reaction time | Median 34 steps (mean 49 ± 39) |
| Recovery rate | 40% |
| EMA gap (Δ) | -0.090 |

### Portfolio Size Scaling

| K | banditGPT | EMA | Δ | Detection |
|---|-----------|-----|---|-----------|
| 2 | 0.478 | 0.764 | -0.286 | 65% |
| 3 | 0.596 | 0.750 | -0.154 | 85% |
| **5** | **0.646** | **0.735** | **-0.090** | **95%** |

Gap narrows 69% from K=2 to K=5. EMA degrades (exploration budget thins); Corralling improves (contextual routing value grows).

---

## Reproduction

```bash
cd experiments/appendix/E_catastrophic_failure_experiment
python generate_figure6_5model.py
# Output: results/figure6_5model.png
# Runtime: ~7 seconds
```

---

## Paper Location

**Appendix D**, Section D.1: Catastrophic Failure Detection. Main paper (Section 5.2) contains a summary referencing the appendix.

---

**Last Updated**: February 15, 2026
**Status**: Ready (K=5, production router, semantic transfer, honest framing)
