# Catastrophic Failure Detection (K=5 and K=10 Portfolios)

**Experiment Goal**: Evaluate Corralling's secondary safety benefit — automatic failover under catastrophic model failure — using production router components with semantic transfer.

**Key Result**: The Corralling router maintains high quality during failure phases and recovers gracefully, while the static baseline collapses to near-zero reward when its preferred model fails.

---

## Overview

This experiment evaluates **Corralling's response to catastrophic model failure** on **K=5** and **K=10** portfolios using the production router: `CostAwareLinUCBRouter` (warmup expert), `CostAwareTabulaRasaRouter` (cold-start expert), and `CorrallingRouter` with importance-weighted meta-learning.

**Warmup priors**: All models have informed priors from either direct RouteLLM battle data or semantic transfer from the nearest known neighbor.

---

## Files

```
E_catastrophic_failure_experiment/
├── generate_figure9_5model.py     # Main experiment (K=5 and K=10)
├── supplementary/
│   └── ablation_learning_rate_catastrophic.py
├── results/
│   ├── catastrophic_failure_results.json
│   ├── figure9_5model.{pdf,png}
│   ├── catastrophic_K5.{pdf,png}
│   └── catastrophic_K10.{pdf,png}
└── README.md
```

**LaTeX write-up**: `../C_extended_results/C1_catastrophic_failure.tex`

---

## Key Results (N=20 Seeds)

### K=5 Phase-by-Phase Performance

| Method | Healthy | Failure | Recovery |
|--------|---------|---------|----------|
| Oracle | 0.990 | 0.975 | 0.988 |
| **banditGPT** | **0.888** | **0.753** | **0.886** |
| EMA Tracker | 0.806 | 0.859 | 0.930 |
| Static Best | 0.950 | 0.154 | 0.958 |

### K=10 Phase-by-Phase Performance

| Method | Healthy | Failure | Recovery |
|--------|---------|---------|----------|
| Oracle | 0.992 | 0.980 | 0.990 |
| **banditGPT** | **0.894** | **0.862** | **0.905** |
| EMA Tracker | 0.810 | 0.859 | 0.925 |
| Static Best | 0.948 | 0.155 | 0.957 |

**Static collapse**: When the best static model fails, quality drops to ~15% — a catastrophic outage. banditGPT maintains 75–86% quality during the failure phase by redistributing traffic across remaining models.

**EMA advantage during failure**: The EMA tracker shows higher failure-phase quality (0.859) because its simpler epsilon-greedy exploration recovers faster in the binary "avoid the failed model" regime. However, banditGPT outperforms EMA in the healthy phase where contextual routing matters.

---

## Reproduction

```bash
cd experiments/appendix/E_catastrophic_failure_experiment
python generate_figure9_5model.py
# Output: results/figure9_5model.png, results/catastrophic_failure_results.json
# Runtime: ~7 seconds
```

---

**Last Updated**: February 2026
