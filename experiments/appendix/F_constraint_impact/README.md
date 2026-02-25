# Hard Per-Request Constraint Impact (K=10 Portfolio)

**Experiment Goal**: Validate that Layer 1 constraint filtering (cost ceilings, latency limits) produces graceful quality degradation, preserves bandit advantage over static baselines, and interacts constructively with the soft cost penalty λ.

---

## Overview

banditGPT's Layer 1 prunes models violating user-specified cost ceilings or latency limits *before* the bandit selects. This experiment evaluates the mechanism across five dimensions:

1. **Cost ceiling sweep** — tighten per-request budget from unconstrained to $0.0001
2. **Latency ceiling sweep** — tighten TTFT limit from unconstrained to 0.4s
3. **Combined production scenarios** — realistic joint cost + latency SLAs
4. **Constrained Pareto frontiers** — hard constraints shift the frontier; λ traces it
5. **Ablation** — always-on constraints vs. eval-only (does constrained training hurt?)

---

## Files

```
F_constraint_impact/
├── run_constraint_experiment.py       # Main experiment script
├── generate_figure.py                 # Figure generation
├── section_constraint_impact.tex      # LaTeX write-up (included via appendix_f.tex)
├── README.md                          # This file
└── results/
    ├── constraint_impact_results.json # Full numerical results
    ├── figure_constraint_impact.pdf   # Two-panel figure
    └── figure_constraint_impact.png
```

---

## Key Results (K=10, N=50 seeds)

### Cost Ceiling Sweep

| Cost Ceiling | Eligible K | Mean Reward (±95% CI) | Mean Cost/req |
|-------------|-----------|----------------------|---------------|
| $0.0001 | 2 | 0.885 ± 0.003 | $0.000047 |
| $0.0003 | 5 | 0.883 ± 0.003 | $0.00017 |
| $0.0005 | 6 | 0.889 ± 0.003 | $0.00026 |
| $0.002 | 7 | 0.888 ± 0.003 | $0.00054 |
| $0.005 | 8 | 0.896 ± 0.004 | $0.00097 |
| $0.01 | 9 | 0.900 ± 0.003 | $0.00175 |
| Unconstrained | 10 | 0.895 ± 0.006 | $0.00217 |

Tightening the cost ceiling by 520× (from unconstrained to $0.0001/req) reduces reward by only 1.0 pp (0.895 → 0.885). At moderate budgets (K'=8), the bandit matches the unconstrained baseline.

### Latency Ceiling Sweep

| TTFT Ceiling | Eligible K | Mean Reward (±95% CI) |
|-------------|-----------|----------------------|
| ≤ 0.4s | 3 | 0.887 ± 0.003 |
| ≤ 0.5s | 5 | 0.885 ± 0.004 |
| ≤ 0.7s | 8 | 0.890 ± 0.006 |
| ≤ 1.0s | 9 | 0.898 ± 0.004 |
| Unconstrained | 10 | 0.895 ± 0.006 |

The 1.0s ceiling (K'=9) *exceeds* the unconstrained mean because the excluded model has below-median quality.

### Production Deployment Scenarios

| Scenario | Cost Ceil. | Lat. Ceil. | K' | banditGPT | Best Static | Cost/req |
|----------|-----------|-----------|-----|-----------|-------------|---------|
| Unconstrained | — | — | 10 | 0.895 ± 0.006 | 0.949 | $0.00217 |
| Premium-SLA | $0.01 | 1.0s | 8 | 0.900 ± 0.004 | 0.949 | $0.00206 |
| Enterprise-SLA | $0.005 | 0.7s | 7 | 0.896 ± 0.004 | 0.949 | $0.00116 |
| Budget-Mid | $0.002 | — | 7 | 0.888 ± 0.003 | 0.914 | $0.00054 |
| Latency-Strict | — | 0.5s | 5 | 0.885 ± 0.004 | 0.906 | $0.00059 |
| Budget-Micro | $0.0003 | — | 5 | 0.883 ± 0.003 | 0.906 | $0.00017 |

Moderate combined constraints (Premium-SLA) *improve* over unconstrained (0.900 vs. 0.895) by pruning low-quality models. Even the most restrictive profile (Budget-Micro) retains 98.6% of unconstrained quality at 13× lower cost.

### Ablation: Always-On vs. Eval-Only Constraints

At cost ≤ $0.0005 (K'=6), 50 paired seeds:
- **Always-on**: 0.889 ± 0.003
- **Eval-only**: 0.887 ± 0.004
- **Wilcoxon p = 0.553**, Cohen's d = 0.08

Constraining exploration to the feasible set during training does *not* sacrifice quality. Operators can enable constraints from day one.

### Zero Violations

Across all conditions (1,650,000 routing decisions), the violation counter is exactly zero. Layer 1 filtering guarantees constraint satisfaction by construction.

---

## Setup

- **Portfolio**: K=10 models spanning 520× cost range ($0.000025–$0.013/req)
- **Architecture**: Full Corralling with warmup priors
- **Online learning**: 533 prompts
- **Holdout evaluation**: 750 prompts
- **Seeds**: 50 independent trials per condition
- **Constraints**: Active during both training and evaluation (unless noted)

---

## Reproduction

```bash
cd experiments/appendix/F_constraint_impact

# Run the full experiment
python run_constraint_experiment.py

# Generate figure
python generate_figure.py
```

**Output**: `results/constraint_impact_results.json`, `results/figure_constraint_impact.{pdf,png}`

---

**Last Updated**: February 2026
