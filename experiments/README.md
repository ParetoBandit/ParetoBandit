# experiments: Budget-Constrained Non-Stationary LLM Routing

Experiments for the paper pivot to **Budget-Constrained Non-Stationary
LLM Routing via Primal-Dual Contextual Bandits**.

All experiments use the canonical K=3 portfolio (Llama-3.1-8B,
Mistral-Large-2512, Gemini-2.5-Pro) unless otherwise noted.

## Directory Layout

```
experiments/
├── utils/                             # Shared utilities (see experiments/utils/)
├── 01_stationary_budget_pacing/       # BudgetPacer vs static cost-penalty Pareto
│   └── results/
├── 02_nonstationary_k3_drift/         # K=3 drift detection + tabula-rasa reset
│   └── results/
├── 03_budget_plus_drift/              # Interaction: budget pacing under model drift
│   └── results/
└── 04_sensitivity_sweep/              # Budget target x drift magnitude heatmap
    └── results/
```

## Experiment Overview

| # | Title | Key Question | Primary Figure |
|---|-------|-------------|----------------|
| 01 | Stationary Budget Pacing | Does BudgetPacer Pareto-dominate a static cost-penalty sweep? | Pareto frontier + lambda convergence |
| 02 | Non-stationary K=3 Drift | Does BanditGPT adapt to model quality shifts better than fixed or naive online routing? | 3-condition cumulative regret |
| 03 | Budget + Drift Interaction | Does the BudgetPacer maintain budget compliance under cost drift where static penalties fail? | 3-condition regret per budget + adaptation dynamics |
| 04 | Sensitivity Sweep | How robust is the system across budget targets and drift magnitudes? | Heatmap (budget x drift) |

### Main-text baselines (Experiments 02 and 03)

Three conditions of increasing sophistication are compared:

1. **Fixed Policy (offline)** — Warmup priors deployed frozen. No online learning.
2. **Naive Bandit (γ=1.0)** — LinUCB with infinite memory and warmup priors.
3. **BanditGPT (γ=0.997)** — Warmup priors + geometric forgetting + BudgetPacer (Exp 03 only).

Full ablation details (Fast forgetting, Tabula Rasa, pacer variants) are in
`appendix/forgetting_ablation/` and `appendix/forgetting_factor_sweep/`.

## Dependencies

Shared utilities live in `experiments/utils/` and are imported from there
via path manipulation (same pattern as the v1 experiments).  Core
components used:

- `bandit_gpt.budget_pacer.BudgetPacer` -- Primal-Dual CBwK pacing
- `bandit_gpt.drift.DriftDetector` -- covariate shift detection
- `bandit_gpt.router.BanditRouter` -- core contextual bandit router

## Relationship to experiments/

The `experiments/` directory contains the original paper experiments
(Figures 1--4, appendix).  This directory (`experiments/`) contains
the new experiments for the budget-constrained non-stationarity pivot.
Both directories coexist; no original experiments are modified.
