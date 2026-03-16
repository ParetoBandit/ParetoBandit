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
├── 02_nonstationary_k3_drift/         # K=3 reward-shift adaptation (5 conditions)
│   └── results/
├── 03_budget_plus_drift/              # Interaction: budget pacing under model drift
│   └── results/
└── 04_hparam_optimization/            # Epsilon-constraint hyperparameter selection
    └── results/
```

## Experiment Overview

| # | Title | Key Question | Primary Figure |
|---|-------|-------------|----------------|
| 01 | Stationary Budget Pacing | Does BudgetPacer Pareto-dominate a static cost-penalty sweep? | Pareto frontier + lambda convergence |
| 02 | Non-stationary K=3 Drift | Does ParetoBandit adapt to model quality shifts better than stationary and non-stationary baselines? | 4-condition cumulative regret |
| 03 | Budget + Drift Interaction | Does the BudgetPacer maintain budget compliance under cost drift where static penalties fail? | 3-condition regret per budget + adaptation dynamics |
| 04 | Hparam Optimization | How should alpha, n_eff, and gamma be jointly selected? | Epsilon-constraint selection (budget-paced AUC + Phase-2 regret) |

### Main-text baselines (Experiments 02 and 03)

Four conditions of increasing sophistication are compared in Experiment 02:

1. **Fixed Policy (offline)** — Warmup priors deployed frozen. No online learning.
2. **Naive Bandit (γ=1.0)** — LinUCB with infinite memory and warmup priors.
3. **SW-UCB (W=200)** — Sliding-Window LinUCB (Garivier & Moulines 2011) without
   priors.  Retains only the last W observations with equal weighting.
   Window size matched to ParetoBandit's effective memory (~200 steps).
4. **ParetoBandit (γ=0.995)** — Warmup priors + geometric forgetting + BudgetPacer (Exp 03 only).

Experiment 03 uses conditions 1, 2, and 4 with BudgetPacer integration.

Full ablation details (Fast forgetting, Tabula Rasa, pacer variants) are in
`appendix/forgetting_ablation/` and `appendix/forgetting_factor_sweep/`.

## Dependencies

Shared utilities live in `experiments/utils/` and are imported from there
via path manipulation (same pattern as the v1 experiments).  Core
components used:

- `pareto_bandit.budget_pacer.BudgetPacer` -- Primal-Dual CBwK pacing
- `pareto_bandit.router.BanditRouter` -- core contextual bandit router

## Relationship to experiments/

The `experiments/` directory contains the original paper experiments
(Figures 1--4, appendix).  This directory (`experiments/`) contains
the new experiments for the budget-constrained non-stationarity pivot.
Both directories coexist; no original experiments are modified.
