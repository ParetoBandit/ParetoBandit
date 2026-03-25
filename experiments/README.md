# experiments: Budget-Constrained Non-Stationary LLM Routing

Experiments for the paper pivot to **Budget-Constrained Non-Stationary
LLM Routing via Primal-Dual Contextual Bandits**.

All experiments use the canonical K=3 portfolio (Llama-3.1-8B,
Mistral-Large-2512, Gemini-2.5-Pro) unless otherwise noted.

## Directory Layout

```
experiments/
├── utils/                             # Shared utilities (simulation, bootstrap, latex gen)
├── tests/                             # Experiment regression tests + pinned references
├── 01_stationary_budget_pacing/       # BudgetPacer vs static cost-penalty Pareto
│   └── results/
├── 02_budget_plus_drift/              # Interaction: budget pacing under model drift
│   └── results/
├── 03_catastrophic_failure/           # Catastrophic model failure (3-phase)
│   └── results/
├── 04_model_onboarding/               # Cold-start model onboarding (K=3→K=4)
│   └── results/
├── appendix/
│   ├── hparam_optimization/           # Epsilon-constraint hyperparameter selection
│   ├── cost_heuristic_validation/     # Cost-target heuristic vs oracle
│   ├── warmup_ablation/               # Warmup prior ablation
│   ├── val_burnin_ablation/           # Validation burn-in ablation
│   ├── prior_mismatch/                # Prior mismatch robustness
│   ├── judge_robustness/              # Judge agreement analysis
│   ├── latency_benchmark/             # Routing latency profiling
│   └── recovery_limit/                # Recovery limit analysis
└── legacy/
    └── 02_nonstationary_k3_drift/     # Original drift experiment (superseded by 02_budget_plus_drift)
```

## Experiment Overview

| # | Title | Key Question | Primary Figure |
|---|-------|-------------|----------------|
| 01 | Stationary Budget Pacing | Does BudgetPacer Pareto-dominate a static cost-penalty sweep? | Pareto frontier + lambda convergence |
| 02 | Budget + Cost Drift | Does the BudgetPacer maintain budget compliance under cost drift where static penalties fail? | 3x1 stacked adaptation dynamics |
| 03 | Catastrophic Model Failure | Can ParetoBandit detect failure, redistribute traffic, and maintain budget compliance? | 3x1 stacked adaptation dynamics |
| 04 | Model Onboarding | Can a single register_model() call onboard a new model with zero offline evaluation? | 3-panel: Flash adoption, arm composition, cost compliance |

### Main-text baselines (Experiments 02 and 03)

Four conditions of increasing sophistication are compared in Experiment 02:

1. **Fixed Policy (offline)** — Warmup priors deployed frozen. No online learning.
2. **Naive Bandit (γ=1.0)** — LinUCB with infinite memory and warmup priors.
3. **SW-UCB (W=200)** — Sliding-Window LinUCB (Garivier & Moulines 2011) without
   priors.  Retains only the last W observations with equal weighting.
   Window size matched to ParetoBandit's effective memory (~200 steps).
4. **ParetoBandit (γ=0.995)** — Warmup priors + geometric forgetting + BudgetPacer (Exp 03 only).

Experiment 03 uses conditions 1, 2, and 4 with BudgetPacer integration.

Full ablation details (warmup priors, validation burn-in, prior mismatch) are in
the `appendix/` subdirectories.

## Reproducing the Experiments

```bash
git clone https://github.com/atabernermiller/paretobandit.git
cd paretobandit
pip install -e ".[experiments]"
```

Each experiment has a `run_*.py` script that writes JSON results to
`results/`, and a `generate_figure.py` / `generate_latex.py` pair that
produces figures and `_autogen.tex` macros from those results.

```bash
# Example: run Experiment 01 and regenerate its figure
python experiments/01_stationary_budget_pacing/run_budget_pacing.py
python experiments/01_stationary_budget_pacing/generate_figure.py
python experiments/01_stationary_budget_pacing/generate_latex.py
```

All experiments are deterministic (seeded RNG, 20 seeds per condition).
Results are pre-computed in `results/` directories so figures and LaTeX
macros can be regenerated without re-running the simulations.

## Dependencies

Shared utilities live in `experiments/utils/` and are imported from there
via path manipulation.  Core components used:

- `pareto_bandit.budget_pacer.BudgetPacer` -- Primal-Dual CBwK pacing
- `pareto_bandit.router.BanditRouter` -- core contextual bandit router

## Legacy Experiments

The `legacy/` subdirectory contains superseded experiments from earlier
iterations of the paper.  They are retained for reference but are not
used in the current manuscript.
