# Appendix: Model Onboarding Under Budget Constraints

**Key question:** Can the BudgetPacer accommodate a newly registered
model (K=3 → K=4) without manual reconfiguration?

## Design

| Phase | Arms | Data | Steps |
|-------|------|------|-------|
| Phase 1 (burn-in) | K=3 (Llama, Mistral, Gemini-Pro) | val split | 1,785 |
| Onboard | `register_model()` adds Gemini-2.5-Flash | — | — |
| Phase 2 (eval) | K=4 (+ Flash) | test_k4 split | varies |

Three budget targets (tight / moderate / loose) + unconstrained baseline.
20 seeds per condition.

## Prerequisites

Flash rewards must be collected using the same DeepSeek-R1 single-judge
v3 rubric used by `build_router_pareto_dataset.py` for all canonical
K=3 data:

```bash
# 1. Collect flash responses + DeepSeek-R1 v3 judgments
python data_collection/scripts/collect_flash_canonical.py

# 2. Merge flash rewards into K=4 val/test splits
python data_collection/scripts/merge_flash_into_splits.py
```

This produces `data_collection/rewards/val_k4.jsonl` and
`data_collection/rewards/test_k4.jsonl`.

## Running

```bash
# Full experiment (20 seeds, ~30 min)
python experiments/appendix/model_onboarding/run_model_onboarding.py

# Quick debugging run (3 seeds)
python experiments/appendix/model_onboarding/run_model_onboarding.py --fast

# Generate figure
python experiments/appendix/model_onboarding/generate_figure.py
```

## Output

```
results/
├── model_onboarding_results.json   # Machine-readable results
├── model_onboarding.pdf            # 3-panel figure
└── model_onboarding.png            # PNG version
```

## Relationship to other experiments

- **Experiment 01** (budget pacing): Tests the BudgetPacer under
  stationary conditions with a fixed K=3 portfolio.
- **Experiment 03** (budget + drift): Tests the BudgetPacer under
  cost drift (existing arm changes price).
- **This experiment**: Tests the BudgetPacer under portfolio expansion
  (a new arm appears).  The exploration starvation finding from Exp 03
  extends to onboarding: tight budgets prevent exploration of new arms.
