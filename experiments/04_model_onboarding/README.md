# Appendix: Model Onboarding Under Budget Constraints

**Key question:** Can ParetoBandit incorporate a newly released model
(K=3 → K=4) via a single `register_model()` call—and correctly
*reject* models that are inferior or too expensive?

## Design

| Phase | Arms | Data | Prompts |
|-------|------|------|---------|
| Phase 1 (online learning) | K=3 (Llama-8B, Mistral-Large, Gemini-Pro) | val split | 1,785 |
| Onboard | `register_model()` adds Gemini-2.5-Flash | — | — |
| Burn-in (forced exploration) | Flash receives 20 unconditional pulls | test_k4 split | 20 |
| Phase 2 (UCB evaluation) | K=4 (+ Flash) | test_k4 split | 1,793 |

**Hyperparameters** (from Experiment 05 epsilon-constraint selection):
`alpha=0.01`, `prior_n_effective=1163.9`, `forgetting_factor=0.997`.

**Budget targets:** tight ($3.0×10⁻⁴), moderate ($6.6×10⁻⁴),
loose ($1.9×10⁻³), plus unconstrained.

**Seeds:** 20 per condition (9000–9019).

### Onboarding scenarios

Three scenarios test whether the router *discriminates* rather than
blindly adopting every new model:

| Scenario | Reward scale | Cost scale | Expected behaviour |
|----------|-------------|------------|-------------------|
| Good & Cheap | 1.0× | 1.0× | Flash adopted |
| Good & Expensive | 1.0× | 10× | Suppressed by pacer under tight/moderate budgets |
| Bad & Cheap | 0.5× | 1.0× | Rejected after burn-in |

### Forced exploration (burn-in)

With `alpha=0.01` and strong K=3 warmup priors (n_eff=1163.9), the UCB
exploration bonus for a cold-start arm is too small to trigger natural
exploration. The first 20 Phase 2 prompts are routed unconditionally to
Flash so the bandit collects real observations before UCB selection
takes over.

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
# Full experiment (20 seeds, all 3 scenarios, ~80 s)
python experiments/04_model_onboarding/run_model_onboarding.py

# Quick debugging run (3 seeds)
python experiments/04_model_onboarding/run_model_onboarding.py --fast

# Single scenario
python experiments/04_model_onboarding/run_model_onboarding.py --scenario bad_cheap

# Generate figure
python experiments/04_model_onboarding/generate_figure.py
```

## Output

```
results/
├── model_onboarding_results.json   # Machine-readable results (all scenarios)
├── model_onboarding.pdf            # 1×3 panel figure (Flash adoption by budget tier)
└── model_onboarding.png            # PNG version
```

## Key results

- **Good & Cheap:** Flash achieves sustained adoption in 20/20 seeds
  across all budget tiers. Final share ranges from 4.4% (tight) to
  10.2% (loose).
- **Bad & Cheap:** Flash is rejected in every seed and budget tier
  (0.0% final share). The 20-prompt burn-in is self-limiting.
- **Good & Expensive:** The BudgetPacer suppresses Flash under tight
  and moderate budgets (0/20 seeds sustain), but permits partial
  adoption under loose (6.1%) and unconstrained (7.3%) budgets.

## Relationship to other experiments

- **Experiment 01** (budget pacing): Tests the BudgetPacer under
  stationary conditions with a fixed K=3 portfolio.
- **Experiment 02** (budget + drift): Tests the BudgetPacer under
  cost drift (existing arm changes price).
- **Experiment 03** (catastrophic failure): Tests recovery when an arm
  degrades suddenly.
- **This experiment**: Tests the BudgetPacer under portfolio expansion
  (a new arm appears) and validates the router's ability to
  discriminate between good, bad, and expensive newcomers.
