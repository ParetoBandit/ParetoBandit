# Experiment 10: Distribution Shift Analysis

Supports the paper claim that "deployment distributions drift from offline training data, which silently degrades static routing policies."

## What this experiment measures

**It does NOT simulate temporal drift.** It quantifies the static distributional gap that already exists between:

- **Offline / Prior distribution**: RouteLLM battle prompts (used to build warmup priors via `priors_warmup_43model.joblib`)
- **Deployment distribution**: LMSYS Arena prompts (the dev + holdout sets used in all paper routing experiments)

These two datasets differ in source, sampling period, and prompt population — making this a realistic test of prior miscalibration under deployment shift.

## Portfolio

K=5 (mirrors Section 4 / Figure 5–6 of the paper):
- Llama-3.1-8B (cheap, meta)
- Mixtral-8x7B (cheap, mistral)
- Gemini-2.5-Flash (mid, google)
- Claude-Sonnet-4 (expensive, anthropic)
- GPT-4.1 (expensive, openai)

## Analyses

### 1. Feature distribution shift (PC1)
- KDE density plots of both distributions on PC1
- **Population Stability Index (PSI)** with bootstrap 95% CI
  - PSI < 0.1: negligible shift
  - PSI 0.1–0.25: moderate shift
  - PSI > 0.25: substantial shift (offline priors unreliable)
- **Kolmogorov–Smirnov test** (D-statistic, p-value)

### 2. Prior miscalibration (K=5)
- Compares offline prior reward estimates (from warmup priors Aθ) vs. observed deployment rewards per model
- Quantifies absolute and relative error per model in the portfolio

### 3. Adaptive router recovery
- **Static (frozen prior)**: N=20 seeds, no online updates, evaluated on holdout
- **Hybrid banditGPT**: trains on dev-set online-learn pool (533 prompts), evaluated at checkpoints [0, 50, 100, 200, 300, 400, 533]
- Reports gap closure: `(adaptive_final - static) / (oracle - static) × 100%`

## Running

```bash
python3 experiments/10_distribution_shift/run_distribution_shift.py
```

Runtime: ~15–20 minutes (dominated by embedding 8k+ prompts + 20-seed router evaluation).

## Output

- `results/distribution_shift_results.json` — all numerical results
- `results/figure_distribution_shift.png` — two-panel figure (shift + recovery)

## Key numbers to cite in paper

From `distribution_shift_results.json`:
- `distribution_shift.psi` — PSI value
- `distribution_shift.psi_ci_95` — bootstrap CI
- `distribution_shift.ks_stat` / `ks_p_value` — KS test
- `prior_miscalibration.<model_id>.relative_error_pct` — per-model prior error
- `gap_closure_pct` — how much of the oracle gap the adaptive router closes
