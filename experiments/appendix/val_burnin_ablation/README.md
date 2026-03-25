# Appendix: Validation Burn-In Ablation

Tests whether the online learning that incidentally occurs on the
validation split inflates held-out test performance.

The val split (n=1,785) serves two roles in the pipeline:
**(i) hyperparameter selection** — alpha, gamma, and n_eff are tuned
on val via the T_adapt-constrained Pareto knee-point method; and
**(ii) online learning** — the bandit processes val prompts before
test, providing reward-model burn-in and BudgetPacer calibration.
The first role is essential (prevents test-set contamination); the
second is incidental.  This experiment ablates the incidental
contribution by varying burn-in from 0% to 100%.

## Motivation

A reviewer may reasonably ask: *Does the bandit need 1,785 online
interactions on val before it performs well on test?*  This experiment
answers that question with three complementary views:

1. **Burn-in fraction sweep** — Shows test regret under 0%, 25%, 50%,
   75%, and 100% val burn-in, isolating the marginal value of each
   additional burn-in increment.

2. **Combined trajectory** — Reports cumulative regret over the full
   val+test stream from step 1 (the "no free lunch" view), making all
   learning costs visible.

3. **2×2 factorial** — Crosses {warmup priors, tabula rasa} with
   {0% burn-in, 100% burn-in} to compare the two systems at extreme
   burn-in levels.

## Key design note: forgetting × burn-in interaction

With γ=0.997, effective memory ≈ 333 steps.  After 100% burn-in
(1,785 val steps), warmup priors are decayed by γ^1785 ≈ 4.7×10⁻³ —
**heavily attenuated but not fully erased**.  This means:

- **0% burn-in**: Priors are fully intact at the start of test.
  The bandit benefits from n_eff=1,163.9 pseudo-observations trained
  on 8,374 examples — a stronger starting point than ~333 effective
  online observations.
- **100% burn-in**: Priors are nearly erased; the bandit runs on the
  last ~333 val observations.  After 100% burn-in, warmup and tabula
  rasa are statistically indistinguishable in the unconstrained
  regime (68.5 vs 60.4, p_adj=0.45), consistent with convergence.

## Setup

- **Arms**: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro
- **Data**: val.jsonl (1,785 prompts, burn-in) + test.jsonl (1,824
  prompts, evaluation); 20 seeds
- **Hyperparameters (warmup)**: alpha=0.01, n_eff=1163.9, gamma=0.997,
  disjoint LinUCB
- **Hyperparameters (tabula rasa)**: alpha=0.01, n_eff=1, gamma=0.995,
  disjoint LinUCB (independently tuned via T_adapt Pareto knee-point)
- **Budget regimes**: Unconstrained, Tight ($3.0e-4 $/req), Moderate
  ($6.62e-4 $/req) — BudgetPacer active for constrained regimes
- **Pacer pre-calibration**: For budget-constrained conditions, the
  BudgetPacer is pre-calibrated on the full val set (routing with the
  initial policy, costs observed, bandit NOT updated). All burn-in
  fractions thus start from an identical λ_t snapshot.  Note: the
  pacer continues to co-adapt with the bandit during burn-in itself,
  so differences at the start of test reflect both reward-model
  learning and pacer adaptation (not reward-model learning alone).
  Pre-calibration ensures a fair *starting point*, not identical
  pacer states at the start of test.
- **Conditions**:
  - *Unconstrained* (7 conditions): Warmup at 0/25/50/75/100% burn-in
    + Tabula Rasa at 0%/100%
  - *Tight + Moderate* (4 conditions each): 2×2 factorial — {Warmup,
    Tabula Rasa} × {0%, 100% burn-in} with BudgetPacer
- **Confound note**: warmup (γ=0.997) and tabula rasa (γ=0.995)
  use independently tuned hyperparameters, so cross-system comparisons
  reflect the difference between two optimised configurations.
  Within-warmup comparisons (varying only burn-in) are cleanly
  single-factor.

## Run

```bash
python experiments/appendix/val_burnin_ablation/run_val_burnin_ablation.py
python experiments/appendix/val_burnin_ablation/generate_figure.py
```

## Output

| File | Description |
|------|-------------|
| `results/val_burnin_ablation_results.json` | Full per-seed metrics and curves |
| `results/val_burnin_test_regret.pdf` | Test-split cumulative regret (unconstrained, full + zoom) |
| `results/val_burnin_budget_summary.pdf` | Budget-stratified 2×2 factorial (unconstrained + tight + moderate) |

## Key Results

**Unconstrained — less burn-in is better (Bonferroni-corrected, 6 tests):**

| Burn-in | Regret | R@200 | Δ% | p_adj |
|---------|--------|-------|----|-------|
| 0% | 56.4±0.3 | 5.6 | -6.6% | < 10⁻⁴ |
| 25% | 56.9±0.4 | 5.8 | -5.8% | < 10⁻⁴ |
| 50% | 58.4±0.5 | 5.8 | -3.2% | 0.029 |
| 75% | 59.4±0.7 | 6.0 | -1.6% | 1.0 |
| 100% (ref) | 60.4±0.5 | 5.9 | — | — |
| TR 0% | 100.2±15.4 | 14.9 | +66.0% | 0.016 |
| TR 100% | 68.5±4.0 | 8.4 | +13.5% | 0.45 |

**Budget-constrained (Bonferroni-corrected, 3 tests per regime):**

| Budget | Condition | Regret | Cost/T | Δ% | p_adj |
|--------|-----------|--------|--------|-----|-------|
| Tight | Warmup 0% | 160.1±1.0 | 1.00 | -1.0% | 1.0 |
| Tight | Warmup 100% | 161.7±1.6 | 1.00 | — | — |
| Tight | TR 0% | 287.9±10.4 | 0.39 | +78.0% | < 10⁻⁵ |
| Tight | TR 100% | 224.8±13.6 | 0.89 | +39.1% | < 10⁻³ |
| Moderate | Warmup 0% | 134.5±2.2 | 1.00 | +10.8% | < 10⁻³ |
| Moderate | Warmup 100% | 121.4±2.2 | 0.98 | — | — |
| Moderate | TR 0% | 236.0±18.7 | 0.42 | +94.5% | < 10⁻⁴ |
| Moderate | TR 100% | 162.5±21.2 | 0.83 | +33.9% | 0.53 |

**Summary — burn-in effect is regime-dependent:**

1. **Unconstrained**: intact priors outperform online estimates → burn-in
   *increases* test regret.
2. **Tight budget**: pacer dominates routing → burn-in is irrelevant.
3. **Moderate budget**: bandit–pacer co-adaptation needs burn-in →
   skipping it *increases* test regret.

In no regime does burn-in artificially inflate test performance.
The val split earns its place through hyperparameter selection and
pacer calibration; the reward-model burn-in it provides is incidental
and, in the unconstrained regime, mildly harmful.

**Tabula rasa fails under budget constraints**: cost/target ratios of
0.39 (tight) and 0.42 (moderate) for TR 0% indicate the cold-start
bandit cannot calibrate with the BudgetPacer, leading to massive
underspending and high regret.
