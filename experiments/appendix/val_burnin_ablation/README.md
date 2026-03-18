# Appendix: Validation Burn-In Ablation

Quantifies how much of the reported test-set performance depends on the
online-learning burn-in that occurs on the validation split before the
test trajectory begins.

The paper's standard protocol is: warmup priors (train, n=8,374) →
online learning on val (n=1,785, no reported metrics) → evaluation on
test (n=1,824, reported metrics).  This experiment decomposes the
contribution of each stage by varying the amount of val burn-in from
0% to 100%.

## Motivation

A reviewer may reasonably ask: *If the bandit needs 1,785 online
interactions on val before it performs well on test, what are the
warmup priors actually contributing?*  This experiment provides two
complementary answers:

1. **Burn-in fraction sweep** — Shows test regret under 0%, 25%, 50%,
   75%, and 100% val burn-in, isolating the marginal value of each
   additional burn-in increment.

2. **Combined trajectory** — Reports cumulative regret over the full
   val+test stream from step 1 (the "no free lunch" view), making all
   learning costs visible.

3. **2×2 factorial** — Crosses {warmup priors, tabula rasa} with
   {0% burn-in, 100% burn-in} to cleanly attribute the burn-in benefit
   to priors vs. online learning.

## Key design note: forgetting × burn-in interaction

With γ=0.995, effective memory ≈ 200 steps.  After 100% burn-in
(1,785 val steps), warmup priors are decayed by γ^1785 ≈ 1.3e−5 —
**effectively erased**.  This means:

- **0% burn-in**: Priors are fully intact at the start of test.
- **100% burn-in**: Priors are forgotten; the bandit runs on the last
  ~200 val observations.  At this point, warmup and tabula rasa
  conditions should converge (both rely on recent online evidence).

This interaction is a feature, not a bug — it reveals that the burn-in
*replaces* the priors rather than merely supplementing them.

## Setup

- **Arms**: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro
- **Data**: val.jsonl (1,785 prompts, burn-in) + test.jsonl (1,824
  prompts, evaluation); 20 seeds
- **Hyperparameters (warmup)**: alpha=0.10, n_eff=1000, gamma=0.995,
  disjoint LinUCB
- **Hyperparameters (tabula rasa)**: alpha=0.10, n_eff=1, gamma=0.995,
  disjoint LinUCB
- **Budget regimes**: Unconstrained, Tight ($2.34e-4 $/req), Moderate
  ($6.62e-4 $/req) — BudgetPacer active for constrained regimes
- **Pacer pre-calibration**: For budget-constrained conditions, the
  BudgetPacer is pre-calibrated on the full val set (routing with the
  initial policy, costs observed, bandit NOT updated). All burn-in
  fractions thus start from an identical λ_t snapshot, decoupling
  pacer calibration from reward-model burn-in.
- **Conditions**:
  - *Unconstrained* (7 conditions): Warmup at 0/25/50/75/100% burn-in
    + Tabula Rasa at 0%/100%
  - *Tight + Moderate* (4 conditions each): 2×2 factorial — {Warmup,
    Tabula Rasa} × {0%, 100% burn-in} with BudgetPacer

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
| `results/val_burnin_combined.pdf` | Combined val+test trajectory + aligned test comparison |
| `results/val_burnin_summary.pdf` | 2×2 factorial bar chart (unconstrained) |
| `results/val_burnin_budget_summary.pdf` | Budget-stratified 2×2 factorial (all regimes) |

## Key Results

**The burn-in finding holds across all budget regimes:**

| Budget | Warmup 0% | Warmup 100% | Δ% | p |
|--------|-----------|-------------|-----|---|
| None | 74.0±0.5 | 74.4±0.8 | -0.6% | 0.55 |
| Tight | 198.2±1.0 | 199.9±1.3 | -0.8% | 0.37 |
| Moderate | 150.8±3.0 | 149.4±1.9 | +0.9% | 0.70 |

**Cold-start penalty (Tabula Rasa 0% vs Warmup 100%):**

| Budget | Tabula Rasa 0% | Warmup 100% | Δ% | p |
|--------|----------------|-------------|-----|---|
| None | 83.5±0.8 | 74.4±0.8 | +12.2% | <10⁻⁵ |
| Tight | 205.1±2.1 | 199.9±1.3 | +2.6% | 0.019 |
| Moderate | 152.2±3.0 | 149.4±1.9 | +1.9% | 0.29 |

**Convergence after burn-in (Tabula Rasa 100% vs Warmup 100%):**

| Budget | Tabula Rasa 100% | Warmup 100% | Δ% | p |
|--------|------------------|-------------|-----|---|
| None | 73.9±0.7 | 74.4±0.8 | -0.8% | 0.67 |
| Tight | 197.2±1.1 | 199.9±1.3 | -1.3% | 0.11 |
| Moderate | 144.4±2.1 | 149.4±1.9 | -3.4% | 0.28 |

**Budget compliance (Cost / Target ratio):**

| Budget | Warmup 0% | Warmup 100% | TR 0% | TR 100% |
|--------|-----------|-------------|-------|---------|
| Tight | 100.0% | 100.0% | 98.8% | 99.9% |
| Moderate | 99.4% | 99.4% | 97.6% | 99.3% |
