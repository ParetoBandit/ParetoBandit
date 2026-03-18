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
- **Conditions** (7 total):
  - Warmup (0% burn-in) — priors only, evaluate directly on test
  - Warmup (25% burn-in) — 446 val steps, then test
  - Warmup (50% burn-in) — 893 val steps, then test
  - Warmup (75% burn-in) — 1,339 val steps, then test
  - Warmup (100% burn-in) — full val (1,785 steps), then test
  - Tabula Rasa (no burn-in) — cold start, no priors, evaluate on test
  - Tabula Rasa (100% burn-in) — cold start + full val, then test

## Run

```bash
python experiments/appendix/val_burnin_ablation/run_val_burnin_ablation.py
python experiments/appendix/val_burnin_ablation/generate_figure.py
```

## Output

| File | Description |
|------|-------------|
| `results/val_burnin_ablation_results.json` | Full per-seed metrics and curves |
| `results/val_burnin_test_regret.pdf` | Test-split cumulative regret (full + zoom) |
| `results/val_burnin_combined.pdf` | Combined val+test trajectory + aligned test comparison |
| `results/val_burnin_summary.pdf` | 2×2 factorial bar chart (priors × burn-in) |

## Expected Findings

- **Warmup priors alone (0% burn-in) should outperform Tabula Rasa**,
  confirming the priors carry genuine value independent of burn-in.
- **Test regret should decrease monotonically** as burn-in fraction
  increases, with diminishing marginal returns.
- **After 100% burn-in, warmup ≈ tabula rasa** on test, because
  forgetting has erased the priors — both conditions operate on the
  same ~200 steps of recent online evidence.
- **The largest gain should come from the first 25–50% of burn-in**,
  reflecting the bandit's rapid online calibration on top of priors.
- **The combined trajectory view should show** that the full-burn-in
  condition shifts regret from the test phase to the val phase,
  not that it avoids it entirely.
