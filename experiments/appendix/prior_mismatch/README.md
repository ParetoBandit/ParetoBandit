# Appendix: Prior Mismatch Sensitivity Analysis

Answers the question: **at what level of prior mismatch do warmup priors
start hurting, and can conservative `n_eff` rescue bad priors?**

The warmup ablation (Appendix: Cold-Start vs Warmup) showed that
well-calibrated priors reduce regret by 11%.  But all existing experiments
used priors trained on a representative training set.  This experiment
stress-tests warmup priors under progressively worse calibration to
quantify the cost of bad priors and identify when tabula rasa is the
safer choice.

## Setup

- **Arms**: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro
- **Data**: test.jsonl (1,824 prompts), 20 seeds; cumulative-regret protocol
- **Seeds**: offset=9000 (matches warmup ablation for paired comparisons)
- **Hyperparameters (warmup conditions)**: alpha=0.01, gamma=0.997, disjoint LinUCB
- **Hyperparameters (tabula rasa)**: alpha=0.01, n_eff=1, gamma=0.995

## Prior Quality Gradient

| Level | Label | Source | N prompts | Why it's mismatched |
|-------|-------|--------|-----------|---------------------|
| Correct | Well-calibrated | Full train set | 8,374 | Baseline — correct arm ranking and magnitude |
| Control | Random-1680 | Random subsample of full train | 1,680 | **Sample-size control** — same n as GSM8K-only, full distribution |
| Mild | MMLU-only | MMLU subset | 1,855 | Correct ranking (Gemini > Mistral >> Llama), knowledge-domain magnitudes |
| Moderate | GSM8K-only | GSM8K subset | 1,680 | All models appear near-equal (~0.95+); no useful arm differentiation |
| Severe | Inverted | Full train, Llama ↔ Gemini swapped | 8,374 | Prior learns cheapest model is best and vice versa |

**Why Random-1680?**  The domain-specific priors (MMLU-only, GSM8K-only)
use fewer training samples than Well-calibrated.  Although the `n_eff`
scaling normalises prior strength (`A_final ≈ n_eff × plasticity × E[xx^T]`,
where the training-set size cancels), a reviewer could argue that
covariance estimation quality differs.  Random-1680 uses the same sample
count as GSM8K-only but preserves the full training distribution, so:

- **Random-1680 ≈ Well-calibrated** → sample size is irrelevant (n_eff controls strength)
- **Random-1680 ≠ GSM8K-only** → the gap is purely domain mismatch

## n_eff Sweep

Each prior-quality level is tested at three `n_eff` values:

| n_eff | Interpretation |
|-------|----------------|
| 10 | Very weak prior (gentle nudge, fast online override) |
| 100 | Moderate prior (informative but correctable) |
| 1000 | Strong prior (the default tuned value; slow to override) |

Total conditions: 5 prior levels × 3 n_eff values + 1 Tabula Rasa = **16 conditions**.

## Statistical Tests

Each condition is compared to Tabula Rasa via two complementary tests:

1. **Sign test** (exact binomial, two-sided, H0: P(condition wins) = 0.5):
   counts per-seed paired wins.  Tests for a location shift — "does the
   condition have lower regret seed-by-seed?"  No distributional assumptions.
2. **Fisher exact test** on the 2×2 catastrophic-failure table (one-sided,
   H0: equal failure rates): tests whether the condition reduces tail risk.
   A seed is "catastrophic" if its regret exceeds 2× the condition's median.

The sign test is used instead of Wilcoxon signed-rank because Tabula Rasa's
heavy-tailed, multimodal per-seed regret (std ≈ 82 vs ≤ 3 for warmup
conditions) violates the Wilcoxon assumption that paired differences are
symmetric.  Results are in the JSON under `pairwise_tests_vs_tabula_rasa`.

## Design Notes

**Hyperparameters.** All warmup conditions use `alpha=0.01`, `gamma=0.997`
(the Exp-05 Pareto knee-point selection for well-calibrated priors);
Tabula Rasa uses `alpha=0.01`, `gamma=0.995` (its own optimal config).
Warmup hyperparameters were **not** re-tuned for mismatched conditions.
This is intentional: it matches the production scenario where
hyperparameters are fixed before deployment and prior quality degrades
post-deployment.  A fairer-to-warmup comparison would re-tune per
condition, but that doesn't reflect the failure mode we're studying.

**Test distribution overlap.** The test split contains ~20% GSM8K and
~22% MMLU prompts.  Domain-specific priors are therefore partially
correct for a fraction of test traffic.  This is realistic (production
traffic is mixed) but means the measured mismatch is conservative.

**Oracle definition.** `oracle = max(reward_a)` — pure quality, no cost
penalty.  Matches the warmup ablation's unconstrained protocol.

## Run

```bash
# Phase 1 generates mismatched prior files (cached after first run)
# Phase 2 runs all 16 conditions × 20 seeds
python experiments/appendix/prior_mismatch/run_prior_mismatch.py

# Generate figures from results
python experiments/appendix/prior_mismatch/generate_figure.py
```

## Outputs

- `results/prior_mismatch_results.json` — full per-seed metrics, curves, sign test + Fisher exact test results
- `results/priors/` — generated `.joblib` prior files (cached)
- `results/prior_mismatch_heatmap.pdf/.png` — regret heatmap (quality × n_eff)
- `results/prior_mismatch_distribution.pdf/.png` — per-seed violin + strip plot
