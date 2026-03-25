# Appendix: Cold-Start vs Warmup Prior Regret

Demonstrates the value of warmup priors by comparing cumulative
regret of ParetoBandit (warmup) vs Tabula Rasa (cold start) on the
K=3 portfolio under stationary conditions across four budget regimes.
A matched-γ mechanistic control isolates the prior contribution from
the confounding effect of different forgetting factors.

## Setup

- **Arms**: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro
- **Data**: test.jsonl (1,824 prompts), 20 seeds; cumulative-regret protocol
- **Hyperparameters (warmup)**: alpha=0.01, n_eff=1163.9, gamma=0.997, disjoint LinUCB (from `BEST_K3_HPARAMS`)
- **Hyperparameters (tabula rasa)**: alpha=0.01, n_eff=1.0, gamma=0.995, disjoint LinUCB (from `BEST_K3_TABULA_RASA_HPARAMS`)
- **Hyperparameters (matched-γ)**: alpha=0.01, n_eff=1.0, gamma=0.997, disjoint LinUCB (mechanistic control — cold start at warmup's forgetting rate)
- **Budget regimes**: Unconstrained, Tight ($3.0e-4 $/req), Moderate ($6.62e-4 $/req), Loose ($1.87e-3 $/req)
- **Conditions**: Warmup, Tabula Rasa, TR matched-γ, Random (× 4 budget levels for bandit conditions)

## Design rationale

The Pareto-knee sweep selects different forgetting factors for warmup
(γ=0.997, ~333-step memory) and tabula rasa (γ=0.995, ~200-step
memory).  The "best vs best" comparison (Q1) answers the deployment
question but confounds priors with memory length.  The matched-γ
control (Q2) runs tabula rasa at γ=0.997 to isolate the prior.

## Statistical methodology

- **Sign test** (exact binomial): location shift (does warmup win seed-by-seed?)
- **Fisher exact test**: tail risk (does warmup reduce catastrophic failures?)
- **Catastrophic threshold**: 2× pooled median across all conditions in each budget regime (condition-independent)
- **Multiple-testing correction**: Holm–Bonferroni across all tests

## Run

```bash
python experiments/appendix/warmup_ablation/run_warmup_ablation.py
python experiments/appendix/warmup_ablation/generate_figure.py
python experiments/appendix/warmup_ablation/generate_uncertainty_figure.py
```

## Key Results

| Budget | Condition | Regret (95% CI) | R@200 (95% CI) | Rwd | p_sign* | Cat. |
|--------|-----------|-----------------|----------------|-----|---------|------|
| None | **Warmup** | **55.0** [54.3, 55.7] | **6.2** [5.8, 6.6] | **0.933** | | 0/20 |
| None | Tabula Rasa | 113.5 [78.2, 148.7] | 17.8 [12.6, 23.0] | 0.901 | < 10⁻⁵ | 6/20 |
| None | TR (matched-γ) | 117.7 [79.8, 155.6] | 17.9 [12.7, 23.1] | 0.899 | < 10⁻³ | 6/20 |
| None | Random | 146.5 [144.3, 148.7] | 16.0 [15.1, 17.0] | 0.883 | --- | --- |
| Tight | **Warmup** | **161.2** [159.2, 163.1] | **19.5** [18.0, 21.0] | **0.875** | | 0/20 |
| Tight | Tabula Rasa | 252.0 [224.6, 279.4] | 29.9 [27.1, 32.7] | 0.825 | 0.001 | 0/20 |
| Tight | TR (matched-γ) | 255.6 [227.0, 284.3] | 29.9 [27.1, 32.7] | 0.823 | 0.006 | 0/20 |
| Moderate | **Warmup** | **133.9** [129.0, 138.7] | **14.7** [12.9, 16.6] | **0.890** | | 0/20 |
| Moderate | Tabula Rasa | 178.8 [138.1, 219.6] | 25.6 [21.2, 30.1] | 0.865 | 0.59 | 6/20 |
| Moderate | TR (matched-γ) | 191.4 [148.2, 234.5] | 26.0 [21.6, 30.4] | 0.858 | 0.50 | 8/20 |
| Loose | **Warmup** | **83.6** [81.3, 85.8] | **8.6** [8.2, 9.1] | **0.918** | | 0/20 |
| Loose | Tabula Rasa | 159.0 [121.9, 196.1] | 24.9 [20.6, 29.1] | 0.876 | 0.083 | 6/20 |
| Loose | TR (matched-γ) | 162.9 [123.5, 202.3] | 25.1 [20.9, 29.4] | 0.874 | 0.083 | 6/20 |

20 seeds; 95% normal-approximation CI; held-out test split (n=1,824). *Holm–Bonferroni corrected.

**Key findings:**
- **Q1 (deployment)**: Warmup priors reduce mean regret by 25–52% across all regimes; significant at unconstrained (p < 10⁻⁵) and tight (p = 0.001) after Holm correction
- **Q2 (mechanism)**: The matched-γ control performs nearly identically to (or slightly worse than) the original Tabula Rasa, ruling out the forgetting-factor confound. Warmup's advantage is attributable to the priors, not to γ=0.997 vs 0.995
- Warmup eliminates catastrophic cold-start failures: 0/20 across all regimes vs up to 8/20 for cold-start conditions
- The warmup benefit is transient by design: geometric forgetting (γ=0.997) replaces priors within ~333 effective-memory steps
- **Caveat**: these gains assume the prior is directionally correct. See [prior mismatch analysis](../prior_mismatch/README.md) for a sensitivity study
