# Appendix: Reward Signal Robustness (Three-Judge Validation)

Validates that the paper's conclusions are robust to evaluator choice.

## Core Question

> Would the paper's conclusions change with a different evaluator?

## Setup

- **Primary judge**: DeepSeek-R1 (used in all main experiments)
- **Supplementary judges**: GPT-4.1-mini (OpenAI), Claude-3.7-Sonnet (Anthropic)
- **Subset**: Stratified 2,000 prompts × 3 models = 6,000 scored pairs per judge
- **Rubric**: Identical v3 continuous rubric for all judges
- **Provider separation**: No judge shares a provider with any routed model

## Argument Structure (for KDD reviewer)

### 1. All judges agree on the expected reward ordering

Every judge ranks **Gemini-Pro > Mistral-Large > Llama-8B** with every
pairwise difference significant at p < 10^{-5}. The bandit's converged
policy depends on this ordering, not on per-prompt accuracy.

### 2. R1's oracle captures 97.5% of other judges' oracle reward

Following R1's routing and evaluating by OTHER judges captures 97.4–97.5%
of their oracle. The reverse direction is worse (95.8–95.9%), making R1
the most consensus-compatible single judge.

### 3. The paper's claims are relative, not absolute

All method comparisons (BanditGPT vs baselines) are conducted within a
single judge. Judge choice cannot flip the relative ordering of methods;
it only affects effect-size magnitude.

### 4. Disagreement is concentrated where it doesn't matter

Per-prompt best-model agreement is ~50%. But gap-conditioned analysis shows:

| R1 gap range | n | Kendall W |
|-------------|-----|-----------|
| [0.00, 0.05) | 603 | 0.17 |
| [0.20, 0.30) | 184 | 0.57 |
| [0.30, 1.00) | 552 | 0.71 |

### Honest limitations

- Per-prompt best-model agreement is ~50% (genuine LLM-as-judge noise)
- R1 sees the largest oracle lift (0.031 vs 0.016–0.020), so main-paper
  effect sizes are an upper bound on absolute routing benefit
- All judges share the same rubric; correlated blind spots are possible

## Reproduction

```bash
# Figures (no API calls)
python experiments/appendix/judge_robustness/generate_figure.py

# Routing-level diagnostics (no API calls)
python experiments/appendix/judge_robustness/diagnostic_agreement.py
python experiments/appendix/judge_robustness/diagnostic_conclusion_sensitivity.py
```

## Outputs

```
results/
├── judge_robustness.pdf          # Two-panel scatter
├── judge_bland_altman.pdf        # Bland-Altman agreement
├── judge_gap_distribution.pdf    # Gap distribution overlay
└── judge_robustness_summary.json # All metrics (machine-readable)
```

## LaTeX

```latex
\input{experiments/appendix/judge_robustness/results_discussion.tex}
```
