# Figure 1: Model Preference Heterogeneity

This experiment establishes the empirical motivation for contextual routing: **model preference varies by prompt, and the router's features predict this variation.**

If model preference were uniform (one model always wins), a static policy would be optimal and learned routing would be unnecessary. This experiment shows that is not the case — the PCA features used by `FeatureService.extract_features()` correlate with model preference, giving the bandit meaningful signal to learn from.

## Core Question

> Do the router's features predict which model will perform better on a given prompt?

## Method

We compute the **Spearman rank correlation** between the router PCA's first principal component (PC1) and the reward gap (GPT-4-Turbo minus Mixtral) on N=750 held-out prompts.

This directly tests the question the bandit needs answered — it matches how BanditGPT actually works (LinUCB regresses features against reward), without introducing artificial clustering, thresholds, or contingency tables.

### Why Spearman?

| Property | Why it matters |
|----------|---------------|
| **Rank-based** | Handles discrete outcomes ({-1, 0, +1}) naturally |
| **No distributional assumptions** | Valid for any monotonic relationship |
| **Directly interpretable** | ρ measures how well features rank prompts by model preference |
| **Matches the router** | The bandit does regression on features → this tests the same relationship |

### Null Baseline

100 independent random orthonormal projections (384 → 2, QR-decomposed). Each projection provides a |ρ| value, forming the null distribution. If the Router PCA |ρ| falls within this distribution, the signal is artifactual.

### Data Independence

| | PCA Training | Holdout Evaluation |
|---|---|---|
| **Source** | RouteLLM battles (HuggingFace) | LMSYS Chatbot Arena |
| **Size** | 80K prompts | 750 prompts |
| **Overlap** | None — disjoint by provenance |

## Key Findings

1. **Features predict preference.** Spearman ρ is significant (p < 0.0001), confirming that PC1 correlates with model preference direction.

2. **Signal exceeds all random baselines.** Router PCA |ρ| exceeds all 100 random projections.

3. **Most prompts are ties.** The majority of prompts show no strong model preference — the cheaper model can serve them with comparable quality. This is the primary source of cost savings.

4. **PC1 is a lower bound.** The router uses all 32 PCA components. The full feature vector captures at least as much predictive signal as PC1 alone.

## Figure Structure

- **Panel A**: PC1 vs Reward Gap scatter — shows features predict model preference
- **Panel B**: Router PCA |ρ| vs 100 random projections — shows signal is real

## Files

| File | Purpose |
|------|---------|
| `plot_figure1.py` | Main analysis and figure generation |
| `results_explanation.tex` | Paper results section |
| `validation_methodology.tex` | Detailed methodology |
| `figure_1_caption.tex` | Figure caption |

## Reproducibility

```bash
# 1. Train PCA (if not present)
python3 scripts/train_pca_from_routellm.py --n-components 32

# 2. Generate Figure 1
python3 experiments/01_figure/plot_figure1.py
```

## Limitations

1. **Stratified holdout.** The holdout was constructed with a difficulty dimension derived from oracle rewards, enriching for prompts with clear preferences. Signal *existence* is robust; *magnitude* in deployment depends on the prompt distribution.

2. **Two-model topology.** This tests Mixtral vs GPT-4-Turbo specifically. A different model pair may show different signal.

3. **Prompt-type mechanism.** The correlation reflects prompt-type variation that correlates with model preference (e.g., instruction-following templates). The PCA captures *prompt type → model preference*, not an abstract routing feature.

## Connection to BanditGPT

This experiment validates the router's **"eyes"** (PCA features). Subsequent experiments validate the **"brain"** (bandit learning):

| Experiment | What it tests |
|-----------|---------------|
| **Figure 1** (this) | Features predict preference → routing is possible |
| **Figure 3** | Corralling design → safety under distribution shift |
| **Figure 4** | Pareto frontier → production cost-quality tradeoffs |
| **Figure 6** | Catastrophic failure detection (appendix) |

## Dependencies

- `sentence-transformers` — Prompt embeddings
- `scikit-learn` — PCA
- `matplotlib` — Visualization
- `scipy` — Spearman correlation, Mann-Whitney U
- `numpy` — Numerical computation
