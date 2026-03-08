# Figure 1: Contextual Sensitivity across K=3 Models

This experiment establishes the empirical motivation for contextual routing: **per-model quality is a function of prompt context, and the sensitivity to context differs across models.**

If all models responded identically to prompt features (shared contextual slope), a static policy would suffice. This experiment shows that slopes are heterogeneous—the optimal model varies with the prompt—giving the bandit meaningful signal to exploit.

## Core Question

> Does prompt context affect model quality differently across models, and do the router's PCA features capture this heterogeneity?

## Method

We evaluate a **K=3 portfolio** spanning three cost tiers on held-out prompts:

| Tier | Model |
|------|-------|
| **Cheap** | Llama-3.1-8B |
| **Mid** | Gemini-2.5-Flash |
| **Expensive** | GPT-4.1 |

The portfolio is loaded from `data_collection/config/models_k3.json` via `bandit_gpt.config.K3_MODELS_PATH`.

### Feature Pipeline (matches production router)

Prompts are encoded with a sentence transformer, projected to PCA dimensions, and **whitened** — each component is scaled by `1/√(explained_variance_j)` so that all components have approximately unit variance under the PCA training distribution. This matches the `embed_prompt()` / `FeatureService` pipeline used by all downstream experiments and the production router. Whitening ensures the LinUCB isotropic prior (`A₀ = λI`) is well-matched to the feature scale.

### Regression Specification

For each model *m*, we fit OLS:

```
r_{i,m} = α_m + γ_m · PC1_std_i + ε_{i,m}
```

where `PC1_std` is the whitened first principal component (further standardised to zero mean and unit variance on the holdout), and `γ_m` is the **contextual slope**—the rate at which model *m*'s reward changes per standard deviation of whitened PC1. If all `γ_m` are equal, a static policy suffices; if they differ significantly, contextual routing can exploit the heterogeneity.

### Likelihood-Ratio Test

A formal LR test compares shared-slope H0 against per-model-slope H1 using up to d=6 PCA components (clamped to the number available), with (K−1) × d degrees of freedom.

### Bootstrap Confidence Intervals

Per-model slopes are estimated with 10,000 prompt-level bootstrap resamples. A slope is declared significant if its 95% CI excludes zero.

### Data Independence

| | PCA Training | Holdout Evaluation |
|---|---|---|
| **Source** | RouteLLM battles (HuggingFace) | banditGPT evaluation pool |
| **Size** | ~46K prompts | holdout split |
| **Overlap** | None — disjoint by provenance |

## Figure Structure

- **Panel A**: Per-model OLS regression lines of reward (%) vs standardised whitened PC1, with 95% confidence bands. Lines fan and cross, showing no single model dominates the full feature space.
- **Panel B**: Forest plot of per-model contextual slopes γ_m (on whitened PC1) with bootstrap 95% CIs. Stars mark significance.

## Files

| File | Purpose |
|------|---------|
| `plot_figure1.py` | Main analysis and figure generation |
| `results_explanation.tex` | Paper results section |
| `validation_methodology.tex` | Detailed methodology |
| `figure_1_caption.tex` | Figure caption (included by paper) |

## Reproducibility

```bash
# 1. Train PCA (if not present)
python3 scripts/train_pca_from_routellm.py --n-components 32

# 2. Generate Figure 1 (uses whitened PCA features)
python3 experiments/01_figure/plot_figure1.py
```

Output: `experiments/01_figure/results/figure1_k3_contextual.png`

The script automatically applies PCA whitening to match the production router pipeline. If the PCA artifact has `whiten=True`, the built-in whitening is used; otherwise external whitening (`1/√(explained_variance)`) is applied. Diagnostic output confirms which path is taken.

## Limitations

1. **Stratified holdout.** The holdout was constructed with a difficulty dimension derived from oracle rewards, enriching for prompts with clear model differences. Signal *existence* is robust; *magnitude* in deployment depends on the prompt distribution.

2. **PC1 is a lower bound.** The router uses all PCA components. Higher components contribute additional signal. The PC1-only visualisation is conservative.

3. **Prompt-type mechanism.** The correlation reflects prompt-type variation that correlates with model preference. The PCA captures *prompt type → model preference*, not an abstract routing feature.

## Connection to banditGPT

This experiment validates the router's **"eyes"** (PCA features). Subsequent experiments validate the **"brain"** (bandit learning):

| Experiment | What it tests |
|-----------|---------------|
| **Figure 1** (this) | Features predict heterogeneous preference → routing is possible |
| **Figure 3** | Prequential evaluation → online learning performance |
| **Figure 4** | Multi-model Pareto frontier → production cost-quality tradeoffs |

## Dependencies

- `sentence-transformers` — Prompt embeddings
- `scikit-learn` — PCA
- `matplotlib` — Visualisation
- `scipy` — Chi-squared distribution, t-distribution
- `numpy` — Numerical computation
- `joblib` — PCA model loading
