# Figure 1: Contextual Sensitivity across K=10 Models

This experiment establishes the empirical motivation for contextual routing: **per-model quality is a function of prompt context, and the sensitivity to context differs across models.**

If all models responded identically to prompt features (shared contextual slope), a static policy would suffice. This experiment shows that slopes are heterogeneous—the optimal model varies with the prompt—giving the bandit meaningful signal to exploit.

## Core Question

> Does prompt context affect model quality differently across models, and do the router's PCA features capture this heterogeneity?

## Method

We evaluate a **K=10 portfolio** spanning three cost tiers on **N=750 held-out prompts**:

| Tier | Models |
|------|--------|
| **Cheap** | Llama-3.1-8B, Mixtral-8x7B, Gemma-3-27B |
| **Mid** | Haiku-4.5, DeepSeek-V3, Gemini-2.5-Flash, Llama-4-Maverick |
| **Expensive** | Claude-Sonnet-4, GPT-4-Turbo, GPT-4.1 |

### Feature Pipeline (matches production router)

Prompts are encoded with a sentence transformer (384D), projected to 32 dimensions via PCA, and **whitened** — each component is scaled by `1/√(explained_variance_j)` so that all components have approximately unit variance under the PCA training distribution. This matches the `embed_prompt()` / `FeatureService` pipeline used by all downstream experiments and the production router. Whitening ensures the LinUCB isotropic prior (`A₀ = λI`) is well-matched to the feature scale.

### Regression Specification

For each model *m*, we fit OLS:

```
r_{i,m} = α_m + γ_m · PC1_std_i + ε_{i,m}
```

where `PC1_std` is the whitened first principal component (further standardised to zero mean and unit variance on the holdout), and `γ_m` is the **contextual slope**—the rate at which model *m*'s reward changes per standard deviation of whitened PC1. If all `γ_m` are equal, a static policy suffices; if they differ significantly, contextual routing can exploit the heterogeneity.

### Likelihood-Ratio Test

A formal LR test compares shared-slope H0 against per-model-slope H1 using d=6 PCA components, with (K−1) × d = 54 degrees of freedom.

### Bootstrap Confidence Intervals

Per-model slopes are estimated with 10,000 prompt-level bootstrap resamples. A slope is declared significant if its 95% CI excludes zero.

### Null Baseline

100 independent random orthonormal projections (384 → 2, QR-decomposed). If the router PCA's |ρ| falls within this null distribution, the signal is artifactual.

### Data Independence

| | PCA Training | Holdout Evaluation |
|---|---|---|
| **Source** | RouteLLM battles (HuggingFace) | banditGPT evaluation pool |
| **Size** | 80K prompts | 750 prompts |
| **Overlap** | None — disjoint by provenance |

## Key Findings

1. **Slopes are heterogeneous (LR test).** The per-model-slope model fits significantly better than the shared-slope null (χ² = 741.4, df = 54, p < 10⁻¹⁵). Context affects models differently.

2. **All ten models are context-sensitive.** Bootstrap 95% CIs (10,000 resamples) exclude zero for all ten slopes. Magnitudes range from |γ| = 0.026 (GPT-4.1, least sensitive) to |γ| = 0.242 (GPT-4-Turbo, most sensitive) — a factor of ~9×.

3. **Oracle gain correlates with PC1.** Spearman ρ between per-prompt oracle gain and whitened PC1 is significantly positive, confirming that the router's features identify where routing helps most.

4. **Signal exceeds random projections.** Router PCA |ρ| exceeds all 100 random projections.

## Figure Structure

- **Panel A**: Per-model OLS regression lines of reward (%) vs standardised whitened PC1, with 95% confidence bands. Lines fan and cross, showing no single model dominates the full feature space.
- **Panel B**: Forest plot of per-model contextual slopes γ_m (on whitened PC1) with bootstrap 95% CIs. All ten CIs exclude zero; stars mark significance.

## Files

| File | Purpose |
|------|---------|
| `plot_figure1.py` | Main analysis and figure generation |
| `results_explanation.tex` | Paper results section (K=10) |
| `validation_methodology.tex` | Detailed methodology |
| `figure_1_caption.tex` | Figure caption (included by paper) |

## Reproducibility

```bash
# 1. Train PCA (if not present)
python3 scripts/train_pca_from_routellm.py --n-components 32

# 2. Generate Figure 1 (uses whitened PCA features)
python3 experiments/01_figure/plot_figure1.py
```

Output: `experiments/01_figure/results/figure1_k10_contextual.png`

The script automatically applies PCA whitening to match the production router pipeline. If the PCA artifact has `whiten=True`, the built-in whitening is used; otherwise external whitening (`1/√(explained_variance)`) is applied. Diagnostic output confirms which path is taken.

## Limitations

1. **Stratified holdout.** The holdout was constructed with a difficulty dimension derived from oracle rewards, enriching for prompts with clear model differences. Signal *existence* is robust; *magnitude* in deployment depends on the prompt distribution.

2. **PC1 is a lower bound.** The router uses all 32 whitened PCA components. Higher components contribute additional signal (e.g., PC3 predicts Mixtral, PC31 predicts Llama-4-Maverick). The PC1-only visualisation is conservative.

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
