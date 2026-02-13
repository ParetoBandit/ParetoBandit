# Figure 1: Routing Signal Validation

This experiment validates that the PCA feature extraction used by the production router (`FeatureService` in `router.py`) captures genuine model preference signal. It is the first experiment in the paper and establishes a necessary precondition for learned routing: that different prompts have different model preferences, and the router's features can detect this.

## Why This Experiment Exists

**For a KDD reviewer:** Before claiming that a learned router outperforms static model selection, we must show that the feature space contains routing-relevant signal. This experiment provides that evidence through a controlled three-condition comparison with a null baseline, using the exact feature extraction pipeline deployed in the production router.

**For a library user:** Before deploying BanditGPT, you want to know: (1) does the router's PCA actually help, or is it noise? (2) what fraction of my traffic will benefit from routing? (3) is the evaluation fair? This experiment answers all three.

## Experimental Design

### Three-Condition Comparison

We project N=750 held-out prompts through three feature extraction conditions using the same sentence encoder (`all-MiniLM-L6-v2`) and test whether PC1 clusters have different win/tie/loss distributions:

| Condition | Training Data | Purpose |
|-----------|--------------|---------|
| **Router PCA (domain-adapted)** | 80K RouteLLM battles (independent dataset) | Tests the production router's actual features |
| **Generic PCA (C4)** | 100K C4 web text | Unbiased baseline — no routing connection |
| **Random projection** | Random orthonormal matrix (seed=42) | Null baseline — how much structure does chance find? |

If the random projection shows similar effect sizes to the router PCA, the finding is a projection artifact. If the router PCA greatly exceeds both baselines, it captures genuine task-relevant variance.

### Data Independence (No Contamination)

The PCA training data and holdout evaluation data are **entirely independent datasets** — disjoint by provenance, not by post-hoc filtering:

| | PCA Training Data | Holdout Evaluation Data |
|---|---|---|
| **Source** | `routellm/gpt4_judge_battles` (HuggingFace) | LMSYS Chatbot Arena general prompt pool |
| **Size** | 80K battle prompts | 750 prompts |
| **Collection** | RouteLLM pairwise battle curation | LMSYS Arena general sampling |
| **Sampling period** | Different | Different |
| **Prompt population** | Battle-format prompts | General user prompts |
| **Model pair** | Mixtral vs GPT-4-Turbo | Mixtral vs GPT-4-Turbo |

The PCA has never seen any evaluation prompt because the two collections were produced independently. No decontamination or exclusion step is needed.

### Statistical Tests

Since reward gaps are discrete (win=+1, tie=0, loss=-1), we use categorical tests as primary:

| Metric | Type | Why |
|--------|------|-----|
| **Cramer's V** | Primary effect size | Appropriate for 2x3 contingency tables |
| **Odds ratio** (Mixtral win) | Primary effect size | Interpretable: "how much more likely is Mixtral to win in High PC1?" |
| **Risk difference** | Primary effect size | Absolute change in win probability |
| **Permutation test** (n=10,000) | Significance | Distribution-free, no large-sample assumptions |
| Chi-squared | Significance | Standard contingency table test |
| Mann-Whitney U | Supplementary | Non-parametric ordinal test |
| Cohen's d | Supplementary | For comparability only — approximate on discrete data |

## Files

### Analysis Scripts

- `plot_figure1.py` — Main analysis and visualization
  - Three-condition comparison (router PCA, generic PCA, random projection)
  - Permutation test for distribution-free p-values
  - Categorical effect sizes (Cramer's V, odds ratio, risk difference)
  - Threshold stability analysis (Cramer's V sweep)
  - Power analysis (Monte Carlo)
  - TF-IDF cluster content analysis
  - Generates Figure 1 (Panel A: scatter, Panel B: grouped bar chart)

- `plot_lmsys_1M_pca.py` — Production-scale spatial analysis
  - 594,199 Chat-1M prompts (317x scale increase)
  - Estimates production routable fraction (~5.9%)
  - Note: no reward labels — spatial structure only

- `download_1M_dataset.py` — Dataset acquisition

### LaTeX Files

- `results_explanation.tex` — Main results section (methodology, findings, practical implications)
- `validation_methodology.tex` — Comprehensive validation (three-condition design, permutation test, threshold stability, high-D analysis)
- `figure_1_caption.tex` — Figure caption
- `figure_1M_analysis.tex` — Scale analysis with production deployment guidance

## Key Findings

### What the Router's PCA Captures

The router PCA separates prompts into two regions of the PC1 axis with significantly different outcome distributions:

- **Low PC1 (~81%, n=606):** Dominated by ties (~82%). Both models perform comparably. This is where cost savings come from — the cheaper model can serve these prompts.
- **High PC1 (~19%, n=144):** ~62% Mixtral wins, ~33% ties, ~5% GPT-4T wins. Strong cheaper-model preference.

The finding: **a minority of prompts show strong cheaper-model preference, while the majority show no model differentiation.** For routing, this means:
1. Most prompts can be served by the cheaper model with comparable quality (cost savings).
2. The router's value is avoiding quality loss on the minority where the stronger model genuinely matters.

### Three-Condition Results

| Condition | Cramer's V | OR (Mixtral) | Risk diff | Perm. p |
|-----------|-----------|-------------|-----------|---------|
| Router PCA (domain-adapted) | **0.667** | **54.3** | **+59.0%** | < 0.0001 |
| Generic PCA (C4) | 0.081 | 4.5 | +12.2% | 0.0771 |
| Random projection | 0.095 | 5.1 | +12.6% | 0.0331 |

*Router PCA captures 7.0x more signal than chance (random projection).*

### Production Estimate

| Metric | Holdout (N=750) | 1M Dataset (N=594,199) |
|--------|-----------------|------------------------|
| High PC1 (%) | 19.2% | **5.9%** |
| Low PC1 (%) | 80.8% | **94.1%** |
| **Reward Labels** | **Available** | **Not available** |

**For deployment planning:** expect ~5-6% of general-purpose traffic in the clearly-routable region. The majority of cost savings come from the tie-dominated majority, where the cheaper model suffices.

## Reproducibility

### Step 1: Train Router PCA (if not already present)

```bash
# Train PCA on RouteLLM battles (independent from holdout data)
python3 scripts/train_pca_from_routellm.py --n-components 32
```

This produces `src/artifacts/pca_32.joblib`.

### Step 2 (Optional): Train Generic PCA

```bash
# Train PCA on C4 corpus (requires internet access for HuggingFace download)
python3 scripts/train_pca_generic.py --n-components 32
```

This produces `src/artifacts/pca_32_generic.joblib`.

### Step 3: Generate Figure 1

```bash
python3 experiments_v1/01_figure/plot_figure1.py
```

The script:
- Uses the standard router PCA (`pca_32.joblib`)
- Runs all three conditions (router PCA, generic PCA if available, random projection)
- Prints comparison table and full statistical analysis
- Saves figure to `results/`

### Scale Analysis (Optional)

```bash
python3 experiments_v1/01_figure/download_1M_dataset.py
python3 experiments_v1/01_figure/plot_lmsys_1M_pca.py
```

## Practical Guidance for Users

### Before Deployment

1. **The router's PCA works.** It captures genuine routing signal, validated against a null baseline.
2. **Expect ~5-6% clearly-routable prompts** in general-purpose traffic. Domain-specific traffic (heavy on instruction-following/structured output) may have more.
3. **Most cost savings come from ties.** ~82% of prompts show no model preference — the cheaper model can serve them.

### Running Your Own Evaluation

1. **The shipped PCA is clean.** It was trained on an independent dataset (RouteLLM battles), separate from the evaluation data.
2. **Retrain PCA for your domain.** If your traffic differs from general chatbot queries, PCA trained on your prompts will capture more domain-relevant variance.
3. **Monitor routable fraction.** Use `FeatureService.extract_features()` to project your prompts and estimate what fraction falls in the high-signal region.

### What This Does NOT Tell You

- **Whether routing actually saves money** — that's tested in the routing evaluation (Table 2).
- **Whether the bandit learns safely under distribution shift** — that's tested in Figure 2.
- **What happens with 3+ models** — that's tested in Figure 4.

## Connection to Overall Contribution

**This experiment establishes:** The router's feature space contains genuine model preference signal — a necessary condition for learned routing.

**What's next:**
1. **Distribution shift** (Figure 2): Does training data match deployment?
2. **Routing evaluation** (Table 2): Does the bandit exploit this signal for practical gains?
3. **Cold-start defense** (Figure 3): How does Corralling handle model mismatch?

## Dependencies

- `sentence-transformers` — Prompt embeddings (same model as production router)
- `scikit-learn` — PCA, clustering, silhouette score
- `matplotlib` — Visualization
- `scipy` — Statistical tests
- `numpy` — Numerical computation
