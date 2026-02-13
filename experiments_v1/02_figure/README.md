# Experiment 02: Feature Distribution Shift Analysis

**Purpose**: Answer a critical question for any user of BanditGPT: *will the shipped warm-start priors still be useful when the library encounters real user traffic?* We quantify covariate shift between the warmup prior (RouteLLM battles — what the library's PCA and LinUCB priors are built from) and the deployment distribution (LMSYS evaluation — simulating new user traffic) using PSI, KS tests, and bootstrap confidence intervals.

**Type**: Statistical analysis establishing the practical motivation for the library's adaptive routing capabilities (online learning via LinUCB, safety via Corralling).

## Overview

This experiment quantifies distribution shift between the warmup prior and deployment data through comprehensive statistical analysis:
- **Warmup Prior**: RouteLLM battle data (all 80K unique prompts — the complete data used for PCA + LinUCB priors)
- **Deployment**: LMSYS dev/holdout prompts (1,871 unique prompts: 1,121 dev + 750 holdout, deduplicated) — the evaluation data that simulates new user traffic

---

### Connection to the Paper and Library

**Motivation from Figure 1:** Figure 1 established that semantic structure makes routing learnable — the PCA features carry a genuine task-difficulty signal. But a critical question remains for anyone deploying the library: **Does the warmup prior distribution (RouteLLM battles) match real deployment traffic?** If it doesn't, the shipped priors may assign suboptimal routing probabilities, degrading the Day 1 advantage that warm-starting provides.

This experiment provides a quantitative answer: shift *does* exist (PSI = 0.225, "significant"), but it is practically small (Cohen's d = 0.33). This "sweet spot" motivates the library's two-part design: (1) ship strong priors for Day 1 intelligence, and (2) adapt online because the priors are imperfect. Table 2 validates that this design works in practice.

---

## Research Questions

1. **How large is the distribution shift?** Quantified via PSI with bootstrap confidence intervals
2. **Is the shift statistically significant?** Validated with Kolmogorov-Smirnov test and Cohen's d effect size
3. **What is the semantic structure?** Analyzed via task category decomposition on ground truth reward gaps

## Methodology

### **CRITICAL: Uses Actual BanditRouter**

This experiment uses the **production BanditRouter** from `src/bandit_gpt/router.py` for feature extraction. This ensures:
- ✅ Analysis reflects actual routing system behavior
- ✅ Features are identical to those used in production
- ✅ No discrepancy between experiment and implementation
- ✅ Router testing is integrated into experimental validation

The router's `_build_routing_features()` method is used to extract features for each prompt, ensuring perfect consistency.

### Statistical Tests

We employ multiple statistical tests for robust validation:

1. **Population Stability Index (PSI)**: Industry-standard metric for distribution monitoring
   - Uses **quantile-based bins from the reference distribution** (industry standard, Yurdakul 2018)
   - Each reference bin has ~equal probability mass, ensuring stable PSI regardless of distribution shape
   ```
   PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
   ```
   - PSI < 0.1: No significant shift
   - 0.1 ≤ PSI < 0.2: Moderate shift
   - 0.2 ≤ PSI < 0.25: Significant shift
   - PSI ≥ 0.25: Substantial shift requiring adaptation

2. **Bootstrap Confidence Intervals**: 1000 resamples for PSI estimation

3. **Kolmogorov-Smirnov Test**: Non-parametric test for distribution equality

4. **Effect Size (Cohen's d)**: Standardized mean difference

### Feature Space

- **Feature Extraction**: `BanditRouter._build_routing_features()` (production code!)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional)
- **Dimensionality Reduction**: PCA with 32 components (35.14% variance explained)
- **Primary Axis**: PC1 (3.10% variance) captures main semantic variation

## Files

- `plot_distribution_shift_improved.py`: Main analysis script with full statistical validation
- `plot_distribution_shift.py`: Deprecated (see improved version)
- `results/distribution_shift_pc1.png`: Main visualization (300 DPI)
- `results/distribution_shift_pc1_hires.png`: High-resolution version (600 DPI)
- `results/distribution_shift_summary.json`: All metrics in machine-readable format
- `figure_distribution_shift.tex`: LaTeX for paper
- `CITATIONS.bib`: Bibliography entries

## Usage

### Running the Analysis

```bash
python3 experiments_v1/02_figure/plot_distribution_shift_improved.py
```

### Prerequisites

1. **PCA Model**: Pre-trained PCA model must exist
   ```bash
   # If not available, train it:
   python3 scripts/train_pca_from_routellm.py
   ```

2. **Data Files**: 
   - Source data: `src/bandit_gpt/data/offline_dataset/dev_rewards_2models.jsonl.gz`
   - Source data: `src/bandit_gpt/data/offline_dataset/holdout_rewards_2models.jsonl.gz`
   - RouteLLM data: `src/bandit_gpt/data/offline_dataset/routellm_battles_rewards.jsonl`

### Configuration

Default settings:
- **Warmup prior data**: All 80K RouteLLM battle prompts (all unique, no duplicates) — the complete distribution the PCA and LinUCB priors were trained on. No subsampling, no selection bias.
- **Deployment data**: LMSYS dev + holdout → 1,871 unique prompts after deduplication (1,121 dev + 750 holdout)
- **PSI bins**: 20 quantile-based bins from the reference distribution (industry standard)
- **Bootstrap samples**: 1,000 (for confidence intervals)
- **KDE bandwidth**: Scott's rule (data-driven, per scipy.stats.gaussian_kde)
- **PCA model**: 32 components pre-trained on RouteLLM data

## Output

### Figure Components

**Top Panel: Overall Distribution Comparison**
- Blue curve: Warmup Prior (RouteLLM battles — what the library's PCA and priors are built from)
- Red curve: Deployment (LMSYS evaluation — simulating new user traffic)
- Dashed lines: Distribution means
- Title includes: PSI with 95% CI, PC1 variance explained

**Bottom Panel: Task Category Decomposition (on Prior/RouteLLM data)**
- Green curve: Mixtral-Sufficient tasks (Gap ≤ 0, i.e., Mixtral wins or ties; 32% of prior data)
- Purple curve: GPT-4-Turbo-Required tasks (Gap = +1, i.e., GPT-4-Turbo wins; 68% of prior data)
- Based on discrete ground truth reward gaps: Gap = R_GPT-4-Turbo − R_Mixtral ∈ {-1, 0, +1}
- **Note:** These are NOT distinct clusters — >99% range overlap, Cohen's d = 0.23 (small effect)

### Saved Files
1. `results/figure2_distribution_shift.png` - Main figure (300 DPI)
2. `results/figure2_distribution_shift_hires.png` - High-res (600 DPI)
3. `results/distribution_shift_summary.json` - All metrics including:
   - PSI with confidence intervals
   - KS test statistics
   - Cohen's d effect sizes (overall + task categories)
   - Sample prompts from each task category
   - PCA statistics and sensitivity analysis

## Key Results

### Distribution Shift Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **PSI** | 0.225 (95% CI: [0.194, 0.285]) | Significant shift (quantile bins, full 80K reference) |
| **Cohen's d** | 0.33 | Small but reliable effect size |
| **KS Statistic** | 0.119 (p < 10⁻²²) | Distributions significantly different (expected at large N) |
| **Mean Shift** | +0.060 | Deployment right-shifted relative to prior |

### Task Category Decomposition (Prior/RouteLLM)

Because battle rewards are discrete (win=1, loss=0), Gap ∈ {-1, 0, +1}:

| Category | Threshold | Maps to | Proportion | Mean Gap |
|----------|-----------|---------|-----------|----------|
| **Mixtral-Sufficient** | Gap ≤ 0 | Mixtral wins or ties | 32% | -0.30 |
| **GPT-4-Turbo-Required** | Gap = +1 | GPT-4-Turbo wins | 68% | 1.00 |

**Important:** These task categories overlap extensively (>99% range overlap, Cohen's d = 0.23). They are NOT distinct clusters — PC1 captures a weak but genuine gradient in model-task compatibility.

### Implications for the Library

1. **The shipped priors are a strong but imperfect initialization.** PSI = 0.225 (significant) means deployment prompts are measurably different from what the priors were trained on. Without adaptation, the router will slowly degrade — but the priors remain a *far* better starting point than cold-start (62% regret reduction, Table 1).
2. **Online learning is necessary, not optional.** The shift (d = 0.33) is too large to ignore but too small to require prior replacement. LinUCB's continuous updates correct for prior miscalibration as deployment data accumulates — the library's default configuration already does this.
3. **Corralling provides safety guarantees.** If the prior happens to be worse than what the shift metrics suggest (e.g., for a particular user population), Corralling bounds worst-case regret. Table 2 validates this.
4. **The feature pipeline carries real routing signal.** Despite extensive overlap, the PCA features distinguish Mixtral-Sufficient from GPT-4-Turbo-Required prompts (d = 0.23). This weak per-prompt signal compounds over many bandit rounds into effective routing decisions.
5. **Shift is concentrated where it matters.** Leading PCs (between-domain variation) show the strongest shift; higher PCs (within-domain variation) are stable. The bandit's adaptation effort is focused on the most informative dimensions.

## Statistical Foundation

### Why PC1?

The first principal component:
1. **Captures most variance**: 3.1% of total variance (384D → 1D)
2. **Carries routing-relevant signal**: Task categories show a weak but statistically reliable separation along PC1 (d = 0.23)
3. **Robust to noise**: PCA filters out high-frequency noise
4. **Interpretable**: Mixtral-Sufficient tasks centered at +0.022, GPT-4-Turbo-Required at −0.016

### Why PSI?

PSI is industry-standard for production ML:
1. **Model-agnostic**: Works with any distribution
2. **Interpretable**: Clear thresholds (0.1, 0.2)
3. **Actionable**: Directly informs retraining decisions
4. **Efficient**: Fast to compute, suitable for monitoring

### Statistical vs. Practical Significance

With N > 10,000 samples, almost any nonzero distributional difference reaches statistical significance. We therefore emphasize **effect sizes** over p-values:
- **Overall shift**: Cohen's d = 0.35 (small effect) — reliable but modest
- **Task categories**: Cohen's d = 0.23 (small effect) — weak per-prompt signal
- **P-values** (KS: p < 10⁻²², t-test: p < 10⁻²⁵) confirm non-zero differences but are expected at this sample size

## Paper Integration

Figure 2 occupies the pivotal position in the paper's narrative arc — it establishes the *problem* that the library's core capabilities are designed to solve:

| Experiment | Role | Key Finding |
|-----------|------|-------------|
| **Figure 1** | Features generalize | PCA captures task-difficulty structure on unseen data |
| **Figure 2** (this) | Generalization is imperfect | Significant distribution shift between priors and deployment |
| **Table 2** | The library handles it | Corralling + LinUCB recover from prior miscalibration |

**Why this matters for the reader:** Without Figure 2, the paper's claim that adaptive routing is *necessary* rests on theoretical arguments alone. This experiment provides the first quantitative evidence that distribution shift is a practical concern for LLM routing — not merely a theoretical risk — and that it arises even within the relatively homogeneous domain of LLM evaluation benchmarks. The implication: if shift occurs here, it will be larger in real deployments where user populations, languages, and task mixes vary more widely.

**Why this matters for the library user:** The experiment directly informs configuration decisions. The priors are valuable (use them), but adaptation is necessary (enable LinUCB, consider Corralling for safety-critical deployments), and monitoring is prudent (track PSI via `router.stats()`).

### Figure Reference

Include as Figure 2 in paper with caption describing:
- Top: Overall distribution comparison with PSI, Cohen's d, and bootstrap CI
- Bottom: Task category decomposition on prior data, with explicit overlap reporting
- Forward-reference to Table 2 for performance consequences and Corralling validation

## Future Enhancements

Potential extensions for deeper analysis:

1. **Multi-dimensional PSI**: Compute PSI jointly on PC1-5 with variance-weighted aggregation
2. **Temporal drift**: Track PSI over time batches to detect concept drift
3. **Stratified analysis**: Compute PSI separately for easy/hard task clusters
4. **Sensitivity analysis**: Test robustness to different embedding models (MPNet, E5)
5. **Causal investigation**: Analyze why shift occurred (user population, time period, use cases)

## Technical Details

### Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimension**: 384
- **Normalization**: L2 normalization applied

### PCA Model
- **Components**: 32
- **Variance Explained**: 35.14% (cumulative)
- **PC1 Variance**: 3.10%
- **PC1-5 Variance**: 10.87%

### Statistical Parameters
- **PSI Bins**: 20 (binning for histogram)
- **Bootstrap Samples**: 1000 (for confidence intervals)
- **Bootstrap Method**: Resampling with replacement
- **Significance Level**: α = 0.05 (95% CI)

### Thresholds
- **Reward Structure**: Battle rewards are discrete (win = 1, loss = 0), so Gap ∈ {-1, 0, +1}
- **Mixtral-Sufficient**: Gap ≤ 0.3 → captures discrete values {-1, 0} (Mixtral wins or ties)
- **GPT-4-Turbo-Required**: Gap > 0.6 → captures discrete value {+1} (GPT-4-Turbo wins)
- **Gap Definition**: Gap = R_GPT-4-Turbo − R_Mixtral

## Troubleshooting

### Issue: PCA file not found

```bash
# Train PCA model first:
python3 scripts/train_pca_from_routellm.py
```

### Issue: No prompts loaded

Check that data files exist:
```bash
ls -lh src/bandit_gpt/data/offline_dataset/
```

### Issue: KDE fails (too few samples)

Increase `max_samples` in the script:
```python
lmsys_prompts = load_lmsys_evaluation_prompts(dev_file, holdout_file, max_samples=20000)
```

### Issue: Memory error

Reduce batch size:
```python
pc1_values = project_to_pc1(prompts, pca_file, batch_size=32)
```

---

## What's Next?

This experiment establishes that the shipped priors are **strong but imperfect**: significant distribution shift exists (PSI = 0.225, Cohen's d = 0.33), but the priors remain valuable as an initialization. The critical follow-up question is: *what are the performance consequences, and can the library handle them?*

**Table 2 answers both questions:**
1. **Performance consequences**: Measures routing quality under the distribution mismatch documented here
2. **Recovery via Corralling**: Validates that adaptive routing overcomes prior miscalibration with formal regret guarantees
3. **Learning rate trade-offs**: Shows how users should configure the adaptation speed (conservative η=0.1 for stability vs. aggressive η=1.0 for faster recovery)

**The library's thesis, validated across three experiments:**
- **Figure 1**: The feature pipeline works — PCA captures real task structure
- **Figure 2**: The priors are imperfect — distribution shift exists in practice
- **Table 2**: The adaptation works — LinUCB + Corralling recover from prior miscalibration

Together: *ship strong priors, then adapt.*

## Contact

For questions about this experiment:
- See main project README
- Check `experiments_v1/README.md` for experiment guidelines

