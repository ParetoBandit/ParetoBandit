# Composite Reasoning Score (CRS): Method Documentation

*For inclusion in KDD Paper - Evaluation & Metrics Section*

---

## Overview

To robustly evaluate and compare reasoning capabilities across models — even when not every model has results on every benchmark — we define a **Composite Reasoning Score (CRS)** as a latent variable aggregating performance on multiple complementary benchmarks. This serves as a principled, reproducible, and KDD-friendly "reasoning quality" metric.

---

## Benchmark Selection & Coverage

We select a diverse suite of reasoning-heavy benchmarks that together cover multiple reasoning dimensions:

| Benchmark | Field | Description | Coverage |
|-----------|-------|-------------|----------|
| **MATH-500** | `math_500` | Formal symbolic mathematics reasoning (algebra, number theory, geometry) with strict ground-truth correctness | 100% |
| **GPQA** | `gpqa` | Graduate-level science/engineering QA requiring deep conceptual reasoning (physics, chemistry, biology) | 99% |
| **HLE** | `hle` | Humanity's Last Exam — expert-level reasoning on questions designed to be extremely challenging | 100% |
| **AIME** | `aime` | American Invitational Mathematics Examination — competition-level multi-step problem solving | 87% |
| **Math Index** | `math_index` | Artificial Analysis composite mathematics score aggregating multiple math benchmarks | 73% |

This mix ensures coverage across:
- **Formal symbolic reasoning** (MATH-500, AIME)
- **Scientific reasoning** (GPQA)
- **Expert-level reasoning** (HLE)
- **Competition mathematics** (AIME, Math Index)

**Note:** We excluded `intelligence_index` from the benchmark set after observing near-zero residual variance (σ ≈ 0.08) during model fitting, which created funnel geometry and convergence issues in the Bayesian model. This benchmark was too perfectly correlated with the latent factor to provide independent information.

---

## Data Normalization and Directional Alignment

Let $y_{i,b}$ denote the raw score of model $i$ on benchmark $b$.

Because benchmarks are heterogeneous in scale and directionality, we apply the following preprocessing pipeline:

### 1. Scale Normalization

Each benchmark is scaled to a common 0-100 range:

| Benchmark | Raw Range | Scale Factor | Normalized Range |
|-----------|-----------|--------------|------------------|
| MATH-500 | 0.0 - 1.0 | ×100 | 0 - 100 |
| GPQA | 0 - 100 | ×1 | 0 - 100 |
| HLE | 0 - 0.5 | ×100 | 0 - 50 |
| AIME | 0 - 100 | ×1 | 0 - 100 |
| Math Index | 0 - 100 | ×1 | 0 - 100 |

### 2. Directional Alignment

For any benchmark where lower values indicate better performance (e.g., error rates), we apply a monotonic inversion:

$$y'_{i,b} = -y_{i,b}$$

All benchmarks are therefore placed on a "higher is better" scale prior to aggregation. In our current benchmark set, all metrics are already "higher is better."

### 3. Statistical Standardization

Each benchmark is independently standardized across all participating models using a z-score transformation:

$$z_{i,b} = \frac{y'_{i,b} - \mu_b}{\sigma_b}$$

where $\mu_b$ and $\sigma_b$ denote the mean and standard deviation of benchmark $b$, computed over all non-missing values.

If a benchmark exhibits zero or undefined variance, its standardized values are set to missing and it is excluded from the latent model.

---

## Latent Variable Model Specification

We assume that the observed standardized score $z_{i,b}$ arises from a benchmark-specific noisy observation of a latent reasoning factor $\theta_i$:

$$z_{i,b} \sim \mathcal{N}(\alpha_b + \lambda_b \theta_i, \sigma_b^2)$$

where:

- $\theta_i$: latent Composite Reasoning Score (CRS) for model $i$
- $\alpha_b$: benchmark-specific intercept
- $\lambda_b$: benchmark-specific loading (factor weight)
- $\sigma_b$: benchmark-specific residual standard deviation

### Prior Specification

We place the following priors:

$$\theta_i \sim \mathcal{N}(0, 1)$$
$$\alpha_b \sim \mathcal{N}(0, 1)$$
$$\lambda_b \sim \text{HalfNormal}(0.7)$$
$$\sigma_b \sim \text{HalfNormal}(1.0)$$

**Identifiability:** We constrain all loadings $\lambda_b$ to be positive via HalfNormal priors. This fixes the direction of the latent factor so that higher $\theta_i$ consistently indicates better reasoning performance across all benchmarks.

This formulation treats each benchmark as a noisy indicator of a common latent reasoning construct, while allowing for different signal strengths (loadings) and noise levels across benchmarks.

---

## Missing Benchmark Handling

Not all models are evaluated on all benchmarks. We treat missing benchmark entries as missing-at-random (MAR) and exclude them from the likelihood. Because inference occurs over the latent variable $\theta_i$, CRS remains identifiable and estimable even under partial benchmark coverage.

This enables fair cross-model comparison without requiring complete benchmark overlap. Models with fewer benchmarks will have wider credible intervals, appropriately reflecting greater uncertainty.

**Minimum coverage:** We require at least 3 benchmarks for inclusion, ensuring sufficient evidence for a reliable latent score estimate.

---

## Bayesian Inference and CRS Estimation

We perform posterior inference via Hamiltonian Monte Carlo (HMC) using PyMC, with the following settings:

- 4 chains
- 2,000 tuning steps
- 2,000 posterior draws per chain
- Target acceptance rate = 0.9
- Random seed = 123 (for reproducibility)

For each model $i$, we report:

- **Posterior mean** of $\theta_i$ → CRS estimate (z-score scale)
- **Posterior standard deviation** → uncertainty estimate
- **95% credible interval:** $[\theta_i^{2.5\%}, \theta_i^{97.5\%}]$
- **Normalized CRS** (0-100 scale via min-max transformation)

CRS is therefore reported with uncertainty, reflecting both benchmark noise and partial coverage.

---

## Learned Benchmark Loadings

The Bayesian model learns the relative importance of each benchmark from the data. Our fitted model produced the following loadings:

| Benchmark | Loading (λ) | Interpretation |
|-----------|-------------|----------------|
| AIME | 1.026 | Most informative for reasoning |
| Math Index | 1.011 | Highly informative |
| GPQA | 0.889 | Moderate signal |
| MATH-500 | 0.774 | Moderate signal |
| HLE | 0.754 | Moderate signal |

**Interpretation:** Competition mathematics (AIME) and the Math Index composite emerged as the strongest signals for the latent reasoning factor. This aligns with intuition — performance on difficult competition math problems is highly discriminating for reasoning capability.

---

## Convergence Diagnostics

We verify model convergence using standard Bayesian diagnostics:

| Metric | Criterion | Result |
|--------|-----------|--------|
| Divergences | < 1% | 2 (0.0%) ✓ |
| R-hat (all parameters) | < 1.01 | Max 1.01 ✓ |
| ESS bulk (theta) | > 400 | Min 853 ✓ |
| ESS tail (theta) | > 400 | Min 2029 ✓ |

Additionally, we perform:

1. **Z-score sanity checks:** Verify no benchmark has extreme outliers (|z| > 5) or insufficient coverage (< 10 observations)
2. **Benchmark coverage analysis:** Report per-benchmark availability
3. **Optional winsorization:** Cap extreme z-scores at ±3 if needed

---

## Results Summary

From our evaluation of 100 models (1 excluded for insufficient benchmarks):

| Statistic | Value |
|-----------|-------|
| Models with valid CRS | 100 |
| Mean CRS | 42.2 |
| Median CRS | 33.6 |
| Std Dev | 29.1 |
| Min | 0.0 |
| Max | 100.0 |

### Top 15 Models by CRS

| Rank | Model | CRS | 95% CI | z-score |
|------|-------|-----|--------|---------|
| 1 | Gemini 3 Pro Preview (high) | 100.0 | [84.6, 116.0] | 1.98 |
| 2 | GPT-5 (high) | 93.5 | [81.1, 105.9] | 1.76 |
| 3 | Claude Opus 4.5 (Reasoning) | 93.1 | [80.8, 105.4] | 1.75 |
| 4 | GPT-5.1 (high) | 92.7 | [77.9, 108.3] | 1.73 |
| 5 | Kimi K2 Thinking | 90.9 | [77.0, 106.2] | 1.67 |
| 6 | Grok 4 | 90.7 | [78.1, 103.1] | 1.67 |
| 7 | o4-mini (high) | 86.9 | [74.6, 99.2] | 1.53 |
| 8 | gpt-oss-120B (high) | 86.2 | [72.2, 101.5] | 1.51 |
| 9 | o3 | 86.1 | [73.4, 97.5] | 1.50 |
| 10 | Gemini 2.5 Pro | 85.6 | [73.6, 97.8] | 1.49 |
| 11 | Claude 4.5 Sonnet (Reasoning) | 84.1 | [72.2, 96.2] | 1.44 |
| 12 | DeepSeek V3.1 Terminus (Reasoning) | 83.9 | [68.3, 98.5] | 1.43 |
| 13 | Grok 3 mini Reasoning (high) | 83.5 | [71.3, 95.5] | 1.42 |
| 14 | DeepSeek V3.1 (Reasoning) | 82.6 | [68.0, 97.5] | 1.39 |
| 15 | o3-mini (high) | 78.6 | [65.5, 92.7] | 1.25 |

---

## Interpretation

- **Higher CRS values** indicate higher latent reasoning capability
- **Credible interval width** reflects uncertainty due to benchmark sparsity and noise
- **CRS integrates:**
  - Symbolic correctness (MATH-500)
  - Competition mathematics (AIME)
  - Scientific multi-step reasoning (GPQA)
  - Expert-level challenge problems (HLE)
  - Composite mathematics (Math Index)

CRS should be interpreted as a **generalized static reasoning factor**, not as an agentic or long-horizon planning capability.

---

## Why CRS Is KDD-Appropriate

1. **Multi-dimensional construct**: By combining symbolic reasoning, scientific reasoning, competition mathematics, and expert-level challenges, CRS reflects a more holistic "reasoning quality" than any single benchmark.

2. **Handles missing data gracefully**: The latent variable model naturally accommodates that not all models are evaluated on the same subset of benchmarks — critical for comparing heterogeneous models.

3. **Learns benchmark importance from data**: Unlike fixed-weight approaches, the Bayesian model learns factor loadings that reflect each benchmark's true discriminative power.

4. **Uncertainty quantification**: CRS is reported with credible intervals, providing principled uncertainty estimates that widen appropriately for models with sparse benchmark coverage.

5. **Reproducible and interpretable**: All benchmarks are standard or public (MATH-500, GPQA, HLE, AIME), and the modeling procedure is fully described with open-source code.

6. **Avoids overfit to a single task**: By learning a common latent factor across multiple benchmarks, CRS is less vulnerable to overfitting or benchmark-specific artifacts.

---

## Paper Text Snippet (Short Version)

> **Composite Reasoning Score (CRS).** To systematically compare reasoning quality across models — even when not every model has results on all benchmarks — we define a Composite Reasoning Score (CRS) as the posterior mean of a latent reasoning factor inferred from standardized benchmark scores (MATH-500, GPQA, HLE, AIME, Math Index). All benchmarks are first scaled to a common range and z-score normalized across models. CRS is estimated using a one-factor Bayesian Gaussian latent variable model with benchmark-specific intercepts, loadings, and noise terms. Missing benchmark results are handled naturally as missing observations in the likelihood. We report CRS with 95% credible intervals and verify convergence via R-hat and effective sample size diagnostics.

---

## Paper Text (Full Version)

### Data Normalization

Let $y_{i,b}$ denote the raw score of model $i$ on benchmark $b$. Because benchmarks are heterogeneous in scale, we first normalize each to a common 0-100 range, then apply z-score standardization:

$$z_{i,b} = \frac{y_{i,b} - \mu_b}{\sigma_b}$$

where $\mu_b$ and $\sigma_b$ are computed over all non-missing values for benchmark $b$.

### Latent Variable Model

We model the standardized score as a noisy observation of a latent reasoning factor:

$$z_{i,b} \sim \mathcal{N}(\alpha_b + \lambda_b \theta_i, \sigma_b^2)$$

with priors $\theta_i \sim \mathcal{N}(0,1)$, $\alpha_b \sim \mathcal{N}(0,1)$, $\lambda_b \sim \text{HalfNormal}(0.7)$, $\sigma_b \sim \text{HalfNormal}(1.0)$. The HalfNormal prior on loadings ensures identifiability by constraining the factor direction.

### Inference

Posterior inference is performed via HMC (4 chains, 2000 tune, 2000 draws, target_accept=0.9). We report posterior mean CRS, standard deviation, and 95% credible intervals per model. Convergence is verified via R-hat < 1.01 and ESS > 400 for all parameters.

### Missing Data

Missing benchmark entries are treated as missing-at-random and excluded from the likelihood. The latent factor $\theta_i$ remains identifiable from available observations, with uncertainty appropriately reflected in wider credible intervals for models with sparse coverage.

---

## References

1. D. Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset," NeurIPS 2021.

2. D. Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark," arXiv:2311.12022, 2023.

3. "Humanity's Last Exam," Scale AI / CAIS, 2024. https://lastexam.ai

4. Mathematical Association of America, "American Invitational Mathematics Examination (AIME)."

5. Y. Chang et al., "A Survey on Evaluation of Large Language Models," arXiv:2307.03109, 2023.

6. Artificial Analysis, "LLM Leaderboard," https://artificialanalysis.ai, 2025.

7. A. Gelman et al., "Bayesian Data Analysis," 3rd ed., CRC Press, 2013.

---

## Code Reference

The CRS computation is implemented in:
```
scripts/compute_reasoning_score.py      # CLI script
llm_jury/analysis/latent_factor.py      # Shared Bayesian latent factor module
```

### Basic Usage

```bash
# Weighted z-score method (fast, no dependencies)
python scripts/compute_reasoning_score.py

# Bayesian latent factor model (requires pymc, arviz)
python scripts/compute_reasoning_score.py --bayesian

# With convergence diagnostics
python scripts/compute_reasoning_score.py --bayesian --diagnostics

# Dry run (preview without saving)
python scripts/compute_reasoning_score.py --bayesian --dry-run

# Winsorize extreme z-scores
python scripts/compute_reasoning_score.py --bayesian --winsorize 3.0
```

### Custom Benchmarks

The scripts support custom benchmark configurations via CLI arguments or JSON config files:

```bash
# List available default benchmarks
python scripts/compute_reasoning_score.py --list-benchmarks

# Custom benchmarks via CLI (format: field:scale:weight)
python scripts/compute_reasoning_score.py --benchmarks math_500:100:0.4 gpqa:1:0.3 aime:1:0.3

# Load from JSON config file
python scripts/compute_reasoning_score.py --config my_reasoning_config.json

# Save current config for later reuse
python scripts/compute_reasoning_score.py --save-config reasoning_config.json

# Override score prefix (e.g., for custom composite scores)
python scripts/compute_reasoning_score.py --benchmarks my_bench1:1:0.5 my_bench2:1:0.5 --score-prefix my_score
```

### JSON Config File Format

```json
{
  "name": "reasoning",
  "description": "Composite Reasoning Score benchmarks",
  "score_prefix": "crs",
  "benchmarks": {
    "math_500": {
      "description": "MATH-500: Mathematical problem solving",
      "scale": 100,
      "invert": false,
      "weight": 0.30
    },
    "gpqa": {
      "description": "GPQA: Graduate-level science questions",
      "scale": 1,
      "invert": false,
      "weight": 0.25
    },
    "hle": {
      "description": "HLE: Humanity's Last Exam",
      "scale": 100,
      "invert": false,
      "weight": 0.20
    },
    "aime": {
      "description": "AIME: Competition mathematics",
      "scale": 1,
      "invert": false,
      "weight": 0.15
    },
    "math_index": {
      "description": "Math Index: AA composite",
      "scale": 1,
      "invert": false,
      "weight": 0.10
    }
  }
}
```

### Benchmark Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `scale` | Multiplier to normalize raw scores (e.g., 100 for 0-1 → 0-100) | 1.0 |
| `invert` | Set `true` if lower is better (e.g., error rate) | false |
| `weight` | Relative importance for weighted z-score method | 1.0 |

### Programmatic Usage

```python
from llm_jury.analysis.latent_factor import (
    BenchmarkSuite,
    REASONING_BENCHMARKS,
    extract_benchmark_matrix,
    fit_latent_factor_model,
    summarize_latent_scores,
)

# Use default suite
suite = REASONING_BENCHMARKS

# Or create custom suite
suite = BenchmarkSuite(name="custom", description="My benchmarks", score_prefix="my_score")
suite.add_benchmark("bench_a", "Benchmark A", scale=100, weight=0.5)
suite.add_benchmark("bench_b", "Benchmark B", scale=1, weight=0.5)

# Extract and process
df_scores, df_z, model_names, bench_names = extract_benchmark_matrix(
    models, suite.get_configs(), min_benchmarks=3
)
```

### Cache Fields

The score is stored in the model cache as:
- `crs`: Normalized CRS (0-100 scale)
- `crs_sd`: Posterior standard deviation
- `crs_100`: Same as `crs` (0-100 scale)
- `crs_method`: `"bayesian"` or `"weighted_zscore"`

---

## Related Documentation

- [Composite Coding Score (CCS)](COMPOSITE_CODING_SCORE.md) — Similar methodology applied to coding benchmarks
- [Quality Scoring](QUALITY_SCORING.md) — Overall quality scoring approach
