# Composite Coding Score (CCS): Method Documentation

*For inclusion in KDD Paper - Evaluation & Metrics Section*

---

## Overview

To robustly evaluate and compare coding capabilities across models — even when not every model has results on every benchmark — we define a **Composite Coding Score (CCS)** as a latent variable aggregating performance on multiple complementary coding benchmarks. This follows the same principled methodology as the [Composite Reasoning Score (CRS)](COMPOSITE_REASONING_SCORE.md).

---

## Benchmark Selection & Coverage

We select a diverse suite of coding-focused benchmarks that together cover multiple dimensions of programming ability:

| Benchmark | Field | Description | Coverage |
|-----------|-------|-------------|----------|
| **HumanEval** | `humaneval_score` | OpenAI's function-level code generation benchmark (pass@1) | ~70% |
| **LiveCodeBench** | `livecodebench` | Real-world coding tasks from competitive programming | ~60% |
| **SciCode** | `scicode` | Scientific computing and numerical programming benchmark | ~50% |

This mix ensures coverage across:
- **Function-level generation** (HumanEval)
- **Competitive programming** (LiveCodeBench)
- **Scientific/numerical computing** (SciCode)

---

## Data Normalization and Directional Alignment

The same preprocessing pipeline as CRS is applied:

### 1. Scale Normalization

| Benchmark | Raw Range | Scale Factor | Normalized Range |
|-----------|-----------|--------------|------------------|
| HumanEval | 0 - 100 | ×1 | 0 - 100 |
| LiveCodeBench | 0.0 - 1.0 | ×100 | 0 - 100 |
| SciCode | 0.0 - 1.0 | ×100 | 0 - 100 |

### 2. Statistical Standardization

Each benchmark is independently z-score normalized:

$$z_{i,b} = \frac{y_{i,b} - \mu_b}{\sigma_b}$$

---

## Latent Variable Model

We use the same Bayesian latent factor model as CRS:

$$z_{i,b} \sim \mathcal{N}(\alpha_b + \lambda_b \theta_i, \sigma_b^2)$$

where:
- $\theta_i$: latent Composite Coding Score (CCS) for model $i$
- $\alpha_b$: benchmark-specific intercept
- $\lambda_b$: benchmark-specific loading (factor weight)
- $\sigma_b$: benchmark-specific residual standard deviation

### Prior Specification

$$\theta_i \sim \mathcal{N}(0, 1)$$
$$\alpha_b \sim \mathcal{N}(0, 1)$$
$$\lambda_b \sim \text{HalfNormal}(0.7)$$
$$\sigma_b \sim \text{HalfNormal}(1.0)$$

---

## Learned Benchmark Loadings

The Bayesian model learns the relative importance of each benchmark from the data. Example fitted loadings:

| Benchmark | Loading (λ) | Interpretation |
|-----------|-------------|----------------|
| SciCode | ~0.9 | Most discriminating for coding ability |
| LiveCodeBench | ~0.8 | Strong signal |
| HumanEval | ~0.7 | Moderate signal |

**Interpretation:** Scientific computing (SciCode) emerged as a strong discriminator, likely because it requires both algorithmic thinking and domain knowledge, making it highly differentiating across models.

---

## Code Reference

The CCS computation is implemented in:
```
scripts/compute_coding_score.py         # CLI script
llm_jury/analysis/latent_factor.py      # Shared Bayesian latent factor module
```

### Basic Usage

```bash
# Weighted z-score method (fast, no dependencies)
python scripts/compute_coding_score.py

# Bayesian latent factor model (requires pymc, arviz)
python scripts/compute_coding_score.py --bayesian

# With convergence diagnostics
python scripts/compute_coding_score.py --bayesian --diagnostics

# Dry run (preview without saving)
python scripts/compute_coding_score.py --bayesian --dry-run
```

### Custom Benchmarks

The scripts support custom benchmark configurations:

```bash
# List available default benchmarks
python scripts/compute_coding_score.py --list-benchmarks

# Custom benchmarks via CLI (format: field:scale:weight)
python scripts/compute_coding_score.py --benchmarks humaneval_score:1:0.5 mbpp_score:1:0.5

# Load from JSON config file
python scripts/compute_coding_score.py --config my_coding_config.json

# Save current config for later reuse
python scripts/compute_coding_score.py --save-config coding_config.json

# Override score prefix
python scripts/compute_coding_score.py --score-prefix my_coding_score
```

### JSON Config File Format

```json
{
  "name": "coding",
  "description": "Composite Coding Score benchmarks",
  "score_prefix": "ccs",
  "benchmarks": {
    "humaneval_score": {
      "description": "HumanEval: Code generation pass@1",
      "scale": 1,
      "invert": false,
      "weight": 0.35
    },
    "livecodebench": {
      "description": "LiveCodeBench: Real-world coding tasks",
      "scale": 100,
      "invert": false,
      "weight": 0.40
    },
    "scicode": {
      "description": "SciCode: Scientific computing benchmark",
      "scale": 100,
      "invert": false,
      "weight": 0.25
    }
  }
}
```

### Programmatic Usage

```python
from llm_jury.analysis.latent_factor import (
    BenchmarkSuite,
    CODING_BENCHMARKS,
    extract_benchmark_matrix,
    fit_latent_factor_model,
    summarize_latent_scores,
)

# Use default suite
suite = CODING_BENCHMARKS

# Or create custom suite
suite = BenchmarkSuite(name="custom", description="My coding benchmarks", score_prefix="my_ccs")
suite.add_benchmark("humaneval_score", "HumanEval", scale=1, weight=0.5)
suite.add_benchmark("mbpp_score", "MBPP", scale=1, weight=0.5)

# Extract and process
df_scores, df_z, model_names, bench_names = extract_benchmark_matrix(
    models, suite.get_configs(), min_benchmarks=2
)
```

### Cache Fields

The score is stored in the model cache as:
- `ccs`: Normalized CCS (0-100 scale)
- `ccs_sd`: Posterior standard deviation
- `ccs_100`: Same as `ccs` (0-100 scale)
- `ccs_method`: `"bayesian"` or `"weighted_zscore"`

---

## Relationship to CRS

CCS and CRS are computed using the same underlying methodology (Bayesian latent factor model) but measure different constructs:

| Score | Domain | Default Benchmarks |
|-------|--------|-------------------|
| CRS | Reasoning | MATH-500, GPQA, HLE, AIME, Math Index |
| CCS | Coding | HumanEval, LiveCodeBench, SciCode |

Both scores:
- Handle missing data gracefully
- Learn benchmark importance from data
- Provide uncertainty quantification via credible intervals
- Are normalized to a 0-100 scale for interpretability

---

## References

1. M. Chen et al., "Evaluating Large Language Models Trained on Code" (HumanEval), arXiv:2107.03374, 2021.

2. J. Austin et al., "Program Synthesis with Large Language Models" (MBPP), arXiv:2108.07732, 2021.

3. LiveCodeBench, https://livecodebench.github.io

4. SciCode Benchmark, https://scicode-bench.github.io

---

## Related Documentation

- [Composite Reasoning Score (CRS)](COMPOSITE_REASONING_SCORE.md) — Same methodology for reasoning benchmarks
- [Quality Scoring](QUALITY_SCORING.md) — Overall quality scoring approach
