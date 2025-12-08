# Bayesian Latent Factor Module

The `llm_jury.analysis.latent_factor` module provides a reusable implementation of Bayesian latent factor modeling for computing composite scores from multiple benchmarks.

---

## Overview

This module enables you to:
- Define custom benchmark suites with configurable scaling, inversion, and weights
- Extract and normalize benchmark data from model caches
- Compute composite scores using either weighted z-scores or Bayesian latent factors
- Save/load benchmark configurations via JSON files

---

## Quick Start

### Using Default Suites

```python
from llm_jury.analysis.latent_factor import (
    REASONING_BENCHMARKS,  # Pre-configured for CRS
    CODING_BENCHMARKS,     # Pre-configured for CCS
)

# View default reasoning benchmarks
print(REASONING_BENCHMARKS.score_prefix)  # "crs"
for name, cfg in REASONING_BENCHMARKS.benchmarks.items():
    print(f"  {name}: {cfg.description}")
```

### Creating Custom Suites

```python
from llm_jury.analysis.latent_factor import BenchmarkSuite

# Create a new suite
suite = BenchmarkSuite(
    name="my_suite",
    description="Custom benchmark suite",
    score_prefix="my_score",
)

# Add benchmarks
suite.add_benchmark(
    name="bench_a",
    description="My first benchmark",
    scale=100,      # Multiply raw scores by 100
    invert=False,   # Higher is better
    weight=0.6,     # 60% weight for weighted z-score method
)
suite.add_benchmark("bench_b", "Second benchmark", scale=1, weight=0.4)

# Save to JSON for reuse
suite.to_json("my_config.json")

# Load from JSON
loaded_suite = BenchmarkSuite.from_json("my_config.json")
```

---

## Benchmark Configuration

Each benchmark has the following configuration options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | str | required | Field name in model cache |
| `description` | str | name | Human-readable description |
| `scale` | float | 1.0 | Multiplier for raw scores |
| `invert` | bool | False | If True, negate scores (for "lower is better" metrics) |
| `weight` | float | 1.0 | Relative weight for weighted z-score method |

### Scaling Examples

| Benchmark Type | Raw Range | Scale | Result |
|----------------|-----------|-------|--------|
| Percentage | 0-100 | 1 | 0-100 |
| Proportion | 0.0-1.0 | 100 | 0-100 |
| Error rate | 0.0-1.0 | 100, invert=True | 0-100 (higher is better) |

---

## Computing Scores

### Extract Benchmark Matrix

```python
from llm_jury.analysis.latent_factor import extract_benchmark_matrix

# models = list of model dicts from cache
df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
    models,
    suite.get_configs(),
    min_benchmarks=2,  # Require at least 2 benchmarks per model
)

# df_scores: Raw scaled scores
# df_z: Z-score normalized scores
# model_names: List of model names included
# benchmark_names: List of benchmarks used
```

### Weighted Z-Score Method (Fast)

```python
from llm_jury.analysis.latent_factor import compute_weighted_zscore, transform_to_0_100

df_result = compute_weighted_zscore(
    df_z,
    model_names,
    suite.get_weights(),
    score_name="my_score",
)

# Transform to 0-100 scale
df_result = transform_to_0_100(
    df_result,
    mean_col="my_score_mean",
    output_col="my_score_100",
)
```

### Bayesian Latent Factor Model

```python
from llm_jury.analysis.latent_factor import (
    prepare_long_data,
    fit_latent_factor_model,
    summarize_latent_scores,
    transform_to_0_100,
)

# Convert to long format for PyMC
z_obs, idx_model, idx_bench, n_models, n_benchmarks = prepare_long_data(
    df_z, model_names, benchmark_names
)

# Fit Bayesian model
idata = fit_latent_factor_model(
    z_obs, idx_model, idx_bench, n_models, n_benchmarks,
    draws=2000,
    tune=2000,
    chains=4,
    target_accept=0.9,
    random_seed=42,
)

# Extract posterior summaries
df_result = summarize_latent_scores(idata, model_names, score_name="my_score")

# Transform to 0-100 scale
df_result = transform_to_0_100(
    df_result,
    mean_col="my_score_mean",
    output_col="my_score_100",
    hdi_low_col="my_score_hdi_low",
    hdi_high_col="my_score_hdi_high",
)
```

---

## CLI Usage

Both `compute_reasoning_score.py` and `compute_coding_score.py` share the same CLI interface:

```bash
# List default benchmarks
python scripts/compute_coding_score.py --list-benchmarks

# Custom benchmarks (format: field:scale:weight)
python scripts/compute_coding_score.py --benchmarks humaneval_score:1:0.5 mbpp_score:1:0.5

# From config file
python scripts/compute_coding_score.py --config my_config.json

# Save config
python scripts/compute_coding_score.py --save-config my_config.json

# Override score prefix
python scripts/compute_coding_score.py --score-prefix custom_score
```

---

## JSON Config Format

```json
{
  "name": "my_suite",
  "description": "My custom benchmark suite",
  "score_prefix": "my_score",
  "benchmarks": {
    "benchmark_field_1": {
      "description": "First benchmark",
      "scale": 100,
      "invert": false,
      "weight": 0.5
    },
    "benchmark_field_2": {
      "description": "Second benchmark",
      "scale": 1,
      "invert": false,
      "weight": 0.5
    }
  }
}
```

---

## API Reference

### Classes

#### `BenchmarkConfig`
```python
@dataclass
class BenchmarkConfig:
    name: str           # Field name in model cache
    description: str    # Human-readable description
    scale: float = 1.0  # Score multiplier
    invert: bool = False  # Negate scores if True
    weight: float = 1.0  # Weight for weighted z-score
```

#### `BenchmarkSuite`
```python
@dataclass
class BenchmarkSuite:
    name: str                              # Suite name
    description: str                       # Suite description
    benchmarks: Dict[str, BenchmarkConfig] # Benchmark configurations
    score_prefix: str = "score"            # Prefix for output fields
    
    def add_benchmark(name, description, scale, invert, weight) -> self
    def remove_benchmark(name) -> self
    def get_configs() -> Dict[str, Dict]
    def get_weights() -> Dict[str, float]  # Normalized weights
    def to_dict() -> Dict
    def to_json(path)
    @classmethod from_dict(data) -> BenchmarkSuite
    @classmethod from_json(path) -> BenchmarkSuite
```

### Functions

| Function | Description |
|----------|-------------|
| `extract_benchmark_matrix(models, configs, min_benchmarks)` | Extract and normalize benchmark scores |
| `prepare_long_data(df_z, model_names, benchmark_names)` | Convert to PyMC-compatible format |
| `fit_latent_factor_model(...)` | Fit Bayesian latent factor model |
| `summarize_latent_scores(idata, model_names, score_name)` | Extract posterior summaries |
| `compute_weighted_zscore(df_z, model_names, weights, score_name)` | Fast weighted average |
| `transform_to_0_100(df, mean_col, output_col, ...)` | Min-max normalize to 0-100 |
| `update_models_cache(cache_data, df_scores, score_prefix, method)` | Update model cache |
| `get_benchmark_diagnostics(idata, benchmark_names)` | Get convergence diagnostics |
| `parse_benchmark_args(benchmark_args, config_path, default_suite)` | Parse CLI arguments |
| `add_benchmark_args(parser)` | Add standard args to argparse |

### Pre-built Suites

| Constant | Score Prefix | Benchmarks |
|----------|--------------|------------|
| `REASONING_BENCHMARKS` | `crs` | math_500, gpqa, hle, aime, math_index |
| `CODING_BENCHMARKS` | `ccs` | humaneval_score, livecodebench, scicode |

---

## Mathematical Background

The Bayesian latent factor model assumes:

$$z_{i,b} \sim \mathcal{N}(\alpha_b + \lambda_b \theta_i, \sigma_b^2)$$

where:
- $\theta_i$: Latent composite score for model $i$
- $\alpha_b$: Benchmark-specific intercept
- $\lambda_b$: Benchmark-specific loading (constrained positive)
- $\sigma_b$: Benchmark-specific residual noise

This allows:
- **Missing data handling**: Models with partial benchmark coverage are scored appropriately
- **Learned importance**: Factor loadings are learned from data, not predetermined
- **Uncertainty quantification**: Posterior credible intervals reflect both noise and data sparsity

---

## Related Documentation

- [Composite Reasoning Score (CRS)](COMPOSITE_REASONING_SCORE.md)
- [Composite Coding Score (CCS)](COMPOSITE_CODING_SCORE.md)
