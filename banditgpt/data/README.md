# BanditGPT Data

This folder contains data files used by the BanditGPT router.

## Files

| File | Size | Description |
|------|------|-------------|
| `models_cache.json` | ~32 KB | Model registry (81 models with cost/latency) |
| `priors/` | ~21 MB | Expert-distilled priors for warm-start |
| `quality_predictor/` | — | Optional: trained model for TieredGrader |

---

## models_cache.json

Minimal model registry with only fields needed for routing:

```json
{
  "models": [
    {
      "openrouter_id": "anthropic/claude-3.5-haiku",
      "display_name": "Claude 3.5 Haiku",
      "input_cost_per_m": 0.8,
      "output_cost_per_m": 4.0,
      "time_to_first_token_seconds": 0.786,
      "output_tokens_per_second": 49.071
    }
  ]
}
```

### Required Fields

| Field | Description | Used For |
|-------|-------------|----------|
| `openrouter_id` | Model identifier (e.g., `openai/gpt-4o`) | Model lookup |
| `display_name` | Human-readable name | Display |

### Cost Fields (at least one required)

| Field | Description |
|-------|-------------|
| `input_cost_per_m` | Cost per 1M input tokens (USD) |
| `output_cost_per_m` | Cost per 1M output tokens (USD) |
| `price_1m_blended` | Blended cost (fallback) |

### Latency Fields (at least one required)

| Field | Description |
|-------|-------------|
| `time_to_first_token_seconds` | TTFT in seconds (mean) |
| `output_tokens_per_second` | Generation speed |
| `median_latency_ms` | Median latency (fallback) |

### TTFT Statistical Fields (optional, for precision)

| Field | Description |
|-------|-------------|
| `ttft_mean` | Mean TTFT in seconds |
| `ttft_std` | Standard deviation |
| `ttft_ci_95_lower` | 95% confidence interval lower bound |
| `ttft_ci_95_upper` | 95% confidence interval upper bound |
| `ttft_p50` | Median (50th percentile) |
| `ttft_p95` | 95th percentile |
| `ttft_p99` | 99th percentile |
| `ttft_samples` | Number of samples (0 = estimated, 100+ = measured) |

---

## Usage

```python
from banditgpt.core import build_registry_from_models_cache
from banditgpt._resources import get_models_cache_path

# Load registry from cache
registry = build_registry_from_models_cache(get_models_cache_path())

# Each model has: cost, latency_s, display_name
print(registry["openai/gpt-4o"])
# {'display_name': 'GPT-4o', 'cost': 0.00375, 'latency_s': 1.2, ...}
```

---

---

## Updating TTFT Measurements

The bundled cache includes estimated TTFT values. For precise measurements (100 samples per model with 95% confidence intervals):

```bash
# Update all models (parallel, ~15-20 min with 10 workers)
python -m banditgpt.core.model_manager batch-update --workers 10

# Update a single model
python -m banditgpt.core.model_manager update-ttft openai/gpt-4o

# Add a new model with TTFT measurement
python -m banditgpt.core.model_manager add openai/gpt-5
```

**Cost**: ~$0.01 for all 81 models (100 samples × 5 output tokens each)

---

## Data Sources

The model data was collected from:

| Source | Data |
|--------|------|
| [Artificial Analysis](https://artificialanalysis.ai/) | Pricing, throughput |
| [OpenRouter](https://openrouter.ai/) | TTFT measurements |

See [ACKNOWLEDGMENTS.md](../../ACKNOWLEDGMENTS.md) for full attribution.

