# LLM Jury Data Sources

This document describes the data sources, collection methods, and fields available in the model cache.

## Data Files

- **`models_cache.json`** - Primary model cache with all collected data
- **`models_complete_composite_indices.json`** - Filtered cache with only models that have complete benchmark data

## Data Sources

### 1. Artificial Analysis API

**Source:** https://artificialanalysis.ai/

**What it provides:**
- Model benchmark scores (intelligence, coding, math indices)
- Pricing information (input/output token costs)
- Performance metrics (throughput, latency)
- Model metadata (provider, context length, etc.)

**Collection method:**
- API calls to Artificial Analysis endpoints
- Requires API key (`ARTIFICIAL_ANALYSIS_API_KEY`)

**Key fields:**
| Field | Description |
|-------|-------------|
| `name` | Model display name |
| `slug` | Model identifier |
| `provider` | Model provider (OpenAI, Anthropic, etc.) |
| `intelligence_index` | General intelligence benchmark score |
| `coding_index` | Coding benchmark score |
| `math_index` | Math benchmark score |
| `input_cost_per_million` | Cost per million input tokens (USD) |
| `output_cost_per_million` | Cost per million output tokens (USD) |
| `output_tokens_per_second` | Throughput (tokens/second) |
| `median_latency_ms` | Median response latency |
| `context_length` | Maximum context window size |

---

### 2. Vectara Hallucination Leaderboard

**Source:** https://github.com/vectara/hallucination-leaderboard

**What it provides:**
- Hallucination rates for LLMs
- Factual consistency scores
- Answer rates (how often models respond vs refuse)

**Collection method:**
- Parses the README.md table from the GitHub repository
- No API key required (public data)

**Key fields:**
| Field | Description |
|-------|-------------|
| `hallucination_rate` | Percentage of responses containing hallucinations |
| `factual_consistency_rate` | 100 - hallucination_rate (reliability score) |
| `hallucination_answer_rate` | Percentage of prompts the model answered |
| `refusal_rate` | 100 - answer_rate (how often model refuses to answer) |
| `hallucination_source` | Always "vectara" for traceability |

---

### 3. OpenRouter TTFT Measurement

**Source:** https://openrouter.ai/

**What it provides:**
- Time To First Token (TTFT) latency measurements
- Real-world latency data via streaming API calls

**Collection method:**
- Streaming API calls to OpenRouter with `stream=True`
- Measures time from request to first token received
- Multiple samples averaged for reliability
- Requires API key (`OPENROUTER_API_KEY`)

**Key fields:**
| Field | Description |
|-------|-------------|
| `measured_ttft_seconds` | Average Time To First Token in seconds |

**Model mapping:**
Models are mapped from our internal names to OpenRouter model IDs. Some models may not be available on OpenRouter and will not have TTFT data.

---

## ETL Pipeline

The data is collected via the ETL (Extract, Transform, Load) pipeline:

```python
from llm_jury.etl import ETLPipeline

pipeline = ETLPipeline()
pipeline.run(
    require_complete_benchmarks=True,  # Only models with all indices
    include_hallucination_data=True,   # Fetch from Vectara
    include_ttft_data=True,            # Measure via OpenRouter
    require_ttft=True,                 # Filter out models without TTFT
    ttft_samples=2,                    # Average 2 samples per model
)
```

### Pipeline Steps

1. **Step 1:** Fetch models from Artificial Analysis API
2. **Step 1b:** Fetch hallucination data from Vectara GitHub
3. **Step 2:** Load existing cache
4. **Step 3:** Merge Artificial Analysis data
5. **Step 3b:** Merge hallucination data
6. **Step 3c:** Measure TTFT via OpenRouter
7. **Step 4:** Save updated cache

---

## Data Quality

### Filtering Criteria

Models are included in the cache only if they have:

1. **Complete benchmark data** (when `require_complete_benchmarks=True`):
   - `intelligence_index`
   - `coding_index`
   - `math_index`

2. **Hallucination data** (when `include_hallucination_data=True`):
   - Models are matched by name to Vectara leaderboard
   - Not all models have hallucination data available

3. **TTFT data** (when `require_ttft=True`):
   - Model must be available on OpenRouter
   - TTFT measurement must succeed

### Current Statistics

As of the last pipeline run:
- **80 models** with complete data
- All models have benchmark scores from Artificial Analysis
- All models have measured TTFT from OpenRouter
- Hallucination data available for models in Vectara leaderboard

---

## Derived Metrics

Some fields are calculated from raw data:

| Derived Field | Calculation |
|---------------|-------------|
| `blended_cost` | `0.75 * input_cost + 0.25 * output_cost` |
| `factual_consistency_rate` | `100 - hallucination_rate` |
| `refusal_rate` | `100 - hallucination_answer_rate` |

---

## Environment Variables

Required API keys (set in `.env` file):

```bash
ARTIFICIAL_ANALYSIS_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

---

## Updating Data

To refresh the cache with latest data:

```bash
cd /path/to/llm_jury
python -c "
from llm_jury.etl import ETLPipeline
pipeline = ETLPipeline()
pipeline.run(require_complete_benchmarks=True)
"
```

Or use the CLI (if available):

```bash
python -m llm_jury.etl.pipeline
```

