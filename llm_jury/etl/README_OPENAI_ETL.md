# OpenAI Direct ETL Client

## Overview

This ETL script collects **complete model data** directly from OpenAI's API, including:

1. ✅ **Pricing** (input/output costs per 1M tokens)
2. ✅ **Latency** (measured TTFT via streaming API calls)
3. ✅ **Context Window** (max tokens)
4. ✅ **Model Metadata** (name, creator, identifiers)

This data populates the fields needed for **HYBRID optimization**:
- **Cost**: Direct from OpenAI pricing
- **Latency**: Measured via actual API calls
- **Quality**: Should be merged from Artificial Analysis benchmarks
- **Hallucination**: Should be merged from Vectara leaderboard
- **Refusal**: Can be measured separately

## Setup

### 1. Add OpenAI API Key to `.env`

```bash
# In project root .env file
OPENAI_API_KEY=sk-...
```

### 2. Install Dependencies

```bash
pip install openai python-dotenv
```

## Usage

### List Available Models

```bash
python -m llm_jury.etl.openai_direct_client --list-models
```

Output:
```
Available OpenAI Models:
============================================================
✓ gpt-3.5-turbo                              $0.5/$1.5 per 1M
✓ gpt-4                                      $30.0/$60.0 per 1M
✓ gpt-4o                                     $2.5/$10.0 per 1M
✓ gpt-4o-mini                                $0.15/$0.6 per 1M
...
```

### Collect Data for Specific Models

```bash
# Collect data for GPT-3.5 and GPT-4o with 5 TTFT samples each
python -m llm_jury.etl.openai_direct_client \
    --models gpt-3.5-turbo gpt-4o \
    --samples 5 \
    --output data/openai_models_data.json
```

### Collect and Update Cache

```bash
# Measure TTFT and update models_cache.json
python -m llm_jury.etl.openai_direct_client \
    --models gpt-3.5-turbo gpt-4o gpt-4o-mini \
    --samples 5 \
    --update-cache
```

### Skip Latency Measurement (Faster)

```bash
# Only update pricing/metadata, skip TTFT measurement
python -m llm_jury.etl.openai_direct_client \
    --models gpt-3.5-turbo \
    --no-latency \
    --update-cache
```

### Collect All Models with Pricing

```bash
# Default behavior: collect all models with known pricing
python -m llm_jury.etl.openai_direct_client \
    --samples 3 \
    --delay 2.0 \
    --output data/openai_complete.json \
    --update-cache
```

## Output Format

The script generates JSON with complete model data:

```json
{
  "name": "GPT-3.5 Turbo",
  "slug": "gpt-3.5-turbo",
  "creator_name": "OpenAI",
  "creator_slug": "openai",
  
  "price_1m_input": 0.5,
  "price_1m_output": 1.5,
  "input_cost_per_m": 0.5,
  "output_cost_per_m": 1.5,
  
  "measured_ttft_seconds": 0.3245,
  
  "context_length": 16385,
  "context_window_k": 16,
  
  "openrouter_id": "openai/gpt-3.5-turbo",
  "data_source": "openai_direct"
}
```

## Data Completeness

### What This Script Provides ✅

| Field | Source | Notes |
|-------|--------|-------|
| **Cost** | OpenAI Pricing | Input/output per 1M tokens |
| **Latency** | Measured API | Actual TTFT via streaming |
| **Context** | OpenAI Docs | Max tokens |
| **Metadata** | OpenAI API | Name, creator, IDs |

### What Needs External Merge ⚠️

| Field | Source | How to Get |
|-------|--------|-----------|
| **Quality Score** | Artificial Analysis | Run `artificial_analysis_client.py` |
| **Hallucination Rate** | Vectara Leaderboard | Run `hallucination_leaderboard_client.py` |
| **Refusal Rate** | Manual Testing | Measure separately or use defaults |
| **Benchmarks** | Multiple Sources | MMLU, GPQA, HumanEval, etc. |

## Integration Workflow

### Step 1: Collect OpenAI Data

```bash
python -m llm_jury.etl.openai_direct_client \
    --models gpt-3.5-turbo gpt-4o gpt-4o-mini \
    --samples 5 \
    --output data/openai_data.json
```

### Step 2: Collect External Benchmarks

```bash
# Artificial Analysis (quality scores, benchmarks)
python -m llm_jury.etl.artificial_analysis_client

# Vectara (hallucination rates)
python -m llm_jury.etl.hallucination_leaderboard_client
```

### Step 3: Merge All Data

```bash
python -m llm_jury.etl.data_merger \
    --openai data/openai_data.json \
    --artificial-analysis data/aa_data.json \
    --vectara data/vectara_data.json \
    --output data/models_cache.json
```

Or manually update cache with OpenAI data:

```python
from llm_jury.etl.openai_direct_client import update_cache_with_openai_data
from pathlib import Path
import json

# Load OpenAI data
with open("data/openai_data.json") as f:
    openai_data = json.load(f)

# Update cache
updated, added = update_cache_with_openai_data(
    Path("data/models_cache.json"),
    openai_data,
    merge_strategy="update"
)

print(f"Updated {updated} models, added {added} new models")
```

## TTFT Measurement Details

### How It Works

1. Creates streaming chat completion request
2. Measures time from request start to first token
3. Repeats N times (default: 5 samples)
4. Returns average TTFT

### Accuracy

- ✅ Real latency (actual API calls)
- ✅ Includes network roundtrip
- ✅ Represents production performance
- ⚠️ Varies by time of day, server load
- ⚠️ Measured from your location (latency to OpenAI servers)

### Cost

Each TTFT measurement uses ~10 tokens:
- Input: ~6 tokens ("Say 'Hello' once.")
- Output: ~4 tokens (model response)

For 5 samples on gpt-3.5-turbo:
- Cost: ~(6 + 4) × 5 × ($0.5 + $1.5) / 1M = $0.0001
- Negligible for measurement purposes

## Troubleshooting

### "OpenAI API key not found"

Add to `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### "Rate limit exceeded"

Add delay between models:
```bash
--delay 5.0  # 5 seconds between models
```

### "Model not found"

Check available models:
```bash
python -m llm_jury.etl.openai_direct_client --list-models
```

### Inconsistent TTFT

Increase samples:
```bash
--samples 10  # More samples = more stable average
```

## Example: Complete GPT-3.5 Data Collection

```bash
# Full workflow for GPT-3.5-turbo
python -m llm_jury.etl.openai_direct_client \
    --models gpt-3.5-turbo \
    --samples 10 \
    --delay 1.0 \
    --output data/gpt35_complete.json \
    --update-cache

# Output:
# [1/1] Processing gpt-3.5-turbo...
# Collecting data for gpt-3.5-turbo...
#   Sample 1/10: 0.3156s
#   Sample 2/10: 0.3289s
#   ...
#   TTFT: 0.3245s ± 0.0089s (10 samples)
#   ✓ Collected: $0.5/$1.5 per 1M, TTFT=0.3245s, Context=16385
#
# ✅ Successfully collected data for 1 models
# ✅ Saved data to data/gpt35_complete.json
# ✅ Updated cache: 1 models updated, 0 models added
#
# SUMMARY
# ================================================================================
# GPT-3.5 Turbo:
#   Pricing: $0.5/$1.5 per 1M tokens
#   Latency: 0.3245s TTFT
#   Context: 16K tokens
```

## Next Steps

After collecting OpenAI data:

1. **Verify in Cache**: Check `data/models_cache.json` for updated fields
2. **Merge External Data**: Run other ETL clients for complete dataset
3. **Run Optimization**: Use LLM Jury with complete data:
   ```python
   from llm_jury import Optimizer, ModelRegistry
   
   registry = ModelRegistry.load_cache()
   # Now has complete data: cost, latency, quality, hallucination, refusal
   ```

4. **Fair Evaluation**: With complete data, comparisons are more accurate

