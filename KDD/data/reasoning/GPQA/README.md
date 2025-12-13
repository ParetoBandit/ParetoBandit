# GPQA (Graduate-Level Google-Proof Q&A) Benchmark Scores

This directory contains GPQA Diamond split benchmark scores and the scripts used to obtain them from the Artificial Analysis API.

## Overview

**GPQA** is a challenging benchmark consisting of graduate-level questions across physics, chemistry, and biology, designed to be extremely difficult even for expert-level PhD holders. The questions are "Google-proof" - meaning they require deep conceptual reasoning rather than simple information retrieval.

**Dataset**: GPQA Diamond (the most challenging split)  
**Source**: Artificial Analysis API (https://artificialanalysis.ai)  
**Coverage**: 98.8% of models (82 out of 83 models)  
**Format**: Accuracy scores on 0.0-1.0 scale (0-100%)

## Directory Structure

```
GPQA/
├── README.md                           # This documentation
├── gpqa_scores.json                    # Extracted GPQA scores for all models
├── artificial_analysis_client.py       # API client for fetching benchmark data
└── etl_pipeline.py                     # ETL pipeline that orchestrates data fetching
```

## GPQA Benchmark Details

### What is GPQA?

GPQA (Graduate-Level Google-Proof Q&A Benchmark) is designed to test expert-level scientific reasoning:

- **Domain**: Graduate-level science (physics, chemistry, biology)
- **Format**: Multiple choice questions with 4 options
- **Difficulty**: Designed to be challenging for PhD experts
- **Google-Proof**: Questions require reasoning, not just fact retrieval
- **Dataset Size**: ~450 expert-vetted questions
- **Split Used**: Diamond (most challenging subset)

### Why GPQA Diamond?

The Diamond split contains the highest-quality, most discriminative questions:
- Vetted by domain experts
- High inter-annotator agreement
- Most resistant to memorization
- Best differentiation between model capabilities

### HuggingFace Dataset

**Dataset ID**: `Idavidrein/gpqa`  
**Config**: `gpqa_diamond`  
**Access**: Gated (requires HuggingFace authentication)  
**URL**: https://huggingface.co/datasets/Idavidrein/gpqa

## Data File: gpqa_scores.json

### Format

```json
{
  "models": [
    {
      "name": "Claude 3.5 Sonnet",
      "slug": "claude-35-sonnet",
      "creator_name": "Anthropic",
      "gpqa": 0.599,
      "source": "artificial_analysis_api",
      "aa_id": "..."
    },
    ...
  ]
}
```

### Fields

- `name`: Human-readable model name
- `slug`: URL-friendly identifier
- `creator_name`: Model creator/provider
- `gpqa`: GPQA Diamond accuracy score (0.0-1.0 scale)
  - `null` if score not available
  - Example: 0.599 = 59.9% accuracy
- `source`: Always "artificial_analysis_api" (confirms direct API source)
- `aa_id`: Artificial Analysis internal model ID

### Coverage Statistics

- **Total Models**: 83
- **Models with GPQA**: 82 (98.8%)
- **Missing GPQA**: 1 model (GPT-4 Turbo)
- **Score Range**: 0.297 to 0.827
  - Lowest: GPT-3.5 Turbo (0.297 / 29.7%)
  - Highest: o3 (0.827 / 82.7%)

## How GPQA Scores Were Obtained

### Source: Artificial Analysis API

All GPQA scores in this dataset were fetched directly from the Artificial Analysis API with **zero imputation or estimation**.

### API Endpoint

```
GET https://artificialanalysis.ai/api/v2/data/llms/models
```

### Authentication

Requires API key set via environment variable:
```bash
export ARTIFICIAL_ANALYSIS_API_KEY="your_api_key_here"
```

### Scripts

#### 1. artificial_analysis_client.py

The main API client for fetching model data including GPQA scores.

**Key Method**:
```python
def get_llm_models(self) -> List[Dict]:
    """Fetch LLM model data from Artificial Analysis API.
    
    Returns:
        List of model dictionaries with benchmarks, pricing, and speed metrics
    """
    response = requests.get(
        self.LLM_ENDPOINT,
        headers={"x-api-key": self.api_key}
    )
    models = response.json()["data"]
    return [self._normalize_model(m) for m in models]
```

**Normalization**:
```python
def _normalize_model(self, raw_model: Dict) -> Dict:
    evaluations = raw_model.get("evaluations", {})
    return {
        "name": raw_model.get("name"),
        "slug": raw_model.get("slug"),
        "gpqa": evaluations.get("gpqa"),  # ← Direct copy from API
        # ... other fields
        "raw_data": raw_model,  # ← Preserves original API response
        "source": "artificial_analysis_api"
    }
```

**Important**: The `gpqa` field is directly copied from the API response without any transformation, imputation, or estimation.

#### 2. etl_pipeline.py

Orchestrates the complete data fetching and merging process.

**Main Pipeline**:
```python
def run(self):
    # Step 1: Fetch models from Artificial Analysis
    aa_models = self.client.get_llm_models()
    
    # Step 2: Load existing cache
    existing_cache = self.merger.load_cache(self.output_file)
    
    # Step 3: Merge with existing cache
    merged_models = self.merger.merge_aa_data(aa_models, existing_cache)
    
    # Step 4: Save to cache
    self.merger.save_cache(merged_models, self.output_file)
```

### Data Quality Assurance

1. **No Imputation**: Models without GPQA scores have `null` values (not estimated)
2. **Raw Data Preserved**: Original API response stored in `raw_data` field
3. **Verification**: All cached GPQA values match original API responses (verified)
4. **Source Tracking**: Every model tagged with `"source": "artificial_analysis_api"`

### Reproducing the Data

To fetch fresh GPQA scores from Artificial Analysis:

```bash
# Set API key
export ARTIFICIAL_ANALYSIS_API_KEY="your_key"

# Run ETL pipeline (from project root)
python -m llm_jury.etl.pipeline
```

Or use the client directly:

```python
from artificial_analysis_client import ArtificialAnalysisClient

client = ArtificialAnalysisClient(api_key="your_key")
models = client.get_llm_models()

# Extract GPQA scores
for model in models:
    print(f"{model['name']}: {model['gpqa']}")
```

## Usage in the Project

### 1. Composite Reasoning Score (CRS)

GPQA is one of 5 benchmarks used to compute the Composite Reasoning Score:

**Benchmarks**:
- MATH-500 (formal mathematics)
- **GPQA** (graduate science reasoning) ← 99% coverage
- HLE (expert-level reasoning)
- AIME (competition mathematics)
- Math Index (composite math score)

**Weight**: Determined by Bayesian latent factor model based on signal strength

### 2. Validation Studies

GPQA is used to validate that CRS correlates with actual reasoning performance:

**Script**: `KDD/composite_quality_scores/llm_judge_reasoning_validation.py`

This script:
- Loads GPQA Diamond questions from HuggingFace
- Tests models with Chain-of-Thought prompting
- Compares predicted performance (CRS) vs actual performance (GPQA accuracy)
- Uses LLM judges to evaluate reasoning quality

### 3. Complexity Analysis

GPQA questions used to study reasoning complexity:

**Script**: `KDD/quality_score_complexity_models/analyze_reasoning_complexity.py`

Classifies GPQA questions by complexity to understand which prompts differentiate model capabilities.

## Benchmark Characteristics

### Score Distribution (Current Cache)

| Percentile | GPQA Score |
|------------|------------|
| Max (100%) | 0.827 (o3) |
| 95th | 0.779 |
| 75th | 0.708 |
| Median | 0.599 |
| 25th | 0.408 |
| 5th | 0.297 |
| Min | 0.297 (GPT-3.5 Turbo) |

### Top Performing Models

1. **o3** - 0.827 (82.7%)
2. **DeepSeek V3.1 Terminus (Reasoning)** - 0.792 (79.2%)
3. **DeepSeek V3.1 (Reasoning)** - 0.779 (77.9%)
4. **o3-mini (high)** - 0.773 (77.3%)
5. **Claude 3.7 Sonnet (Reasoning)** - 0.772 (77.2%)

### Reasoning Model Performance

Models with reasoning modes (o-series, DeepSeek R1, Claude thinking) show significantly higher GPQA scores, confirming the benchmark's ability to measure deep reasoning capabilities.

## References

- **GPQA Paper**: "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"
- **Artificial Analysis**: https://artificialanalysis.ai
- **HuggingFace Dataset**: https://huggingface.co/datasets/Idavidrein/gpqa
- **Project Documentation**: `quality_scoring/docs/COMPOSITE_REASONING_SCORE.md`

## Related Files

- **Main Cache**: `data/models_cache.json` (full model data including GPQA)
- **Validation Script**: `KDD/composite_quality_scores/llm_judge_reasoning_validation.py`
- **CRS Documentation**: `quality_scoring/docs/COMPOSITE_REASONING_SCORE.md`
- **Complexity Analysis**: `KDD/quality_score_complexity_models/analyze_reasoning_complexity.py`

## Data Authenticity

✅ **All GPQA scores in this dataset are authentic from Artificial Analysis API**  
✅ **Zero imputation or estimation performed**  
✅ **Original API responses preserved in raw_data field**  
✅ **100% of cached values match original API values (verified)**

Models without GPQA scores have `null` values and are not estimated. Only 1 model (GPT-4 Turbo) lacks a GPQA score.
