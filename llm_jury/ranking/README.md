# Multi-Objective Optimization for LLM Selection

This document explains the multi-objective optimization algorithms used to rank and recommend LLM models based on multiple competing objectives.

## Overview

Selecting the "best" LLM is inherently a multi-objective optimization problem. Users care about:
- **Quality** - How well the model performs on benchmarks
- **Cost** - How much it costs per token
- **Latency** - How fast it responds (Time To First Token)
- **Reliability** - How often it hallucinates or refuses to answer

These objectives often conflict: the highest quality models tend to be expensive and slow, while fast cheap models may hallucinate more. This module provides two optimization approaches:

1. **Chebyshev Scalarization** - Minimizes worst-case regret across objectives
2. **Knee Point Detection** - Finds the "best bang for buck" on the Pareto frontier

## Quick Start

```python
from llm_jury import Optimizer, OptimizationStrategy

# Basic usage
optimizer = Optimizer(
    baseline_model=gpt4,
    all_models_data=models_data,
    strategy=OptimizationStrategy.BALANCED
)

results = optimizer.rank(models, decision, top_k=5)
```


---

## Extensibility: Adding Custom Objectives

The optimizer uses a pluggable **Objective** system that makes adding new metrics trivial. This is useful when you want to optimize for additional criteria like ethics scores, safety ratings, or domain-specific benchmarks.

### The Objective System

Each optimization objective is defined by an `Objective` dataclass:

```python
from llm_jury import Objective, NormalizationMethod

objective = Objective(
    name="ethics",                          # Unique identifier
    display_name="Ethics Score",            # Human-readable name
    direction="maximize",                   # "maximize" or "minimize"
    default_weight=0.15,                    # Weight in balanced optimization
    default_value=50.0,                     # Fallback when data missing
    extractor=lambda m, d, ctx: ...,        # Function to extract value
    normalization=NormalizationMethod.PERCENTAGE,  # How to normalize
)
```

### Adding a Custom Objective

Here's a complete example of adding an ethics metric:

```python
from llm_jury import (
    Optimizer, OptimizationStrategy,
    Objective, ObjectiveRegistry, NormalizationMethod
)

# 1. Define the objective
ethics_objective = Objective(
    name="ethics",
    display_name="Ethics Score",
    direction="maximize",
    default_weight=0.15,
    default_value=50.0,
    # Extractor function: (model, decision, context) -> float
    extractor=lambda m, d, ctx: getattr(m, 'ethics_score', 50.0),
    normalization=NormalizationMethod.PERCENTAGE,
)

# 2. Create registry with defaults and add custom objective
registry = ObjectiveRegistry.default()  # Has 5 built-in objectives
registry.register(ethics_objective)

# 3. Create optimizer with custom objectives
optimizer = Optimizer(
    baseline_model=baseline,
    all_models_data=models_data,
    objectives=registry,
    custom_weights={
        "quality": 0.30,
        "ethics": 0.15,
        "cost": 0.15,
        "latency": 0.10,
        "hallucination": 0.20,
        "refusal": 0.10,
    }
)

# 4. Use as normal
results = optimizer.rank(models, decision, top_k=5)
```

### Normalization Methods

| Method | Use Case | Formula |
|--------|----------|---------|
| `PERCENTAGE` | Value is 0-100, higher is better | `v / 100` |
| `INVERSE_PERCENTAGE` | Value is 0-100, lower is better | `1 - (v / 100)` |
| `RATIO_TO_BASELINE` | Relative to baseline model | `1 / (1 + ratio)` for minimize |
| `MIN_MAX` | Use population range, higher is better | `(v - min) / (max - min)` |
| `INVERSE_MIN_MAX` | Use population range, lower is better | `1 - (v - min) / (max - min)` |
| `CUSTOM` | Provide your own function | User-defined |

### Custom Normalization

For complex normalization logic:

```python
def my_normalizer(value: float, baseline_value: float, stats: dict) -> float:
    # Custom logic here
    # Must return 0-1 where 1 is best
    return min(1.0, value / 100.0) ** 0.5  # Square root scaling

objective = Objective(
    name="custom_metric",
    ...
    normalization=NormalizationMethod.CUSTOM,
    custom_normalizer=my_normalizer,
)
```

---

## Model Data Structure

For the optimizer to extract metrics, model data must be structured correctly. The system uses `ModelMetadata` objects with specific attributes.

### Required Model Attributes

| Attribute | Type | Used By | Description |
|-----------|------|---------|-------------|
| `name` | `str` | All | Model identifier |
| `input_cost_per_m` | `float` | Cost objective | Cost per 1M input tokens |
| `output_cost_per_m` | `float` | Cost objective | Cost per 1M output tokens |

### Optional Model Attributes (Built-in Objectives)

| Attribute | Type | Default | Used By |
|-----------|------|---------|---------|
| `intelligence_index` | `float` | None | Quality |
| `coding_index` | `float` | None | Quality |
| `math_index` | `float` | None | Quality |
| `mmlu_pro` | `float` | None | Quality |
| `gpqa` | `float` | None | Quality |
| `measured_ttft_seconds` | `float` | 1.0 | Latency |
| `time_to_first_token_seconds` | `float` | 1.0 | Latency |
| `median_latency_ms` | `float` | 1000 | Latency (fallback) |
| `hallucination_rate` | `float` | 15.0 | Hallucination |
| `refusal_rate` | `float` | 5.0 | Refusal |
| `context_window_k` | `int` | 8 | Context constraint |

---

## Context Window & Capability Constraints

Beyond quality/cost optimization, real-world use cases often require specific model capabilities like sufficient context windows, function calling, or vision support. The constraint system provides both **hard constraints** (pre-filtering) and **soft constraints** (scoring objectives).

### Quick Start with Constraints

```python
from llm_jury.ranking import (
    ConstraintConfig,
    CapabilityRequirement,
    apply_constraints,
    create_context_objective,
)

# Define requirements for a RAG pipeline
constraints = ConstraintConfig(
    min_context_k=100,  # Hard requirement: at least 100K context
    target_context_k=200,  # Soft: bonus for 200K+
    capabilities=[CapabilityRequirement.FUNCTION_CALLING],
    preferred_capabilities=[CapabilityRequirement.REASONING],
)

# Filter models by hard constraints
filtered_models = apply_constraints(models, constraints)

# Add context as a soft objective in optimization
context_obj = create_context_objective(target_context_k=200, weight=0.15)
optimizer.objectives.register(context_obj)
```

### Setting Capability Flags in Your Model Cache

For accurate capability detection, add explicit capability flags to your model cache. The constraint system will use these flags when available.

#### Recommended Capability Fields

Add these fields to each model in your `models_cache.json`:

```json
{
  "name": "GPT-4o",
  "context_length": 128000,
  "context_window_k": 128,
  
  "supports_functions": true,
  "supports_vision": true,
  "supports_audio": false,
  "supports_json_mode": true,
  "supports_streaming": true,
  "supports_embeddings": false,
  "is_reasoning_model": false,
  
  "tool_use_ability": 0.95,
  
  "input_modalities": ["text", "image"],
  "output_modalities": ["text"]
}
```

#### Capability Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `context_length` | `int` | Context window in tokens |
| `context_window_k` | `int` | Context window in thousands (e.g., 128 for 128K) |
| `supports_functions` | `bool` | Supports function/tool calling |
| `supports_vision` | `bool` | Accepts image inputs |
| `supports_audio` | `bool` | Accepts audio inputs |
| `supports_json_mode` | `bool` | Supports structured JSON output |
| `supports_streaming` | `bool` | Supports streaming responses |
| `supports_embeddings` | `bool` | Can generate embeddings |
| `is_reasoning_model` | `bool` | Extended thinking / chain-of-thought model |
| `tool_use_ability` | `float` | Tool use proficiency score (0-1) |

#### Adding Capability Flags to Your Cache

```python
import json

# Load existing cache
with open("data/models_cache.json", "r") as f:
    cache = json.load(f)

# Add capability flags
for model in cache:
    name = model.get("name", "").lower()
    
    # Example: set vision support for known vision models
    model["supports_vision"] = any(v in name for v in ["4o", "vision", "gemini-pro"])
    
    # Example: set reasoning flag
    model["is_reasoning_model"] = any(r in name for r in ["reasoning", "think", " r1"])
    
    # Example: set function calling for major providers
    creator = model.get("creator_slug", "").lower()
    model["supports_functions"] = creator in ["openai", "anthropic", "google", "cohere"]

# Save updated cache
with open("data/models_cache.json", "w") as f:
    json.dump(cache, f, indent=2)
```

### Fallback: Provider-Based Heuristic Detection

**When explicit capability flags are not set**, the constraint system falls back to **provider-based heuristics**. This allows the system to work even with incomplete capability data.

#### How Fallback Detection Works

The system uses model name patterns and known provider capabilities:

```python
# Providers known to support function calling
_FUNCTION_CALLING_PROVIDERS = {'anthropic', 'openai', 'google', 'cohere', 'mistral', 'meta'}

# Model families known to support function calling
_FUNCTION_CALLING_MODELS = {'claude', 'gpt', 'gemini', 'command', 'llama', 'mixtral'}

# Vision detection patterns
_VISION_MODELS = {'gpt-4o', 'claude-3', 'gemini', 'llava', 'vision'}

# Reasoning model patterns (checked in model name)
_REASONING_PATTERNS = ['reason', 'think', ' r1']
```

#### Detection Priority

For each capability, the system checks in order:

1. **Explicit flag** (e.g., `supports_functions: true`) → Use directly
2. **Ability score** (e.g., `tool_use_ability > 0.3`) → Infer capability
3. **Model name patterns** (e.g., "vision" in name) → Heuristic match
4. **Provider heuristics** (e.g., OpenAI models support function calling) → Fallback

#### Example: Function Calling Detection

```python
def _has_function_calling(model: dict) -> bool:
    # 1. Check explicit flag
    if model.get('supports_functions'):
        return True
    
    # 2. Check ability score
    if model.get('tool_use_ability', 0) > 0.3:
        return True
    
    # 3. Check provider (heuristic fallback)
    creator = model.get('creator_slug', '').lower()
    if creator in {'anthropic', 'openai', 'google', 'cohere', 'mistral', 'meta'}:
        return True
    
    # 4. Check model family name
    name = model.get('name', '').lower()
    if any(family in name for family in {'claude', 'gpt', 'gemini', 'llama'}):
        return True
    
    return False
```

### Available Capability Requirements

Use these with `ConstraintConfig.capabilities` (hard) or `preferred_capabilities` (soft):

```python
from llm_jury.ranking import CapabilityRequirement

# Output capabilities
CapabilityRequirement.FUNCTION_CALLING  # Tool use / function calling
CapabilityRequirement.JSON_MODE         # Structured JSON output
CapabilityRequirement.STREAMING         # Streaming responses

# Input capabilities
CapabilityRequirement.VISION            # Image input
CapabilityRequirement.AUDIO             # Audio input
CapabilityRequirement.FILE_UPLOAD       # Document processing

# Special capabilities
CapabilityRequirement.EMBEDDINGS        # Embedding generation
CapabilityRequirement.REASONING         # Extended thinking models

# Context-related
CapabilityRequirement.LONG_CONTEXT      # 100K+ context
CapabilityRequirement.VERY_LONG_CONTEXT # 200K+ context  
CapabilityRequirement.MILLION_CONTEXT   # 1M+ context
```

### Use Case Presets

Common configurations are available as presets:

```python
from llm_jury.ranking import UseCaseConstraints

# RAG pipeline with 50K documents
constraints = UseCaseConstraints.rag_pipeline(document_size_tokens=50_000)

# Agentic workflows requiring function calling
constraints = UseCaseConstraints.function_calling()

# Long document analysis (100K+ docs)
constraints = UseCaseConstraints.long_document_analysis(document_tokens=100_000)

# Vision/image analysis tasks
constraints = UseCaseConstraints.vision_analysis()

# Complex reasoning tasks
constraints = UseCaseConstraints.reasoning_heavy()

# Budget-conscious selection
constraints = UseCaseConstraints.budget_conscious(max_cost_per_m=1.0)
```

### Complete Constrained Optimization Example

```python
import json
from llm_jury.ranking import (
    Optimizer,
    OptimizationStrategy,
    ConstraintConfig,
    CapabilityRequirement,
    apply_constraints,
    create_context_objective,
)
from llm_jury.core.models import ModelMetadata, RoutingDecision, PromptCategory, ProductArchetype

# Load models
with open("data/models_cache.json") as f:
    models = json.load(f)

# 1. Define constraints for an agentic RAG system
constraints = ConstraintConfig(
    min_context_k=100,           # Need 100K minimum for documents
    target_context_k=200,        # Ideal: 200K for headroom
    capabilities=[
        CapabilityRequirement.FUNCTION_CALLING,  # Required for tool use
    ],
    preferred_capabilities=[
        CapabilityRequirement.REASONING,  # Prefer reasoning models
    ],
    max_input_cost_per_m=10.0,   # Budget: max $10/M input tokens
)

# 2. Apply hard constraints (pre-filter)
filtered = apply_constraints(models, constraints, verbose=True)
# Output: Context >= 100K: 41/51 models
#         FUNCTION_CALLING: 41/41 models
#         Max input cost: 38 models
#         Final: 38/51 models pass constraints

# 3. Setup optimizer with soft context objective
baseline = ModelMetadata(name="Gemini 2.5 Pro", context_window_k=1048)

optimizer = Optimizer(
    baseline_model=baseline,
    all_models_data=filtered,
    strategy=OptimizationStrategy.HYBRID,
)

# Add context as 6th objective (rewards excess context)
context_obj = create_context_objective(
    target_context_k=200,
    weight=0.12,        # 12% weight
    excess_bonus=True,  # Bonus for context > target
)
optimizer.objectives.register(context_obj)

# 4. Run optimization
decision = RoutingDecision(
    archetype=ProductArchetype.RAG_SPECIALIST,
    category=PromptCategory.GENERAL,
    reason="Agentic RAG pipeline",
)

# Convert dicts to ModelMetadata for ranking
model_objects = [dict_to_model(d) for d in filtered]
results = optimizer.rank(model_objects, decision, top_k=5)

# Results now consider: quality, cost, latency, hallucination, refusal, AND context
```

### Verifying Capability Detection

Check how capabilities are being detected for your models:

```python
from llm_jury.ranking import check_model_capability, CapabilityRequirement
import json

with open("data/models_cache.json") as f:
    models = json.load(f)

for model in models[:10]:
    name = model["name"]
    fn_call = check_model_capability(model, CapabilityRequirement.FUNCTION_CALLING)
    vision = check_model_capability(model, CapabilityRequirement.VISION)
    reasoning = check_model_capability(model, CapabilityRequirement.REASONING)
    
    print(f"{name}: fn_call={fn_call}, vision={vision}, reasoning={reasoning}")
```

If a model is being incorrectly classified, add explicit flags to your cache:

```json
{
  "name": "My Custom Model",
  "supports_functions": true,
  "supports_vision": false,
  "is_reasoning_model": true
}
```

---

### Adding Custom Attributes for New Objectives

To support a custom objective, add the relevant attribute to your model data:

```python
# Option 1: Add attribute to ModelMetadata objects
model.ethics_score = 85.0
model.safety_rating = 92.0

# Option 2: Use a dict accessor in the extractor
ethics_objective = Objective(
    name="ethics",
    extractor=lambda m, d, ctx: (
        m.ethics_score if hasattr(m, 'ethics_score') 
        else m.__dict__.get('ethics_score', 50.0)
    ),
    ...
)

# Option 3: Use context for complex lookups
def extract_ethics(model, decision, context):
    # Access external data source
    ethics_db = context.get("ethics_database", {})
    return ethics_db.get(model.name, 50.0)

# Pass custom context to optimizer
optimizer = Optimizer(...)
optimizer._context["ethics_database"] = load_ethics_data()
```

---

## Adding New Metrics to the Model Cache

The model cache (`data/models_cache.json`) is the primary data store. To add new metrics for custom objectives, you'll need to extend the cache with your data.

### Model Cache Structure

The cache is a JSON array where each model is an object:

```json
[
  {
    "name": "GPT-4o",
    "slug": "gpt-4o",
    "creator_name": "OpenAI",
    
    // Benchmark indices (from Artificial Analysis)
    "intelligence_index": 45.2,
    "coding_index": 42.1,
    "math_index": 48.3,
    
    // Pricing
    "input_cost_per_m": 2.50,
    "output_cost_per_m": 10.00,
    
    // Performance
    "output_tokens_per_second": 85.4,
    "time_to_first_token_seconds": 0.45,
    "measured_ttft_seconds": 0.38,
    
    // Reliability (from Vectara)
    "hallucination_rate": 3.5,
    "factual_consistency_rate": 96.5,
    "refusal_rate": 2.1,
    
    // Your custom metrics go here
    "ethics_score": 92.0,
    "safety_rating": 95.0
  },
  // ... more models
]
```

### Method 1: Manual Cache Extension

Add your metrics directly to the cache file:

```python
import json

# Load existing cache
with open("data/models_cache.json", "r") as f:
    cache = json.load(f)

# Your custom metric data (from your data source)
ethics_scores = {
    "GPT-4o": 92.0,
    "Claude 3.5 Sonnet": 94.0,
    "Gemini 1.5 Pro": 88.0,
    # ... more models
}

# Merge into cache
for model in cache:
    model_name = model.get("name", "")
    model["ethics_score"] = ethics_scores.get(model_name, 50.0)  # Default value

# Save updated cache
with open("data/models_cache.json", "w") as f:
    json.dump(cache, f, indent=2)
```

### Method 2: Extend the ETL Pipeline

For repeatable data collection, add a new step to the ETL pipeline:

```python
# In llm_jury/etl/custom_metrics.py
import requests

class EthicsDataClient:
    """Fetch ethics scores from your data source."""
    
    def fetch_ethics_scores(self) -> dict:
        """Returns dict mapping model names to ethics scores."""
        # Example: fetch from API
        response = requests.get("https://your-api.com/ethics-scores")
        data = response.json()
        return {item["model"]: item["score"] for item in data}

# In your ETL run script
from llm_jury.etl import ETLPipeline
from your_module import EthicsDataClient

# Run standard ETL
pipeline = ETLPipeline()
cache = pipeline.run(require_complete_benchmarks=True)

# Add custom metrics
ethics_client = EthicsDataClient()
ethics_scores = ethics_client.fetch_ethics_scores()

for model in cache:
    model["ethics_score"] = ethics_scores.get(model["name"], 50.0)

# Save
pipeline.save_cache(cache)
```

### Method 3: Runtime Injection

If you don't want to modify the cache file, inject metrics at runtime:

```python
from llm_jury import ModelRegistry, Optimizer, Objective, ObjectiveRegistry

# Load models
registry = ModelRegistry()
models = registry.get_all_models()

# Load your custom data
ethics_data = load_ethics_from_csv("ethics_scores.csv")

# Inject into model objects
for model in models:
    model.ethics_score = ethics_data.get(model.name, 50.0)

# Now create optimizer with custom objective
registry = ObjectiveRegistry.default()
registry.register(Objective(
    name="ethics",
    extractor=lambda m, d, ctx: getattr(m, 'ethics_score', 50.0),
    ...
))

optimizer = Optimizer(baseline, registry.get_raw_data(), objectives=registry)
```

### Naming Conventions

When adding new metrics to the cache, follow these conventions:

| Pattern | Example | Use Case |
|---------|---------|----------|
| `*_score` | `ethics_score`, `safety_score` | 0-100 scores (higher is better) |
| `*_rate` | `error_rate`, `bias_rate` | 0-100 percentages (lower is better) |
| `*_index` | `reasoning_index` | Composite benchmark indices |
| `*_seconds` | `cold_start_seconds` | Time measurements |
| `*_per_m` | `cache_cost_per_m` | Per-million-token costs |

### Connecting Cache Fields to Objectives

Once your field is in the cache, connect it to an Objective:

```python
# Field in cache: "ethics_score": 92.0

ethics_objective = Objective(
    name="ethics",
    display_name="Ethics Score",
    direction="maximize",           # Higher ethics_score is better
    default_weight=0.15,
    default_value=50.0,             # Fallback if field missing
    extractor=lambda m, d, ctx: getattr(m, 'ethics_score', 50.0),
    normalization=NormalizationMethod.PERCENTAGE,  # 0-100 → 0-1
)
```

For rate-based metrics (lower is better):

```python
# Field in cache: "bias_rate": 8.5

bias_objective = Objective(
    name="bias",
    display_name="Bias Rate",
    direction="minimize",           # Lower bias_rate is better
    default_weight=0.10,
    default_value=15.0,
    extractor=lambda m, d, ctx: getattr(m, 'bias_rate', 15.0),
    normalization=NormalizationMethod.INVERSE_PERCENTAGE,  # Inverts: 0% → 1.0, 100% → 0.0
)
```

### Verifying Your Data

After adding metrics, verify they're accessible:

```python
from llm_jury import ModelRegistry

models = ModelRegistry.load_cache()

# Check a few models
for model in models[:5]:
    print(f"{model.name}:")
    print(f"  ethics_score: {getattr(model, 'ethics_score', 'NOT FOUND')}")
    print(f"  safety_rating: {getattr(model, 'safety_rating', 'NOT FOUND')}")
```

---

## Using Custom Model Caches

You can provide your own model cache file instead of using the default. This is useful for:
- Testing with a subset of models
- Using proprietary model data
- Offline/air-gapped environments

### Loading from Custom Cache

```python
from llm_jury import ModelRegistry, get_recommendations

# Load models from custom cache
models = ModelRegistry.load_cache("/path/to/my_models.json")

# Get raw data for QualityScorer
raw_data = ModelRegistry.load_raw_cache("/path/to/my_models.json")

# Use custom cache with get_recommendations
results = get_recommendations(
    "Write a Python function",
    cache_path="/path/to/my_models.json"
)
```

### Custom Cache Format

Your cache file should be a JSON array of model objects:

```json
[
  {
    "name": "My Custom Model",
    "slug": "my-custom-model",
    "creator_name": "MyCompany",
    
    "intelligence_index": 45.0,
    "coding_index": 42.0,
    "math_index": 38.0,
    
    "input_cost_per_m": 1.00,
    "output_cost_per_m": 2.00,
    
    "time_to_first_token_seconds": 0.5,
    "output_tokens_per_second": 80.0,
    
    "hallucination_rate": 5.0,
    "refusal_rate": 2.0,
    
    "my_custom_metric": 92.0
  }
]
```

### Required Fields for Optimization

The optimizer uses **5 built-in objectives**. Each objective extracts specific fields from your models. Here's what you need:

#### Absolute Minimum (optimizer will run but with limited accuracy)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Model identifier (required for all models) |
| `input_cost_per_m` | float | Cost per 1M input tokens (USD) |
| `output_cost_per_m` | float | Cost per 1M output tokens (USD) |

> ⚠️ Models without `input_cost_per_m` and `output_cost_per_m` will be **filtered out** of optimization.

#### Complete Field Reference by Objective

**1. Quality Objective** (benchmark-based scoring)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `intelligence_index` | float | 0-100 | General intelligence composite score |
| `coding_index` | float | 0-100 | Coding benchmark composite score |
| `math_index` | float | 0-100 | Math benchmark composite score |
| `mmlu_pro` | float | 0-1 | MMLU-Pro benchmark score |
| `gpqa` | float | 0-1 | GPQA benchmark score |
| `hle` | float | 0-1 | HLE benchmark score |
| `livecodebench` | float | 0-1 | LiveCodeBench score |
| `scicode` | float | 0-1 | SciCode benchmark score |
| `math_500` | float | 0-1 | MATH-500 benchmark score |
| `aime` | float | 0-1 | AIME benchmark score |

> If quality fields are missing, the model will receive a lower quality score based on available data.

**2. Cost Objective** (pricing)

| Field | Type | Description |
|-------|------|-------------|
| `input_cost_per_m` | float | Cost per 1M input tokens (USD) - **REQUIRED** |
| `output_cost_per_m` | float | Cost per 1M output tokens (USD) - **REQUIRED** |

> Alternative field names also accepted: `price_1m_input`, `price_1m_output`

**3. Latency Objective** (time to first token)

| Field | Type | Description | Priority |
|-------|------|-------------|----------|
| `measured_ttft_seconds` | float | Measured TTFT in seconds | 1st (preferred) |
| `time_to_first_token_seconds` | float | API-reported TTFT | 2nd |
| `median_latency_ms` | float | Median latency in milliseconds | 3rd (converted) |

> Default if missing: 1.0 seconds

**4. Hallucination Objective**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hallucination_rate` | float | 0-100 | Percentage of responses with hallucinations |

> Default if missing: 15.0%

**5. Refusal Objective**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `refusal_rate` | float | 0-100 | Percentage of prompts the model refuses |

> Default if missing: 5.0%

### Minimum Viable Cache Example

Here's the absolute minimum cache that will work:

```json
[
  {
    "name": "Model A",
    "input_cost_per_m": 1.00,
    "output_cost_per_m": 3.00
  },
  {
    "name": "Model B", 
    "input_cost_per_m": 0.50,
    "output_cost_per_m": 1.50
  }
]
```

> ⚠️ This will run but quality scores will be poor due to missing benchmarks.

### Recommended Complete Cache Example

For accurate optimization, include all objective fields:

```json
[
  {
    "name": "My Model",
    "input_cost_per_m": 1.00,
    "output_cost_per_m": 3.00,
    
    "intelligence_index": 45.0,
    "coding_index": 42.0,
    "math_index": 38.0,
    
    "measured_ttft_seconds": 0.45,
    
    "hallucination_rate": 5.2,
    "refusal_rate": 1.8
  }
]
```

### Validating Your Cache

Use this script to check if your cache has all required fields:

```python
import json
from pathlib import Path

def validate_cache(cache_path: str) -> dict:
    """Validate a model cache for optimization compatibility."""
    
    with open(cache_path) as f:
        models = json.load(f)
    
    if not isinstance(models, list):
        models = models.get('models', [])
    
    # Required fields
    required = ['name', 'input_cost_per_m', 'output_cost_per_m']
    
    # Fields for each objective
    quality_fields = [
        'intelligence_index', 'coding_index', 'math_index',
        'mmlu_pro', 'gpqa', 'hle', 'livecodebench'
    ]
    latency_fields = ['measured_ttft_seconds', 'time_to_first_token_seconds', 'median_latency_ms']
    
    results = {
        'total_models': len(models),
        'valid_models': 0,
        'missing_required': [],
        'quality_coverage': 0,
        'latency_coverage': 0,
        'hallucination_coverage': 0,
        'refusal_coverage': 0,
    }
    
    for model in models:
        name = model.get('name', 'Unknown')
        
        # Check required
        missing = [f for f in required if not model.get(f)]
        if missing:
            results['missing_required'].append((name, missing))
        else:
            results['valid_models'] += 1
        
        # Check coverage
        if any(model.get(f) for f in quality_fields):
            results['quality_coverage'] += 1
        if any(model.get(f) for f in latency_fields):
            results['latency_coverage'] += 1
        if model.get('hallucination_rate') is not None:
            results['hallucination_coverage'] += 1
        if model.get('refusal_rate') is not None:
            results['refusal_coverage'] += 1
    
    # Print report
    print(f"Cache Validation Report: {cache_path}")
    print(f"{'='*50}")
    print(f"Total models: {results['total_models']}")
    print(f"Valid for optimization: {results['valid_models']}")
    print()
    print("Objective Coverage:")
    print(f"  Quality:       {results['quality_coverage']}/{results['total_models']} ({100*results['quality_coverage']/max(1,results['total_models']):.0f}%)")
    print(f"  Latency:       {results['latency_coverage']}/{results['total_models']} ({100*results['latency_coverage']/max(1,results['total_models']):.0f}%)")
    print(f"  Hallucination: {results['hallucination_coverage']}/{results['total_models']} ({100*results['hallucination_coverage']/max(1,results['total_models']):.0f}%)")
    print(f"  Refusal:       {results['refusal_coverage']}/{results['total_models']} ({100*results['refusal_coverage']/max(1,results['total_models']):.0f}%)")
    
    if results['missing_required']:
        print()
        print(f"⚠️  {len(results['missing_required'])} models missing required fields:")
        for name, missing in results['missing_required'][:5]:
            print(f"  - {name}: missing {missing}")
        if len(results['missing_required']) > 5:
            print(f"  ... and {len(results['missing_required'])-5} more")
    
    return results

# Usage
validate_cache("/path/to/my_models.json")
```

### What Happens When Fields Are Missing

The optimizer supports two strategies for handling missing data:

#### MissingDataStrategy.STRICT (Default)

Only includes models with complete data for all objectives. This gives the highest confidence results but may reduce the candidate pool.

```python
from llm_jury import Optimizer, MissingDataStrategy

optimizer = Optimizer(
    baseline, models_data,
    missing_data=MissingDataStrategy.STRICT  # Default
)
```

| Missing Field | Behavior |
|---------------|----------|
| `input_cost_per_m` | Model excluded from optimization |
| `output_cost_per_m` | Model excluded from optimization |
| `hallucination_rate` | Model excluded from optimization |
| `refusal_rate` | Model excluded from optimization |

#### MissingDataStrategy.IMPUTE

Includes all models, using default or custom values for missing data. More models available but lower confidence in rankings.

```python
from llm_jury import Optimizer, MissingDataStrategy

# Use default imputation values
optimizer = Optimizer(
    baseline, models_data,
    missing_data=MissingDataStrategy.IMPUTE
)

# Or specify custom imputation values
optimizer = Optimizer(
    baseline, models_data,
    missing_data=MissingDataStrategy.IMPUTE,
    imputation_values={
        "hallucination": 25.0,  # Assume worse if unknown
        "refusal": 15.0,        # Assume higher refusal if unknown
    }
)
```

| Missing Field | Default Imputation |
|---------------|-------------------|
| Latency fields | 1.0 seconds |
| `hallucination_rate` | 15.0% |
| `refusal_rate` | 5.0% |
| Quality benchmarks | Scored based on available data |

#### Checking Data Completeness

You can inspect which models have complete vs incomplete data:

```python
# Check if a specific model has complete data
optimizer.has_complete_data(model)

# Get detailed missing data report
report = optimizer.get_missing_data_report(model)
# Returns: {"hallucination": ["hallucination_rate"], ...}

# Filter to only complete models
complete_models = optimizer.filter_complete_models(models, verbose=True)
```

### Configuration

Set a default custom cache path via config:

```python
from llm_jury import get_config

config = get_config()
config.cache_file_path = "/path/to/my_models.json"

# Now all calls use your custom cache by default
from llm_jury import get_recommendations
results = get_recommendations("My prompt")  # Uses custom cache
```

Or via environment:

```bash
# In .env or shell
export LLM_JURY_CACHE_PATH="/path/to/my_models.json"
```

### Data Source Integration

The built-in objectives pull from these sources:

| Metric | Source | Fields |
|--------|--------|--------|
| Quality | Artificial Analysis API | `intelligence_index`, `coding_index`, `math_index`, `mmlu_pro`, `gpqa`, etc. |
| Cost | Artificial Analysis API | `input_cost_per_m`, `output_cost_per_m` |
| Latency | OpenRouter (measured) | `measured_ttft_seconds` |
| Hallucination | Vectara Leaderboard | `hallucination_rate` |
| Refusal | Vectara Leaderboard | `refusal_rate` |

To add a new data source for custom metrics, you can:

1. **Extend the ETL pipeline** to fetch and merge new data
2. **Add attributes** to `ModelMetadata` in `core/models.py`
3. **Register objectives** that extract from those attributes

---

## The Utopia Point

The algorithm optimizes toward an ideal "utopia point" where all objectives are perfect:

| Objective | Utopia Value | Direction | Meaning |
|-----------|--------------|-----------|---------|
| Quality | 100% | Maximize | Perfect benchmark scores |
| Cost | $0 | Minimize | Free to use |
| Latency (TTFT) | 0ms | Minimize | Instant response |
| Hallucination | 0% | Minimize | Never makes things up |
| Refusal | 0% | Minimize | Always attempts to answer |

No real model achieves the utopia point, but the algorithm finds models that get closest.

## Chebyshev Scalarization

### Why Chebyshev?

Traditional weighted-sum approaches have a critical flaw: they can miss Pareto-optimal solutions in non-convex regions of the objective space. Chebyshev scalarization solves this by minimizing the **maximum weighted deviation** from the utopia point.

### The Algorithm

For each model, we calculate:

1. **Normalize each metric** to a 0-1 scale where 1 = utopia (best)
2. **Calculate regret** for each objective: `regret = 1.0 - normalized_value`
3. **Apply weights** based on the chosen strategy
4. **Chebyshev distance** = `max(weight_i × regret_i)` for all objectives

The model with the **lowest Chebyshev distance** is ranked first.

### Mathematical Formulation

```
minimize: max{ w_i × (1 - ñ_i) } for all objectives i

where:
  ñ_i = normalized value for objective i (0-1, higher is better)
  w_i = weight for objective i (sum to 1.0)
```

## Normalization

Each metric is normalized differently based on its nature:

### Quality (0-100 → 0-1)
```python
normalized_quality = quality_score / 100.0
```

### Cost (logarithmic, relative to baseline)
```python
cost_ratio = model_cost / baseline_cost
normalized_cost = 1.0 / (1.0 + cost_ratio)
# Result: 0x (free) = 1.0, 0.5x = 0.67, 1x = 0.5, 2x = 0.33, 10x = 0.09
```

### Latency (min-max normalization, inverted)
```python
normalized_latency = 1.0 - (latency - min_latency) / (max_latency - min_latency)
# Fastest model = 1.0, slowest = 0.0
```

### Hallucination & Refusal (percentage, inverted)
```python
normalized_hallucination = 1.0 - (hallucination_rate / 100.0)
normalized_refusal = 1.0 - (refusal_rate / 100.0)
# 0% = 1.0 (perfect), 100% = 0.0 (worst)
```

## Ranking Strategies

Different use cases call for different weight distributions. Use this decision guide to choose the right strategy:

### Quick Decision Guide

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Which Strategy Should I Use?                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  "I don't know what I want"                                             │
│      └──► KNEE (default - best bang for buck on Pareto frontier)       │
│                                                                         │
│  "I want the best bang for my buck"                                     │
│      └──► KNEE (finds diminishing returns sweet spot)                  │
│                                                                         │
│  "I have specific requirements"                                         │
│      ├── Quality is #1 priority ──► QUALITY_FOCUSED                    │
│      ├── Budget is tight ──► COST_FOCUSED                              │
│      ├── Need real-time responses ──► SPEED_FOCUSED                    │
│      └── Can't afford mistakes ──► RELIABILITY_FOCUSED                 │
│                                                                         │
│  "I want X% quality at Y% cost"                                         │
│      └──► VALUE_OPTIMIZED (with custom constraints)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Strategy Comparison Table

| Strategy | Best For | Prioritizes | Sacrifices |
|----------|----------|-------------|------------|
| **BALANCED** | General use, unsure users | Nothing specific | Nothing specific |
| **KNEE** | Cost-conscious, pragmatic users | Best trade-off point | May miss extremes |
| **QUALITY_FOCUSED** | Research, medical, legal | Accuracy | Cost, speed |
| **COST_FOCUSED** | High-volume, budget apps | Savings | Quality |
| **SPEED_FOCUSED** | Real-time, interactive | Latency | Quality, cost |
| **RELIABILITY_FOCUSED** | Factual content, customer-facing | Trust | Cost, speed |
| **VALUE_OPTIMIZED** | Specific requirements | User-defined | Flexibility |

---

### BALANCED
**Use when:** You want equal weight on all dimensions (quality, cost, speed, reliability).

**How it works:** Gives roughly equal weight to all objectives, preventing any single factor from dominating.

| Quality | Cost | Latency | Hallucination | Refusal |
|---------|------|---------|---------------|---------|
| 35% | 20% | 15% | 20% | 10% |

**Example result:** A model that's good (not great) at everything.

---

### KNEE (Default)
**Use when:** You want the best "bang for your buck" - the point where paying more gives diminishing returns.

**How it works:** Finds the point of maximum curvature on the Pareto frontier. This is where the trade-off curve bends - before this point, you get a lot of value for your money; after it, improvements are expensive.

**Key insight:** KNEE is different from BALANCED. BALANCED minimizes worst-case regret across objectives. KNEE finds where the trade-off curve "bends" - the natural sweet spot.

**Why it's the default:** Most users want to maximize value. KNEE automatically finds the optimal quality/cost tradeoff without requiring manual constraint tuning.

```
Quality ▲
         │         ● Premium (GPT-5, Claude Opus)
         │       ╱   ← Diminishing returns zone
         │     ╱     (pay a lot more, get a little better)
         │   ★ ← KNEE POINT (best value)
         │  ╱        (optimal trade-off)
         │╱ ← Value zone
         ●──────────────────► Cost
       Budget models
```

**When KNEE beats BALANCED:**
- You're cost-conscious but don't want to sacrifice too much quality
- You want a data-driven "default" rather than subjective weights
- You're comparing many models and want the natural break point

**When BALANCED beats KNEE:**
- You have specific priorities (use focused strategies instead)
- You want predictable, weight-based optimization
- The model landscape is sparse (few options)

---

### QUALITY_FOCUSED
**Use when:** Accuracy is paramount and cost/speed are secondary. Research, medical, legal applications.

| Quality | Cost | Latency | Hallucination | Refusal |
|---------|------|---------|---------------|---------|
| 50% | 10% | 10% | 20% | 10% |

**Trade-off:** You'll pay more and wait longer, but get the best results.

---

### COST_FOCUSED
**Use when:** Budget is the primary constraint. High-volume applications, prototyping, non-critical tasks.

| Quality | Cost | Latency | Hallucination | Refusal |
|---------|------|---------|---------------|---------|
| 20% | 40% | 15% | 15% | 10% |

**Trade-off:** Quality may suffer, but you'll save significantly on API costs.

---

### SPEED_FOCUSED
**Use when:** Real-time applications requiring low latency. Chatbots, interactive tools, gaming.

| Quality | Cost | Latency | Hallucination | Refusal |
|---------|------|---------|---------------|---------|
| 20% | 15% | 40% | 15% | 10% |

**Trade-off:** May sacrifice quality for faster Time To First Token.

---

### RELIABILITY_FOCUSED
**Use when:** Factual accuracy and responsiveness are critical. Customer-facing content, fact-checking, knowledge bases.

| Quality | Cost | Latency | Hallucination | Refusal |
|---------|------|---------|---------------|---------|
| 25% | 10% | 10% | 35% | 20% |

**Trade-off:** Prioritizes models that don't hallucinate and don't refuse to answer.

---

### VALUE_OPTIMIZED
**Use when:** You have specific requirements like "I want 80-95% of GPT-4 quality at 10-30% of the cost."

| Quality | Cost | Latency | Hallucination | Refusal |
|---------|------|---------|---------------|---------|
| 35% | 25% | 15% | 15% | 10% |

**How it works:** First filters to models meeting your constraints, then optimizes within that set.

```python
optimizer = Optimizer(
    baseline_model=gpt4,
    strategy=OptimizationStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),  # 80-95% of baseline
    cost_range=(0.10, 0.30),     # 10-30% of baseline cost
)
```

---

## BALANCED vs KNEE: When to Use Each

This is the most common question. Here's a detailed comparison:

| Aspect | BALANCED | KNEE |
|--------|----------|------|
| **Philosophy** | "Minimize worst-case regret" | "Find diminishing returns point" |
| **Math** | Chebyshev distance (max weighted regret) | Perpendicular distance from Pareto line |
| **Weights** | User-defined (35/20/15/20/10) | Implicit (based on frontier shape) |
| **Best for** | Known priorities | Unknown priorities, exploration |
| **Predictability** | High (same weights = same ranking) | Medium (depends on model landscape) |
| **Sensitivity** | To weight choices | To model distribution |

### Use BALANCED when:
- ✅ You know your priorities (quality vs cost vs speed)
- ✅ You want consistent, reproducible rankings
- ✅ You're building a production system with fixed requirements
- ✅ You want to tune weights over time

### Use KNEE when:
- ✅ You're exploring and don't know what trade-offs exist
- ✅ You want a "smart default" without choosing weights
- ✅ You're cost-conscious but quality-aware
- ✅ You want to find the natural break point in value
- ✅ You're presenting options to non-technical stakeholders

## The KNEE Algorithm in Detail

The KNEE strategy uses a geometric approach to find the optimal trade-off point:

### Step 1: Calculate Composite Scores
For each model, compute:
- **Benefit** = sum of all normalized objective scores (higher = better overall)
- **Cost** = sum of all regrets (higher = more compromises)

### Step 2: Approximate Pareto Frontier
Sort models by benefit (descending). This gives an approximation of the Pareto frontier.

### Step 3: Find the Knee
Draw a line from the worst model (low benefit, low cost) to the best model (high benefit, high cost). The "knee" is the model with the maximum perpendicular distance *below* this line.

```
Benefit
    ▲
    │              ● Best model
    │            ╱
    │          ╱
    │        ╱   ← Line from worst to best
    │      ★ ← Knee (max distance below line)
    │    ╱
    │  ╱
    │╱
    ●─────────────────────────► Cost (regret)
  Worst model
```

### Step 4: Position Bonus
Models in the middle of the frontier get a bonus. Extreme models (all quality or all cost) are penalized because they represent edge cases, not balanced trade-offs.

### Why This Works
The knee point is where the "slope" of the trade-off changes most dramatically. Before the knee, you get a lot of quality per dollar. After the knee, you pay a lot more for marginal improvements. The knee is the inflection point.

### KNEE Customization Options

The KNEE strategy supports two customization parameters:

#### 1. `knee_objective_weights` - Emphasize certain objectives

By default, all objectives contribute equally to the benefit/cost calculation. You can weight them differently:

```python
# Emphasize quality in knee detection
optimizer = Optimizer(
    baseline, models_data,
    strategy=OptimizationStrategy.KNEE,
    knee_objective_weights={
        "quality": 2.0,        # Double weight for quality
        "cost": 1.0,
        "latency": 1.0,
        "hallucination": 1.5,  # 1.5x weight for reliability
        "refusal": 1.0,
    }
)
```

This changes how "benefit" and "cost" are calculated for each model, shifting the knee point toward models that excel in your prioritized objectives.

#### 2. `knee_position_weight` - Control middle-of-frontier preference

The algorithm prefers models in the middle of the Pareto frontier (not extremes). You can control this:

```python
# Default: moderate preference for middle (0.3)
optimizer = Optimizer(..., knee_position_weight=0.3)

# No position preference - pure distance from line
optimizer = Optimizer(..., knee_position_weight=0.0)

# Strong preference for middle of frontier
optimizer = Optimizer(..., knee_position_weight=0.8)
```

| Value | Effect |
|-------|--------|
| 0.0 | Only considers distance from the line (may select extremes) |
| 0.3 | Default - mild preference for middle |
| 0.5 | Balanced between distance and position |
| 0.8+ | Strong preference for middle (avoids edge cases) |

---

## Constrained Optimization

The `VALUE_OPTIMIZED` strategy supports constraint-based filtering:

```python
optimizer = Optimizer(
    baseline_model=gemini_3_pro,
    all_models_data=models_data,
    strategy=OptimizationStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),  # 80-95% of baseline quality
    cost_range=(0.10, 0.30),     # 10-30% of baseline cost
    speed_range=(1.0, 2.0),      # 1-2x baseline speed (optional)
)
```

This first filters to models meeting the constraints, then applies Chebyshev optimization within the feasible set.

## Custom Weights

You can provide custom weights for any strategy:

```python
optimizer = Optimizer(
    baseline_model=baseline,
    all_models_data=models_data,
    custom_weights={
        "quality": 0.40,
        "cost": 0.30,
        "latency": 0.10,
        "hallucination": 0.15,
        "refusal": 0.05,
    }
)
```

When you add custom objectives, include them in `custom_weights`:

```python
optimizer = Optimizer(
    ...
    objectives=registry_with_ethics,
    custom_weights={
        "quality": 0.35,
        "ethics": 0.15,  # Include your custom objective
        "cost": 0.15,
        ...
    }
)
```

---

## Complete Example: Adding a Safety Metric

Here's a full example of adding a "safety" objective that reads from a custom attribute:

```python
from llm_jury import (
    Optimizer, OptimizationStrategy,
    Objective, ObjectiveRegistry, NormalizationMethod,
    ModelRegistry
)
from llm_jury.core.models import RoutingDecision, PromptCategory, ProductArchetype

# 1. Load models and add safety scores
registry = ModelRegistry()
models = registry.get_all_models()

# Add safety scores (in practice, load from your data source)
safety_scores = {
    "gpt-4o": 95.0,
    "claude-3-opus": 92.0,
    "gemini-1.5-pro": 88.0,
    # ... more models
}

for model in models:
    model.safety_score = safety_scores.get(model.name, 70.0)

# 2. Define safety objective
safety_objective = Objective(
    name="safety",
    display_name="Safety",
    direction="maximize",
    default_weight=0.10,
    default_value=70.0,
    extractor=lambda m, d, ctx: getattr(m, 'safety_score', 70.0),
    normalization=NormalizationMethod.PERCENTAGE,
)

# 3. Create registry with safety objective
objectives = ObjectiveRegistry.default()
objectives.register(safety_objective)

# 4. Create optimizer
baseline = registry.get_model("gpt-4o")
optimizer = Optimizer(
    baseline_model=baseline,
    all_models_data=registry.get_raw_data(),
    objectives=objectives,
    strategy=OptimizationStrategy.BALANCED,
    custom_weights={
        "quality": 0.30,
        "safety": 0.15,
        "cost": 0.15,
        "latency": 0.10,
        "hallucination": 0.20,
        "refusal": 0.10,
    }
)

# 5. Rank models
decision = RoutingDecision(
    category=PromptCategory.GENERAL,
    archetype=ProductArchetype.FRONTIER,
    cot_template="",
    reason="General task"
)

results = optimizer.rank(models, decision, top_k=5, return_detailed=True)

for r in results:
    print(f"{r.name}: Score={r.chebyshev_score:.4f}")
    print(f"  {r.tradeoff_summary}")
```

---

## Interpreting Results

### Chebyshev Score
- **Lower is better** (closer to utopia)
- Scores typically range from 0.02 to 0.20
- Score < 0.05: Excellent trade-off
- Score 0.05-0.10: Good trade-off
- Score > 0.10: Significant compromises

### Tradeoff Summary
Each result includes a human-readable summary:
```
Quality: 85.6 (-14.4) | Cost: 94% cheaper | TTFT: 632ms (+68%) | Halluc: 19.2% (-5.6)
```

This shows:
- Absolute quality score and difference from baseline
- Cost savings percentage
- TTFT in milliseconds and improvement percentage
- Hallucination rate and difference from baseline

Custom objectives are automatically included in the summary.

## Why This Approach?

### Academic Foundation
Chebyshev scalarization is a well-established technique in multi-objective optimization, proven to find all Pareto-optimal solutions including those in non-convex regions.

### Practical Benefits
1. **Interpretable**: Weights directly correspond to user priorities
2. **Flexible**: Easy to adjust for different use cases
3. **Fair**: Prevents any single objective from dominating
4. **Complete**: Considers all relevant factors simultaneously
5. **Extensible**: Add new objectives without changing core algorithm

### vs. Simple Ranking
Simple approaches like "sort by quality" or "filter by cost then sort" miss the nuanced trade-offs. A model that's 5% worse in quality but 80% cheaper and 50% faster might be the better choice—multi-objective optimization captures this.

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `Optimizer` | Main optimizer class |
| `OptimizationStrategy` | Enum of available strategies |
| `Objective` | Dataclass defining a single optimization objective |
| `ObjectiveRegistry` | Collection of objectives with chainable API |
| `NormalizationMethod` | Enum of normalization approaches |

### Optimizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline_model` | `ModelMetadata` | Required | Reference model for comparisons |
| `all_models_data` | `List[Dict]` | Required | Raw model data for QualityScorer |
| `strategy` | `OptimizationStrategy` | `KNEE` | Optimization strategy (KNEE = best value tradeoff) |
| `objectives` | `ObjectiveRegistry` | `default()` | Custom objectives |
| `quality_range` | `tuple` | `(0.80, 0.95)` | For VALUE_OPTIMIZED |
| `cost_range` | `tuple` | `(0.10, 0.30)` | For VALUE_OPTIMIZED |
| `speed_range` | `tuple` | `None` | For VALUE_OPTIMIZED |
| `custom_weights` | `Dict[str, float]` | `None` | Override objective weights |
| `knee_position_weight` | `float` | `0.3` | For KNEE strategy |
| `knee_objective_weights` | `Dict[str, float]` | `None` | For KNEE strategy |

## References

- Miettinen, K. (1999). *Nonlinear Multiobjective Optimization*. Springer.
- Bowman, V.J. (1976). "On the Relationship of the Tchebycheff Norm and the Efficient Frontier of Multiple-Criteria Objectives"
- Artificial Analysis: https://artificialanalysis.ai/
- Vectara Hallucination Leaderboard: https://github.com/vectara/hallucination-leaderboard
