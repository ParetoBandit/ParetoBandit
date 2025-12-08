# Quality Scoring System

This document explains how LLM Jury calculates quality scores for language models.

## Overview

The quality scoring system evaluates models across multiple dimensions using benchmark data from Artificial Analysis. It provides:

- **Task-specific scoring** with different weights for coding, data science, creative, and general use cases
- **Non-saturating distributions** that prevent score clustering at the top
- **Multiple optimization strategies** (Chebyshev, Knee Point) for different use cases

## Benchmarks Used

We use 12 benchmarks to evaluate models:

| Benchmark | Description | Scale |
|-----------|-------------|-------|
| `intelligence_index` | General reasoning ability | 0-100 |
| `coding_index` | Programming across difficulties | 0-100 |
| `math_index` | Mathematical reasoning | 0-100 |
| `mmlu_pro` | Massive Multitask Language Understanding | 0-1 |
| `gpqa` | Graduate-level science questions | 0-1 |
| `hle` | Hard language understanding | 0-1 |
| `livecodebench` | LeetCode-style algorithmic problems | 0-1 |
| `scicode` | Scientific/research-level coding | 0-1 |
| `math_500` | Competition math problems | 0-1 |
| `aime` | American Invitational Math Exam | 0-1 |
| `hallucination_rate` | Tendency to hallucinate (lower = better) | 0-100 |
| `factual_consistency_rate` | Factual accuracy (higher = better) | 0-100 |

### Trust Score

We derive a `trust_score` from hallucination metrics:

```python
trust_score = 100 - hallucination_rate
```

This inverted metric ensures higher values = better (consistent with other benchmarks).

## Task-Specific Weights

Different use cases prioritize different benchmarks:

### CODING
| Benchmark | Weight | Why |
|-----------|--------|-----|
| `coding_index` | 35% | Primary capability |
| `livecodebench` | 25% | Algorithmic problem-solving |
| `scicode` | 15% | Expert-level indicator |
| `math_index` | 10% | Logic/reasoning |
| `intelligence_index` | 8% | General capability |
| Other | 7% | Minor factors |

**Complexity Ladder:** `coding_index` → `livecodebench` → `scicode`

### DATA_SCIENCE
| Benchmark | Weight | Why |
|-----------|--------|-----|
| `math_index` | 30% | Primary capability |
| `coding_index` | 18% | Implementation ability |
| `gpqa` | 12% | Scientific knowledge |
| `intelligence_index` | 12% | General reasoning |
| `math_500` | 8% | Competition math |
| `aime` | 5% | Olympiad-level |
| Other | 15% | Minor factors |

**Complexity Ladder:** `math_index` → `math_500` → `aime`

### CREATIVE
| Benchmark | Weight | Why |
|-----------|--------|-----|
| `intelligence_index` | 40% | Creativity proxy |
| `mmlu_pro` | 28% | Knowledge breadth |
| `gpqa` | 12% | Domain knowledge |
| `hle` | 10% | Language nuance |
| `trust_score` | 8% | Reliability |
| Other | 2% | Minor factors |

### GENERAL (with Trust)
| Benchmark | Weight | Why |
|-----------|--------|-----|
| `intelligence_index` | 25% | Primary capability |
| `trust_score` | 18% | **Reliability critical** |
| `mmlu_pro` | 15% | Knowledge breadth |
| `coding_index` | 12% | Versatility |
| `math_index` | 10% | Reasoning |
| Other | 20% | Balanced factors |

**Note:** The `trust_score` at 18% means GENERAL has a lower maximum score (~82) than other categories (~100).

## Score Calculation

### Step 1: Percentile Normalization

Each benchmark value is normalized based on its percentile in the model population:

```python
if value >= p95:
    # Top 5%: Logarithmic spacing prevents saturation
    score = 0.95 + log_position * 0.05
elif value >= p90:
    # 90-95th percentile: Linear
    score = 0.90 + linear_position * 0.05
elif value >= p75:
    # 75-90th percentile: Linear
    score = 0.75 + linear_position * 0.15
elif value >= p50:
    # 50-75th percentile: Linear
    score = 0.50 + linear_position * 0.25
else:
    # Bottom 50%: Linear
    score = linear_position * 0.50
```

This prevents saturation where all top models get the same score.

### Step 2: Weighted Composite

```python
composite = sum(normalized[key] * weight[key] for key in benchmarks)
composite /= sum(weights)  # Normalize for missing data
```

### Step 3: Final Transformation

```python
if composite < 0.5:
    # Bottom half: logarithmic spread
    score = 50 * log1p(composite * 2) / log1p(1)
else:
    # Top half: power law (α=1.5) prevents saturation
    score = 50 + 50 * pow((composite - 0.5) * 2, 1.5)
```

**Result:** Scores 0-100 with only 2-8% of models scoring ≥90.

## Optimization Strategies

### 1. Quality Ranking

Simple: Sort by quality score.

**Best for:** "I want the best model regardless of cost"

### 2. Chebyshev Optimization

Minimizes the maximum regret across multiple dimensions:

```python
chebyshev_distance = max(
    w_quality * quality_regret,
    w_cost * cost_regret,
    w_latency * latency_regret,
    w_trust * trust_regret
)
```

Where regret = normalized distance from ideal (0 = best, 1 = worst).

**Best for:** "I need balanced performance with no weak spots"

**Weights:** Quality 35%, Cost 30%, Latency 15%, Trust 20%

### 3. Knee Point Optimization

Finds the point of maximum curvature on the Pareto frontier:

1. Build Pareto frontier (non-dominated models by quality/cost)
2. Find knee using curvature analysis (angle change between adjacent points)
3. Score models by proximity to knee point

```python
# Knee = point where marginal quality/cost ratio changes most
for i in range(1, len(pareto) - 1):
    angle_change = 1 - dot(v1_normalized, v2_normalized)
    if angle_change > max_angle:
        knee_point = pareto[i]
```

**Best for:** "Where do I get the best bang for my buck?"

## Complexity-Aware Routing

The system can route to different model tiers based on task complexity:

| Complexity | CODING Threshold | DATA_SCIENCE Threshold |
|------------|-----------------|----------------------|
| Simple | `coding_index` ≥ 20 | `math_index` ≥ 30 |
| Medium | `livecodebench` ≥ 0.70 | `math_index` ≥ 50 |
| Hard | `livecodebench` ≥ 0.80 | `math_500` ≥ 0.80 |
| Expert | `scicode` ≥ 0.45 | `aime` ≥ 0.50 |

```python
# Get cheapest model that can handle Expert-level coding
model = scorer.get_minimum_model_for_task(
    TaskComplexity.EXPERT, 
    PromptCategory.CODING
)
# Returns: o4-mini at $1.93/M (scicode ≥ 0.45)
```

## Example Results

### Quality Scores by Use Case

| Model | CODING | DATA_SCIENCE | CREATIVE | GENERAL |
|-------|--------|--------------|----------|---------|
| Gemini 3 Pro | 100.0 | 100.0 | 91.9 | 82.1 |
| Claude Opus 4.5 | 94.8 | 92.1 | 89.2 | 80.3 |
| GPT-5.1 | 92.0 | 93.4 | 86.6 | 79.3 |
| GPT-5 | 86.8 | 91.4 | 84.8 | 77.8 |

### Optimization Comparison (CODING)

| Method | Top Pick | Score | Cost |
|--------|----------|-------|------|
| Quality | Gemini 3 Pro | 100.0 | $4.50/M |
| Chebyshev | Gemini 3 Pro | 100.0 | $4.50/M |
| Knee Point | GPT-5 | 100.0 | $3.44/M |

**Insight:** Knee point finds GPT-5 at $3.44 as the sweet spot - 87% quality at 76% of the cost.

## Color Coding

Visualizations use this color scheme:

| Color | Quality | Label |
|-------|---------|-------|
| 🟢 Green | ≥90 | Excellent |
| 🔵 Blue | ≥80 | Very Good |
| 🩵 Cyan | ≥70 | Good |
| 🟡 Gold | ≥60 | Above Average |
| 🟠 Orange | ≥50 | Average |
| 🔴 Red | ≥40 | Below Average |
| ⚫ Gray | <40 | Low |

## API Usage

```python
from llm_jury.ranking.quality_scorer import QualityScorer, TaskComplexity
from llm_jury.core.models import PromptCategory

# Initialize with model data
scorer = QualityScorer(models_data)

# Get quality score for a specific use case
score = scorer.calculate_quality_score(model, PromptCategory.CODING)

# Get complexity capability
capability = scorer.get_model_complexity_capability(model, PromptCategory.CODING)
# Returns: TaskComplexity.EXPERT

# Get cheapest model for a complexity level
cheapest = scorer.get_minimum_model_for_task(
    TaskComplexity.HARD, 
    PromptCategory.CODING
)

# Get value-optimized recommendations
recommendations = scorer.recommend_for_complexity(
    TaskComplexity.MEDIUM,
    PromptCategory.CODING,
    max_cost=2.0,
    top_n=5
)
```

## Files

- `llm_jury/ranking/quality_scorer.py` - Main scoring logic
- `llm_jury/optimization/chebyshev_scorer.py` - Chebyshev optimization
- `llm_jury/ranking/optimizer.py` - Knee point and multi-objective optimization
- `data/models_cache.json` - Benchmark data for all models


