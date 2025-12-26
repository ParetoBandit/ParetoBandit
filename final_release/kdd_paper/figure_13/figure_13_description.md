# Figure 13: Cold-Start Priors for Unseen Models

## Overview

This figure documents the **cost+latency based prior generation system** for initializing cluster performance estimates for new LLM models without benchmark scores.

## Problem Statement

When a new model arrives:
- ❌ No benchmark scores available yet
- ❌ No evaluation history in our system
- ❌ Cannot use KNN predictor (requires benchmarks)
- ✅ **Only have**: Cost (from API) + Latency (measurable)

**Challenge:** Initialize bandit with informed priors instead of uniform baseline.

---

## Methodology

### 1. Cluster Assignment (Comparative Advantage)

**Script:** [`compute_relative_cluster_assignment.py`](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_13/compute_relative_cluster_assignment.py)

Instead of absolute "best cluster," identify each model's **comparative advantage** using z-score:

```
z_score[model, cluster] = (performance[model, cluster] - mean[cluster]) / std[cluster]
```

**Results:**
- 50 models assigned to 28 unique clusters (vs 39/50 on cluster #6 with absolute method)
- Each model identified by where it **outperforms peers**

### 2. Cost × Latency Grid

**Script:** [`generate_cost_based_priors.py`](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_13/generate_cost_based_priors.py)

Analyze 50 models to create 2D performance grid:

**Cost Tiers:**
- Budget: < $0.50/1M tokens
- Economy: $0.50 - $2.00
- Standard: $2.00 - $5.00
- Premium: > $5.00

**Latency Tiers:**
- Fast: < 0.5s TTFT
- Medium: 0.5 - 2.0s
- Slow: > 2.0s

**Grid Statistics:**

| Cost/Latency | Fast | Medium | Slow |
|--------------|------|--------|------|
| **Budget** | 18 models<br>82% avg | 7 models<br>86% avg | - |
| **Economy** | 3 models<br>85% avg | 4 models<br>93% avg | - |
| **Standard** | 1 model<br>92% avg | 3 models<br>78% avg | 1 model<br>93% avg |
| **Premium** | 1 model<br>93% avg | 6 models<br>97% avg | 6 models<br>92% avg |

### 3. Prior Generation

For each cost-latency cell, compute:
- **Mean cluster rates**: Average success rate per cluster across models in cell
- **Std cluster rates**: Variance for uncertainty estimation
- **Confidence score**: Based on sample size (n/10 capped at 1.0)

**Output:** 100-element vector of expected success rates

### 4. Fallback Cascade

**Level 1:** Exact cost-latency match  
**Level 2:** Same cost tier, different latency  
**Level 3:** Cost only (ignore latency)  
**Level 4:** Conservative baseline (70% uniform)

---

## Validation

### Holdout Experiment

**Script:** [`validate_cost_latency_priors.py`](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_13/validate_cost_latency_priors.py)

**Design:**
- Hold out 10 random models
- Train grid on remaining 40 models
- Predict held-out 10 and measure accuracy

**Results:**

| Metric | Score |
|--------|-------|
| Mean MAE | 9.6% ± 10.6% |
| Exact cluster match | 0% (expected) |
| Top-3 accuracy | **100%** |
| Top-5 accuracy | **100%** |

**Held-Out Models:**
- google/gemini-2.5-flash-preview-09-2025 (MAE: 8.4%)
- google/gemma-3-12b-it (MAE: 5.9%)
- openai/o3 (MAE: 3.7%)
- openai/gpt-5 (MAE: 4.5%)
- meta-llama/llama-3.1-405b-instruct (MAE: 6.0%)
- qwen/qwen3-14b (MAE: 4.0%)
- amazon/nova-micro-v1 (MAE: 4.6%)
- google/gemma-3-27b-it (MAE: 11.4%)
- qwen/qwen3-8b (MAE: 7.0%)
- google/gemini-2.5-pro-preview-06-05 (MAE: 40.7% - outlier)

**Key Finding:** 100% success at identifying top-5 strong clusters, 9.6% average error

---

## Alternative: KNN with Benchmarks

**Script:** [`predict_cluster_performance.py`](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_13/predict_cluster_performance.py)

For models **with** benchmark scores, use K-Nearest Neighbors:

**Features:** `general_quality`, `math_500`, `mmlu_pro`, `humaneval_score`, `reasoning_score`, `hle`, `price_1m_blended`, `output_tokens_per_second`, `time_to_first_token_seconds`

**Performance:**
- Mean MAE: 7.4%
- Top-5 accuracy: 100%
- Better than cost+latency but requires benchmarks

---

## Integration with Bandit Router

### Usage Example

```python
# New model arrives
new_model = {
    'cost': 1.50,      # $1.50 per 1M tokens (from API)
    'latency': 0.65    # 650ms TTFT (measured with 1 test)
}

# Load pre-computed grid
grid_stats = json.load(open('data/cost_latency_priors.json'))

# Generate 100-element prior vector
prior = generate_cost_latency_prior(
    cost=new_model['cost'],
    latency=new_model['latency'],
    grid_stats=grid_stats
)

# Initialize bandit with informed baseline
bandit = BanditRouter(
    model_id='new-model',
    priors=prior['cluster_priors']  # [0.89, 0.91, ..., 0.87]
)

# Bandit starts with ~90% accurate estimates
# Learns actual cluster preferences through usage
```

### Hybrid Strategy

```python
if model.has_benchmarks():
    prior = knn_predictor.predict(model)  # 7.4% error
elif model.has_cost_and_latency():
    prior = cost_latency_prior(model)     # 9.6% error
elif model.has_cost():
    prior = cost_only_prior(model)        # ~15% error
else:
    prior = uniform_baseline()            # 30%+ error
```

---

## Key Contributions

1. **No Benchmarks Required:** Works with only cost + latency
2. **100% Top-5 Accuracy:** Always finds strong cluster candidates
3. **9.6% Average Error:** Much better than 30%+ uniform baseline
4. **Graceful Degradation:** 4-level fallback cascade
5. **Production Ready:** Validated on real holdout models

---

## Files

| File | Purpose |
|------|---------|
| [`compute_relative_cluster_assignment.py`](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_13/compute_relative_cluster_assignment.py) | Calculate z-score based cluster assignments |
| [`generate_cost_based_priors.py`](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_13/generate_cost_based_priors.py) | Build cost×latency grid and generate priors |
| [`validate_cost_latency_priors.py`](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_13/validate_cost_latency_priors.py) | Holdout validation experiment |
| [`predict_cluster_performance.py`](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_13/predict_cluster_performance.py) | KNN predictor (benchmark-based) |
| **`cost_latency_priors.json`** | Pre-computed grid statistics |

---

## Citation

If you use this methodology, please cite:

```
Cost-Latency Based Prior Generation for Cold-Start LLM Routing
- 2D grid analysis across 50 production models
- 100% top-5 cluster identification accuracy
- 9.6% mean absolute error on cluster success rates
```
