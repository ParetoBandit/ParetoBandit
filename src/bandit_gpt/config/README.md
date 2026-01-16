# Model Configuration Files

This directory contains various model registry configurations for the BanditRouter.

## 📁 Production Files

### `models_pareto.json` ⭐ **RECOMMENDED**
- **Models**: 5 Pareto-optimal models
- **Purpose**: Production routing with mathematically optimal quality/cost trade-offs
- **Quality Range**: 94.7% - 97.9%
- **Cost Range**: $0.02 - $1.25 per 1M tokens
- **Efficiency**: 100% of models are Pareto optimal

**Models**:
1. `openai/gpt-oss-120b` ($0.02, 94.7%) - Ultra-efficient
2. `google/gemini-2.5-flash-preview-09-2025` ($0.30, 95.1%) - Fast & cheap
3. `moonshotai/kimi-k2-0905` ($0.60, 96.0%) - Balanced
4. `x-ai/grok-3-mini` ($0.80, 97.5%) - Sweet spot
5. `openai/gpt-5.1` ($1.25, 97.9%) - Quality leader

**Benefits**:
- No dominated models (100% efficient)
- Covers full quality spectrum
- Optimal cost/quality trade-offs
- Backed by Pareto frontier analysis

**Usage**:
```python
from bandit_gpt import BanditRouter

router = BanditRouter.create(
    model_registry_path="models_pareto.json",
    alpha=0.5
)
```

---

### `models.json`
- **Models**: 42 models with `initial_quality` scores
- **Purpose**: Full portfolio for research and experimentation
- **Quality Range**: 37.7% - 97.9%
- **Cost Range**: $0.02 - $15.00 per 1M tokens
- **Efficiency**: 11.9% of models are Pareto optimal

**Use Cases**:
- Research experiments
- Ablation studies (Pareto vs non-Pareto)
- Benchmarking against larger portfolios
- Testing router behavior with dominated models

**Note**: 88.1% of these models are dominated by Pareto models.

---

## 📚 Archive Files

### `models_full.json`
- **Models**: 84 models (42 with quality scores, 42 without)
- **Purpose**: Complete model registry before filtering
- **Status**: Source file for quality scores

### `models_orig.json`
- **Models**: 84 models from git history
- **Purpose**: Historical reference with all benchmark fields

### `models_6models_backup.json`
- **Models**: Original 6 manually curated models
- **Purpose**: Backup before expansion to 42 models
- **Date**: Before January 15, 2026

### `models_2_models.json`
- **Models**: 2 models (minimal test configuration)
- **Purpose**: Testing and debugging

### `models_full_backup.json`
- **Models**: 84 models before adding `initial_quality`
- **Purpose**: Backup before quality score calculation

---

## 🎯 Choosing the Right Configuration

| Use Case | Recommended File | Why |
|----------|------------------|-----|
| **Production routing** | `models_pareto.json` | Optimal efficiency, no wasted cost/quality |
| **KDD paper experiments** | `models_pareto.json` | Mathematically grounded, Pareto theory |
| **Research ablations** | `models.json` | Compare Pareto vs full portfolio |
| **Benchmarking** | `models.json` | Test against larger model sets |
| **Development/testing** | `models_2_models.json` | Fast iteration |

---

## 📊 Quality Score Sources

All `initial_quality` scores are calculated from:
- **Dataset**: `dev_rewards_complete.jsonl.gz`
- **Metric**: Mean success rate across 1,121 prompts
- **Coverage**: 100% model coverage (every model evaluated on every prompt)
- **Method**: Sigmoid transform of reward logits to probability

---

## 🔄 Model Registry Structure

Each model entry contains:

```json
{
  "openrouter_id": "x-ai/grok-3-mini",
  "display_name": "Grok 3 Mini",
  "initial_quality": 0.9751,
  "initial_quality_source": "dev_rewards_complete",
  "initial_quality_n_obs": 1121,
  "price_1m_input": 0.80,
  "price_1m_output": 3.20,
  "context_length": 131072,
  ...
}
```

**Key Fields**:
- `openrouter_id`: Unique model identifier
- `initial_quality`: Dev set success rate (0-1)
- `price_1m_input`: Cost per 1M input tokens ($)
- `price_1m_output`: Cost per 1M output tokens ($)

---

## 🌟 Pareto Frontier Analysis

See `experiments/08_arbitrage_frontier/PARETO_ANALYSIS.md` for detailed analysis.

**Key Findings**:
- Only 5/42 models (11.9%) are Pareto optimal
- 88.1% of models are dominated (waste money or quality)
- Pareto models span 94.7% - 97.9% quality
- Cost range: $0.02 - $1.25 (62.5x difference)

**Dominated Models Example**:
- `openai/gpt-4.1` (97.5%, $2.00) dominated by `x-ai/grok-3-mini` (97.5%, $0.80)
  - Same quality, 60% cheaper!

---

## 🔄 Update History

**January 15, 2026**:
- ✅ Created `models_pareto.json` with 5 Pareto-optimal models
- ✅ Expanded `models.json` from 6 to 42 models
- ✅ Added `initial_quality` scores from dev set to `models_full.json`
- ✅ Collected missing Gemini 2.5 & GPT-5 rewards
- ✅ Filtered dev/holdout to 100% model coverage

**Previous**:
- Original 6-model manual curation
- 84-model full registry from git

---

## 📖 Related Documentation

- **Pareto Analysis**: `experiments/08_arbitrage_frontier/PARETO_ANALYSIS.md`
- **Dataset Info**: `src/bandit_gpt/data/offline_dataset/README.md`
- **Quality Calculation**: `experiments/calculate_initial_quality.py`
- **Pareto Finder**: `experiments/find_pareto_frontier.py`

---

## 🚀 Next Steps

1. **Use `models_pareto.json`** for production routing
2. **Run experiments** comparing Pareto vs full portfolio
3. **Measure cost savings** from routing to Pareto models only
4. **Document results** in KDD paper with Pareto efficiency claims
