# Pareto Frontier Update (FCI-Based)

**Date**: 2026-01-15  
**Experiment**: 08_arbitrage_frontier

## Summary

Updated the arbitrage frontier visualization to use the **new FCI-based Pareto frontier** from Experiment 10.

## Changes

### 1. New Pareto Models Selected

**Previous Models:**
- Expensive: `openai/gpt-5.1` ($1.25/1M)
- Cheap: `openai/gpt-oss-120b` ($0.06/1M)

**New Models (FCI-Based):**
- **Expensive**: `google/gemini-3-pro-preview` ($7.00/1M, FCI=1.000) 🏆
- **Cheap**: `openai/gpt-oss-120b` ($0.06/1M, FCI=0.740) ⭐

### 2. Trade-off Analysis

| Metric | Old Range | New Range | Improvement |
|--------|-----------|-----------|-------------|
| **Cost Ratio** | 20.8x | **116.7x** | 5.6x wider range |
| **Cost Range** | $0.06 - $1.25 | **$0.06 - $7.00** | Much broader spectrum |
| **Quality Metric** | initial_quality | **FCI (composite)** | More robust |

**Key Insight**: The new trade-off shows a **117x cost difference** between the cheapest and most expensive Pareto-optimal models, making the arbitrage decision much more dramatic and impactful.

### 3. Files Updated

1. **`src/bandit_gpt/config/models_binary.json`**
   - Updated to contain the 2 extreme Pareto models
   - Now uses Gemini-3-Pro-Preview instead of GPT-5.1

2. **`experiments/08_arbitrage_frontier/plot_rational_boundary.py`**
   - Updated `expensive_id` to `google/gemini-3-pro-preview`
   - Updated all labels, comments, and documentation
   - Updated plot titles and legends

3. **`artifacts/priors_warmup_binary.joblib`**
   - Regenerating warmup priors for the new binary model set
   - Uses 1,000 synthetic prompts for faster generation

### 4. Why These Models?

**Gemini-3-Pro-Preview** is now the strongest Pareto model because:
- Highest FCI score: **1.000** (normalized maximum)
- Best performance on all 3 benchmarks:
  - HLE: 0.372 (highest)
  - GPQA: 0.91 (highest)
  - LiveBench: 0.92 (highest)
- Represents the absolute frontier of quality

**GPT-OSS-120B** remains the cheapest Pareto model because:
- Ultra-low cost: **$0.06/1M** tokens
- Still maintains strong FCI: **0.740**
- Dominates all cheaper models (ministral-3b, gemma-3-4b-it have negative FCI under old normalization, much lower under new)

### 5. Why Ultra-Cheap Models Didn't Make the Frontier

We investigated models under $0.10/1M and found:

| Model | Cost | FCI | Why Dominated? |
|-------|------|-----|----------------|
| ministral-3b | $0.08/1M | 0.180 | Saves $0.02 but loses **0.56 FCI points** |
| gemma-3-4b-it | $0.085/1M | 0.094 | Saves $0.015 but loses **0.65 FCI points** |
| ministral-8b | $0.10/1M | 0.240 | **Costs more** but delivers 0.50 FCI points less |

**Conclusion**: GPT-OSS-120B at $0.06/1M offers such exceptional value that no cheaper alternative makes economic sense.

## Normalization Method

We recomputed FCI using **full-range normalization** (including weak models):

```
NEW Normalization Ranges:
  HLE:       [0.033, 0.372]
  GPQA:      [0.200, 0.910]
  LiveBench: [0.020, 0.920]
```

This eliminates negative FCI scores while maintaining the same Pareto frontier.

## Next Steps

1. ✅ Update `models_binary.json` with new extreme models
2. ✅ Update `plot_rational_boundary.py` with new model IDs and labels
3. 🔄 Generate warmup priors for binary models (in progress)
4. ⏳ Run the arbitrage frontier visualization
5. ⏳ Compare results with old frontier

## Expected Results

The new visualization should show:
- **Wider cost spread**: 117x vs 21x (more dramatic decisions)
- **Clearer quality gaps**: FCI-based quality is more defensible
- **Better alignment**: Matches the Pareto frontier from Experiment 10

This makes the arbitrage frontier visualization more compelling for the KDD paper, as it demonstrates routing decisions across a **2-order-of-magnitude cost range**.

