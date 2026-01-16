# FCI-Based Pareto Frontier Analysis

**Date:** 2026-01-16  
**Quality Metric:** FCI (Frontier Capability Index) = (HLE + GPQA + LiveBench) / 3

## Executive Summary

Using pure FCI composite scores, the **true Pareto-optimal frontier contains 7 models**, not 4. The current Pareto configuration is **suboptimal** because:

1. ❌ **Claude Opus 4.5 is NOT Pareto-optimal** - it's dominated by cheaper models with similar or better quality
2. ❌ **GPT-OSS-120B cannot be evaluated** - missing LiveBench score (no FCI calculation possible)
3. ⭐ **4 high-value models are missing** from the current frontier

---

## Current vs. Recommended Pareto Frontier

### Current Frontier (4 models)
```
1. openai/gpt-oss-120b                   ❌ Missing LiveBench (no FCI)
2. google/gemini-2.5-pro-preview-06-05   ✅ FCI=0.6183, Cost=$3.44/M (KEEP)
3. anthropic/claude-opus-4.5             ❌ FCI=0.6747, Cost=$18.33/M (DOMINATED)
4. google/gemini-3-pro-preview           ✅ FCI=0.7340, Cost=$7.00/M (KEEP)
```

### Recommended FCI-Based Frontier (7 models)
```
1. mistralai/ministral-8b                ⭐ FCI=0.2710, Cost=$0.10/M   (cheapest)
2. mistralai/mistral-small-3.2-24b       ⭐ FCI=0.2760, Cost=$0.15/M
3. google/gemini-2.5-flash-lite          ⭐ FCI=0.4597, Cost=$0.18/M
4. openai/gpt-oss-20b                    ⭐ FCI=0.5227, Cost=$0.40/M
5. google/gemini-2.5-pro-preview-06-05   ✅ FCI=0.6183, Cost=$3.44/M   (keep)
6. openai/gpt-5                          ⭐ FCI=0.6550, Cost=$5.63/M   (NEW!)
7. google/gemini-3-pro-preview           ✅ FCI=0.7340, Cost=$7.00/M   (highest quality)
```

---

## Key Findings

### 1. Claude Opus 4.5 is NOT Pareto-Optimal 🚫

**Why it's dominated:**
- **FCI:** 0.6747 at **$18.33/M** (expensive!)
- **GPT-5:** FCI=0.6550 (-3% quality) at **$5.63/M** (-69% cost) ← Better value
- **Gemini 3 Pro:** FCI=0.7340 (+9% quality) at **$7.00/M** (-62% cost) ← Better quality AND cheaper

**Verdict:** Claude Opus 4.5 offers worse quality-cost trade-off than alternatives. Remove from Pareto frontier.

---

### 2. Missing High-Value Models

Four models offer excellent quality-cost ratios that are currently missing:

#### A. **Mistral Small 3.2** ($0.15/M)
- FCI: 0.2760
- **Best budget option** for simple tasks
- Quality/Dollar: 1.84 (12x better than Gemini 2.5 Pro)

#### B. **Gemini 2.5 Flash-Lite** ($0.18/M)
- FCI: 0.4597 (+67% vs Mistral Small)
- Excellent mid-budget choice
- Quality/Dollar: 2.63 (15x better than Gemini 2.5 Pro)

#### C. **GPT-OSS-20B** ($0.40/M)
- FCI: 0.5227 (+14% vs Flash-Lite)
- Strong reasoning at ultra-low cost
- Benchmarks: HLE=0.098, GPQA=0.690, LiveBench=0.780

#### D. **GPT-5** ($5.63/M)
- FCI: 0.6550 (+6% vs Gemini 2.5 Pro)
- **Fills the quality gap** between Gemini 2.5 Pro and Gemini 3 Pro
- Near-flagship quality at mid-tier price

---

### 3. Quality Distribution

| FCI Range | Models | Description |
|-----------|--------|-------------|
| 0.70-0.75 | 1 | **Flagship** (Gemini 3 Pro) |
| 0.65-0.70 | 2 | **Near-Flagship** (GPT-5, Claude Opus 4.5) |
| 0.60-0.65 | 2 | **High-Tier** (Gemini 2.5 Pro, o3, Grok 4) |
| 0.50-0.60 | 3 | **Mid-Tier** (GPT-OSS-20B, Claude Sonnet 4.5, Claude Sonnet 4) |
| 0.40-0.50 | 2 | **Budget-Plus** (Flash-Lite, Claude Haiku 4.5) |
| 0.25-0.40 | 9 | **Budget** (Mistral, Llama 4, DeepSeek, etc.) |
| 0.00-0.25 | 15 | **Ultra-Budget** (Small models) |

---

## Pareto Analysis: Quality vs Cost Trade-offs

### Quality Improvements Along Frontier

| From → To | ΔQ (FCI) | ΔCost | Cost per Quality Point |
|-----------|----------|-------|------------------------|
| Ministral 8B → Mistral Small | +0.0050 | +$0.05/M | $10/point |
| Mistral Small → Flash-Lite | +0.1837 | +$0.03/M | $0.14/point ⭐ |
| Flash-Lite → GPT-OSS-20B | +0.0630 | +$0.22/M | $3.49/point |
| GPT-OSS-20B → Gemini 2.5 Pro | +0.0956 | +$3.04/M | $31.80/point |
| Gemini 2.5 Pro → GPT-5 | +0.0367 | +$2.19/M | $59.67/point |
| GPT-5 → Gemini 3 Pro | +0.0790 | +$1.38/M | $17.47/point |

**Best value jumps:**
1. ⭐ **Mistral Small → Flash-Lite:** +18% quality for only +$0.03/M
2. ⭐ **Gemini 2.5 Pro → GPT-5:** +6% quality for +$2.19/M (fills quality gap)

---

## Recommendations

### Option 1: Conservative (Minimal Changes)
Keep current models but **replace Claude Opus 4.5 with GPT-5**:
```
1. openai/gpt-oss-120b              (if LiveBench data becomes available)
2. google/gemini-2.5-pro-preview    FCI=0.6183, $3.44/M
3. openai/gpt-5                     FCI=0.6550, $5.63/M (replaces Claude)
4. google/gemini-3-pro-preview      FCI=0.7340, $7.00/M
```

**Pros:** Minimal disruption, removes dominated model  
**Cons:** Missing ultra-budget options

---

### Option 2: Expanded Frontier (Recommended ⭐)
Use the full **7-model Pareto frontier**:
```
1. mistralai/ministral-8b                $0.10/M  (ultra-budget)
2. mistralai/mistral-small-3.2-24b       $0.15/M
3. google/gemini-2.5-flash-lite          $0.18/M  (budget champion)
4. openai/gpt-oss-20b                    $0.40/M
5. google/gemini-2.5-pro-preview         $3.44/M
6. openai/gpt-5                          $5.63/M  (quality bridge)
7. google/gemini-3-pro-preview           $7.00/M  (flagship)
```

**Pros:**
- Complete quality spectrum coverage
- Excellent ultra-budget options ($0.10-0.40/M)
- Better granularity for router optimization
- All models are Pareto-optimal (no dominated choices)

**Cons:** 
- More models to warm up and maintain
- Larger state space for LinUCB

---

### Option 3: Practical Subset (5 models)
Balance coverage with simplicity:
```
1. mistralai/ministral-8b                $0.10/M  (cheapest)
2. google/gemini-2.5-flash-lite          $0.18/M  (best budget)
3. google/gemini-2.5-pro-preview         $3.44/M  (mid-tier)
4. openai/gpt-5                          $5.63/M  (high-tier)
5. google/gemini-3-pro-preview           $7.00/M  (flagship)
```

**Pros:** Balanced coverage, manageable size, removes redundancy  
**Cons:** Skips GPT-OSS-20B and Mistral Small

---

## Technical Notes

### FCI Calculation
```python
FCI = (HLE + GPQA + LiveBench) / 3
```

### Pareto Optimality Criterion
A model is Pareto-optimal if and only if:
```
∄ model M such that:
    FCI(M) ≥ FCI(model) AND Cost(M) ≤ Cost(model)
    with at least one strict inequality
```

### Models Excluded (Missing FCI Data)
- `openai/gpt-oss-120b` - No LiveBench score
- `x-ai/grok-3-mini` - No LiveBench score  
- `google/gemini-2.5-flash-preview-09-2025` - No LiveBench score
- Many others missing GPQA or LiveBench

Total evaluated: **34 models** with complete FCI data  
Total in `models.json`: **70+ models**

---

## Visualization

See `fci_pareto_frontier.png` for the complete scatter plot showing:
- 🔴 **Red dots:** Pareto-optimal models
- ⚪ **Gray dots:** Dominated models
- 🟢 **Green circles:** Current Pareto models that are optimal
- 🟠 **Orange X:** Current Pareto models that are dominated (Claude Opus 4.5)
- 🔴 **Dashed line:** Pareto frontier boundary

---

## Implementation Steps

1. **Update `models_pareto.json`** with recommended frontier
2. **Retrain warmup priors** with new model set
3. **Update plot_rational_boundary.py** to use 7-model frontier
4. **Validate routing decisions** on test set
5. **Monitor empirical performance** to confirm FCI predictions

---

## Conclusion

The FCI-based analysis reveals that:
1. **Claude Opus 4.5 should be removed** - it's dominated by GPT-5 and Gemini 3 Pro
2. **4 ultra-budget models are missing** - huge value proposition at $0.10-0.40/M
3. **GPT-5 fills a critical gap** between mid-tier and flagship quality
4. **The 7-model frontier is optimal** from a pure quality-cost perspective

**Recommendation:** Adopt the 7-model Pareto frontier (Option 2) for maximum routing efficiency.

