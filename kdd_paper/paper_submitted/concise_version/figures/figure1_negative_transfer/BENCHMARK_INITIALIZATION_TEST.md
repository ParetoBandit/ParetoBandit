# Figure 1: Testing Value of Benchmark Initialization

## What We're Testing

**Research Question**: Does initializing the bandit with public benchmark scores (Math-500, MMLU-Pro, Reasoning) reduce regret compared to starting with uniform/zero knowledge?

## Experimental Setup

### 5-Fold Cross-Validation
- **Dataset**: 497 prompts, 81 models
- **Splits**: 80% train, 20% test per fold
- **Evaluation**: 2000 routing decisions per fold

### Two Conditions

#### Condition 1: **With Benchmarks** (Metadata-Guided)
- Initialize each model with its 3-benchmark average
- Example: GPT-4o-mini starts with belief ≈ 0.566
- Example: Gemini-3-Pro starts with belief ≈ 0.954
- **This is our library's default behavior**

#### Condition 2: **Without Benchmarks** (Pure Cold Start)
- Initialize all models with uniform belief = 0.5
- No prior knowledge about model capabilities
- Pure exploration from scratch

### Metrics
- **Primary**: Cumulative Regret at 2000 steps
- **Comparison**: % Regret Reduction = (Without - With) / Without × 100
- **Statistical Test**: One-sided t-test (H1: benchmarks reduce regret)

## Expected Outcomes

### If Benchmarks Help (Expected)
- Regret Reduction > 0% across most/all folds
- Statistically significant p-value (< 0.05)
- Demonstrates value of metadata-guided initialization

### If Benchmarks Don't Help (Unexpected)
- Regret Reduction ≈ 0% or negative
- Would suggest online learning quickly overcomes any initialization advantage
- Still validates our approach (no harm from benchmarks)

## Why This Matters for the Paper

### Current Claim (Needs Validation)
> "BanditGPT uses metadata-guided initialization from public benchmarks..."

**We need to show**: This actually provides value compared to pure cold start

### What We're NOT Testing
- ❌ Complex offline training (overtrained priors)
- ❌ Task-specific calibration data
- ❌ Manual expert configuration

### What We ARE Testing
- ✅ Simple public benchmarks (3 scores per model)
- ✅ Zero task-specific training
- ✅ Immediate deployment advantage

## Script Details

**File**: `generate_figure1_library.py`

**Key Features**:
- Uses actual `BanditRouter` from library
- Loads `model_registry.json` with 81 models
- Creates two versions of registry (with/without benchmarks)
- 5-fold CV with proper train/test splits
- Generates 2-panel figure + statistics

**Output**:
- `figure1_library_version.png` - Visualization
- `figure1_library_stats.json` - Numerical results

## Interpretation Guide

### Strong Positive Result (Ideal)
- Mean reduction: 10-20%
- p-value < 0.01
- 5/5 folds show benefit
- **Paper claim**: "Benchmarks reduce Day-1 regret by X%"

### Moderate Positive Result (Good)
- Mean reduction: 5-10%
- p-value < 0.05
- 4-5/5 folds show benefit
- **Paper claim**: "Benchmarks provide modest but consistent advantage"

### Neutral Result (Still OK)
- Mean reduction: 0-5%
- p-value > 0.05
- Mixed fold results
- **Paper claim**: "Benchmarks provide negligible cost, enabling safe initialization"

### Negative Result (Requires Reframing)
- Mean reduction: < 0%
- Would need to explain why benchmarks don't help
- May need to adjust approach or paper narrative

## Current Status

**Running**: 5-fold validation (takes 5-10 minutes)

**Using**: Real BanditRouter library with proper benchmark initialization

**Testing**: Scientific validation of our core initialization strategy

