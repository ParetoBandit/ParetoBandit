# Metadata-Guided Initialization (NOT Cold Start)

## Critical Terminology for Paper

**IMPORTANT**: The BanditGPT router does NOT start from "cold start" (zero knowledge). It uses **Metadata-Guided Initialization** based on public benchmark scores.

## What We Actually Do

### Initialization Strategy:
1. **Load Public Benchmarks** for each model:
   - Math-500 (mathematical reasoning)
   - MMLU-Pro (broad knowledge)
   - Reasoning benchmark
   
2. **Compute 3-Benchmark Average**:
   ```
   initial_quality_estimate = (math_500 + mmlu_pro + reasoning) / 3.0
   ```

3. **Use as Prior Belief**:
   - The bandit starts with these scores as initial quality estimates
   - NOT starting from uniform/zero knowledge
   - Models with better benchmarks get explored earlier

### Why This Matters for the Paper:

**Before (INCORRECT)**:
> "Our cold-start approach requires no training data..."

**After (CORRECT)**:
> "Our metadata-guided initialization uses only public benchmark scores (Math-500, MMLU-Pro, Reasoning) to seed the bandit's initial beliefs, requiring no task-specific training data or graded examples..."

## Comparison with Other Methods

| Method | Training Data Required | Initialization |
|--------|----------------------|----------------|
| **FrugalGPT** | 500-2000 labeled examples | Calibrated scoring function |
| **RouteLLM** | 1000-5000 preference pairs | Trained router model |
| **Semantic Router** | Manual intent definitions + examples | Pre-defined decision boundaries |
| **BanditGPT** | **0 labeled examples** | **3 public benchmarks** (Math, MMLU, Reasoning) |

## Key Claims We CAN Make:

✅ **Zero Task-Specific Training**: No need for graded examples on YOUR data
✅ **No Manual Configuration**: No intent definitions or decision boundaries
✅ **Public Metadata Only**: Uses widely available benchmark scores
✅ **Online Learning**: Adapts from real feedback, not offline training

## Key Claims We CANNOT Make:

❌ ~~"Pure cold start"~~ → Use "metadata-guided initialization"
❌ ~~"Zero knowledge"~~ → Use "zero task-specific training"
❌ ~~"Starts from scratch"~~ → Use "starts from public benchmarks"

## Paper Text Recommendations

### Abstract:
Replace:
> "...eliminates cold-start calibration..."

With:
> "...eliminates task-specific calibration by initializing from public benchmarks..."

### Method Section (3.6):
**Title**: "Metadata-Guided Initialization" (NOT "Cold Start")

**Content**:
> Unlike offline routing methods that require hundreds of graded examples, our approach leverages publicly available benchmark scores (Math-500, MMLU-Pro, Reasoning) to initialize the bandit's quality beliefs. Each model m is assigned an initial quality estimate:
>
> q_init(m) = (Math500(m) + MMLU(m) + Reasoning(m)) / 3
>
> This metadata-guided initialization serves two purposes: (1) it biases early exploration toward models with strong general capabilities, reducing Day-1 regret, and (2) it requires zero task-specific training data, enabling immediate deployment across arbitrary domains.

### Evaluation Section:
When discussing Figure 1, clarify:
> "To isolate the impact of metadata initialization, we compare against a **true cold-start baseline** (uniform priors, no benchmark information)..."

## Bottom Line

**What we're avoiding**: The cost and delay of collecting hundreds/thousands of graded examples on the user's specific task.

**What we're using instead**: A few public benchmark scores that are available for free and work across all tasks.

This is still a massive practical advantage, but we must be scientifically precise about what "zero training" means.

