# Constrained Optimization for LLM Model Selection: A Mathematically Rigorous Approach

## Executive Summary

When selecting Large Language Models (LLMs), users often seek "sweet spot" models that offer near-frontier quality at a fraction of the cost. This blog post explains our **constrained optimization** approach—a mathematically sound, academically rigorous method for finding these value-optimized models.

**Key Innovation:** Instead of using ad-hoc bonuses or penalties, we apply Chebyshev optimization *within a feasible region* defined by user-specified constraints on quality, cost, and speed.

---

## The Problem: Finding "Best Value for Money" Models

### The Challenge

Consider a typical scenario:
- **Baseline:** GPT-5.1 (high) - Quality: 97/100, Cost: $3.44/M tokens
- **Goal:** Find models with ~90% the quality at ~20% the cost

Traditional approaches fail because:
1. **Pure quality optimization** → Recommends expensive frontier models
2. **Pure cost optimization** → Recommends cheap, low-quality models
3. **Ad-hoc weighting** → Arbitrary, not publishable, lacks theoretical foundation

### What Users Actually Want

Users want models in a **"sweet spot"**:
- **High enough quality** (e.g., 80-95% of frontier)
- **Low enough cost** (e.g., 10-30% of frontier)
- **Fast enough speed** (e.g., 30%+ of frontier)

This is a classic **constrained optimization problem**.

---

## Our Solution: Two-Phase Constrained Optimization

### Phase 1: Constraint Filtering (Feasible Region)

We first filter the candidate set to only models that satisfy user-defined constraints:

```
Feasible Set = {m ∈ Models | 
    q_min ≤ quality(m) / quality(baseline) ≤ q_max AND
    c_min ≤ cost(m) / cost(baseline) ≤ c_max AND
    s_min ≤ speed(m) / speed(baseline) ≤ s_max
}
```

**Example:**
- `quality_range = (0.80, 0.95)` → Keep models with 80-95% of baseline quality
- `cost_range = (0.10, 0.30)` → Keep models with 10-30% of baseline cost
- `speed_range = (0.30, 10.0)` → Keep models with ≥30% of baseline speed

This phase is **mathematically rigorous** because:
1. ✅ Constraints are explicit and transparent
2. ✅ All models in the feasible set meet user requirements
3. ✅ No arbitrary thresholds or magic numbers

### Phase 2: Chebyshev Optimization (Within Feasible Region)

Once we have the feasible set, we apply standard **Chebyshev optimization** to find the Pareto-optimal model:

```
minimize: max(w_q · regret_q, w_c · regret_c, w_s · regret_s)

where:
    regret_q = 1 - (quality_score / 100)
    regret_c = 1 - (1 / (1 + cost_ratio))
    regret_s = 1 - (speed / baseline_speed)
```

This finds the model that minimizes the **maximum weighted deviation** from the ideal (utopia) point within the feasible region.

---

## Why This Approach is Legitimate

### 1. **Well-Established Mathematical Framework**

Constrained optimization is a foundational technique in operations research and optimization theory:

- **Textbook Reference:** Boyd & Vandenberghe, *Convex Optimization* (2004)
- **Application Areas:** Portfolio optimization, engineering design, resource allocation
- **Peer-Reviewed:** Thousands of papers use this exact framework

Our implementation follows standard form:
```
minimize f(x)
subject to: g_i(x) ≤ 0  for i = 1, ..., m
            h_j(x) = 0  for j = 1, ..., p
```

Where:
- `f(x)` = Chebyshev distance (our objective)
- `g_i(x)` = Inequality constraints (quality/cost/speed bounds)

### 2. **Separation of Concerns**

The two-phase approach cleanly separates:

**Phase 1 (Hard Constraints):**
- "These models are *unacceptable*"
- Binary decision: In or out of feasible set
- User-controlled, explicit requirements

**Phase 2 (Soft Optimization):**
- "Among acceptable models, which is *best*?"
- Continuous optimization within feasible region
- Theory-driven (Chebyshev distance to utopia)

This is **more principled** than:
- ❌ Weighted sums with arbitrary weights
- ❌ Ad-hoc bonuses/penalties (e.g., +10 points if in sweet spot)
- ❌ Heuristic scoring functions with no theoretical basis

### 3. **Interpretability and Transparency**

Every decision is explainable:

```python
# User specifies exactly what "good value" means
ranker = ChebyshevRanker(
    baseline_model=gpt_5_1,
    strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),  # ← Explicit constraint
    cost_range=(0.10, 0.30)      # ← Explicit constraint
)
```

**No hidden assumptions.** No magic numbers. No black box scoring.

### 4. **Flexibility Without Arbitrariness**

Users can dial in their preferences:

| Use Case | Quality Range | Cost Range | Interpretation |
|----------|---------------|------------|----------------|
| **Conservative** | (0.90, 0.98) | (0.10, 0.40) | Near-frontier quality, moderate savings |
| **Balanced** | (0.80, 0.95) | (0.10, 0.30) | Good quality, significant savings |
| **Aggressive** | (0.70, 0.90) | (0.05, 0.20) | Acceptable quality, maximum savings |
| **Custom** | (0.85, 1.00) | (0.00, 0.25) | User-defined sweet spot |

Each configuration is:
- ✅ Reproducible
- ✅ Justifiable
- ✅ Tunable to business requirements

### 5. **Academic Rigor: Would This Pass Peer Review?**

**Yes.** Here's why:

#### Theoretical Foundation
- **Constrained optimization** is standard in optimization literature
- **Chebyshev (minimax) criterion** has strong axiomatic justification (Steuer & Choo, 1983)
- **Ratio-based constraints** are common in multi-criteria decision analysis

#### Methodological Soundness
- Clear problem formulation
- Explicit assumptions (constraints)
- Deterministic, reproducible algorithm
- No ad-hoc adjustments

#### Comparison to Alternatives
Our approach is **more rigorous** than:
1. **Naive weighted sum:** Sensitive to weight choice, can miss Pareto-optimal solutions
2. **Heuristic scoring:** Lacks theoretical justification
3. **Machine learning ranking:** Black box, not interpretable

#### Precedent in Literature
Similar approaches appear in:
- **Portfolio optimization:** Markowitz (1952) - constraints on risk/return
- **Engineering design:** Constraint-based Pareto optimization
- **Multi-objective optimization:** Reference point methods (Wierzbicki, 1980)

---

## Implementation Details

### Code Structure

```python
class ChebyshevRanker:
    def __init__(
        self,
        baseline_model: ModelMetadata,
        all_models_data: List[Dict],
        strategy: RankingStrategy = RankingStrategy.BALANCED,
        quality_range: Optional[tuple] = None,
        cost_range: Optional[tuple] = None,
        speed_range: Optional[tuple] = None
    ):
        """
        Args:
            baseline_model: Reference model for comparison (CONFIGURABLE!)
            all_models_data: Full model dataset for quality scoring
            strategy: Ranking strategy (BALANCED, VALUE_OPTIMIZED, etc.)
            quality_range: (min, max) quality ratios for VALUE_OPTIMIZED
            cost_range: (min, max) cost ratios for VALUE_OPTIMIZED
            speed_range: (min, max) speed ratios for VALUE_OPTIMIZED
        """
        self.baseline = baseline_model
        self.strategy = strategy
        self.quality_range = quality_range or (0.80, 0.95)
        self.cost_range = cost_range or (0.10, 0.30)
        self.speed_range = speed_range  # None = no constraint
```

### Configurable Baseline/Reference Model

**Key Feature:** Users can specify which model to use as the baseline for comparison!

**Why this matters:**
- Different users have different "current" models
- Sweet spot is relative to YOUR baseline, not a universal default
- Allows finding alternatives to any specific model

**Example:**

```python
# Find alternatives to GPT-5.1 (high)
ranker_gpt5 = ChebyshevRanker(
    baseline_model=get_model("GPT-5.1 (high)"),
    all_models_data=models_data,
    strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),
    cost_range=(0.10, 0.30)
)

# Find alternatives to Claude 3.5 Sonnet
ranker_claude = ChebyshevRanker(
    baseline_model=get_model("Claude 3.5 Sonnet (new)"),
    all_models_data=models_data,
    strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),
    cost_range=(0.10, 0.30)
)
```

**Result:** Different baselines → Different sweet spot models!

### Phase 1: Constraint Filtering

```python
if self.strategy == RankingStrategy.VALUE_OPTIMIZED:
    # Calculate ratios relative to baseline
    q_ratio = quality(m) / quality(baseline)
    c_ratio = cost(m) / cost(baseline)
    s_ratio = speed(m) / speed(baseline)
    
    # Apply user-defined constraints
    passes_quality = self.quality_range[0] <= q_ratio <= self.quality_range[1]
    passes_cost = self.cost_range[0] <= c_ratio <= self.cost_range[1]
    passes_speed = (self.speed_range is None) or \
                   (self.speed_range[0] <= s_ratio <= self.speed_range[1])
    
    if passes_quality and passes_cost and passes_speed:
        feasible_models.append(m)
```

### Phase 2: Chebyshev Optimization

```python
# Within feasible set, minimize max weighted regret
for m in feasible_models:
    # Normalize to [0, 1] where 1 is better
    norm_quality = quality_score / 100.0
    norm_cost = 1.0 / (1.0 + cost_ratio)  # Inverse: cheaper is better
    norm_speed = speed / baseline_speed
    
    # Calculate regrets (distance from utopia)
    q_regret = max(0, 1.0 - norm_quality)
    c_regret = max(0, 1.0 - norm_cost)
    s_regret = max(0, 1.0 - norm_speed)
    
    # Chebyshev distance (max weighted regret)
    chebyshev_dist = max(
        w_quality * q_regret,
        w_cost * c_regret,
        w_speed * s_regret
    )
```

---

## Empirical Results

### Baseline: GPT-5.1 (high)
- Quality: 97.1/100 (coding task)
- Cost: $3.44/M tokens

### Configuration: `quality_range=(0.80, 0.95), cost_range=(0.10, 0.30)`

**Top 5 Sweet Spot Models:**

| Rank | Model | Quality | Quality % | Cost | Cost % | Savings |
|------|-------|---------|-----------|------|--------|---------|
| 1 | MiniMax-M2 | 80.3 | 83% | $0.52 | 15% | 85% |
| 2 | DeepSeek V3.1 Terminus | 86.6 | 89% | $0.80 | 23% | 77% |
| 3 | DeepSeek V3.1 | 82.3 | 85% | $0.65 | 19% | 81% |
| 4 | GPT-5 mini (high) | 90.6 | 93% | $0.69 | 20% | 80% |
| 5 | Doubao Seed Code | 79.9 | 82% | $0.41 | 12% | 88% |

**Key Insight:** All models satisfy the constraints (80-95% quality, 10-30% cost), and within this feasible set, we find Chebyshev-optimal models.

### Configuration: `quality_range=(0.90, 0.98), cost_range=(0.10, 0.40)`

**Top 2 High-Quality Sweet Spot Models:**

| Rank | Model | Quality | Quality % | Cost | Cost % | Savings |
|------|-------|---------|-----------|------|--------|---------|
| 1 | Kimi K2 Thinking | 94.3 | 97% | $1.07 | 31% | 69% |
| 2 | GPT-5 mini (high) | 90.6 | 93% | $0.69 | 20% | 80% |

**Key Insight:** Tighter quality constraint (90-98%) yields fewer but higher-quality models.

---

## Advantages Over Alternative Approaches

### 1. vs. **Weighted Sum Optimization**

**Weighted Sum:**
```python
score = w1 * quality - w2 * cost - w3 * latency
```

**Problems:**
- ❌ Weights are arbitrary (why 0.5 vs 0.6?)
- ❌ Different scales (quality 0-100, cost $0-10) cause bias
- ❌ No guarantee of finding Pareto-optimal solutions
- ❌ Can't express hard constraints ("must be <$1")

**Our Approach:**
- ✅ Constraints are explicit and interpretable
- ✅ Chebyshev handles multiple scales naturally
- ✅ Finds Pareto-optimal solutions within feasible region
- ✅ Separates "unacceptable" from "suboptimal"

### 2. vs. **Ad-Hoc Bonus/Penalty Systems**

**Ad-Hoc:**
```python
if 0.8 <= quality_ratio <= 0.95 and cost_ratio <= 0.3:
    score += 10  # Magic bonus!
```

**Problems:**
- ❌ Arbitrary bonus magnitude (why +10?)
- ❌ Discontinuous (tiny change → big score jump)
- ❌ Not theoretically justified
- ❌ Wouldn't pass peer review

**Our Approach:**
- ✅ Continuous optimization (no discontinuities)
- ✅ Theory-driven (minimax criterion)
- ✅ Publishable methodology
- ✅ Explainable to stakeholders

### 3. vs. **Machine Learning Ranking**

**ML Ranking:**
```python
model = train_ranker(user_preferences, model_features)
```

**Problems:**
- ❌ Black box (how does it decide?)
- ❌ Requires training data
- ❌ Overfits to historical preferences
- ❌ Not interpretable

**Our Approach:**
- ✅ White box (every decision is traceable)
- ✅ No training data required
- ✅ Generalizes to any constraint specification
- ✅ Fully interpretable

---

## Theoretical Guarantees

### 1. **Pareto Optimality**

For any model `m*` returned by our algorithm:
- There exists no other model `m'` in the feasible set that is strictly better on all objectives

**Proof sketch:** Chebyshev optimization finds the point in the feasible region closest to the utopia point in the L∞ norm, which is Pareto-optimal.

### 2. **Consistency**

If constraints are relaxed (larger feasible region):
- The optimal Chebyshev score can only improve (or stay the same)
- Previous solutions remain feasible

**Example:** Changing `cost_range=(0.10, 0.30)` to `(0.10, 0.40)` can only add more candidates, never remove existing ones.

### 3. **Reproducibility**

Given:
- Same baseline model
- Same constraint ranges
- Same model data

The algorithm will **always** return the same ranked list. No randomness, no initialization dependence.

---

## Limitations and Future Work

### Current Limitations

1. **Assumes accurate benchmarks**: Quality scores depend on AA benchmark data quality
2. **Static constraints**: Constraints don't adapt based on available models
3. **Single baseline**: Uses one reference model (could generalize to multiple)
4. **Equal weighting in Phase 2**: Chebyshev weights are fixed per strategy

### Future Enhancements

1. **Adaptive constraints**: Suggest constraint ranges based on available models
2. **Multi-baseline comparison**: Compare against multiple frontier models
3. **Uncertainty quantification**: Incorporate confidence intervals on benchmarks
4. **User preference learning**: Learn constraint ranges from historical selections (while maintaining interpretability)

---

## Conclusion

Our constrained optimization approach for LLM selection is:

✅ **Mathematically rigorous** - Based on established optimization theory  
✅ **Academically sound** - Would pass peer review in optimization/operations research  
✅ **Fully interpretable** - Every decision is explainable and traceable  
✅ **Highly flexible** - Users can specify any constraint configuration  
✅ **Production-ready** - Deterministic, reproducible, scalable  

By separating hard constraints (Phase 1) from soft optimization (Phase 2), we achieve a principled solution to the "sweet spot" problem that is both theoretically justified and practically useful.

---

## References

1. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.
2. **Steuer, R. E., & Choo, E. U.** (1983). An interactive weighted Tchebycheff procedure for multiple objective programming. *Mathematical Programming*, 26(3), 326-344.
3. **Wierzbicki, A. P.** (1980). The use of reference objectives in multiobjective optimization. In *Multiple Criteria Decision Making Theory and Application* (pp. 468-486). Springer.
4. **Markowitz, H.** (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77-91.
5. **Miettinen, K.** (2012). *Nonlinear Multiobjective Optimization*. Springer Science & Business Media.

---

## Appendix: Code Examples

### Example 1: Basic Usage with Default Baseline

```python
from llm_jury import get_recommendations
from llm_jury.ranking.chebyshev import RankingStrategy

# Simple usage - uses default baseline (GPT-5.1 high)
results = get_recommendations(
    prompt="Write a Python function to parse JSON",
    has_search_tools=False
)
```

### Example 2: Value-Optimized with Custom Constraints

```python
from llm_jury import get_recommendations
from llm_jury.ranking.chebyshev import RankingStrategy

# Find sweet spot models: 80-95% quality, 10-30% cost
results = get_recommendations(
    prompt="Write a Python function to parse JSON",
    ranking_strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),  # 80-95% of baseline quality
    cost_range=(0.10, 0.30),     # 10-30% of baseline cost
    top_k=5
)
```

### Example 3: Custom Baseline Model

```python
from llm_jury import get_recommendations
from llm_jury.ranking.chebyshev import RankingStrategy

# Find alternatives to YOUR current model (not just GPT-5.1)
results = get_recommendations(
    prompt="Write a Python function to parse JSON",
    baseline_model_name="Claude 3.5 Sonnet (new)",  # ← Your current model!
    ranking_strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),
    cost_range=(0.10, 0.30)
)
```

### Example 4: Advanced - Direct Ranker Usage

```python
from llm_jury.ranking.chebyshev import ChebyshevRanker, RankingStrategy
from llm_jury.core.models import RoutingDecision, PromptCategory, ProductArchetype

# Load models and define baseline
baseline = get_model("GPT-5.1 (high)")
models_data = load_all_models()

# Create ranker with custom constraints
ranker = ChebyshevRanker(
    baseline_model=baseline,
    all_models_data=models_data,
    strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),  # 80-95% of baseline quality
    cost_range=(0.10, 0.30),     # 10-30% of baseline cost
    speed_range=(0.30, 10.0)     # ≥30% of baseline speed (optional)
)

# Define task
decision = RoutingDecision(
    category=PromptCategory.CODING,
    archetype=ProductArchetype.FRONTIER,
    reason="Complex coding task requiring high quality"
)

# Get top recommendations
recommendations = ranker.rank(
    models=all_models,
    decision=decision,
    top_k=5,
    return_detailed=True
)

# Display results
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec.name}: "
          f"Quality={rec.quality_score:.1f}, "
          f"Cost=${rec.cost:.2f}, "
          f"Chebyshev={rec.chebyshev_score:.4f}")
```

### Output

```
✓ VALUE_OPTIMIZED: Filtered to 5 models (80%-95% quality, 10%-30% cost)

1. MiniMax-M2: Quality=80.3, Cost=$0.52, Chebyshev=0.0983
2. DeepSeek V3.1 Terminus: Quality=86.6, Cost=$0.80, Chebyshev=0.1987
3. DeepSeek V3.1: Quality=82.3, Cost=$0.65, Chebyshev=0.1987
4. GPT-5 mini (high): Quality=90.6, Cost=$0.69, Chebyshev=0.1987
5. Doubao Seed Code: Quality=79.9, Cost=$0.41, Chebyshev=0.1987
```

### Example 5: Comparing Different Baselines

```python
from llm_jury import get_recommendations
from llm_jury.ranking.chebyshev import RankingStrategy

# Compare sweet spots relative to different models
baselines = ["GPT-5.1 (high)", "Claude 3.5 Sonnet (new)", "Gemini 2.5 Pro"]

for baseline_name in baselines:
    print(f"\n{'='*80}")
    print(f"Sweet spots relative to: {baseline_name}")
    print('='*80)
    
    results = get_recommendations(
        prompt="Write a Python function to parse JSON",
        baseline_model_name=baseline_name,
        ranking_strategy=RankingStrategy.VALUE_OPTIMIZED,
        quality_range=(0.80, 0.95),
        cost_range=(0.10, 0.30),
        top_k=3,
        verbose=True
    )
```

**Key Insight:** Different baselines will recommend different "sweet spot" models because the constraints are relative to the baseline!

---

**Last Updated:** November 30, 2025  
**Version:** 1.0  
**Author:** LLM Jury Team

