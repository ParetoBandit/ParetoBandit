# LLM Jury Blog & Visualizations

This directory contains all documentation, visualizations, and test scripts demonstrating the LLM Jury system.

---

## 📊 Visualizations

### Configurable Optimization Suite (New!)

**Purpose:** Demonstrate the value of configurable constrained optimization for finding "sweet spot" LLM models.

| Visualization | File | Description |
|---------------|------|-------------|
| **Sweet Spot Zones** | `sweet_spot_zones.png` | Shows 3 different constraint configurations (Conservative, Balanced, Aggressive) overlaid on cost-quality landscape |
| **Baseline Comparison** | `baseline_comparison.png` | Demonstrates how different baseline models produce different sweet spot recommendations |
| **Pareto Frontier** | `pareto_frontier_sweet_spot.png` | Mathematical foundation: Pareto frontier with sweet spot zone and utopia point |
| **Savings vs Quality** | `savings_vs_quality_tradeoff.png` | Quantifies the tradeoff between quality retention and cost savings |
| **Task-Specific Sweet Spots** | `task_specific_sweet_spots.png` | Shows how the same constraints produce different models for different tasks |

**Guide:** See [`VISUAL_GUIDE.md`](VISUAL_GUIDE.md) for detailed explanations of each visualization.

**Regenerate:** Run `python blog/visualize_configurable_optimization.py`

### Quality Scorer Distributions

| Visualization | File | Description |
|---------------|------|-------------|
| **Production Scorer** | `production_scorer_distributions.png` | Quality score distributions across different task categories |

**Regenerate:** Run `python blog/test_production_scorer.py`

### Chebyshev Optimization Landscape

| Visualization | File | Description |
|---------------|------|-------------|
| **4-Panel Overview** | `chebyshev_optimization_landscape.png` | Comprehensive view of Chebyshev optimization with quality, cost, speed, and combined view |
| **Coding Detailed** | `chebyshev_coding_detailed.png` | Detailed analysis for coding tasks |

**Regenerate:** Run `python blog/visualize_chebyshev_landscape.py`

---

## 📚 Documentation

### Core Concepts

| Document | Description |
|----------|-------------|
| [`CONSTRAINED_OPTIMIZATION_EXPLAINED.md`](CONSTRAINED_OPTIMIZATION_EXPLAINED.md) | **Deep dive** into constrained optimization methodology, academic justification, and why it's legitimate |
| [`FEATURE_SUMMARY.md`](FEATURE_SUMMARY.md) | **Quick reference** with examples, API docs, use cases, and business value |
| [`VISUAL_GUIDE.md`](VISUAL_GUIDE.md) | **Detailed explanations** of all visualizations with key insights and business applications |

### Implementation Details

| Document | Description |
|----------|-------------|
| `GAUSSIAN_SCORING_SUMMARY.md` | Analysis of different scoring approaches (historical) |
| `FILTERED_DATASET_SUMMARY.md` | Impact of filtering on quality score distributions |
| `COMPLETE_DATA_FILTER_GUIDE.md` | How to use `--complete-only` filter in ETL |

---

## 🧪 Test Scripts

### Configurable Optimization Tests

| Script | Purpose |
|--------|---------|
| `test_configurable_sweet_spot.py` | Test configurable constraint ranges (quality, cost, speed) |
| `test_configurable_baseline.py` | Test configurable baseline/reference models |

**Run:** `python blog/test_configurable_sweet_spot.py`

### Quality Scorer Tests

| Script | Purpose |
|--------|---------|
| `test_production_scorer.py` | Test production quality scorer with visualizations |

**Run:** `python blog/test_production_scorer.py`

### Visualization Generators

| Script | Purpose |
|--------|---------|
| `visualize_configurable_optimization.py` | **Generate all 5 optimization visualizations** |
| `visualize_chebyshev_landscape.py` | Generate Chebyshev optimization landscape plots |

**Run:** `python blog/visualize_configurable_optimization.py`

---

## 🎯 Key Features Demonstrated

### 1. Configurable Constraints ✅

Users can specify ANY constraint ranges:

```python
results = get_recommendations(
    prompt="Your task",
    quality_range=(0.80, 0.95),  # 80-95% of baseline quality
    cost_range=(0.10, 0.30),     # 10-30% of baseline cost
    speed_range=(0.30, 10.0)     # Optional speed constraint
)
```

**Visualized in:** `sweet_spot_zones.png`, `savings_vs_quality_tradeoff.png`

### 2. Configurable Baseline ✅

Users can specify ANY model as the reference:

```python
results = get_recommendations(
    prompt="Your task",
    baseline_model_name="YOUR_CURRENT_MODEL",  # Not hardcoded!
    ranking_strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),
    cost_range=(0.10, 0.30)
)
```

**Visualized in:** `baseline_comparison.png`

### 3. Mathematical Rigor ✅

Two-phase constrained optimization:
- **Phase 1:** Filter to feasible region (constraint satisfaction)
- **Phase 2:** Chebyshev optimization (minimize max weighted regret)

**Visualized in:** `pareto_frontier_sweet_spot.png`

### 4. Task-Specific Optimization ✅

Same constraints, different models for different tasks:
- Coding tasks get coding-optimized models
- Creative tasks get writing-optimized models
- Data science tasks get math-optimized models

**Visualized in:** `task_specific_sweet_spots.png`

---

## 📈 Business Value

### Cost Savings
- **Conservative:** 70-80% savings, 90%+ quality retention
- **Balanced:** 75-85% savings, 80-95% quality retention
- **Aggressive:** 85-90% savings, 70-90% quality retention

### Flexibility
- **No one-size-fits-all:** Adapt to different teams, projects, budgets
- **Personalized:** Find alternatives to YOUR specific model
- **Transparent:** All constraints are explicit and interpretable

### Academic Rigor
- ✅ Based on established optimization theory (Boyd & Vandenberghe, 2004)
- ✅ Would pass peer review in operations research
- ✅ Fully reproducible and deterministic
- ✅ No black boxes, no magic numbers

---

## 🚀 Quick Start

### 1. Basic Usage

```python
from llm_jury import get_recommendations

# Simple - uses default baseline (GPT-5.1 high)
results = get_recommendations(
    prompt="Write a Python function to parse JSON",
    has_search_tools=False
)
```

### 2. Value-Optimized with Constraints

```python
from llm_jury import get_recommendations
from llm_jury.ranking.chebyshev import RankingStrategy

# Find sweet spot models
results = get_recommendations(
    prompt="Write a Python function to parse JSON",
    ranking_strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),
    cost_range=(0.10, 0.30),
    top_k=5
)
```

### 3. Custom Baseline

```python
# Find alternatives to YOUR current model
results = get_recommendations(
    prompt="Write a Python function to parse JSON",
    baseline_model_name="Claude 3.5 Sonnet (new)",
    ranking_strategy=RankingStrategy.VALUE_OPTIMIZED,
    quality_range=(0.80, 0.95),
    cost_range=(0.10, 0.30)
)
```

---

## 📊 Empirical Results

### Balanced Sweet Spot (Default)

**Baseline:** GPT-5.1 (high) @ $3.44/M tokens  
**Constraints:** 80-95% quality, 10-30% cost

**Top 5 Models:**

| Model | Quality | Cost | Savings |
|-------|---------|------|---------|
| MiniMax-M2 | 80.3 | $0.52 | 85% |
| DeepSeek V3.1 Terminus | 86.6 | $0.80 | 77% |
| DeepSeek V3.1 | 82.3 | $0.65 | 81% |
| GPT-5 mini (high) | 90.6 | $0.69 | 80% |
| Doubao Seed Code | 79.9 | $0.41 | 88% |

---

## 🔄 Regenerating Everything

To regenerate all visualizations and run all tests:

```bash
# Generate optimization visualizations
python blog/visualize_configurable_optimization.py

# Test configurable constraints
python blog/test_configurable_sweet_spot.py

# Test configurable baseline
python blog/test_configurable_baseline.py

# Test quality scorer
python blog/test_production_scorer.py

# Generate Chebyshev landscape
python blog/visualize_chebyshev_landscape.py
```

---

## 📁 File Structure

```
blog/
├── README.md                                    # This file
├── CONSTRAINED_OPTIMIZATION_EXPLAINED.md        # Deep dive documentation
├── FEATURE_SUMMARY.md                           # Quick reference
├── VISUAL_GUIDE.md                              # Visualization guide
│
├── visualize_configurable_optimization.py       # Generate 5 optimization plots
├── test_configurable_sweet_spot.py              # Test constraint ranges
├── test_configurable_baseline.py                # Test baseline models
├── test_production_scorer.py                    # Test quality scorer
├── visualize_chebyshev_landscape.py             # Generate Chebyshev plots
│
├── sweet_spot_zones.png                         # Constraint configurations
├── baseline_comparison.png                      # Different baselines
├── pareto_frontier_sweet_spot.png               # Mathematical foundation
├── savings_vs_quality_tradeoff.png              # ROI analysis
├── task_specific_sweet_spots.png                # Task-specific optimization
├── production_scorer_distributions.png          # Quality distributions
├── chebyshev_optimization_landscape.png         # 4-panel Chebyshev view
└── chebyshev_coding_detailed.png                # Detailed coding analysis
```

---

## 🎓 Academic References

1. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.
2. **Steuer, R. E., & Choo, E. U.** (1983). An interactive weighted Tchebycheff procedure for multiple objective programming. *Mathematical Programming*, 26(3), 326-344.
3. **Wierzbicki, A. P.** (1980). The use of reference objectives in multiobjective optimization. *Multiple Criteria Decision Making Theory and Application*, 468-486.
4. **Miettinen, K.** (2012). *Nonlinear Multiobjective Optimization*. Springer Science & Business Media.

---

## 📧 Contact

For questions or contributions, please open an issue on GitHub.

---

**Last Updated:** November 30, 2025  
**Version:** 1.0  
**Author:** LLM Jury Team

