# Shared Evaluation Architecture

## Purpose

Ensure consistency between Figure 8, Figure 9, and Table 3 by using a single canonical implementation of router evaluation metrics.

## Implementation

### Core Module: `router_evaluation.py`

Contains the `RouterEvaluator` class with canonical methods:

1. **`classify_policy_restricted()`**: Classify queries as medical/legal/financial
   - Uses `HighRiskPromptClassifier` with threshold=5.0
   - Returns boolean array marking restricted queries

2. **`calculate_safety_violation_at_efficiency()`**: Budget-based violation rate
   - Dithering (1e-6 noise) to break ties
   - Top-K selection for target efficiency
   - Returns violation percentage (0-100)

3. **`calculate_leakage_at_target_efficiency()`**: For curve plotting
   - Same as above but returns (efficiency, violation_rate) tuple
   - Used by Figure 9 for smooth compliance curves

4. **`calculate_apgr()`**: APGR metric calculation
   - Area under Pareto curve
   - Normalized by theoretical maximum

### Usage

#### Figure 8 (Histogram): `check_score_spread.py`
```python
from router_evaluation import get_evaluator

evaluator = get_evaluator()
restricted_mask = evaluator.classify_policy_restricted(queries)
# Histogram shows bimodal distribution
```

#### Figure 9 (Compliance Curves): `plot_safety.py`
```python
from router_evaluation import get_evaluator

evaluator = get_evaluator()
restricted_mask = evaluator.classify_policy_restricted(queries)

# For each efficiency target:
actual_eff, violation = evaluator.calculate_leakage_at_target_efficiency(
    df, prob_col, restricted_mask, target_efficiency
)
```

#### Table 3 (Performance): `generate_table_3_final.py`
```python
from router_evaluation import get_evaluator

evaluator = get_evaluator()
restricted_mask = evaluator.classify_policy_restricted(queries)

violation_rate = evaluator.calculate_safety_violation_at_efficiency(
    scores, restricted_mask, target_efficiency=0.95
)
```

## Benefits

1. **Zero Drift**: All figures/tables use identical algorithms
2. **Single Source of Truth**: One place to update evaluation logic
3. **Testable**: Can unit test the evaluator in isolation
4. **Documented**: Clear API for each metric

## Consistency Guarantees

- **Policy Definition**: All use `HighRiskPromptClassifier(threshold=5.0)`
- **Budget-Based Selection**: All use same dithering (1e-6) and top-K algorithm
- **Safety Metric**: All calculate violation rate identically
- **No Algorithm Drift**: Changes to one automatically apply to all

This architecture ensures that the numbers in Table 3 and the curves in Figure 9 are mathematically consistent and come from the exact same evaluation pipeline.
