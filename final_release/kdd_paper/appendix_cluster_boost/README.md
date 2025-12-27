# Appendix: Cluster Boost Weight Tuning

This appendix documents the hyperparameter tuning process for the `cluster_boost_weight` parameter in the cluster-aware reward boosting system.

## Overview

The cluster boost weight controls how strongly models are rewarded/penalized based on their cluster-specific performance:

```python
boosted_reward = base_reward × (1 + z_score × cluster_boost_weight)
```

- **z_score**: Model's comparative advantage in detected cluster (can be positive or negative)
- **cluster_boost_weight**: Tunable parameter (range: 0.0 to 0.5)

## Methodology

**Optimization Approach:** 5-fold cross-validation on training data

**Grid Search Range:** `[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]`

**Evaluation Metric:** Cumulative regret (lower is better)

**Secondary Metrics:**
- Validation accuracy (% optimal selections)
- Mean regret per decision
- Convergence speed

**Configuration:**
- **Dataset**: 4,000 training prompts with ground truth rewards
- **Priors**: HLE benchmark priors (`priors="benchmark"`)
- **Profile**: Balanced cost/quality tradeoff (`profile="balanced"`)
- **Folds**: 5-fold stratified cross-validation

## Script Usage

```bash
cd final_release/kdd_paper/appendix_cluster_boost
python3 tune_cluster_boost_weight.py
```

**Output:**
- Console: Detailed results for each weight
- Plot: `cluster_boost_5fold_cv.png` showing regret and accuracy curves
- Recommendation: Optimal weight based on minimum regret

## Interpretation

**Weight = 0.0** (Baseline)
- No cluster-specific boosting
- Models learn from raw feedback only

**Weight = 0.1** (Conservative)
- 10% boost for models with z=1.0
- Gradual specialization

**Weight = 0.3** (Moderate)
- 30% boost for models with z=1.0
- Faster specialization, risk of over-fitting to priors

**Weight = 0.5** (Aggressive)
- 50% boost for models with z=1.0
- Maximum specialization speed, highest risk

## Expected Results

Typical outcomes from tuning:
- **Best weight**: Usually between 0.05 and 0.15
- **Improvement**: 5-15% reduction in cumulative regret vs baseline
- **Trade-off**: Higher weights → faster learning but lower stability

## References

See implementation in:
- [`cluster_detector.py`](file:///Users/annette/repostitories/llm_jury/final_release/cluster_detector.py) - Cluster detection
- [`bandit.py`](file:///Users/annette/repostitories/llm_jury/final_release/bandit.py) - Reward boosting logic
