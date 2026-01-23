# Metrics Guide: Cold-Start Ablation

## Overview

This experiment tracks **15 key metrics** across 6 visualization panels to provide comprehensive analysis of warmup value.

---

## Core Performance Metrics

### 1. Cumulative Regret
**Definition:** Total regret accumulated over all samples  
**Formula:** `Σ(oracle_reward - actual_reward)`  
**Interpretation:**
- Lower is better
- Measures total cost of suboptimal decisions
- Key metric for comparing learning efficiency

**Typical Values:**
- Warmup: 40-50
- Tabula Rasa: 70-90
- Reduction: 30-50%

**Where to Find:**
- Panel 1: Cumulative Regret Over Time
- JSON: `warmup.cumulative_regret`, `tabula_rasa.cumulative_regret`

---

### 2. Average Reward
**Definition:** Mean reward per sample  
**Formula:** `total_reward / num_samples`  
**Interpretation:**
- Higher is better
- Measures quality of routing decisions
- Converges to optimal policy reward

**Typical Values:**
- Warmup: 0.87-0.89
- Tabula Rasa: 0.82-0.85
- Improvement: 5-10%

**Where to Find:**
- Panel 2: Average Reward Over Time
- JSON: `warmup.avg_reward`, `tabula_rasa.avg_reward`

---

### 3. Day 1 Cumulative Regret
**Definition:** Regret accumulated in first 100 samples  
**Formula:** `Σ(oracle_reward - actual_reward)` for samples 1-100  
**Interpretation:**
- Critical cold-start metric
- Measures early deployment quality
- Most important for user adoption

**Typical Values:**
- Warmup: 10-15
- Tabula Rasa: 20-30
- Reduction: 40-60%

**Where to Find:**
- Panel 5: Day 1 Performance
- JSON: `warmup.day1_cumulative_regret`, `comparison.day1_regret_reduction_pct`

---

### 4. Day 1 Average Reward
**Definition:** Mean reward in first 100 samples  
**Formula:** `mean(rewards[0:100])`  
**Interpretation:**
- Quality during critical early phase
- Directly impacts user experience
- Justifies warmup investment

**Typical Values:**
- Warmup: 0.85-0.87
- Tabula Rasa: 0.78-0.82
- Improvement: 5-15%

**Where to Find:**
- Panel 2: Average Reward (first 100 samples)
- JSON: `warmup.day1_avg_reward`, `comparison.day1_quality_improvement_pct`

---

## Convergence Metrics

### 5. Convergence Sample
**Definition:** Sample where performance gap falls below 1%  
**Formula:** `min(sample where |warmup_reward - tabula_reward| / warmup_reward < 0.01)`  
**Interpretation:**
- Defines "time-to-value" of warmup
- Shows how long warmup advantage lasts
- Critical for ROI analysis

**Typical Values:**
- 200-400 samples
- ~7 hours at 1k queries/day
- ~17 minutes at 1k queries/hour

**Where to Find:**
- Panel 2: Green vertical line
- JSON: `comparison.convergence_sample`, `comparison.time_to_value_samples`

---

### 6. Convergence Gap
**Definition:** Performance gap at convergence point  
**Formula:** `|warmup_reward - tabula_reward| / warmup_reward * 100` at convergence  
**Interpretation:**
- Should be < 1% by definition
- Confirms true convergence
- Lower values = more complete convergence

**Typical Values:**
- 0.5-1.0%
- Validates convergence threshold

**Where to Find:**
- Panel 2: Annotation
- JSON: `comparison.convergence_gap_pct`

---

## Numerical Stability Metrics

### 7. Initial Uncertainty
**Definition:** Average UCB uncertainty in first 10 samples  
**Formula:** `mean(sqrt(x^T A^{-1} x))` for samples 1-10  
**Interpretation:**
- Measures numerical stability
- Higher = more erratic exploration
- Distinguishes stability from semantic guidance

**Typical Values:**
- Warmup: 0.5-1.0
- Tabula Rasa: 2.0-5.0
- Ratio: 3-5×

**Where to Find:**
- Panel 3: Uncertainty Analysis
- JSON: `warmup.avg_initial_uncertainty`, `tabula_rasa.avg_initial_uncertainty`

---

### 8. Initial Uncertainty Ratio
**Definition:** Ratio of tabula rasa to warmup initial uncertainty  
**Formula:** `tabula_initial_uncertainty / warmup_initial_uncertainty`  
**Interpretation:**
- Quantifies numerical instability of cold-start
- Ratio > 1 confirms tabula rasa is less stable
- Separates stability from semantic effects

**Typical Values:**
- 3-5×
- Higher = more instability

**Where to Find:**
- Panel 3: Annotation
- JSON: `comparison.numerical_stability.initial_uncertainty_ratio`

---

### 9. Uncertainty Evolution
**Definition:** UCB uncertainty over first 50 samples  
**Formula:** Time series of `sqrt(x^T A^{-1} x)`  
**Interpretation:**
- Shows how quickly uncertainty stabilizes
- Tabula rasa should converge by sample 30-40
- If performance gap persists after, proves semantic value

**Typical Pattern:**
- Warmup: Stable, low variance
- Tabula Rasa: High initial, rapid decay

**Where to Find:**
- Panel 3: Full time series

---

## Policy Evolution Metrics

### 10. Strong Model Usage %
**Definition:** Percentage of selections for strong model  
**Formula:** `(strong_model_selections / total_selections) * 100`  
**Interpretation:**
- Shows policy adaptation over time
- Both should converge to similar values
- Trajectory matters more than endpoint

**Typical Values:**
- Initial: 50% (random)
- Final: 80-85% (both routers)
- Convergence validates calibration

**Where to Find:**
- Panel 4: Policy Evolution
- JSON: `warmup.final_model_usage`, `tabula_rasa.final_model_usage`

---

### 11. Instantaneous Regret Rate
**Definition:** Change in cumulative regret per sample  
**Formula:** `cumulative_regret[i] - cumulative_regret[i-1]`  
**Interpretation:**
- Derivative of cumulative regret
- Should converge to ~0 (optimal policy)
- Shows learning dynamics

**Typical Pattern:**
- Warmup: Low, stable
- Tabula Rasa: High initially, gradual decline
- Both converge to near-zero

**Where to Find:**
- Panel 6: Regret Rate (smoothed)

---

## Experimental Design Metrics

### 12. Alpha (α)
**Definition:** Exploration parameter in UCB  
**Formula:** `UCB = expected_reward + α * uncertainty`  
**Interpretation:**
- Controls exploration vs. exploitation
- Held constant across routers
- Typically 1.0

**Value:**
- Default: 1.0
- Documented in title and JSON

**Where to Find:**
- Figure title: "α=1.0"
- JSON: `experimental_parameters.alpha`

---

### 13. Total Samples
**Definition:** Number of calibration samples used  
**Typical Value:** 1,121 (matches paper)  
**Where to Find:** JSON: `warmup.total_samples`

---

### 14. Total Regret Reduction %
**Definition:** Overall regret reduction over full calibration  
**Formula:** `(tabula_regret - warmup_regret) / tabula_regret * 100`  
**Interpretation:**
- Overall efficiency gain
- Typically 30-50%
- Sustained advantage beyond Day 1

**Where to Find:**
- Console output
- JSON: `comparison.total_regret_reduction_pct`

---

### 15. Final Model Usage
**Definition:** Model selection percentages at end of calibration  
**Interpretation:**
- Should be similar for both routers
- Validates convergence to same policy
- Proves warmup doesn't bias final policy

**Typical Values:**
- Warmup: 85% strong, 15% weak
- Tabula Rasa: 81% strong, 19% weak
- Difference: 3-5 percentage points

**Where to Find:**
- Console output: Final Model Usage
- JSON: `warmup.final_model_usage`, `tabula_rasa.final_model_usage`

---

## Metric Relationships

### Primary Relationships

```
Cumulative Regret ↔ Average Reward
- Inverse relationship
- Both measure quality, different perspectives

Day 1 Metrics ↔ Overall Metrics
- Day 1 shows cold-start advantage
- Overall shows sustained advantage

Uncertainty ↔ Regret
- High uncertainty → erratic exploration → higher regret
- But relationship decouples after stabilization
```

### Key Insights from Metric Combinations

**Insight 1: Semantic vs. Numerical**
```
IF: Initial Uncertainty Ratio = 3-5×
AND: Day 1 Regret Reduction = 40-60%
THEN: Semantic guidance dominates (numerical stability would only explain ~10-20%)
```

**Insight 2: Convergence Validation**
```
IF: Convergence Sample = 287
AND: Final Model Usage differs by < 5%
THEN: Both routers converge to same policy (validates experiment)
```

**Insight 3: Time-to-Value**
```
IF: Convergence Sample = 287
AND: Day 1 Regret Reduction = 47%
THEN: Warmup provides 287 samples of superior performance
      (critical deployment window)
```

---

## Metric Thresholds

### Strong Results (Publication-Ready)

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Day 1 Regret Reduction | > 40% | Strong cold-start advantage |
| Day 1 Quality Improvement | > 5% | Measurable user impact |
| Total Regret Reduction | > 30% | Sustained advantage |
| Convergence Sample | 200-500 | Reasonable time-to-value |
| Initial Uncertainty Ratio | > 3× | Clear numerical instability |
| Final Usage Difference | < 5% | Proper convergence |

### Concerning Results (Investigate)

| Metric | Threshold | Action |
|--------|-----------|--------|
| Day 1 Regret Reduction | < 20% | Check warmup quality |
| Convergence Sample | > 800 | May indicate domain mismatch |
| Initial Uncertainty Ratio | < 2× | Check A matrix initialization |
| Final Usage Difference | > 10% | Investigate convergence failure |

---

## Metric Checklist for Paper

Before submission, verify:

- [ ] All 15 metrics computed and reported
- [ ] Convergence point explicitly stated
- [ ] Alpha documented in figure and JSON
- [ ] Uncertainty analysis included
- [ ] Time-to-value quantified
- [ ] Final policies compared (convergence validation)
- [ ] Day 1 metrics prominently featured
- [ ] Numerical stability separated from semantic guidance

---

## Quick Reference: Where to Find Each Metric

| Metric | Panel | JSON Path |
|--------|-------|-----------|
| Cumulative Regret | 1 | `*.cumulative_regret` |
| Average Reward | 2 | `*.avg_reward` |
| Day 1 Regret | 5 | `*.day1_cumulative_regret` |
| Day 1 Reward | 2, 5 | `*.day1_avg_reward` |
| Convergence Sample | 2 | `comparison.convergence_sample` |
| Convergence Gap | 2 | `comparison.convergence_gap_pct` |
| Initial Uncertainty | 3 | `*.avg_initial_uncertainty` |
| Uncertainty Ratio | 3 | `comparison.numerical_stability.initial_uncertainty_ratio` |
| Uncertainty Evolution | 3 | `*.uncertainty_history` |
| Strong Model Usage | 4 | `*.final_model_usage` |
| Regret Rate | 6 | Computed from metrics |
| Alpha | Title | `experimental_parameters.alpha` |
| Total Samples | - | `*.total_samples` |
| Total Regret Reduction | Console | `comparison.total_regret_reduction_pct` |
| Final Model Usage | Console | `*.final_model_usage` |

---

## Example Interpretation

**Sample Results:**
```json
{
  "comparison": {
    "day1_regret_reduction_pct": 47.4,
    "day1_quality_improvement_pct": 9.2,
    "total_regret_reduction_pct": 42.7,
    "convergence_sample": 287,
    "convergence_gap_pct": 0.8,
    "numerical_stability": {
      "initial_uncertainty_ratio": 3.2
    }
  }
}
```

**Interpretation:**

1. **Strong cold-start advantage:** 47.4% Day 1 regret reduction proves warmup prevents early disasters
2. **Measurable quality impact:** 9.2% quality improvement translates to real user experience gains
3. **Sustained advantage:** 42.7% total regret reduction shows benefits persist beyond Day 1
4. **Clear time-to-value:** 287 samples (~7 hours at 1k queries/day) of superior performance
5. **Proper convergence:** 0.8% gap confirms both routers reach similar final policies
6. **Semantic dominates numerical:** 3.2× uncertainty ratio explains ~10-20% of advantage; remaining 27-37% is semantic guidance

**Conclusion:** Warmup provides both numerical stability and semantic guidance, with semantic effects dominating. The 287-sample time-to-value justifies the 80k-sample warmup investment for production deployments.

