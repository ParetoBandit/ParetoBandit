# Prior Strength Ablation Study - Changes Summary

## What Changed

The `n_eff_ablation.py` script has been updated to properly test **γ_prior (prior_n_effective)** while comparing CSR vs. HLE priors.

## Key Modifications

### 1. **Experimental Design**
- **γ_structure = 20**: Fixed at N_target = 20 (constant baseline covariance structure)
- **γ_prior varies**: From 0 to 50 to test the incremental value of prior means

This tests whether **prior means add value** on top of the covariance structure baseline.

### 2. **Configuration Changes**

**Before** (Matched Scaling):
```python
csr_router = BanditRouter.create(
    registry,
    priors="benchmark",
    prior_n_effective=float(n_eff),
    prior_structure_n_effective=float(n_eff)  # Both scaled together
)
```

**After** (Fixed Structure + Variable Means):
```python
csr_router = BanditRouter.create(
    registry,
    priors="benchmark",
    prior_n_effective=float(gamma_prior),  # γ_prior: b vector strength (varies)
    prior_structure_n_effective=20.0  # γ_structure = 20: fixed baseline
)
```

### 3. **What This Tests**

- **CSR Priors**: Use task-specific cluster success rates as prior means
- **HLE Priors**: Use generic benchmark scores as prior means
- **Both**: Use the same covariance structure (N_target = 20)

**Expected Result**: CSR should show greater improvement as γ_prior increases, demonstrating that task-specific prior means are more valuable than generic benchmarks when combined with the same structural priors.

### 4. **Output Changes**

- **Plot file**: `prior_strength_ablation.png` (was `n_eff_ablation.png`)
- **X-axis label**: "Prior Strength (γ_prior)" (was "Prior Strength (N_eff)")
- **Title**: "CSR vs. HLE: Quality of Prior Means"
- **Metrics**: Now reports "improvement over cold start %" instead of "variance %"

### 5. **Variables Renamed**

- `n_eff_values` → `gamma_prior_values`
- `N_EFF_VALUES` → `GAMMA_PRIOR_VALUES`
- `"Hybrid Arch (CSR)"` → `"CSR Priors"`
- Labels updated throughout to use γ_prior instead of N_eff

## Running the Ablation

```bash
cd /Users/annette/repostitories/llm_jury
python banditgpt/experiments/ablation/n_eff_ablation.py
```

The script will:
1. Test γ_prior = [0, 1, 5, 10, 20, 40, 50] with γ_structure = 20 (fixed)
2. Run 20 trials per configuration
3. Compare CSR vs. HLE across all values
4. Generate a plot showing cumulative regret vs. prior strength
5. Report which approach benefits more from stronger priors

## Interpretation

- **If CSR shows steeper improvement**: Task-specific prior means are more informative than generic ones
- **If HLE plateaus or improves less**: Generic benchmarks have limited incremental value beyond structure
- **Gap between curves**: Quantifies the "domain specificity advantage" of CSR over HLE
- **Both start better than pure cold start**: The γ_structure = 20 baseline provides a foundation that both build upon
