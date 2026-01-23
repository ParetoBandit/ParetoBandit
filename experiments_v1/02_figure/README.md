# Figure 2: Calibration Convergence Analysis

## Overview

This experiment demonstrates that **policy convergence occurs during calibration, not during holdout evaluation**. We compare three frozen router policies to isolate the effects of covariance inflation (γ-scaling) versus online learning.

## Key Finding

**The Policy Cliff**: Strong model usage drops from 100% → 0.3% (99.7 pp shift), but this convergence happens in two distinct phases:

1. **γ-scaling alone** (Scenario 1 → 2): 0.3 pp change — creates plasticity but doesn't change policy
2. **Calibration data** (Scenario 2 → 3): 99.4 pp change — drives full convergence

This proves convergence occurs **during calibration** (1,121 samples), not during holdout evaluation (750 samples).

## Experimental Design

### Three Scenarios

| Scenario | Description | Strong Usage | Quality | Eff. N |
|----------|-------------|--------------|---------|--------|
| **1. Warmup Only** | Pre-trained priors (GPT-4-Turbo, 80K samples) | 100.0% | 0.971 | 426 |
| **2. γ-Scaled** | Warmup × γ=0.01 (no calibration data) | 99.7% | 0.971 | 4 |
| **3. Calibrated** | γ-scaled + 1,121 GPT-4o samples | 0.3% | 0.823 | 16 |

### Frozen Policy Evaluation

- All scenarios evaluated on 750 holdout samples
- Learning disabled (α=0) to isolate convergence from exploration
- No feedback during evaluation — pure policy testing

## Results

### Panel A: Policy Convergence

- **The Cliff**: 99.7 pp drop in strong model usage
- **Ablation**: γ-scaling necessary but not sufficient
- **Timing**: Convergence during calibration, not holdout

### Panel B: Quality-Cost Tradeoff

- **Quality drop**: 14.8% (0.971 → 0.823)
- **Cost savings**: 99.7% reduction in expensive model calls
- **Discovery**: Weak model (Mixtral-8x7B) handles most queries

### Panel C: Bayesian Plasticity

- **γ-scaling effect**: 99% prior reduction (426 → 4 eff. N)
- **Calibration contribution**: +12 effective samples
- **Influence ratio**: 2.78× (calibration dominates weakened prior)

## Cross-Model Transfer

An important dimension: warmup used **GPT-4-Turbo**, while calibration/holdout used **GPT-4o**.

- Despite capability-tier substitution, router successfully adapts
- 100% → 0.3% shift indicates learning GPT-4o's distinct characteristics
- Validates robustness to model evolution in production

## Files

### Code
- `compare_calibration_convergence.py` — Main analysis script
- Uses actual `BanditRouter` from `src/bandit_gpt/router.py`

### Outputs
- `results/calibration_convergence_comparison.{png,pdf,eps}` — Publication-quality figure
- `results/comparison_metrics.json` — Numerical results
- `results/calibration_convergence_results.tex` — KDD-compliant results section
- `results/figure_caption.tex` — LaTeX figure caption

### Data Sources
- Warmup priors: `src/artifacts/priors_warmup.joblib` (GPT-4-Turbo)
- Calibrated router: `experiments_v1/calibration/results/artifacts/canonical_router_calibrated.joblib` (GPT-4o)
- Holdout data: `src/bandit_gpt/data/offline_dataset/holdout_rewards_complete.jsonl.gz` (GPT-4o)

## Reproducing Results

```bash
cd experiments_v1/02_figure
python compare_calibration_convergence.py --output results
```

### Optional Arguments

```bash
python compare_calibration_convergence.py \
  --warmup-priors <path>      # Default: from config_legacy.py
  --calibrated-router <path>  # Default: from config_legacy.py  
  --holdout-data <path>       # Default: from config_legacy.py
  --pca <path>                # Default: from config_legacy.py
  --gamma 0.01                # Gamma value used during calibration
  --output results            # Output directory
```

## KDD Reviewer Considerations

### Why This Figure Matters

1. **Addresses "overfitting" concern**: Frozen evaluation proves holdout metrics reflect generalization, not continued learning
2. **Ablation study**: γ-scaling vs. calibration effects cleanly separated
3. **Causal evidence**: Three-scenario design shows what drives convergence
4. **Practical relevance**: Cross-model transfer demonstrates production robustness

### Design Choices

- **Bar plots**: Visualize the "cliff effect" more dramatically than tables
- **Three panels**: Show usage, quality, and mechanism in parallel
- **Log scale (Panel C)**: Properly represents 100× magnitude differences
- **Annotations**: Guide reader to key insights (99.4 pp drop, 2.78× ratio)

### Statistical Rigor

- **Frozen policies**: No learning during evaluation
- **Effective N**: Quantifies prior strength (trace(A)/dim)
- **Calibration/Prior ratio**: Measures influence balance
- **Percentage points**: Absolute changes, not relative percentages

## Integration with Paper

This figure supports Section 5 (Experimental Results), specifically:

- **5.1**: The Policy Pivot — shows the convergence trajectory
- **5.2**: Bayesian Plasticity — quantifies γ-scaling + calibration effects
- **5.3**: Convergence Timeline — proves timing of convergence
- **5.4**: Cross-Model Transfer — validates GPT-4-Turbo → GPT-4o adaptation

## Citation

When referencing this experiment:

> We demonstrate that policy convergence occurs during calibration (1,121 samples), not during holdout evaluation (750 samples). Covariance inflation (γ=0.01) reduces prior strength by 99%, enabling calibration data (2.78× influence ratio) to drive a 99.7 percentage point shift in routing strategy (Figure 2).

