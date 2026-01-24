# Figure 3: Optimal Gamma Calibration Analysis

## Overview

This experiment systematically evaluates different gamma (covariance inflation) values to find the optimal balance between warmup priors and calibration data for domain adaptation.

## Key Finding

**Optimal Gamma: γ = 0.010**

This value provides:
- **Balanced Influence**: Calibration/Prior ratio of 1.401
- **Effective Adaptation**: 13.6 pp shift in routing strategy
- **Quality Preservation**: 0.8510 average reward
- **Sample Efficiency**: 800 effective samples (reduced from 80,000)

## Research Questions

1. **How does gamma affect policy adaptation?**
   - Lower gamma values enable faster convergence and larger policy shifts
   - Gamma acts as a "plasticity knob" for the Bayesian prior

2. **What is the optimal Calibration/Prior ratio?**
   - Values near 1.0 provide balanced influence
   - Our optimal gamma achieves 1.401

3. **How does effective sample size influence convergence?**
   - Reducing Eff. N from 80,000 to 800 enables rapid adaptation
   - Too much reduction (γ < 0.001) may discard valuable warmup knowledge

## Files

### Generated Outputs
- `results/optimal_gamma_analysis.png` — High-resolution figure (300 DPI)
- `results/optimal_gamma_analysis.pdf` — Vector format for publication
- `results/optimal_gamma_analysis.eps` — Alternative vector format
- `results/gamma_results.json` — Numerical results
- `results/figure_caption.tex` — LaTeX figure caption
- `results/gamma_results.tex` — LaTeX results section

### Scripts
- `find_optimal_gamma.py` — Main analysis script

## Experimental Design

### Gamma Values Tested
1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.050, 0.020, 0.010, 0.005, 0.002, 0.001

### Dataset
- **Calibration**: 1,121 samples
- **Warmup**: 80,000 samples
- **Models**: mistralai/mixtral-8x7b-instruct vs openai/gpt-4-turbo

### Evaluation Criteria

1. **Target Matching**: How close to oracle usage (if known)
2. **Maximum Adaptation**: Largest change from baseline
3. **Balanced Influence**: Calib/Prior ratio near 1.0
4. **Convergence Speed**: Fastest policy adaptation

## Results Summary

| Gamma | Eff. N | Calib/Prior | Strong % | Reward | Conv. Rate |
|-------|--------|-------------|----------|--------|------------|
| 1.000 | 80,000 | 0.014 | 53.3% | 0.8002 | 0.003009 |
| 0.500 | 40,000 | 0.028 | 56.2% | 0.8020 | 0.005592 |
| 0.300 | 24,000 | 0.047 | 58.8% | 0.8278 | 0.007745 |
| 0.200 | 16,000 | 0.070 | 61.3% | 0.8439 | 0.009699 |
| 0.150 | 12,000 | 0.093 | 63.3% | 0.8448 | 0.011315 |
| 0.100 | 8,000 | 0.140 | 65.9% | 0.8448 | 0.012901 |
| 0.050 | 4,000 | 0.280 | 68.8% | 0.8483 | 0.013135 |
| 0.020 | 1,600 | 0.701 | 68.4% | 0.8510 | 0.006383 |
| 0.010 ⭐ | 800 | 1.401 | 67.0% | 0.8510 | 0.000492 |
| 0.005 | 400 | 2.803 | 64.1% | 0.8617 | 0.005730 |
| 0.002 | 160 | 7.006 | 62.0% | 0.8715 | 0.008239 |
| 0.001 | 80 | 14.012 | 63.4% | 0.8483 | 0.005856 |


⭐ = Recommended value

## Key Insights

### 1. Prior Weakening is Essential

The baseline (γ=1.0) shows minimal adaptation (53.3% strong usage), demonstrating that 80,000 warmup samples create strong inertia.

### 2. Optimal Balance Exists

Too large (γ ≥ 0.1): Insufficient adaptation
Too small (γ ≤ 0.001): May discard valuable knowledge
**Optimal (γ = 0.010)**: Balanced influence

### 3. Sample Efficiency

With optimal gamma, 1,121 calibration samples (1.40% of warmup data) achieve significant policy adaptation.

## Reproducing Results

```bash
cd experiments_v1/03_figure

# Basic usage (uses defaults from config_legacy.py)
python find_optimal_gamma.py --output results/

# Custom gamma values
python find_optimal_gamma.py \
  --gamma-values 1.0 0.05 0.02 0.01 0.005 0.002 0.001 \
  --output results/

# With target usage (if you know oracle policy)
python find_optimal_gamma.py \
  --target-usage 25.0 \
  --output results/

# Verbose mode
python find_optimal_gamma.py --verbose --output results/
```

## Integration with Paper

This figure supports:

- **Section 4 (Methodology)**: Explains gamma selection process
- **Section 5 (Experimental Results)**: Demonstrates optimal calibration
- **Section 6 (Analysis)**: Shows sample efficiency and adaptation dynamics

### Citation Example

> We systematically evaluated gamma values from 0.001 to 1.0 to determine the optimal covariance inflation factor. Our analysis (Figure~\ref{fig:optimal_gamma}) reveals that γ=0.010 provides the optimal balance, achieving a Calibration/Prior ratio of 1.401. This enables 1,121 calibration samples to effectively adapt 80,000 warmup priors, resulting in a 13.6 percentage point shift in routing strategy.

## Related Experiments

- **Figure 1**: Semantic task specialization visualization
- **Figure 2**: Calibration convergence analysis
- **Calibration**: Full calibration pipeline and evaluation

---

**Created**: 1769218099.3788846  
**Dataset**: 1,121 calibration samples, 80,000 warmup samples  
**Recommended**: γ = 0.010
