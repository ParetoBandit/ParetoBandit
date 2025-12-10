# BLF Model: Quick Reference Guide

## Model Specification

### Likelihood
```
z_{i,b} ~ Normal(α_b + λ_b * θ_i, σ_b²)
```

where:
- `z_{i,b}`: Standardized benchmark score for model i on benchmark b
- `θ_i`: Latent composite score for model i (standardized, mean=0, sd=1)
- `α_b`: Benchmark intercept (difficulty offset)
- `λ_b`: Benchmark loading (learned weight, > 0)
- `σ_b`: Residual noise (measurement error)

### Priors
```
θ_i  ~ Normal(0, 1)           # Standard normal for identifiability
α_b  ~ Normal(0, 2²)          # Weakly informative
λ_b  ~ HalfNormal(1)          # Positive loadings, mean ≈ 0.80
σ_b  ~ HalfNormal(1)          # Positive noise
```

## Interpretation

### Latent Score (θ_i)
- **Mean = 0**: Average model
- **θ_i > 0**: Above-average model
- **θ_i < 0**: Below-average model
- **Transform to 0-100**: Score = 50 + 10*θ_i

### Loadings (λ_b)
- **High λ_b (0.85-1.0)**: Benchmark strongly correlates with composite score
- **Medium λ_b (0.6-0.85)**: Moderate correlation
- **Low λ_b (< 0.6)**: Weak correlation (e.g., auxiliary benchmarks)

### Noise (σ_b)
- **Low σ_b (< 0.4)**: Reliable benchmark with low measurement error
- **High σ_b (> 0.7)**: Noisy benchmark (e.g., stochastic evaluation)

### Posterior Uncertainty
- **Narrow HDI (< 5 points)**: High confidence in score
- **Wide HDI (> 10 points)**: Low confidence (missing data or conflicting signals)

## Convergence Criteria

✅ **Good Convergence:**
- R̂ < 1.01 for all parameters
- ESS > 400 per chain (1,600 total for 4 chains)
- No divergent transitions
- BFMI > 0.3
- Trace plots show good mixing

⚠️ **Convergence Issues:**
- R̂ > 1.05: Chains haven't converged
- Divergences > 1%: Problematic posterior geometry
- Low ESS: High autocorrelation, need more samples

## Model Validation Checklist

1. **Posterior Predictive Checks**
   - [ ] R² > 0.80 (observed vs. predicted)
   - [ ] Residuals centered at 0
   - [ ] No systematic patterns in residuals

2. **External Validation**
   - [ ] Spearman ρ > 0.85 with Arena ELO
   - [ ] Reasonable correlation with other quality metrics

3. **Convergence**
   - [ ] All R̂ < 1.01
   - [ ] All ESS > 400 per chain
   - [ ] Zero divergences

4. **Sensitivity**
   - [ ] Prior sensitivity: ρ > 0.995 across variants
   - [ ] Benchmark ablation: largest Δρ < 0.10

## Comparison with Baselines

| Method | Pros | Cons |
|--------|------|------|
| **BLF** | • Handles missing data<br>• Learns weights<br>• Quantifies uncertainty<br>• High coverage (95%) | • Computationally expensive<br>• Requires MCMC |
| **Weighted Z-Score** | • Fast<br>• Simple | • Requires complete data (68% coverage)<br>• Manual weights<br>• No uncertainty |
| **Arithmetic Mean** | • Very simple | • Ignores benchmark quality<br>• Requires complete data |
| **Best Single** | • Simple<br>• Good coverage (90%) | • Ignores other signals<br>• Lower correlation (0.82 vs 0.89) |

## When to Use BLF

✅ **Use BLF when:**
- Models have heterogeneous benchmark coverage
- You need uncertainty quantification
- Benchmark quality varies substantially
- You want data-driven weighting
- You have time for offline computation

❌ **Don't use BLF when:**
- All models have complete data (use weighted z-score)
- Real-time scoring is required (precompute instead)
- < 50 models available (insufficient data)
- Benchmarks measure fundamentally different constructs

## Common Issues and Solutions

### Issue 1: Divergent Transitions
**Symptom:** Warning about divergences during sampling  
**Solution:**
- Increase `target_accept` (0.90 → 0.95 → 0.99)
- Reparameterize model (non-centered parameterization)
- Increase tuning steps (2000 → 3000)

### Issue 2: Low ESS
**Symptom:** ESS < 100 per chain  
**Solution:**
- Increase samples (2000 → 5000)
- Check for high autocorrelation (thinning may help)
- Reparameterize model

### Issue 3: R̂ > 1.01
**Symptom:** Chains haven't converged  
**Solution:**
- Run more tuning steps
- Check for multimodality
- Increase chains from 4 to 8
- Check initialization (use MAP estimate)

### Issue 4: Wide HDIs
**Symptom:** All models have wide credible intervals  
**Solution:**
- Check if benchmarks are too noisy (high σ_b)
- Add auxiliary benchmarks for imputation
- Verify standardization is correct
- May be legitimate uncertainty (not enough data)

## Key Equations

### Posterior Mean (Point Estimate)
```
θ̂_i = E[θ_i | z] = ∫ θ_i p(θ_i | z) dθ_i
```
Computed via MCMC: average of posterior samples.

### Credible Interval (Uncertainty)
```
HDI_95% = [θ_2.5%, θ_97.5%]
```
Highest Density Interval: narrowest interval containing 95% of posterior mass.

### Posterior Predictive
```
z_{i,b}^{rep} ~ ∫ p(z | θ, α, λ, σ) p(θ, α, λ, σ | z) d(θ, α, λ, σ)
```
Used for model checking: simulated data should look like observed data.

## Computational Requirements

**Typical Runtime** (250 models, 5 benchmarks):
- Warm-up: 1-2 minutes
- Sampling: 2-3 minutes
- Total: 3-5 minutes on single CPU

**Memory:**
- Data: ~1 MB
- Posterior samples: ~10 MB (2000 samples × 4 chains × 255 parameters)

**Scaling:**
- Linear in N (models)
- Linear in B (benchmarks)
- Linear in S (MCMC samples)

## References

See `CITATION.bib` for complete references.

**Key Papers:**
- Hoffman & Gelman (2014): NUTS sampler
- Rubin (1987): Multiple imputation
- Gelman & Rubin (1992): R̂ convergence diagnostic
- Spearman (1904): Factor analysis foundations
