# Covariance Ablation: Experimental Design & Interpretation Guide

## Research Question
**Do off-diagonal correlations in the CSR covariance matrix provide routing value in PCA-reduced space?**

## Why This Matters
The user's theoretical framework posits that even in PCA space, the **success-weighted covariance** is NOT diagonal:
- Global PCA components are uncorrelated by definition
- BUT successful examples may show correlated performance across PCs
- Example: If DeepSeek-Coder excels at both PC1 (Coding) and PC5 (Logic), then successful examples will show positive correlation between these PCs

## Experimental Design

### Test Matrix (4 Conditions)

| Configuration | prior_n_effective | prior_structure_n_effective | Covariance Type |
|---------------|-------------------|------------------------------|-----------------|
| Struct + Full | 0.0 | 20.0 | Full Σ_CSR |
| Struct + Diag | 0.0 | 20.0 | diag(Σ_CSR) |
| CSR + Full | 20.0 | 20.0 | Full Σ_CSR |
| CSR + Diag | 20.0 | 20.0 | diag(Σ_CSR) |

### What Each Parameter Does

**prior_n_effective** (controls b vector):
- `0.0`: No prior beliefs about model quality → cold start for means
- `20.0`: Inject CSR prior beliefs (cluster success rates)

**prior_structure_n_effective** (controls A matrix):
- `20.0`: Scale covariance to equivalent of 20 samples → allows learning

**Covariance Type**:
- **Full**: Complete 45×45 matrix with all correlations
- **Diagonal**: Only 45 variances, zero off-diagonals

## Interpretation Framework

### Scenario 1: Both Experiments Show Benefit
```
Structure-Only: Full < Diagonal (benefit > 10%)
Full CSR:       Full < Diagonal (benefit > 10%)
```
**Conclusion**: Off-diagonal correlations are valuable **independently**
- Structure itself encodes useful transfer learning
- Correlations help even without prior beliefs
- **Architecture is the hero** - the "map" works alone

### Scenario 2: Only Full CSR ShowsBenefit
```
Structure-Only: Full ≈ Diagonal (benefit < 10%)
Full CSR:       Full < Diagonal (benefit > 10%)
```
**Conclusion**: Correlations require prior means to be useful
- Off-diagonals work **synergistically** with beliefs
- The "map" needs the "compass" (prior means) to navigate
- Combined effect is greater than sum of parts

### Scenario 3: Neither Shows Benefit
```
Structure-Only: Full ≈ Diagonal (benefit < 10%)
Full CSR:       Full ≈ Diagonal (benefit < 10%)
```
**Conclusion**: Diagonal variances are sufficient
- Off-diagonal correlations add minimal value
- Simpler diagonal covariance may be preferable
- Challenges the theoretical framework

### Scenario 4: Unexpected Pattern
```
Structure-Only: Full < Diagonal (benefit > 10%)
Full CSR:       Full ≈ Diagonal (benefit < 10%)
```
**Conclusion**: Correlations are noise, not signal
- Off-diagonals interfere when combined with strong priors
- Diagonal may help avoid overfitting
- Prior means dominate the routing decision

## Implementation Details

**Feature Space**: 46 dimensions total
- 32 PCA components (384→32 via trained PCA)
- 8 explicit features (code density, JSON requirements, etc.)
- 5 cluster distances (anchor clusters)
- 1 bias term

**Covariance Matrix**: 45×45 (excludes bias)
- Loaded from `priors_meta_pca.npz`
- Built from 21,719 training prompts
- Mean diagonal: ~7,309
- Mean |off-diagonal|: ~634

**Scaling**: `gamma = 20.0 / 21,719 ≈ 0.000921`

## Files

- **Structure-Only Script**: `covariance_ablation_csr.py`
- **Comprehensive Comparison**: `covariance_ablation_comparison.py`
- **Documentation**: `covariance_ablation_README.md`

## Expected Runtime

- Each trial: ~60 seconds (981 prompts × 2 models × 2 routing + update)
- Structure-Only (20 trials): ~40 minutes
- Full Comparison (4 conditions × 20 trials): ~120 minutes
