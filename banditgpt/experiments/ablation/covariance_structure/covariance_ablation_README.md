# Covariance Ablation Experiment for CSR

## Objective
Test the impact of **off-diagonal correlations** in the CSR covariance matrix within PCA-reduced space.

## Hypothesis
In PCA space, the covariance of successful examples is NOT diagonal, even though global PCA components are uncorrelated. The off-diagonal elements encode critical transfer learning signals:
- **PC-to-PC Transfer**: "Success in PC1 (Coding) → Success in PC5 (Logical Reasoning)"
- **PC-to-Explicit Transfer**: "Success in Code Density (explicit feature) → Success in PC1 (Coding semantics)"

## Experimental Design

### Configuration
- **Feature Space**: 45 dimensions
  - 32 PCA components (from 384-dim SBERT embeddings)
  - 8 explicit features (code density, JSON requirements, etc.)
  - 5 cluster distances (anchor clusters: Math, Coding, Creative Writing, etc.)
  - 1 bias term
- **Total Dimension**: 46 (45 + 1 bias)

### Parameters (CSR Defaults)
- `prior_n_effective = 0.0` - **No prior means** (isolates structure contribution)
- `prior_structure_n_effective = 20.0` - **CSR default** structure strength
- `exploration = "safe"` - alpha = 0.1
- `ridge_lambda = 1.0` - Regularization
- `forgetting_factor = 1.0` - No temporal decay

### Scaling Formula
```
gamma_structure = N_target / N_offline
              = 20.0 / 21719
              ≈ 0.000921

A_matrix = (ridge_lambda * I) + (gamma_structure * cov_matrix)
b_vector = 0  (prior_n_effective = 0)
```

### Conditions

#### Condition A: Full Covariance
- Uses complete Σ_CSR with all off-diagonal correlations
- Matrix: 45×45, ~2025 unique elements
- Example correlation: Cov(PC1_Coding, PC5_Logic) ≠ 0

#### Condition B: Diagonal Only  
- Uses only diagonal variances: `diag(Σ_CSR)`
- Matrix: 45×45 diagonal, 45 unique elements
- Destroys all PC-to-PC and PC-to-Explicit correlations

## Test Data
- **Prompts**: 981 test prompts
- **Models**: 36 Pareto-optimal models
- **Trials**: 20 (with different shuffles)

## Metrics
- **Cumulative Regret**: Sum of (Best Reward - Selected Reward) over 981 steps
- **Correlation Benefit**: `(Regret_Diagonal - Regret_Full) / Regret_Diagonal × 100%`

## Interpretation

### If Correlation Benefit > 30%
**Off-diagonal correlations are CRITICAL** - The router leverages cross-feature transfer learning to make superior routing decisions.

### If Correlation Benefit: 10-30%
**Moderate benefit** - Correlations provide useful but not essential information.

### If Correlation Benefit < 10%
**Weak effect** - Diagonal variances dominate; correlations add minimal value.

## Files
- **Script**: [`covariance_ablation_csr.py`](file:///Users/annette/repostitories/llm_jury/banditgpt/experiments/ablation/covariance_ablation_csr.py)
- **Covariance Matrix**: `banditgpt/priors/priors_meta_pca.npz`
- **PCA Model**: `banditgpt/data/pca_32.joblib`
- **Test Data**: `banditgpt/data/test_rewards_pareto_dedup.jsonl`

## Results
*To be filled after experiment completes*

## Notes
- This experiment isolates **structure** from **prior means** by setting `prior_n_effective=0`
- The router still learns online via LinUCB updates
- PCA is applied to embeddings (384→32) before concatenating explicit features
