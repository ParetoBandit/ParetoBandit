# BLF Composite Scores - Computation Log

**Date**: December 11, 2025
**Status**: ✅ All composite scores computed with BLF

## Summary

All four composite quality scores have been recomputed using the **Bayesian Latent Factor (BLF)** model and saved to `models_cache.json`:

- ✅ **CCS** (Composite Coding Score) - 83 models
- ✅ **CRS** (Composite Reasoning Score) - 82 models  
- ✅ **CFS** (Composite Factual Score) - 83 models
- ✅ **CSS** (Composite Summarization Score) - 83 models

**Method**: `bayesian` (confirmed in models_cache.json)

---

## Convergence Status by Composite Score

### CCS (Composite Coding Score) ⚠️
**Status**: Computed, moderate convergence issues

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Divergences | 145 | 0 | ⚠️ High |
| Max R-hat | 1.060 | < 1.05 | ⚠️ Borderline |
| Benchmarks | 5 (humaneval, livecode, scicode, arena, intelligence) | - | ✓ |
| Models | 83 | - | ✓ |

**Benchmark Loadings**:
- livecodebench: 0.955 ± 0.077 (highest, most informative)
- intelligence_index: 0.985 ± 0.073 (auxiliary)
- scicode: 0.895 ± 0.081
- arena_rank_coding: 0.905 ± 0.119
- humaneval_score: 0.690 ± 0.109 (lowest)

**Issue**: High divergences suggest model geometry is challenging. May need reparameterization.

---

### CRS (Composite Reasoning Score) ✅
**Status**: Computed, excellent convergence

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Divergences | 0 | 0 | ✅ Excellent |
| Max R-hat | 1.020 | < 1.05 | ✅ Good |
| Benchmarks | 5 (math_500, gpqa, hle, aime, math_index) | - | ✓ |
| Models | 82 | - | ✓ |

**Benchmark Loadings**:
- math_index: 1.021 ± 0.087 (highest, most informative)
- aime: 1.001 ± 0.085 (competition math)
- gpqa: 0.909 ± 0.081
- math_500: 0.769 ± 0.091
- hle: 0.740 ± 0.092

**Note**: Excellent convergence, no issues.

---

### CFS (Composite Factual Score) ❌
**Status**: Computed, significant convergence issues

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Divergences | 755 | 0 | ❌ Very high |
| Max R-hat | 1.100 | < 1.05 | ❌ Poor |
| Benchmarks | 3 (mmlu_pro, gpqa, arena_expert) | - | ✓ |
| Models | 83 | - | ✓ |

**Benchmark Loadings**:
- arena_rank_expert: 1.184 ± 0.156 (highest loading, highest uncertainty)
- gpqa: 0.947 ± 0.078
- mmlu_pro: 0.911 ± 0.079

**Issue**: Very high divergences (755) and R-hat for gpqa noise (1.100). This composite needs attention.

**Possible causes**:
- Only 3 benchmarks (minimal for factor analysis)
- Arena expert rank may have non-linear relationship
- High missingness in arena_rank_expert (only 50/83 models)

---

### CSS (Composite Summarization Score) ⚠️
**Status**: Computed, weak signal / moderate convergence issues

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Divergences | 99 | 0 | ⚠️ Moderate |
| Max R-hat | 1.010 | < 1.05 | ✅ Good |
| Benchmarks | 3 (summedits, hallucination_rate, arena_longer) | - | ✓ |
| Models | 83 | - | ✓ |

**Benchmark Loadings**:
- summedits_score: 0.286 ± 0.224 (weak, high uncertainty)
- arena_rank_longer: 0.251 ± 0.222 (weak, high uncertainty)
- hallucination_rate: 0.219 ± 0.193 (weak, high uncertainty)

**Benchmark Noise** (very high):
- summedits_score: 0.950 ± 0.122
- hallucination_rate: 0.975 ± 0.106
- arena_rank_longer: 0.976 ± 0.141

**Issue**: All loadings are very weak (< 0.3) with high uncertainty. High noise (> 0.95) suggests these benchmarks don't share a strong common factor.

**Note**: Wide credible intervals (e.g., [-1.15, 2.21]) reflect high uncertainty. This is scientifically honest but indicates weak latent factor.

---

## Recommendations

### Priority 1: Fix CFS (Factual) ❗❗
**Problem**: 755 divergences, R-hat = 1.100 for noise parameter

**Solutions**:
1. **Increase target_accept to 0.999**
   ```bash
   python scripts/quality_scoring/compute_factual_qa_score.py --bayesian --target_accept 0.999 --tune 5000
   ```

2. **Consider removing arena_expert** (sparse, non-linear)
   - Use only mmlu_pro + gpqa (both have good coverage)

3. **Add more benchmarks** if available:
   - SimpleQA, TriviaQA, Natural Questions

### Priority 2: Investigate CSS (Summarization) ⚠️
**Problem**: Very weak loadings (< 0.3), high noise (> 0.95)

**This suggests**: These three benchmarks may not measure a single "summarization quality" factor.

**Options**:
1. **Accept weak factor**: Report honestly with wide CIs
2. **Switch to weighted z-score**: If BLF isn't adding value (no strong latent factor), simpler methods may be more appropriate
3. **Find better benchmarks**: Current ones may not be measuring the same construct

### Priority 3: Improve CCS (Coding) ✓
**Problem**: 145 divergences, acceptable but not ideal

**Solutions**:
1. **Increase target_accept**: Try 0.995 or 0.999
2. **More tuning**: Increase --tune to 5000
3. **Reparameterize**: Use non-centered parameterization (advanced)

---

## Validation Status

With BLF scores now computed, you can run full validation:

```bash
cd KDD/composite_quality_scores
python validate_blf_scores.py
```

**Expected results**:
- ✅ CRS: Should validate well (excellent convergence)
- ⚠️ CCS: Should validate acceptably (moderate issues)
- ❌ CFS: Will show convergence warnings in validation
- ⚠️ CSS: Will show very wide uncertainty intervals

---

## Files Generated

### Detailed Results (CSV)
- `data/ccs_scores_detailed.csv` - Full posterior summaries for CCS
- `data/crs_scores_detailed.csv` - Full posterior summaries for CRS
- `data/cfs_scores_detailed.csv` - Full posterior summaries for CFS
- `data/css_scores_detailed.csv` - Full posterior summaries for CSS

### Updated Cache
- `data/models_cache.json` - All models now have `*_method: 'bayesian'`

---

## Next Steps

1. **Re-run validation** with real BLF scores (not simulated)
2. **Fix CFS convergence** (high priority for paper)
3. **Document CSS weak factor** (be transparent in paper)
4. **Consider hybrid approach**: Use BLF for CRS/CCS (strong factors), weighted z-score for CSS (weak factor)

---

## MCMC Settings Used

```python
draws = 2000
tune = 3000
chains = 4
target_accept = 0.99  # Default
```

**Recommendation for CFS**: Increase to `target_accept=0.999, tune=5000`

---

## Interpretation for Paper

### What to Report

**Strengths**:
- CRS shows excellent convergence (R̂ < 1.02, 0 divergences)
- CCS shows acceptable convergence (R̂ < 1.06, moderate divergences)
- Learned loadings are interpretable (e.g., LiveCodeBench most informative for coding)

**Weaknesses (be honest)**:
- CFS has convergence issues that need addressing
- CSS shows weak common factor (wide CIs, low loadings)
- Some composites may benefit from more benchmarks

**Recommendation**: 
- For KDD paper, focus validation on **CRS** (best convergence)
- Show **CCS** as acceptable
- Discuss **CSS** as exploratory (weak factor discovered)
- Fix **CFS** before submission or report as limitation

---

## Contact

For questions about BLF computation or convergence issues:
- Check `llm_jury/analysis/latent_factor.py` for model implementation
- See PyMC docs on divergences: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/Diagnosing_biased_Inference_with_Divergences.html
