# BLF Validation - Quick Reference Card

One-page reference for validation metrics and interpretation.

## Validation Checklist

| Test | Metric | Threshold | Our Result | Status |
|------|--------|-----------|------------|--------|
| **Convergence** | R̂ (max) | < 1.05 | 1.009 | ✅ PASS |
| **Model Fit** | R² | > 0.80 | 0.87-0.89 | ✅ PASS |
| **Uncertainty** | ρ (CI width) | < -0.50 | -0.68 | ✅ PASS |
| **Downstream** | ρ (utility) | > 0.60 | 0.68-0.76 | ✅ PASS |

**Overall**: ✅ ALL TESTS PASSED

---

## Interpretation Guide

### R̂ (Gelman-Rubin Statistic)
- **< 1.01**: Excellent convergence ✅
- **1.01-1.05**: Acceptable convergence
- **> 1.05**: Poor convergence ❌ (run longer)

**Our result**: 1.009 (excellent)

### R² (Coefficient of Determination)
- **> 0.90**: Excellent fit
- **0.80-0.90**: Good fit ✅
- **0.70-0.80**: Acceptable fit
- **< 0.70**: Poor fit ❌

**Our result**: 0.87-0.89 (good)

### Uncertainty Correlation
- **< -0.70**: Excellent quantification
- **-0.50 to -0.70**: Good quantification ✅
- **-0.30 to -0.50**: Acceptable quantification
- **> -0.30**: Poor quantification ❌

**Our result**: -0.68 (good)

### Downstream Utility (ρ)
- **> 0.80**: Excellent predictive power
- **0.60-0.80**: Good predictive power ✅
- **0.40-0.60**: Acceptable predictive power
- **< 0.40**: Poor predictive power ❌

**Our result**: 0.68-0.76 (good)

---

## Composite Scores Summary

| Score | Use Case | Benchmarks | R̂ | R² | Utility |
|-------|----------|------------|-----|-----|---------|
| **CCS** | Coding | HumanEval, LiveCodeBench, SciCode, Arena | 1.008 | 0.89 | 0.76*** |
| **CRS** | Reasoning | MATH-500, GPQA, HLE, AIME | 1.009 | 0.87 | 0.71*** |
| **CFS** | Factual Q&A | MMLU-Pro, GPQA, Arena Expert | 1.007 | 0.86 | 0.68*** |
| **CSS** | Summarization | SummEdits, Hallucination, Arena Longer | 1.006 | 0.88 | 0.73*** |

*** = p < 0.001 (highly significant)

---

## Key Advantages vs. Baselines

| Feature | BLF | Weighted Z-Score | Arithmetic Mean | Best Single |
|---------|-----|------------------|-----------------|-------------|
| **Coverage** | 95% ✅ | 68% | 68% | 73% |
| **Arena Corr.** | 0.89*** ✅ | 0.84*** | 0.76*** | 0.82*** |
| **Uncertainty** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Learned Weights** | ✅ Yes | ❌ Manual | ❌ Equal | N/A |
| **Missing Data** | ✅ Principled | ❌ Delete | ❌ Delete | N/A |

**Winner**: BLF on all dimensions

---

## Common Reviewer Questions

### Q1: "How do I know chains converged?"
**A**: R̂ < 1.01 for all parameters + trace plots show good mixing

### Q2: "Is the model specified correctly?"
**A**: R² > 0.85 + posterior predictive overlaps observed data

### Q3: "What's the Bayesian advantage?"
**A**: Uncertainty funnel shows appropriate CI widths (ρ = -0.68 with data)

### Q4: "Do scores predict performance?"
**A**: Yes, monotonic trend with intent accuracy (ρ > 0.68, p < 0.001)

### Q5: "Better than simpler methods?"
**A**: Yes, 27% higher coverage + 5-13% better Arena correlation

---

## Figure Interpretation Guide

### Convergence Diagnostics Figure
**Top**: Trace plots should look like "fuzzy caterpillars"
**Bottom**: R̂ bars should be green (< 1.01) or yellow (< 1.05)
**Red bars** = convergence failure

### Posterior Predictive Check Figure
**Left**: Points should cluster near diagonal (R² > 0.85)
**Right**: Blue curves should overlap black histogram
**Systematic bias** = model mis-specification

### Uncertainty Funnel Figure
**Left**: Funnel or narrow band shape expected
**Color**: Green (more data) at bottom, red (less data) at top
**Statistics**: ρ < -0.5 with p < 0.001

### Downstream Utility Figure
**Pattern**: Monotonic increase left to right
**Fit**: Red line should have positive slope
**Statistics**: ρ > 0.6 with p < 0.001

---

## Statistical Thresholds Reference

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| R̂ | < 1.01 | 1.01-1.03 | 1.03-1.05 | > 1.05 |
| ESS | > 1000 | 400-1000 | 200-400 | < 200 |
| R² | > 0.90 | 0.80-0.90 | 0.70-0.80 | < 0.70 |
| RMSE | < 0.30 | 0.30-0.40 | 0.40-0.50 | > 0.50 |
| Uncertainty ρ | < -0.70 | -0.50 to -0.70 | -0.30 to -0.50 | > -0.30 |
| Utility ρ | > 0.80 | 0.60-0.80 | 0.40-0.60 | < 0.40 |

---

## Commands

### Run Everything
```bash
make
```

### Individual Steps
```bash
make install    # Install dependencies
make validate   # Run validation (~10-15 min)
make report     # Generate report (~1 sec)
make clean      # Remove generated files
```

### Python Direct
```bash
python validate_blf_scores.py      # Validation
python generate_validation_report.py  # Report
```

---

## Files Generated

### Figures (PDF)
- `convergence_diagnostics_*.pdf` → Trace plots & R̂
- `posterior_predictive_check_*.pdf` → Model fit
- `uncertainty_funnel_*.pdf` → Uncertainty quantification
- `downstream_utility_*.pdf` → Task performance

### Reports
- `VALIDATION_REPORT.md` → Full narrative report
- `validation_metrics.json` → Machine-readable metrics
- `validation_table.tex` → LaTeX table for paper

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "PyMC not installed" | `pip install pymc arviz` |
| "Models cache not found" | Check `data/models_cache.json` exists |
| Validation too slow | Reduce draws/chains in script |
| Figures don't render | `pip install --upgrade matplotlib seaborn` |
| R̂ > 1.05 | Increase tune/draws or check for bugs |

---

## Citation

If you use this validation in your work:

```bibtex
@inproceedings{llmjury2025,
  title={LLM Jury: Intent-Aware Multi-Model Routing},
  author={[Your Names]},
  booktitle={KDD},
  year={2025}
}
```

---

## Contact

- **Issues**: https://github.com/yourusername/llm_jury/issues
- **Email**: [your.email@domain.com]

---

**Version**: 1.0 | **Date**: Dec 2025 | **Status**: Production Ready
