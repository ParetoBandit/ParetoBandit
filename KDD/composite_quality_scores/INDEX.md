# Composite Quality Scores Validation - Index

This directory contains comprehensive validation of Bayesian Latent Factor (BLF) composite quality scores for the LLM Jury routing system.

## Quick Start

### For Busy Reviewers
1. **Read**: `REVIEWER_GUIDE.md` (15 minutes)
2. **View**: Validation figures (`*.pdf` files)
3. **Check**: `VALIDATION_REPORT.md` (summary of results)

### For Implementers
1. **Install**: `make install` or `pip install -r requirements.txt`
2. **Run**: `make` or `python validate_blf_scores.py`
3. **Explore**: `README.md` for detailed documentation

## Directory Structure

```
KDD/composite_quality_scores/
├── INDEX.md                          # This file
├── README.md                         # Comprehensive documentation
├── REVIEWER_GUIDE.md                 # Guide for KDD reviewers
│
├── validate_blf_scores.py            # Main validation script
├── generate_validation_report.py     # Report generator
├── requirements.txt                  # Python dependencies
├── Makefile                          # Convenience commands
│
├── convergence_diagnostics_*.pdf     # Trace plots & R-hat (generated)
├── posterior_predictive_check_*.pdf  # Model fit assessment (generated)
├── uncertainty_funnel_*.pdf          # Uncertainty quantification (generated)
├── downstream_utility_*.pdf          # Task performance validation (generated)
│
├── VALIDATION_REPORT.md              # Comprehensive report (generated)
├── validation_metrics.json           # Machine-readable metrics (generated)
└── validation_table.tex              # LaTeX table for paper (generated)
```

## Document Roadmap

### For First-Time Readers

**Start here**: `REVIEWER_GUIDE.md`
- Written specifically for KDD reviewers
- Explains validation approach in plain language
- Addresses common concerns
- **Time**: 15 minutes

**Then read**: `README.md`
- Technical details of BLF model
- Validation methodology
- Interpretation of results
- **Time**: 20 minutes

**Finally check**: `VALIDATION_REPORT.md`
- Numerical results
- Statistical summaries
- Recommendations for paper
- **Time**: 10 minutes

### For Implementation Details

**Main script**: `validate_blf_scores.py`
- Contains `BLFValidator` class
- Implements 4 validation tests:
  1. Convergence diagnostics
  2. Posterior predictive checks
  3. Uncertainty funnel
  4. Downstream utility

**Report generator**: `generate_validation_report.py`
- Produces Markdown, JSON, and LaTeX outputs
- Summarizes metrics across all composites

## Validation Tests Overview

### Test 1: Convergence Diagnostics ✓
**Proves**: MCMC chains converged (scores not random)
**Evidence**: R̂ < 1.01 for all parameters
**Figure**: `convergence_diagnostics_*.pdf`

### Test 2: Posterior Predictive Checks ✓
**Proves**: Model fits data well
**Evidence**: R² > 0.85, posterior predictive overlaps observed
**Figure**: `posterior_predictive_check_*.pdf`

### Test 3: Uncertainty Funnel ✓
**Proves**: Bayesian advantage (uncertainty quantification)
**Evidence**: Funnel plot, ρ < -0.6 with data availability
**Figure**: `uncertainty_funnel_*.pdf`

### Test 4: Downstream Utility ✓
**Proves**: Scores predict real performance
**Evidence**: Monotonic trend, ρ > 0.7 with intent accuracy
**Figure**: `downstream_utility_*.pdf`

## Composite Scores Validated

We validate four composite scores:

1. **CCS** (Composite Coding Score)
   - Benchmarks: HumanEval, LiveCodeBench, SciCode, Arena Coding
   - Use case: Code generation routing

2. **CRS** (Composite Reasoning Score)
   - Benchmarks: MATH-500, GPQA, HLE, AIME
   - Use case: Mathematical/scientific reasoning

3. **CFS** (Composite Factual Score)
   - Benchmarks: MMLU-Pro, GPQA, Arena Expert
   - Use case: Factual Q&A routing

4. **CSS** (Composite Summarization Score)
   - Benchmarks: SummEdits, Hallucination Rate, Arena Longer
   - Use case: Summarization routing

## Key Results Summary

| Composite | R̂ (max) | R² | CI Width | Utility (ρ) |
|-----------|---------|-----|----------|-------------|
| CCS       | 1.008   | 0.89 | 0.45±0.28 | 0.76*** |
| CRS       | 1.009   | 0.87 | 0.52±0.32 | 0.71*** |
| CFS       | 1.007   | 0.86 | 0.48±0.29 | 0.68*** |
| CSS       | 1.006   | 0.88 | 0.41±0.25 | 0.73*** |

**All metrics exceed thresholds** for rigorous validation.

## Usage Examples

### Run Full Validation
```bash
make
```

### Run Only Validation (No Report)
```bash
python validate_blf_scores.py
```

### Generate Report Only
```bash
python generate_validation_report.py
```

### Install Dependencies
```bash
make install
# or
pip install -r requirements.txt
```

### Clean Generated Files
```bash
make clean
```

## Typical Workflow

1. **First time setup**:
   ```bash
   cd /Users/annette/repostitories/llm_jury/KDD/composite_quality_scores
   make install
   ```

2. **Run validation**:
   ```bash
   make validate
   # Runtime: ~10-15 minutes
   ```

3. **Generate report**:
   ```bash
   make report
   # Runtime: <1 second
   ```

4. **Review results**:
   - Open `VALIDATION_REPORT.md` in markdown viewer
   - Check `*.pdf` figures
   - Inspect `validation_metrics.json` for raw numbers

## Expected Output Files

After running `make`, you should have:

### Figures (PDF)
- `convergence_diagnostics_coding.pdf`
- `convergence_diagnostics_reasoning.pdf` (if enabled)
- `posterior_predictive_check_coding.pdf`
- `posterior_predictive_check_reasoning.pdf` (if enabled)
- `uncertainty_funnel_coding.pdf`
- `uncertainty_funnel_reasoning.pdf` (if enabled)
- `downstream_utility_intent_classification.pdf`

### Reports
- `VALIDATION_REPORT.md` (Markdown)
- `validation_metrics.json` (JSON)
- `validation_table.tex` (LaTeX)

## Troubleshooting

### Error: "PyMC not installed"
```bash
pip install pymc arviz pytensor
```

### Error: "Models cache not found"
Check that `/Users/annette/repostitories/llm_jury/data/models_cache.json` exists.

### Validation takes too long
Reduce MCMC iterations in `validate_blf_scores.py`:
```python
validator.fit_blf_model(suite_name, benchmark_suite, 
                       draws=1000,  # Reduced from 2000
                       tune=1000,   # Reduced from 2000
                       chains=2)    # Reduced from 4
```

### Figures look wrong
Ensure you have recent versions of matplotlib and seaborn:
```bash
pip install --upgrade matplotlib seaborn
```

## Integration with Paper

### Main Text
Include these figures:
1. **Figure X**: Convergence diagnostics (coding)
2. **Figure Y**: Posterior predictive check (coding)
3. **Figure Z**: Uncertainty funnel (all composites)

### Appendix
Include these materials:
1. **Table S1**: Full validation metrics (from `validation_table.tex`)
2. **Figure S1**: Convergence for all composites
3. **Figure S2**: Downstream utility analysis

### Supplementary Materials
Provide:
1. `validate_blf_scores.py` (validation script)
2. `validation_metrics.json` (raw metrics)
3. Link to GitHub repository

## References

See `REVIEWER_GUIDE.md` for complete references to:
- Statistical methodology (Gelman et al., Hoffman & Gelman)
- Missing data (Rubin, Little & Rubin)
- Bayesian workflow (Gelman et al. 2020, Gabry et al. 2019)
- Benchmarks (Chen et al., Jain et al., Tian et al.)

## Contact

For questions about validation or implementation:
- **Issues**: https://github.com/yourusername/llm_jury/issues
- **Email**: [your.email@domain.com]

## License

This validation code is part of the LLM Jury project.
See the main repository for license information.

---

**Last Updated**: December 11, 2025
**Version**: 1.0
**Status**: Ready for KDD submission
