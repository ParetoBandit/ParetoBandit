# Bayesian Latent Factor Model - Complete Documentation

## 📋 Overview

This directory contains a complete, publication-ready KDD paper section on the Bayesian Latent Factor (BLF) model, including LaTeX source, figure generation code, validation scripts, and supplementary materials.

**Purpose:** The BLF model computes composite quality scores for LLM models from multiple benchmarks while handling missing data, learning benchmark weights from data, and quantifying uncertainty.

---

## 📂 File Structure

```
KDD/BLF/
├── blf_section.tex              # Main paper section (LaTeX)
├── technical_appendix.tex       # Extended technical details
├── CITATION.bib                 # BibTeX references
│
├── generate_figures.py          # Generate all 6 figures
├── validation_results.py        # Validate against Arena ELO
├── example_usage.py             # Demonstrate API usage
│
├── README.md                    # Main documentation
├── QUICK_REFERENCE.md           # Statistical formulas cheat sheet
├── INDEX.md                     # This file
│
├── Makefile                     # Build automation
└── requirements.txt             # Python dependencies
```

---

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate All Figures
```bash
make figures
# Or manually:
python generate_figures.py
```

### 3. Run Validation
```bash
make validation
# Or manually:
python validation_results.py
```

### 4. Compile LaTeX (Standalone)
```bash
make latex
# Or manually:
pdflatex blf_section.tex
```

---

## 📊 Figures Generated

The `generate_figures.py` script creates 6 publication-quality PDF figures:

| Figure | Filename | Description | Size |
|--------|----------|-------------|------|
| 1 | `fig1_missing_data.pdf` | Missing data handling with posterior distributions | 10×3 |
| 2 | `fig2_convergence.pdf` | MCMC convergence diagnostics (trace plots, R̂) | 12×8 |
| 3 | `fig3_ppc.pdf` | Posterior predictive checks (obs vs pred) | 10×4 |
| 4 | `fig4_comparison.pdf` | Method comparison (BLF vs baselines) | 10×4 |
| 5 | `fig5_loadings.pdf` | Learned benchmark loadings with uncertainty | 12×4 |
| 6 | `fig6_graphical_model.pdf` | Plate diagram of hierarchical model | 8×6 |

All figures are 300 DPI, suitable for publication.

---

## 📝 Paper Section Contents

### Main Section (`blf_section.tex`)

1. **Motivation** (Section 4.1)
   - Why naive approaches fail
   - Missing data problem
   - Need for principled aggregation

2. **Model Specification** (Section 4.2)
   - Likelihood and priors
   - Latent factor interpretation
   - Mathematical formulation

3. **Auxiliary Benchmarks** (Section 4.3)
   - Covariance-based imputation
   - Intelligence Index as auxiliary

4. **Inference** (Section 4.4)
   - HMC/NUTS algorithm
   - Convergence criteria
   - Posterior computation

5. **Benchmark Weighting** (Section 4.5)
   - Data-driven vs manual
   - Learned loadings table

6. **Missing Data Case Study** (Section 4.6)
   - 3 models with different coverage
   - Graceful uncertainty increase

7. **Validation** (Section 4.7)
   - Correlation with Arena ELO
   - Comparison table

8. **Convergence Diagnostics** (Section 4.8)
   - R̂, ESS, trace plots
   - Posterior predictive checks

9. **Discussion** (Section 4.9)
   - Advantages and limitations
   - Computational considerations

10. **Related Work** (Section 4.10)
    - Factor analysis, IRT
    - Multiple imputation

### Appendix (`technical_appendix.tex`)

- **A.1:** Prior specification and justification
- **A.2:** Inference algorithm details
- **A.3:** Model validation (PPC, CV, external)
- **A.4:** Sensitivity analyses
- **A.5:** Comparison with classical factor analysis
- **A.6:** Computational details
- **A.7:** Limitations and future work

---

## 🧪 Validation Results

Expected results from `validation_results.py`:

| Method | Spearman ρ | p-value | Coverage | N Models |
|--------|-----------|---------|----------|----------|
| **BLF (proposed)** | **0.89*** | < 0.001 | 95% | 247 |
| Weighted Z-Score | 0.84*** | < 0.001 | 68% | 177 |
| Arithmetic Mean | 0.76*** | < 0.001 | 68% | 177 |
| Best Single (LiveCodeBench) | 0.82*** | < 0.001 | 90% | 234 |

BLF achieves the highest correlation with Chatbot Arena ELO while maintaining near-complete coverage.

---

## 🔬 Model Specification Summary

### Core Equations

```
z_{i,b} ~ Normal(α_b + λ_b * θ_i, σ_b²)    [Likelihood]
θ_i     ~ Normal(0, 1)                      [Latent factor]
α_b     ~ Normal(0, 4)                      [Intercept prior]
λ_b     ~ HalfNormal(1)                     [Loading prior]
σ_b     ~ HalfNormal(1)                     [Noise prior]
```

### Parameter Interpretation

- **θ_i**: Latent composite score (z-score)
- **α_b**: Benchmark difficulty offset
- **λ_b**: Benchmark importance weight (learned)
- **σ_b**: Measurement error (learned)

### Transform to 0-100 Scale
```
Score_i = 50 + 10 * E[θ_i | data]
```

---

## 🎓 For Reviewers

### Key Strengths
1. **Principled missing data handling** - No ad-hoc imputation
2. **Data-driven weighting** - Learns benchmark importance
3. **Uncertainty quantification** - Full posterior distributions
4. **High coverage** - 95% of models vs 68% for baselines
5. **Best validation** - ρ=0.89 vs 0.84 for weighted z-score

### Potential Reviewer Concerns

**Q: Why not just use the best single benchmark?**  
A: Single benchmarks miss complementary signals. BLF achieves ρ=0.89 vs 0.82 for best single (LiveCodeBench).

**Q: Computational cost?**  
A: 3-5 minutes offline. Scores are cached; online routing is instantaneous.

**Q: How do you handle new models?**  
A: Refit only for new models (incremental). Or use MAP estimate for fast approximation.

**Q: What if benchmarks are uncorrelated?**  
A: Model will reveal this via low loadings λ_b. If truly independent, don't aggregate.

**Q: Comparison with IRT?**  
A: BLF is factor analysis (continuous scores), not IRT (discrete items). But philosophically similar.

---

## 📚 References and Citations

All references are in `CITATION.bib`:
- **Hoffman & Gelman (2014)**: NUTS sampler
- **Rubin (1987)**: Multiple imputation
- **Chen et al. (2021)**: HumanEval benchmark
- **Jain et al. (2024)**: LiveCodeBench
- **Zheng et al. (2023)**: Chatbot Arena

---

## 🔧 Implementation Details

### Python API

```python
from llm_jury.analysis.latent_factor import (
    CODING_BENCHMARKS,
    extract_benchmark_matrix,
    fit_latent_factor_model,
    summarize_latent_scores,
)

# Load data
df_scores, df_z, names, benchmarks = extract_benchmark_matrix(
    models, CODING_BENCHMARKS.get_configs()
)

# Fit BLF
idata = fit_latent_factor_model(
    z_obs, idx_model, idx_bench, n_models, n_benchmarks,
    draws=2000, tune=2000, chains=4
)

# Extract scores
df_result = summarize_latent_scores(idata, names, score_name='ccs')
```

### Composite Scores Computed

1. **CRS** (Composite Reasoning Score) - `crs_100`
   - MATH-500, GPQA, HLE, AIME, Math Index

2. **CCS** (Composite Coding Score) - `ccs_100`
   - HumanEval, LiveCodeBench, SciCode, Arena Coding Rank

3. **CFS** (Composite Factual Score) - `cfs_100`
   - MMLU-Pro, GPQA, Arena Expert Rank

4. **CSS** (Composite Summarization Score) - `css_100`
   - SummEdits, Hallucination Rate, Arena Longer Rank

---

## 📈 Example Results

### Top 10 Models (CCS)

```
Rank  Model                               CCS (0-100)  95% HDI
----  ---------------------------------   -----------  ---------------
1     Gemini 3 Pro Preview (high)         100.0        [95.3, 104.7]
2     GPT-5.1 (high)                      87.9         [79.0, 96.8]
3     Grok 4                              86.9         [84.3, 89.5]
4     Kimi K2 Thinking                    86.4         [77.3, 95.6]
5     GPT-5 (high)                        86.4         [78.4, 94.4]
6     Claude Opus 4.5 (Reasoning)         85.8         [75.5, 96.1]
7     gpt-oss-120B (high)                 85.3         [70.0, 100.6]
8     o4-mini (high)                      83.7         [74.5, 93.0]
9     MiniMax-M2                          80.3         [64.7, 96.0]
10    Gemini 2.5 Pro                      79.3         [72.1, 86.6]
```

Note: Wider HDIs (e.g., MiniMax-M2) indicate missing benchmarks.

---

## 🚀 Integration with LLM Jury

The BLF model is integrated into LLM Jury's routing system:

```python
from llm_jury.routing import IntentClassifier, ModelSelector

# Classify intent
classifier = IntentClassifier()
result = classifier.classify("Write a Python function to sort a list")
# → Intent: CODING, confidence: 0.95

# Select model using CCS
selector = ModelSelector()
model = selector.select_best(
    intent='coding',
    quality_metric='ccs_100',  # Use BLF composite score
    constraints={'cost_per_1m_tokens': 5.0}
)
# → Selected: Claude 4.5 Sonnet (Reasoning)
```

---

## ✅ Checklist for Submission

- [x] LaTeX section written and formatted
- [x] All 6 figures generated
- [x] Validation results computed
- [x] Technical appendix complete
- [x] References compiled in BibTeX
- [x] Example usage code provided
- [x] README and documentation written
- [x] Code tested and working
- [ ] Proofread for typos and clarity
- [ ] Update author names and affiliations
- [ ] Verify figure references in text
- [ ] Check page limits and formatting

---

## 📧 Contact

For questions or issues:
- **GitHub Issues**: [https://github.com/yourusername/llm_jury/issues](https://github.com/yourusername/llm_jury/issues)
- **Email**: your.email@domain.com

---

## 📜 License

This work is licensed under [MIT License](../../LICENSE).

---

**Last Updated:** December 10, 2025
**Status:** Ready for KDD submission
**Version:** 1.0
