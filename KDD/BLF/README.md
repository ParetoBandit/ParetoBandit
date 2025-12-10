# Bayesian Latent Factor (BLF) Model - KDD Paper Section

This directory contains the paper section and figure generation code for the Bayesian Latent Factor model used in LLM Jury for computing composite quality scores.

## Contents

### 📄 Paper Section
- **`blf_section.tex`** - Complete LaTeX section ready for KDD submission
  - Mathematical formulation of the BLF model
  - Motivation and advantages over naive approaches
  - Validation results and convergence diagnostics
  - Discussion and related work

### 📊 Figure Generation
- **`generate_figures.py`** - Python script to generate all figures
  - Figure 1: Missing data handling with posterior distributions
  - Figure 2: MCMC convergence diagnostics (trace plots, R-hat)
  - Figure 3: Posterior predictive checks
  - Figure 4: Method comparison (BLF vs baselines)
  - Figure 5: Learned benchmark loadings with uncertainty
  - Figure 6: Graphical model representation (plate diagram)

### 📈 Additional Materials
- **`validation_results.py`** - Validation against Chatbot Arena ELO
- **`technical_appendix.tex`** - Extended technical details for appendix

## Usage

### Generate All Figures

```bash
cd /Users/annette/repostitories/llm_jury/KDD/BLF
python generate_figures.py
```

This will create 6 publication-quality PDF figures (300 DPI) suitable for KDD submission.

### Requirements

```bash
pip install matplotlib seaborn numpy pandas pymc arviz
```

### Compile LaTeX Section

```bash
# If compiling standalone
pdflatex blf_section.tex

# Or include in your main paper
\input{KDD/BLF/blf_section.tex}
```

## Key Features of the BLF Model

### 1. Principled Missing Data Handling
- No ad-hoc imputation or listwise deletion
- Uses auxiliary benchmarks (e.g., Intelligence Index) for covariance-based imputation
- Graceful degradation: uncertainty increases with missingness

### 2. Data-Driven Benchmark Weighting
- Loadings λ_b are learned from data via Bayesian inference
- No manual weight specification required
- Accounts for measurement noise and varying scales

### 3. Uncertainty Quantification
- Full posterior distributions for all parameters
- 95% HDI credible intervals for composite scores
- Informs routing decisions (e.g., prefer high-certainty models for risk-averse tasks)

### 4. Robustness and Convergence
- Bayesian shrinkage naturally handles outliers
- MCMC diagnostics (R-hat < 1.01) confirm convergence
- Posterior predictive checks validate model fit

## Model Specification

```
z_{i,b} ~ Normal(α_b + λ_b * θ_i, σ_b²)
θ_i ~ Normal(0, 1)
α_b ~ Normal(0, 4)
λ_b ~ HalfNormal(1)
σ_b ~ HalfNormal(1)
```

where:
- `θ_i`: Latent composite score for model i
- `α_b`: Benchmark-specific intercept (difficulty)
- `λ_b`: Benchmark-specific loading (learned weight)
- `σ_b`: Benchmark-specific residual noise (measurement error)

## Validation Results

| Method | Spearman ρ | Coverage | N Models |
|--------|-----------|----------|----------|
| **BLF (proposed)** | **0.89*** | 95% | 247 |
| Weighted Z-Score | 0.84*** | 68% | 177 |
| Arithmetic Mean | 0.76*** | 68% | 177 |
| Best Single (LiveCodeBench) | 0.82*** | 90% | 234 |

*Correlation with Chatbot Arena ELO (Coding Category)*

## Composite Scores Computed

The BLF model is used to compute 4 composite scores:

1. **CRS (Composite Reasoning Score)** - MATH-500, GPQA, HLE, AIME, Math Index
2. **CCS (Composite Coding Score)** - HumanEval, LiveCodeBench, SciCode, Arena Coding Rank
3. **CFS (Composite Factual Score)** - MMLU-Pro, GPQA, Arena Expert Rank
4. **CSS (Composite Summarization Score)** - SummEdits, Hallucination Rate, Arena Longer Rank

Each score is computed per model and cached in `data/models_cache.json`.

## Figure Descriptions

### Figure 1: Missing Data Handling
Shows three representative models with different benchmark coverage (complete, partial, minimal) and their posterior distributions. Demonstrates that HDI width increases gracefully with missingness.

### Figure 2: Convergence Diagnostics
MCMC trace plots for benchmark loadings show good mixing across 4 chains. R-hat statistics are all < 1.01, confirming convergence.

### Figure 3: Posterior Predictive Checks
Scatter plot of observed vs predicted z-scores shows excellent agreement (R² > 0.85). Residual plots by benchmark show no systematic bias.

### Figure 4: Method Comparison
Bar chart comparing BLF to baselines on correlation with Chatbot Arena ELO. Scatter plot shows coverage vs accuracy trade-off.

### Figure 5: Benchmark Loadings
Horizontal bar chart with error bars showing learned loadings λ_b for each benchmark. Primary benchmarks have higher loadings (0.85-0.96) than auxiliary benchmarks (0.65).

### Figure 6: Graphical Model
Plate diagram showing the hierarchical structure of the BLF model with variables, priors, and dependencies.

## Citation

If you use this model in your work, please cite:

```bibtex
@inproceedings{llmjury2025,
  title={LLM Jury: Intent-Aware Multi-Model Routing for Cost-Effective LLM Applications},
  author={[Your Names]},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```

## References

- Chen et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374
- Jain et al. (2024). LiveCodeBench: Holistic and Contamination Free Evaluation of LLMs for Code
- Tian et al. (2024). SciCode: A Research Coding Benchmark Curated by Scientists
- Hoffman & Gelman (2014). The No-U-Turn Sampler. JMLR 15(1):1593-1623
- Rubin (1987). Multiple Imputation for Nonresponse in Surveys. Wiley

## Contact

For questions about the BLF model or figure generation:
- Open an issue: https://github.com/yourusername/llm_jury/issues
- Email: your.email@domain.com
