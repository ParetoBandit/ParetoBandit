# banditGPT: Paper

This directory contains the LaTeX source for the paper submission.

## Paper Title
**banditGPT: Lifelong Learning for LLM Routing via Latent Semantic Transfer and Expert Corralling**

## Structure

```
paper/
├── main.tex                 # Main paper file (includes title and abstract)
├── references.bib           # Bibliography
├── sections/                # Individual sections
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   └── conclusion.tex
└── figures/                 # Figures for the paper
```

## Compilation

To compile the paper:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk for automatic compilation:

```bash
latexmk -pdf main.tex
```

## Key Results

- **State-of-the-art reward**: 0.91 (vs. GPT-4-Turbo: 0.81, RouteLLM: 0.87)
- **Oracle gap closure**: 66.2%
- **Cost reduction**: 27%
- **Hyperparameter robustness**: Perfect stability across 20× range of n_eff

## Formatting Compliance

This paper follows the ACM conference formatting requirements:
- Document class: `acmart` with `sigconf` option
- Anonymous submission format
- CCS concepts included
- Keywords specified
- ACM reference format for bibliography

## Abstract Summary

The paper introduces banditGPT, an open-source cost-aware contextual bandit framework for LLM routing:

1. **Model Preference Heterogeneity**: Expensive models aren't always better; prompt structure predicts preference
2. **Cost-Aware Online Learning**: Warmup priors + online adaptation surpasses RouteLLM in 400 prompts
3. **Corralling Meta-Learning**: Safety insurance against prior mismatch with catastrophic failure detection
4. **Honest Null Result**: Semantic transfer for cold-start provides no significant improvement (reported transparently)

## Contact

For questions about the paper, please contact the authors.

