# banditGPT: Paper

This directory contains the LaTeX source for the paper submission.

## Paper Title
**banditGPT: Cost-Aware Online Learning for LLM Routing via Expert Corralling**

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

- **Peak quality**: 0.914 ± 0.006 (vs. GPT-4-Turbo: 0.812, RouteLLM: 0.883)
- **Oracle gap closure**: 70.0% (vs. 46.2% for RouteLLM)
- **Crossover**: Surpasses RouteLLM after ~200 label-free prompts (at moderate-to-high budgets)
- **Two-regime finding**: RouteLLM competitive at low budgets; banditGPT dominates at higher budgets
- **Evaluation**: 750 held-out prompts, 20 independent trials per configuration

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
3. **Corralling Meta-Learning**: Complementary alpha schedules achieve 30-40% lower regret; catastrophic failure detection
4. **Honest Null Result**: Semantic transfer for cold-start provides no significant improvement (reported transparently)

## Contact

For questions about the paper, please contact the authors.

