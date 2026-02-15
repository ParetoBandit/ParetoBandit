# LaTeX Sections for Paper Integration

**Last Updated:** February 14, 2026

---

## Available LaTeX Files

| File | Purpose | Priority |
|------|---------|----------|
| `latex_figure3_combined_caption.tex` | Main 2-panel figure (crossover + weights) | HIGH |
| `latex_section_results_meta_learning_cost.tex` | "When Does Adaptive Safety Pay Off?" | HIGH |
| `latex_table_strategy_guide.tex` | Strategy selection decision table | HIGH |
| `latex_section_5.3_practical_recommendations.tex` | Deployment guidelines | HIGH |
| `figure_3_caption.tex` | Architecture diagram caption | MEDIUM |

---

## Integration

```latex
% Main figure
\input{experiments_v1/03_figure/latex_figure3_combined_caption}

% Results narrative
\input{experiments_v1/03_figure/latex_section_results_meta_learning_cost}

% Strategy table
\input{experiments_v1/03_figure/latex_table_strategy_guide}

% Practical recommendations
\input{experiments_v1/03_figure/latex_section_5.3_practical_recommendations}
```

---

## Required Packages

```latex
\usepackage{booktabs}     % Professional tables
\usepackage{multirow}     % Strategy table
\usepackage{enumitem}     % Compact lists
```
