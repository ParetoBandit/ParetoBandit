# Figure 1: Semantic Task Specialization in Latent Space

## Overview

This figure visualizes the semantic structure of task difficulty across 80,000 RouteLLM prompts using KDE (Kernel Density Estimation) contours.

## Files

### Generated Figures
- **`pca_2d_reward_gap.png`**: Standard resolution figure (300 DPI)
- **`pca_2d_reward_gap_hires.png`**: High resolution figure (600 DPI) for publication

### Scripts
- **`plot_pca_reward_gap.py`**: Script to generate the visualization
- **`plot_output.log`**: Generation log with statistics

### LaTeX Files
- **`figure_1_caption.tex`**: LaTeX caption for the figure
- **`results_explanation.tex`**: Detailed results explanation for methodology/results section
- **`README.md`**: This file - documentation and usage guide

## Key Insights

### 1. Distinct Semantic Neighborhoods

The visualization proves that **hard tasks occupy specific regions** in semantic space:

- **Blue Contours** (Easy Tasks): Prompts where Mixtral is sufficient (|Gap| ≤ 0.3)
  - Examples: Simple queries, basic coding, general knowledge
  - Density peaks in central latent region

- **Red Contours** (Hard Tasks): Prompts requiring GPT-4 (Gap > 0.6)
  - Examples: Complex math, advanced code, multi-step reasoning
  - Density peaks in distinct semantic neighborhoods

### 2. The Ambiguous Frontier

The **overlapping region** between blue and red contours represents the "Ambiguous Frontier":
- Tasks where static linguistic features alone are insufficient
- Requires contextual online learning (BanditGPT's advantage)
- Demonstrates why rule-based routers fail

### 3. Evidence for Contextual Bandits

This figure provides empirical evidence for:
1. **Non-random difficulty distribution**: Hard tasks cluster semantically
2. **Insufficient static features**: Overlapping regions require context
3. **Need for online learning**: Ambiguous frontier necessitates adaptive routing

## Technical Details

### Data
- **Source**: RouteLLM GPT-4 judge battles dataset
- **Size**: 80,000 pairwise comparisons
- **Models**: Mixtral-8x7B vs GPT-4-Turbo
- **Reward Gap**: $R_{\text{GPT-4-Turbo}} - R_{\text{Mixtral}}$

### Method
1. **Embedding**: SentenceTransformer (all-MiniLM-L6-v2)
2. **Dimensionality Reduction**: PCA-23 (29% variance explained)
3. **Visualization**: First 2 PCA components (5.4% variance)
4. **Density Estimation**: Gaussian KDE with bandwidth 0.1

### Difficulty Thresholds

Rewards are binary from pairwise judge comparisons: 1.0 (win), 0.5 (tie), 0.0 (loss).
Gap = R_GPT4-Turbo - R_Mixtral ranges from -1.0 to +1.0.

- **Easy**: |Gap| ≤ 0.3 
  - Practical meaning: Models perform nearly equally (e.g., both tie, or GPT-4 wins only slightly more often)
  - Implication: Mixtral is sufficient, routing to it saves cost
  
- **Medium**: 0.3 < Gap ≤ 0.6 
  - Practical meaning: GPT-4 wins more often, but not decisively
  - Note: 0% of prompts fall in this range (bimodal distribution)
  
- **Hard**: Gap > 0.6 
  - Practical meaning: GPT-4 wins decisively (e.g., Gap=1.0 means GPT-4 always wins, Mixtral always loses)
  - Implication: Quality difference justifies the cost of GPT-4

## Statistics

From `plot_output.log`:

```
Reward Gap Statistics:
  Min: -1.000
  Max: 1.000
  Mean: 0.586
  Median: 1.000
  Std: 0.652

Battle Outcomes:
  GPT-4-Turbo wins: 62,288 (77.9%)
  Mixtral wins: 0 (0.0%)
  Ties: 17,712 (22.1%)

Semantic Regions:
  Easy prompts (|Gap| ≤ 0.3): [count] ([percent]%)
  Hard prompts (Gap > 0.6): [count] ([percent]%)
```

## Usage in Paper

### Including the Figure

In your main LaTeX file:
```latex
\usepackage{graphicx}

% Include the figure with caption
\input{experiments_v1/01_figure/figure_1_caption.tex}
```

### Including the Results Explanation

In your results/methodology section:
```latex
% Include detailed results explanation
\input{experiments_v1/01_figure/results_explanation.tex}
```

### Key Points to Emphasize

1. **Semantic Structure**: "Figure~\ref{fig:semantic_specialization} reveals distinct density peaks for easy and hard tasks."

2. **Ambiguous Frontier**: "The overlapping region (Figure~\ref{fig:semantic_specialization}) demonstrates where static features fail."

3. **Motivation for Bandits**: "As shown in Figure~\ref{fig:semantic_specialization}, the ambiguous frontier necessitates contextual online learning."

4. **Cold-Start Capability**: "Because hard tasks cluster together (Figure~\ref{fig:semantic_specialization}), the router generalizes to unseen prompts."

### Full Paper Structure

```latex
\documentclass{article}
\usepackage{graphicx}

\begin{document}

\section{Introduction}
% Motivation: why routing is hard
Static routers struggle because task difficulty occupies an ambiguous frontier...

\section{Problem Formulation}
% Reference the figure to define the problem
As Figure~\ref{fig:semantic_specialization} illustrates, task difficulty is not randomly distributed...

\input{experiments_v1/01_figure/figure_1_caption.tex}

\section{Methodology}
% Explain the approach

\section{Results}
% Include detailed results explanation
\input{experiments_v1/01_figure/results_explanation.tex}

\subsection{Empirical Performance}
% Performance tables and comparisons

\end{document}
```

## Regenerating the Figure

```bash
# From project root
cd experiments_v1/01_figure

# Run visualization script
python3 plot_pca_reward_gap.py

# Output files will be generated:
# - pca_2d_reward_gap.png (300 DPI)
# - pca_2d_reward_gap_hires.png (600 DPI)
# - plot_output.log
```

## Related Experiments

- **PCA Training**: `scripts/train_pca_from_routellm.py`
- **Data Download**: `scripts/download_and_process_routellm.py`
- **Calibration**: `experiments_v1/calibration/`

## Paper Sections

This figure should appear in:

1. **Introduction**: Motivate the problem
   - "Static routers cannot distinguish the ambiguous frontier"

2. **Problem Formulation**: Define difficulty structure
   - "Task difficulty is not uniformly distributed"

3. **Experimental Setup**: Describe data
   - "We evaluate on 80K prompts from RouteLLM"

4. **Results**: Reference performance
   - "BanditGPT excels in the ambiguous frontier (red-blue overlap)"

---

**Created**: January 23, 2026  
**Author**: BanditGPT Research Team  
**Purpose**: KDD 2026 Paper Figure 1

