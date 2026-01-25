# BanditGPT LaTeX Documentation

This directory contains three KDD-compliant LaTeX files that document the BanditGPT router architecture and experimental results.

## Files Overview

### 1. `prior_initialization.tex`
**Focus:** Three-Layer Warm-Start Architecture

**Key Sections:**
- Layer 1: Static Warm-Start (80k battle priors)
- Layer 2: Semantic Transfer (DNA matching and θ-only transfer)
- Layer 3: T-Shirt Sizing (business logic injection)
- Mathematical foundations and empirical validation
- Figure 4 results (92% cost reduction)

**Compile:** `pdflatex prior_initialization.tex`

---

### 2. `system_architecture.tex`
**Focus:** Prior Initialization and Latent Semantic Transfer

**Key Sections:**
- The "Confident Transfer Trap" problem
- θ-Transfer via Identity Reset (avoiding negative transfer)
- Semantic DNA Matching algorithm
- Business Logic Injection with confidence scaling
- Complete three-layer initialization protocol
- Empirical validation on N=1,871 prompts
- Ablation study showing impact of each layer

**Compile:** `pdflatex system_architecture.tex`

---

### 3. `conclusion_section.tex`
**Focus:** The Case for Principled Uncertainty

**Key Sections:**
- Redefining production stability (jagged weight evolution is a feature)
- Closing the "Pareto Gap" (87% Pareto efficiency)
- Architectural innovations summary
- Future directions (multi-objective optimization, hierarchical clustering)
- Broader impact (environmental sustainability, democratization)
- Limitations and open challenges

**Compile:** `pdflatex conclusion_section.tex`

---

## Compilation Instructions

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex

# Or use Overleaf (web-based, no installation)
```

### Compile Individual Files
```bash
cd src/bandit_gpt/

# Compile each file
pdflatex prior_initialization.tex
pdflatex system_architecture.tex
pdflatex conclusion_section.tex

# Clean auxiliary files
rm *.aux *.log *.out
```

### Create Combined Document
To combine all three sections into a single paper, create a master file:

```latex
% master.tex
\documentclass[sigconf]{acmart}
\begin{document}

\title{BanditGPT: Cost-Aware LLM Routing \\
via Three-Layer Warm-Start and Principled Uncertainty}

\author{[Your Name]}
\affiliation{\institution{[Your Institution]}}

\begin{abstract}
[Combined abstract from all three files]
\end{abstract}

\maketitle

% Include sections from each file
\input{prior_initialization_content}
\input{system_architecture_content}
\input{conclusion_section_content}

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}

\end{document}
```

---

## Key Results Summary

### Performance Metrics
- **Cost Reduction:** 92% at production quality (0.90 reward)
- **Pareto Efficiency:** 87% (only 13% gap from oracle)
- **Convergence Speed:** 4.25× faster than cold-start (200 vs 850 samples)
- **Domain Shift Recovery:** Robust to PSI = 0.42

### Architectural Contributions
1. **Three-Layer Warm-Start:** Eliminates cold-start penalty
2. **θ-Only Transfer:** Avoids "confident transfer trap"
3. **Confidence-Scaled Bias Injection:** Ensures business logic matters
4. **Dynamic α-Decay:** Optimal exploration-exploitation balance

### Empirical Validation
- **Dataset:** N=1,871 real prompts (1,121 train, 750 test)
- **Models:** GPT-4-turbo + Mixtral-8x7B
- **Baselines:** Static routing, RouteLLM-MF, cold-start LinUCB
- **Cluster Detection:** 94.2% Easy Cluster → Mixtral, 5.8% Hard Cluster → GPT-4

---

## Figure References

### Figure 3: Jagged Weight Evolution
Demonstrates "principled uncertainty" through bounded exploration and self-healing properties.

### Figure 4: Steady-State Pareto Frontier
Shows 92% cost reduction and 87% Pareto efficiency across all budget tiers.

---

## Citations

These documents are formatted for **ACM SIGKDD Conference** submission:
- Two-column format
- ACM Reference Format bibliography
- Algorithm pseudocode (algorithmic package)
- Tables with booktabs styling

### Key References
1. Li et al. (2010) - LinUCB contextual bandits
2. Ong et al. (2024) - RouteLLM baseline
3. Agrawal & Goyal (2013) - Thompson Sampling
4. Reimers & Gurevych (2019) - Sentence-BERT for DNA matching

---

## Contact & Contribution

For questions or contributions to the LaTeX documentation:
1. Check implementation in `src/bandit_gpt/router.py`
2. Refer to experiments in `experiments_v1/04_figure/`
3. See markdown docs in `experiments_v1/04_figure/THREE_LAYER_WARMSTART.md`

---

## License

These LaTeX files document the BanditGPT system architecture and are intended for academic publication and technical documentation purposes.

