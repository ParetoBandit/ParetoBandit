# KDD 2026 Paper Compilation Guide

## Quick Start

### Option 1: Using Make (Recommended)
```bash
cd paper
make          # Compile the paper
make view     # Open the PDF
make clean    # Remove auxiliary files
```

### Option 2: Manual Compilation
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 3: Using latexmk (Auto-compilation)
```bash
cd paper
latexmk -pdf main.tex           # Compile once
latexmk -pdf -pvc main.tex      # Watch mode (recompiles on changes)
```

## Requirements

### LaTeX Distribution
- **macOS**: MacTeX (https://www.tug.org/mactex/)
- **Linux**: TeX Live (`sudo apt-get install texlive-full`)
- **Windows**: MiKTeX (https://miktex.org/)

### Required Packages
The following packages are used in the paper:
- `acmart` - ACM article template (KDD format)
- `amsmath`, `amssymb`, `amsfonts` - Math symbols
- `algorithm`, `algorithmic` - Algorithm formatting
- `graphicx` - Figure inclusion
- `booktabs` - Professional tables
- `hyperref` - Hyperlinks and cross-references

Most LaTeX distributions include these by default.

## File Structure

```
paper/
├── main.tex                 # Main document (EDIT THIS)
├── references.bib           # Bibliography entries
├── sections/                # Individual sections (to be created)
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   └── conclusion.tex
├── figures/                 # Place figures here
├── Makefile                 # Build automation
├── .gitignore              # Git ignore rules
└── README.md               # Project overview
```

## Current Status

### ✅ Completed
- [x] Title and abstract formatted
- [x] KDD-compliant document structure
- [x] Bibliography file created
- [x] Build system (Makefile)
- [x] Custom LaTeX commands defined

### 🚧 To Do
- [ ] Write Introduction section
- [ ] Write Related Work section
- [ ] Write Methodology section
- [ ] Write Experiments section
- [ ] Write Results section
- [ ] Write Conclusion section
- [ ] Add figures from experiments_v1/
- [ ] Complete bibliography entries
- [ ] Appendices (hyperparameter robustness, etc.)

## KDD 2026 Formatting Notes

### Anonymous Submission
The current template is configured for **anonymous submission**:
- Author names: "Anonymous Authors"
- Institution: "Anonymous Institution"
- Copyright notice removed
- ACM reference removed

### Camera-Ready Version
For the camera-ready version, uncomment the author block in `main.tex`:
```latex
\author{First Author}
\affiliation{%
  \institution{Your Institution}
  \city{City}
  \state{State}
  \country{Country}
}
\email{first.author@institution.edu}
```

### Page Limit
- **Main paper**: 9 pages (including references)
- **Appendices**: Unlimited (separate from main paper)

### Figures and Tables
- Use `\includegraphics` for figures
- Use `booktabs` package for professional tables
- Caption format: `\caption{Your caption here}`
- Label format: `\label{fig:your-label}` or `\label{tab:your-label}`

## Custom Commands

The following custom commands are defined in `main.tex`:

```latex
\neff          % Renders as: n_{eff}
\thetavec      % Renders as: \boldsymbol{\theta}
\Amat          % Renders as: \mathbf{A}
\bvec          % Renders as: \mathbf{b}
```

Usage example:
```latex
The effective sample size $\neff$ controls the confidence in the prior $\thetavec$.
```

## Troubleshooting

### Common Issues

1. **Missing package error**
   ```
   Solution: Install the missing package via your LaTeX distribution
   - MacTeX: Use TeX Live Utility
   - MiKTeX: Packages install automatically
   - TeX Live: sudo tlmgr install <package>
   ```

2. **Bibliography not showing**
   ```
   Solution: Run the full compilation sequence:
   pdflatex → bibtex → pdflatex → pdflatex
   ```

3. **Figures not found**
   ```
   Solution: Check that figure paths are correct relative to main.tex
   Example: \includegraphics{figures/figure1.pdf}
   ```

4. **Overfull/underfull hbox warnings**
   ```
   Solution: These are usually harmless. Adjust text if they're severe.
   ```

## Integration with Experiments

### Copying Figures from experiments_v1/

```bash
# Example: Copy Figure 7 (hyperparameter sensitivity)
cp ../experiments_v1/07_figure/results/figure7_sensitivity.png figures/

# In LaTeX:
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\columnwidth]{figures/figure7_sensitivity.png}
  \caption{Hyperparameter sensitivity analysis showing perfect robustness.}
  \label{fig:sensitivity}
\end{figure}
```

### Importing LaTeX from experiments_v1/

Many experiments already have LaTeX snippets:
- `experiments_v1/01_figure/figure1_caption.tex`
- `experiments_v1/appendix_e/hyperparameter_robustness.tex`
- `experiments_v1/appendix_d/hyperparameter_sensitivity.tex`

These can be directly included or adapted.

## Next Steps

1. **Outline the paper structure** - Decide on section organization
2. **Import existing LaTeX** - Leverage content from experiments_v1/
3. **Create figures** - Copy and reference experimental results
4. **Write narrative** - Connect the technical contributions
5. **Proofread** - Check for consistency and clarity

## Contact

For questions about the paper compilation, see the main README.md or contact the authors.

