# Concise Version - File Manifest

This document lists all files included in the concise version package.

## Core LaTeX Files

### Main Document
- **main_CONCISE.tex** - Master LaTeX file that orchestrates the entire document

### Section Files (Concise Versions)
- **abstract_CONCISE.tex** - Concise abstract (197 words, included in main_CONCISE.tex)
- **introduction_CONCISE.tex** - Concise introduction (~1.0 page)
- **use_cases_CONCISE.tex** - Concise use cases (~1.5 pages)
- **related_work_CONCISE.tex** - Concise related work (~0.65 pages)
- **conclusion_CONCISE.tex** - Concise conclusion (~0.5 pages)

### Section Files (Unchanged from Full Version)
- **method.tex** - Technical methodology (~2.0 pages)
- **evaluation.tex** - Experimental results (~2.5 pages)
- **appendix.tex** - Supplementary material

## Bibliography and References
- **references.bib** - BibTeX bibliography database

## Compilation Artifacts (Generated)
- **main_CONCISE.pdf** - Compiled PDF output
- **main_CONCISE.aux** - LaTeX auxiliary file
- **main_CONCISE.bbl** - BibTeX bibliography file
- **main_CONCISE.blg** - BibTeX log file
- **main_CONCISE.log** - LaTeX compilation log
- **main_CONCISE.out** - Hyperref outline file

## Scripts
- **compile.sh** - Automated compilation script

## Documentation
- **README.md** - Package overview and compilation instructions
- **CONCISE_VERSION_SUMMARY.md** - Detailed compression strategy and statistics
- **FILE_MANIFEST.md** - This file

## Figures Directory
- **figures/** - All figures and diagrams referenced in the paper
  - PDF figures (architecture diagrams, decision trees, results plots)
  - PNG figures (plots, comparisons, domain breakdowns)
  - Source `.tex` files for TikZ diagrams
  - Compilation scripts for diagrams

## File Dependencies

```
main_CONCISE.tex
├── introduction_CONCISE.tex
├── use_cases_CONCISE.tex
├── method.tex
├── evaluation.tex
├── related_work_CONCISE.tex
├── conclusion_CONCISE.tex
├── appendix.tex
├── references.bib
└── figures/
    ├── architecture_diagram.pdf
    ├── decision_tree_diagram.pdf
    ├── distillation_diagram.pdf
    ├── figure_pareto_frontier.pdf
    ├── figure_confident_failure.pdf
    ├── figure5_ood_generalization.pdf
    ├── figure6_sota_comparison.png
    ├── figure7_domain_breakdown.png
    └── [other figures as referenced in .tex files]
```

## Verification Checklist

✅ All `.tex` files referenced by `\input{}` commands are present  
✅ Bibliography file (`references.bib`) is included  
✅ All figures directory is included with all referenced figures  
✅ Compilation script is included and configured  
✅ Documentation files are included  
✅ PDF output is pre-generated for reference  

## Self-Contained Package

This folder is **completely self-contained** and does not require any files from the parent directory. All dependencies are included, and the paper can be compiled independently.

## Quick Start

To compile the paper:

```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/paper_submitted/concise_version
./compile.sh
```

Or manually:

```bash
pdflatex -interaction=nonstopmode main_CONCISE.tex
bibtex main_CONCISE
pdflatex -interaction=nonstopmode main_CONCISE.tex
pdflatex -interaction=nonstopmode main_CONCISE.tex
```

The output will be `main_CONCISE.pdf`.

