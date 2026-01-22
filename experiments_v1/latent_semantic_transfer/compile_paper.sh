#!/bin/bash
# Compile LaTeX paper to PDF

echo "Compiling paper.tex..."
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex  # Run twice for references

# Clean up auxiliary files
rm -f paper.aux paper.log paper.out

echo "✅ Paper compiled: paper.pdf"

