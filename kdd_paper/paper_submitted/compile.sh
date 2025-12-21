#!/bin/bash

# ============================================================================
# LaTeX Compilation Script for KDD Paper
# ============================================================================

TEXBIN="/usr/local/texlive/2025/bin/universal-darwin"
PAPER="main_RESTRUCTURED"

echo "🔧 Compiling ${PAPER}.tex..."
echo ""

# First pass
echo "📄 Pass 1: Initial compilation..."
${TEXBIN}/pdflatex -interaction=nonstopmode ${PAPER}.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Pass 1 complete"
else
    echo "❌ Pass 1 failed - check ${PAPER}.log"
    exit 1
fi

# BibTeX
echo "📚 Processing bibliography..."
${TEXBIN}/bibtex ${PAPER} 2>&1 | grep -E "(Warning|Error)" || echo "✅ BibTeX complete"

# Second pass
echo "📄 Pass 2: Resolving references..."
${TEXBIN}/pdflatex -interaction=nonstopmode ${PAPER}.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Pass 2 complete"
else
    echo "❌ Pass 2 failed - check ${PAPER}.log"
    exit 1
fi

# Third pass
echo "📄 Pass 3: Final compilation..."
${TEXBIN}/pdflatex -interaction=nonstopmode ${PAPER}.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Pass 3 complete"
else
    echo "❌ Pass 3 failed - check ${PAPER}.log"
    exit 1
fi

# Check output
if [ -f "${PAPER}.pdf" ]; then
    PAGES=$(${TEXBIN}/pdfinfo ${PAPER}.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    SIZE=$(ls -lh ${PAPER}.pdf | awk '{print $5}')
    echo ""
    echo "🎉 Compilation successful!"
    echo "📄 Output: ${PAPER}.pdf"
    echo "📊 Pages: ${PAGES}"
    echo "💾 Size: ${SIZE}"
    echo ""
    echo "To view: open ${PAPER}.pdf"
else
    echo "❌ PDF not generated - check ${PAPER}.log"
    exit 1
fi

