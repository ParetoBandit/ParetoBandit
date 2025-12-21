#!/bin/bash

# Compile TikZ diagrams to PDF

TEXBIN="/usr/local/texlive/2025/bin/universal-darwin"
DIAGRAMS=("architecture_diagram" "distillation_diagram" "decision_tree_diagram")

echo "🎨 Compiling TikZ diagrams to PDF..."
echo ""

for diagram in "${DIAGRAMS[@]}"; do
    echo "📄 Compiling ${diagram}.tex..."
    
    # Compile with pdflatex
    ${TEXBIN}/pdflatex -interaction=nonstopmode ${diagram}.tex > /dev/null 2>&1
    
    if [ $? -eq 0 ] && [ -f "${diagram}.pdf" ]; then
        SIZE=$(ls -lh ${diagram}.pdf | awk '{print $5}')
        echo "✅ ${diagram}.pdf created (${SIZE})"
        
        # Clean up auxiliary files
        rm -f ${diagram}.aux ${diagram}.log
    else
        echo "❌ ${diagram}.pdf failed to compile"
        echo "   Check ${diagram}.log for errors"
    fi
    echo ""
done

echo "🎉 Diagram compilation complete!"
echo ""
echo "Generated PDFs:"
ls -lh architecture_diagram.pdf distillation_diagram.pdf decision_tree_diagram.pdf 2>/dev/null

