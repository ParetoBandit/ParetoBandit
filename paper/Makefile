# Makefile for banditGPT KDD 2026 Paper

# Main file
MAIN = main
TEX = pdflatex
BIB = bibtex

# Targets
.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	open $(MAIN).pdf

# Quick compile (no bibtex)
quick:
	$(TEX) $(MAIN)

# Watch mode (requires latexmk)
watch:
	latexmk -pdf -pvc $(MAIN).tex

