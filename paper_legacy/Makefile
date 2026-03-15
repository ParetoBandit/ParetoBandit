# Makefile for banditGPT Paper

# Main file
MAIN = main
TEX = pdflatex
BIB = bibtex

# Auto-track all .tex sources so changes to \input files trigger a rebuild
SOURCES = $(wildcard sections/*.tex) body_full.tex body_core.tex preamble.tex

# Targets
.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib $(SOURCES)
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
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

