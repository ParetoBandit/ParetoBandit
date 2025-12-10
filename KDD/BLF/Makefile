# Makefile for BLF paper section and figures
#
# Usage:
#   make all          - Generate all figures and compile LaTeX
#   make figures      - Generate all PDF figures
#   make validation   - Run validation against Arena ELO
#   make examples     - Run example usage scripts
#   make clean        - Remove generated files
#   make latex        - Compile LaTeX section (standalone)

PYTHON = python3
FIGURES = fig1_missing_data.pdf fig2_convergence.pdf fig3_ppc.pdf \
          fig4_comparison.pdf fig5_loadings.pdf fig6_graphical_model.pdf

.PHONY: all figures validation examples clean latex help

all: figures validation
	@echo "✓ All artifacts generated"

figures: $(FIGURES)
	@echo "✓ All figures generated"

$(FIGURES): generate_figures.py
	$(PYTHON) generate_figures.py

validation: validation_results.csv
	@echo "✓ Validation complete"

validation_results.csv: validation_results.py
	$(PYTHON) validation_results.py

examples:
	$(PYTHON) example_usage.py

latex: blf_section.pdf
	@echo "✓ LaTeX compiled"

blf_section.pdf: blf_section.tex
	pdflatex blf_section.tex
	pdflatex blf_section.tex  # Run twice for references

clean:
	rm -f *.pdf *.aux *.log *.out *.toc
	rm -f validation_results.csv custom_config.json
	@echo "✓ Cleaned generated files"

help:
	@echo "BLF Paper Section Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Generate all figures and run validation"
	@echo "  figures     - Generate all PDF figures (6 total)"
	@echo "  validation  - Run validation against Chatbot Arena ELO"
	@echo "  examples    - Run example usage demonstrations"
	@echo "  latex       - Compile LaTeX section to PDF"
	@echo "  clean       - Remove generated files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  pip install -r requirements.txt"
