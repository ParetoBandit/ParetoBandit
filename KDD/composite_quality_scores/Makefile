# Makefile for BLF Validation
# Makes it easy to run all validation scripts

.PHONY: all validate report clean help install

# Default target
all: validate report

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Run validation (generates all figures)
validate:
	@echo "=========================================="
	@echo "Running BLF Validation"
	@echo "=========================================="
	python validate_blf_scores.py

# Generate validation report
report:
	@echo "=========================================="
	@echo "Generating Validation Report"
	@echo "=========================================="
	python generate_validation_report.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f *.pdf
	rm -f validation_metrics.json
	rm -f VALIDATION_REPORT.md
	rm -f validation_table.tex
	@echo "✓ Cleaned"

# Show help
help:
	@echo "BLF Validation Makefile"
	@echo "======================="
	@echo ""
	@echo "Targets:"
	@echo "  all       - Run validation and generate report (default)"
	@echo "  install   - Install Python dependencies"
	@echo "  validate  - Run validation (generates figures)"
	@echo "  report    - Generate validation report"
	@echo "  clean     - Remove generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make         # Run all (validate + report)"
	@echo "  make install # Install dependencies first"
	@echo "  make clean   # Clean up generated files"
