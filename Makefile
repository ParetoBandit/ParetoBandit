.PHONY: help install dev test test-fast lint typecheck format docs clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in editable mode
	pip install -e .

dev:  ## Install with all development extras
	pip install -e ".[dev,experiments,embeddings,docs]"
	pre-commit install

test:  ## Run the full test suite
	python -m pytest tests/ -v

test-fast:  ## Run tests excluding slow markers
	python -m pytest tests/ -v -m "not slow and not stress and not performance and not live"

lint:  ## Run ruff linter
	ruff check src/ tests/

typecheck:  ## Run mypy type checker
	mypy src/pareto_bandit/

format:  ## Auto-format code with ruff
	ruff check --fix src/ tests/
	ruff format src/ tests/

docs:  ## Build documentation site locally
	mkdocs serve

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
