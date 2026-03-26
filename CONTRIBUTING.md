# Contributing to ParetoBandit

Thank you for your interest in contributing to ParetoBandit! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/atabernermiller/paretobandit.git
cd paretobandit
pip install -e ".[dev]"
```

This installs the package in editable mode with development dependencies (pytest, ruff, mypy, pre-commit).

## Running Tests

```bash
# Full test suite (~2 min)
python -m pytest tests/ -v

# Skip slow tests
python -m pytest tests/ -v -m "not slow"

# Specific test file
python -m pytest tests/test_router.py -v

# With coverage
python -m pytest tests/ --cov=pareto_bandit --cov-report=term-missing
```

## Code Quality

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy.readthedocs.io/) for type checking.

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/pareto_bandit/
```

All pull requests must pass lint and type checks. The CI pipeline runs these automatically.

## Code Style

- **Line length**: 100 characters
- **Python version**: 3.10+ (use `X | Y` union syntax, not `Union[X, Y]`)
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Required for all public classes and methods; use Google-style format
- **Imports**: Sorted by isort (configured via ruff)

## Making Changes

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for any new functionality
3. **Update docstrings** and the API reference (`docs/API_REFERENCE.md`) if you change the public API
4. **Run the full test suite** before submitting
5. **Open a pull request** with a clear description of the change

### Commit Messages

Use concise, descriptive commit messages:

```
Add adaptive decay schedule for exploration rate
Fix numerical instability in low-traffic arms when update_lambda=0
Improve Figure 4 annotations and crossover detection
```

## Architecture Overview

```
src/pareto_bandit/
├── router.py            # BanditRouter, LinUCB policies
├── feature_service.py   # Prompt embedding + PCA compression
├── storage.py           # Context persistence (SQLite, ephemeral)
├── calibration.py       # train_pca(), generate_warmup_priors()
├── cli.py               # CLI entry point
├── config/              # Model registry and default configurations
└── utils/               # Warmup priors, heuristics
```

### Key Design Principles

- **Separation of concerns**: FeatureService handles embedding, Router handles bandit math, Storage handles persistence
- **Self-healing**: Missing artifacts trigger JIT recovery rather than crashes
- **Lock-minimal concurrency**: Snapshot-swap pattern keeps routing fast during matrix updates
- **Progressive API**: Users can start simple (`BanditRouter.create()`) and add complexity as needed

## Reporting Issues

When filing a bug report, please include:

- Python version (`python --version`)
- ParetoBandit version (`paretobandit --version`)
- Minimal reproduction code
- Full traceback

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
