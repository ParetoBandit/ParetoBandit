# Contributing to BanditGPT

Thank you for your interest in contributing to BanditGPT! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/atabernermiller/banditgpt.git
   cd banditgpt
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Pytest**: Testing

Run all checks locally before submitting:

```bash
# Linting
ruff check banditgpt tests

# Type checking
mypy banditgpt --ignore-missing-imports

# Tests
pytest tests/ -v
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed

3. **Run the test suite**
   ```bash
   pytest tests/ -v
   ```

4. **Submit a pull request**
   - Provide a clear description of the changes
   - Reference any related issues

## Release Checklist

1. Bump version in `pyproject.toml` and `banditgpt/__init__.py`.
2. Update `CHANGELOG.md` with changes and include current priors manifest checksums from `banditgpt/data/priors/manifest.json`.
3. Run full tests: `python -m pytest tests/ -v`.
4. Build artifacts: `python -m build` (wheel + sdist).
5. Clean-venv smoke test: new venv, install the wheel from `dist/`, run `banditgpt verify-priors`, and a minimal `BanditRouter.create(..., priors="bundled").route("hello", profile="balanced")`.
6. Publish (trusted publishing or `twine upload dist/*`).
7. Tag release: `git tag vX.Y.Z` and push tags.

## Reporting Issues

When reporting issues, please include:

- Python version (`python --version`)
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Any relevant error messages

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person

## License

By contributing to BanditGPT, you agree that your contributions will be licensed under the **Apache License 2.0**. This includes an explicit patent grant, ensuring enterprise users can safely use this library in commercial products.

See [LICENSE](LICENSE) for the full license text.

## Questions?

Feel free to open an issue for any questions about contributing.
