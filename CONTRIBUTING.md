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

## Questions?

Feel free to open an issue for any questions about contributing.
