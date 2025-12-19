"""
Allow running the package directly: python -m banditgpt

This delegates to the CLI module.
"""

from banditgpt.async_bandit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
