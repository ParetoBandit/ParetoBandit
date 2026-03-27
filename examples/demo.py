#!/usr/bin/env python3
"""Thin wrapper — delegates to ``pareto_bandit.demo.main()``.

This script lets git-clone users run the demo directly without
installing the package::

    python examples/demo.py
    python examples/demo.py --scenario 2

For pip-installed users, the equivalent command is::

    paretobandit-demo

See ``paretobandit-demo --help`` for all available options.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit.demo import main  # noqa: E402

if __name__ == "__main__":
    main()
