"""Shipped data assets for pareto_bandit.

This package bundles lightweight artefacts (PCA models, warmup priors,
example evaluation data) so that the library and its demo work out of
the box from a ``pip install`` without cloning the full repository.
"""

from __future__ import annotations

from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent


def _require(path: Path, label: str) -> Path:
    """Return *path* if it exists, otherwise raise ``FileNotFoundError``."""
    if not path.exists():
        raise FileNotFoundError(
            f"Shipped {label} data not found at {path}. "
            "Re-install pareto_bandit or clone the repository."
        )
    return path


def get_example_holdout_path() -> Path:
    """Return the absolute path to the shipped test-holdout JSONL file.

    The file contains 1,824 prompts from public benchmarks (GSM8K,
    WinoGrande, etc.) with per-arm rewards and costs for the default
    K=3 model set.  Used as the **evaluation** split by the demo.

    Returns
    -------
    Path
        Absolute path to ``data/examples/test_holdout.jsonl``.

    Raises
    ------
    FileNotFoundError
        If the file is missing (e.g. a custom build that excluded it).
    """
    return _require(_DATA_DIR / "examples" / "test_holdout.jsonl", "holdout")


def get_example_val_path() -> Path:
    """Return the absolute path to the shipped validation JSONL file.

    The file contains 1,785 prompts (same schema as the holdout) used
    as the **training** split by the demo — matching the paper's
    train-on-val / evaluate-on-holdout protocol.

    Returns
    -------
    Path
        Absolute path to ``data/examples/val.jsonl``.

    Raises
    ------
    FileNotFoundError
        If the file is missing (e.g. a custom build that excluded it).
    """
    return _require(_DATA_DIR / "examples" / "val.jsonl", "validation")
