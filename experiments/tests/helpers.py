"""Shared helpers for experiment regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

REFERENCE_DIR = Path(__file__).parent / "references"


def load_reference(name: str) -> Dict[str, Any]:
    """Load a reference JSON file, raising ``FileNotFoundError`` if absent."""
    path = REFERENCE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def save_reference(name: str, data: Dict[str, Any]) -> Path:
    """Save a reference JSON file and return its path."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def assert_metrics_match(
    actual: Dict[str, Any],
    reference: Dict[str, Any],
    *,
    atol: float = 1e-10,
) -> None:
    """Assert all scalar metrics in *reference* match *actual* within *atol*.

    For nested dicts (e.g. ``model_fractions``), comparison is applied
    recursively to leaf float values.
    """
    for key, ref_val in reference.items():
        act_val = actual[key]
        if isinstance(ref_val, dict):
            assert isinstance(act_val, dict), f"{key}: expected dict, got {type(act_val)}"
            for sub_key, sub_ref in ref_val.items():
                assert act_val[sub_key] == pytest.approx(sub_ref, abs=atol), (
                    f"{key}.{sub_key}: {act_val[sub_key]} != {sub_ref} (atol={atol})"
                )
        elif isinstance(ref_val, float):
            assert act_val == pytest.approx(ref_val, abs=atol), (
                f"{key}: {act_val} != {ref_val} (atol={atol})"
            )
        else:
            assert act_val == ref_val, f"{key}: {act_val} != {ref_val}"
