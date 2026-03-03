"""
Centralized model pricing for experiments.

Motivation
----------
Many experiment scripts historically hard-coded model prices (input/output
$/1M tokens). This is brittle: a single typo silently changes cost-quality
frontiers and can confound comparisons across routing methods.

This module provides a single source of truth by loading prices from the
project's JSON registries:
  - ``src/bandit_gpt/config/models.json`` (small/production registry)
  - ``src/bandit_gpt/config/models_all.json`` (extended registry for experiments)

All experiments should use :func:`build_model_registry_from_json` (or
:func:`get_prices_for_models`) so that updating a JSON file updates every
downstream experiment consistently.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

from bandit_gpt.config import DEFAULT_MODEL_REGISTRY_PATH


DEFAULT_MODEL_REGISTRY_ALL_PATH: Path = DEFAULT_MODEL_REGISTRY_PATH.with_name(
    "models_all.json"
)

# Experiment-level pricing file (single source of truth for all experiment scripts).
# This is intentionally separate from the package registry so experiments can add
# models without modifying the pip-installed package data.
EXPERIMENT_MODEL_PRICING_PATH: Path = (
    Path(__file__).resolve().parent.parent / "config" / "model_prices.json"
)


def _as_float(x: Any, *, field: str, model_id: str, registry_path: Path) -> float:
    if not isinstance(x, (int, float)):
        raise TypeError(
            f"Invalid type for {field} of {model_id} in {registry_path}: "
            f"expected number, got {type(x).__name__}"
        )
    return float(x)


@lru_cache(maxsize=8)
def _load_registry_by_model_id(registry_path_str: str) -> Dict[str, Dict[str, Any]]:
    """Load a registry JSON and index by ``model_id`` (cached)."""
    registry_path = Path(registry_path_str)
    payload = json.loads(registry_path.read_text())
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(
            f"Malformed registry at {registry_path}: expected top-level 'models' list."
        )
    by_id: Dict[str, Dict[str, Any]] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = m.get("model_id")
        if isinstance(mid, str):
            by_id[mid] = m
    return by_id


def _default_registry_search_paths() -> Tuple[Path, Path, Path]:
    # Prefer the experiment pricing table, then fall back to package registries.
    return (
        EXPERIMENT_MODEL_PRICING_PATH,
        DEFAULT_MODEL_REGISTRY_ALL_PATH,
        DEFAULT_MODEL_REGISTRY_PATH,
    )


def get_prices_for_models(
    model_ids: Iterable[str],
    *,
    registry_paths: Iterable[Path] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Return input/output prices for each model from JSON registries.

    Args:
        model_ids: Iterable of model IDs to look up (e.g. ``openai/gpt-4-turbo``).
        registry_paths: Optional ordered list of JSON registry paths to search.
            If omitted, searches ``experiments/config/model_prices.json`` first,
            then ``models_all.json`` and ``models.json``.

    Returns:
        Dict mapping ``model_id`` to a dict with:
          - ``input_cost_per_m``: float ($ per 1M input tokens)
          - ``output_cost_per_m``: float ($ per 1M output tokens)

    Raises:
        KeyError: If any model_id is not found in the provided registries.
        TypeError/ValueError: If registry contents are malformed.
    """
    search_paths = tuple(registry_paths) if registry_paths is not None else _default_registry_search_paths()

    out: Dict[str, Dict[str, float]] = {}
    missing: list[str] = []
    for model_id in model_ids:
        found: Dict[str, Any] | None = None
        found_path: Path | None = None
        for p in search_paths:
            reg = _load_registry_by_model_id(str(p))
            if model_id in reg:
                found = reg[model_id]
                found_path = p
                break
        if found is None or found_path is None:
            missing.append(model_id)
            continue
        inp = _as_float(found.get("input_cost_per_m"), field="input_cost_per_m", model_id=model_id, registry_path=found_path)
        outp = _as_float(found.get("output_cost_per_m"), field="output_cost_per_m", model_id=model_id, registry_path=found_path)
        out[model_id] = {"input_cost_per_m": inp, "output_cost_per_m": outp}

    if missing:
        searched = ", ".join(str(p) for p in search_paths)
        raise KeyError(
            "Missing pricing for model_ids: "
            + ", ".join(missing)
            + f". Searched registries: {searched}"
        )
    return out


def build_model_registry_from_json(
    model_ids: Iterable[str],
    *,
    registry_paths: Iterable[Path] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Build the ``model_registry`` dict expected by ``create_experiment_router``.

    Args:
        model_ids: Model IDs to include.
        registry_paths: Optional ordered registry paths to search.

    Returns:
        Dict mapping model_id -> {"input_cost_per_m": float, "output_cost_per_m": float}.
    """
    prices = get_prices_for_models(model_ids, registry_paths=registry_paths)
    return {m: dict(prices[m]) for m in prices}

