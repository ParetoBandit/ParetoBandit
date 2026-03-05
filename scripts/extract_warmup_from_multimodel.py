#!/usr/bin/env python3
"""
Extract portfolio-specific warmup priors from the 43-model warmup artifact.

Motivation
----------
The multi-model warmup priors artifact (``priors_warmup_43model*.joblib``)
contains per-model LinUCB sufficient statistics (A, b) for all 43 models
observed in the prior-train pool.  Experiments that run on a smaller
portfolio (K=2, K=10, etc.) should use priors filtered to *only* the models
in their portfolio, avoiding information leakage from models outside the
test set and keeping the comparison clean.

This script reads a model list (either from a JSON config or a comma-
separated CLI argument) and produces a filtered warmup priors joblib that is
compatible with ``BanditRouter.create(..., warmup_path=...)``.

Examples
--------
# K=10 from config file (6-component PCA):
python scripts/extract_warmup_from_multimodel.py \
    --input  src/artifacts/priors_warmup_43model_6comp.joblib \
    --output data_collection/warmup_priors/priors_warmup_k10_6comp.joblib \
    --model-config data_collection/config/models_k10.json

# K=2 with explicit model list:
python scripts/extract_warmup_from_multimodel.py \
    --input  src/artifacts/priors_warmup_43model_6comp.joblib \
    --output data_collection/warmup_priors/priors_warmup_k2_6comp.joblib \
    --models meta-llama/llama-3.1-8b-instruct,openai/gpt-4.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib


def extract_priors(
    priors: Dict[str, Any],
    *,
    models: List[str],
    reward_source_suffix: str,
) -> Dict[str, Any]:
    """Filter a warmup priors dict to a subset of models.

    Args:
        priors: Dict containing at least keys ``A``, ``b``, and ``models``.
        models: Model IDs to retain (must all exist in priors).
        reward_source_suffix: String appended to ``reward_source`` for
            provenance tracking.

    Returns:
        A new priors dict with A/b filtered to ``models`` and metadata
        updated.

    Raises:
        ValueError: If ``models`` is empty.
        KeyError: If any requested model is absent from the priors.
    """
    if not models:
        raise ValueError("models list must be non-empty.")
    for m in models:
        if m not in priors.get("A", {}) or m not in priors.get("b", {}):
            available = sorted(priors.get("A", {}).keys())
            raise KeyError(
                f"Model '{m}' not found in priors A/b keys. "
                f"Available: {available}"
            )

    new_priors = dict(priors)
    new_priors["A"] = {m: priors["A"][m] for m in models}
    new_priors["b"] = {m: priors["b"][m] for m in models}
    new_priors["models"] = list(models)

    rs = str(priors.get("reward_source", "unknown"))
    new_priors["reward_source"] = f"{rs}{reward_source_suffix}"
    return new_priors


def _load_model_ids_from_config(config_path: Path) -> List[str]:
    """Read model IDs from a JSON config file (``{"models": [...]}``)."""
    with open(config_path) as f:
        data = json.load(f)
    entries = data.get("models", data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise TypeError(f"Expected list of models, got {type(entries)}")
    return [
        e["model_id"] if isinstance(e, dict) else str(e) for e in entries
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract portfolio-specific warmup priors from the "
                    "43-model artifact.",
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to multi-model warmup priors joblib.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write filtered warmup priors joblib.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model-config",
        type=str,
        help="Path to a JSON config file with a 'models' array "
             "(each entry has a 'model_id' field).",
    )
    source.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model IDs to extract.",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Custom reward_source suffix.  Defaults to "
             "'|filtered_k<N>_from_43model'.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if args.model_config:
        models = _load_model_ids_from_config(Path(args.model_config))
    else:
        models = [m.strip() for m in str(args.models).split(",") if m.strip()]

    suffix = args.suffix or f"|filtered_k{len(models)}_from_43model"

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Models ({len(models)}):")
    for m in models:
        print(f"  - {m}")

    priors = joblib.load(in_path)
    if not isinstance(priors, dict):
        raise TypeError(f"Expected dict priors, got {type(priors)}")

    filtered = extract_priors(
        priors,
        models=models,
        reward_source_suffix=suffix,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(filtered, out_path)

    ctx_dim = filtered.get("context_dim", "unknown")
    print(f"\nWrote: {out_path}")
    print(f"  K={len(filtered['models'])}, context_dim={ctx_dim}")
    print(f"  reward_source: {filtered.get('reward_source', 'N/A')}")


if __name__ == "__main__":
    main()
