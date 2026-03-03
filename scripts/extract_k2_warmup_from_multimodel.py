#!/usr/bin/env python3
"""
Extract K=2 warmup priors from the 43-model warmup artifact.

Motivation
----------
The multi-model warmup priors artifact (`priors_warmup_43model.joblib`) contains
per-model LinUCB sufficient statistics (A, b) for all models observed in the
prior-train pool. For K=2 experiments, we can reuse the exact same warmup signal
by filtering to the two target models, avoiding any separate warmup generation
pipeline (and avoiding dependence on external corpora).

This script produces a new warmup priors joblib that is compatible with
`BanditRouter.create(..., warmup_path=...)` / `create_experiment_router(...)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import joblib


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract 2-model warmup priors from multi-model priors.")
    p.add_argument(
        "--input",
        type=str,
        default="src/artifacts/priors_warmup_43model.joblib",
        help="Path to multi-model warmup priors joblib.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="src/artifacts/priors_warmup_k2_from_43model.joblib",
        help="Path to write filtered K=2 warmup priors joblib.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="mistralai/mixtral-8x7b-instruct,openai/gpt-4-turbo",
        help="Comma-separated list of exactly two model IDs to extract.",
    )
    return p.parse_args()


def extract_priors(
    priors: Dict[str, Any],
    *,
    models: List[str],
    reward_source_suffix: str,
) -> Dict[str, Any]:
    """Filter a warmup priors dict to a subset of models.

    Args:
        priors: Dict containing at least keys `A`, `b`, and `models`.
        models: Model IDs to retain (must all exist in priors).
        reward_source_suffix: String appended to `reward_source` for provenance.

    Returns:
        A new priors dict with A/b filtered to `models` and metadata updated.
    """
    if len(models) != 2:
        raise ValueError(f"Expected exactly 2 models, got {len(models)}: {models}")
    for m in models:
        if m not in priors.get("A", {}) or m not in priors.get("b", {}):
            raise KeyError(f"Model '{m}' not found in priors A/b keys.")

    new_priors = dict(priors)
    new_priors["A"] = {m: priors["A"][m] for m in models}
    new_priors["b"] = {m: priors["b"][m] for m in models}
    new_priors["models"] = list(models)

    rs = str(priors.get("reward_source", "unknown"))
    new_priors["reward_source"] = f"{rs}{reward_source_suffix}"
    return new_priors


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]

    priors = joblib.load(in_path)
    if not isinstance(priors, dict):
        raise TypeError(f"Expected dict priors, got {type(priors)}")

    filtered = extract_priors(
        priors,
        models=models,
        reward_source_suffix="|filtered_k2_from_43model",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(filtered, out_path)
    print(f"Wrote: {out_path} (models={filtered['models']}, context_dim={filtered.get('context_dim')})")


if __name__ == "__main__":
    main()

