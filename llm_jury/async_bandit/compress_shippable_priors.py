#!/usr/bin/env python3
"""
Compress a warmed router state into a tiny, check-in-friendly priors bundle.

Input:
  - router_state_synthetic.json (v1 huge JSON, or v2 small JSON + bandit.npz sidecar)

Output:
  - shippable_priors.npz  (A_shared float16 + b_vectors float16 + meta)

This implements the "Shared-Covariance Trick":
  store one shared A and per-model b vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llm_jury.async_bandit.bandit_router import BanditRouter, build_registry_from_models_cache


PROJECT_ROOT = Path(__file__).parent.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Compress warmed state to shippable priors (<1MB)")
    ap.add_argument("--state", type=str, required=True, help="Path to router_state_*.json")
    ap.add_argument("--cache", type=str, default=str(PROJECT_ROOT / "data" / "models_cache.json"))
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "priors" / "shippable_priors.npz"))
    args = ap.parse_args()

    state_path = Path(args.state)
    cache_path = Path(args.cache)
    out_path = Path(args.out)

    registry = build_registry_from_models_cache(cache_path)
    router = BanditRouter.load_state(state_path)
    # Ensure registry is present (for completeness)
    router.registry = dict(registry)

    router.save_shippable_priors(out_path, dtype=np.float16)
    print(f"Wrote shippable priors: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

