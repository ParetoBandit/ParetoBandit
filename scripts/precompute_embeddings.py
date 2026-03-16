#!/usr/bin/env python3
"""
Pre-compute and cache embeddings for all experiment prompts.

Produces two cache layers in a single encoder pass:

1. **Raw embeddings** — the full-dimensional SentenceTransformer output
   (e.g. 384-dim for ``all-MiniLM-L6-v2``).  These are PCA-agnostic and can
   be projected through any PCA truncation downstream.

2. **PCA-projected embeddings** — whitened, bias-appended context vectors
   ready for LinUCB (e.g. 16-dim for PCA-15).

Loads every unique prompt from the dev and holdout reward files, encodes
them with the default SentenceTransformer, and writes both ``.npz``
archives keyed by SHA-256 hash of the prompt text.  Sidecar ``.meta.json``
files store provenance metadata so downstream consumers can verify the
cache matches their encoder/PCA.

The caches are prompt-centric and model-agnostic — a single file covers
K=2, K=10, and any other portfolio that draws from the same prompt pool.

Usage
-----
::

    python scripts/precompute_embeddings.py
    python scripts/precompute_embeddings.py --pca src/pareto_bandit/data/artifacts/pca_32.joblib \\
                                             --output data_collection/embeddings/embeddings_pca32.npz

Outputs
-------
``data_collection/embeddings/raw_embeddings.npz``
    Compressed archive mapping ``sha256(prompt) -> raw_embedding``.
``data_collection/embeddings/raw_embeddings.meta.json``
    Provenance: encoder name, raw embedding dim, n_prompts, timestamp.
``data_collection/embeddings/embeddings_pca15.npz``
    Compressed archive mapping ``sha256(prompt) -> context_vector``.
``data_collection/embeddings/embeddings_pca15.meta.json``
    Provenance: encoder name, PCA path, n_components, n_prompts, timestamp.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    DEV_DATA_PATH_ALL_MODELS,
    EMBEDDINGS_CACHE_PATH,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    K4_TRAIN_DATA_PATH,
    K4_VAL_DATA_PATH,
    K4_HOLDOUT_DATA_PATH,
    RAW_EMBEDDINGS_CACHE_PATH,
)


def _hash_prompt(prompt: str) -> str:
    """Deterministic SHA-256 hex digest of a prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _collect_unique_prompts(data_paths: List[Path]) -> List[str]:
    """Extract all unique prompt strings from gzipped JSONL reward files."""
    seen: set = set()
    prompts: List[str] = []
    for path in data_paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                entry = json.loads(line)
                if not entry.get("ok"):
                    continue
                p = entry["prompt"]
                if p not in seen:
                    seen.add(p)
                    prompts.append(p)
    return prompts


def _embed_batch(
    prompts: List[str],
    encoder: SentenceTransformer,
    pca_model,
    *,
    batch_size: int = 256,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Encode prompts in batches, returning both raw and PCA-projected caches.

    Returns:
        A ``(raw_cache, pca_cache)`` tuple where each dict maps
        ``sha256(prompt)`` to the corresponding embedding vector.
        ``raw_cache`` contains the full-dimensional SentenceTransformer
        output; ``pca_cache`` contains whitened PCA projections with an
        appended bias term.
    """
    ev = getattr(pca_model, "explained_variance_", None)
    do_whiten = not bool(getattr(pca_model, "whiten", False)) and ev is not None
    if do_whiten:
        scale = 1.0 / np.sqrt(np.maximum(np.asarray(ev, dtype=np.float64), 1e-12))

    raw_cache: Dict[str, np.ndarray] = {}
    pca_cache: Dict[str, np.ndarray] = {}
    n = len(prompts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = prompts[start:end]
        raw = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        projected = pca_model.transform(raw)
        if do_whiten:
            projected = projected * scale
        for i, prompt in enumerate(batch):
            key = _hash_prompt(prompt)
            raw_cache[key] = raw[i]
            pca_cache[key] = np.append(projected[i], 1.0)
        print(f"  Embedded {end}/{n} prompts", end="\r", flush=True)
    print(flush=True)
    return raw_cache, pca_cache


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute PCA-projected embeddings for experiment prompts.",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default=DEFAULT_SENTENCE_TRANSFORMER,
        help="SentenceTransformer model name.",
    )
    p.add_argument(
        "--pca",
        type=str,
        default=str(DEFAULT_PCA_PATH),
        help="Path to PCA joblib artifact.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(EMBEDDINGS_CACHE_PATH),
        help="Output .npz path for PCA-projected cache (default: EMBEDDINGS_CACHE_PATH).",
    )
    p.add_argument(
        "--raw-output",
        type=str,
        default=str(RAW_EMBEDDINGS_CACHE_PATH),
        help="Output .npz path for raw embedding cache (default: RAW_EMBEDDINGS_CACHE_PATH).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size.",
    )
    return p.parse_args()


def _write_cache(
    cache: Dict[str, np.ndarray],
    out_path: Path,
    meta: dict,
) -> None:
    """Write an ``.npz`` cache and its sidecar ``.meta.json``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)
    print(f"  Wrote: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    meta_path = out_path.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  Wrote: {meta_path}")


def main() -> None:
    args = _parse_args()
    pca_path = Path(args.pca)
    pca_out_path = Path(args.output)
    raw_out_path = Path(args.raw_output)

    print(f"Encoder:     {args.encoder}")
    print(f"PCA:         {pca_path}")
    print(f"PCA output:  {pca_out_path}")
    print(f"Raw output:  {raw_out_path}")

    data_paths = [
        DEV_DATA_PATH_ALL_MODELS,
        HOLDOUT_DATA_PATH_ALL_MODELS,
        K4_TRAIN_DATA_PATH,
        K4_VAL_DATA_PATH,
        K4_HOLDOUT_DATA_PATH,
    ]
    print(f"\nCollecting unique prompts from {len(data_paths)} files ...")
    prompts = _collect_unique_prompts(data_paths)
    print(f"  {len(prompts)} unique prompts")

    print(f"\nLoading encoder: {args.encoder} ...")
    encoder = SentenceTransformer(args.encoder)

    print(f"Loading PCA: {pca_path.name} ...")
    pca = joblib.load(pca_path)
    n_comp = getattr(pca, "n_components_", getattr(pca, "n_components", "?"))
    print(f"  {n_comp} components")

    print(f"\nEmbedding {len(prompts)} prompts (batch_size={args.batch_size}) ...")
    t0 = time.perf_counter()
    raw_cache, pca_cache = _embed_batch(
        prompts, encoder, pca, batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({len(prompts) / elapsed:.0f} prompts/sec)")

    sample_raw = next(iter(raw_cache.values()))
    sample_pca = next(iter(pca_cache.values()))
    print(f"  Raw dim: {sample_raw.shape[0]}, PCA dim: {sample_pca.shape[0]}")

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    source_strs = [str(p) for p in data_paths]

    print("\nWriting raw embedding cache ...")
    _write_cache(raw_cache, raw_out_path, {
        "encoder": args.encoder,
        "vector_dim": int(sample_raw.shape[0]),
        "n_prompts": len(raw_cache),
        "timestamp": ts,
        "data_sources": source_strs,
    })

    print("Writing PCA-projected embedding cache ...")
    _write_cache(pca_cache, pca_out_path, {
        "encoder": args.encoder,
        "pca_path": str(pca_path),
        "n_components": int(n_comp) if isinstance(n_comp, (int, np.integer)) else str(n_comp),
        "vector_dim": int(sample_pca.shape[0]),
        "n_prompts": len(pca_cache),
        "timestamp": ts,
        "data_sources": source_strs,
    })


if __name__ == "__main__":
    main()
