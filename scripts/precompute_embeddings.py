#!/usr/bin/env python3
"""
Pre-compute and cache PCA-projected embeddings for all experiment prompts.

Loads every unique prompt from the dev and holdout reward files, encodes
them with the default SentenceTransformer + PCA pipeline, and writes the
results to a NumPy ``.npz`` archive keyed by SHA-256 hash of the prompt
text.  A sidecar JSON stores provenance metadata so downstream consumers
can verify the cache matches their encoder/PCA.

The cache is prompt-centric and model-agnostic — a single file covers
K=2, K=10, and any other portfolio that draws from the same prompt pool.

Usage
-----
::

    python scripts/precompute_embeddings.py
    python scripts/precompute_embeddings.py --pca src/bandit_gpt/data/artifacts/pca_32.joblib \\
                                             --output data_collection/embeddings/embeddings_pca32.npz

Outputs
-------
``data_collection/embeddings/embeddings_pca6.npz``
    Compressed archive mapping ``sha256(prompt) -> context_vector``.
``data_collection/embeddings/embeddings_pca6_meta.json``
    Provenance: encoder name, PCA path, n_components, n_prompts, timestamp.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
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
) -> Dict[str, np.ndarray]:
    """Encode prompts in batches and apply PCA + whitening + bias.

    Returns a dict mapping ``sha256(prompt) -> context_vector``.
    """
    ev = getattr(pca_model, "explained_variance_", None)
    do_whiten = not bool(getattr(pca_model, "whiten", False)) and ev is not None
    if do_whiten:
        scale = 1.0 / np.sqrt(np.maximum(np.asarray(ev, dtype=np.float64), 1e-12))

    cache: Dict[str, np.ndarray] = {}
    n = len(prompts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = prompts[start:end]
        raw = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        projected = pca_model.transform(raw)
        if do_whiten:
            projected = projected * scale
        for i, prompt in enumerate(batch):
            vec = np.append(projected[i], 1.0)
            cache[_hash_prompt(prompt)] = vec
        print(f"  Embedded {end}/{n} prompts", end="\r", flush=True)
    print(flush=True)
    return cache


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
        default=str(PROJECT_ROOT / "data_collection" / "embeddings" / "embeddings_pca6.npz"),
        help="Output .npz path.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pca_path = Path(args.pca)
    out_path = Path(args.output)

    print(f"Encoder:  {args.encoder}")
    print(f"PCA:      {pca_path}")
    print(f"Output:   {out_path}")

    data_paths = [DEV_DATA_PATH_ALL_MODELS, HOLDOUT_DATA_PATH_ALL_MODELS]
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
    cache = _embed_batch(prompts, encoder, pca, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({len(prompts) / elapsed:.0f} prompts/sec)")

    sample_vec = next(iter(cache.values()))
    print(f"  Vector dim: {sample_vec.shape[0]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)
    print(f"\nWrote: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    meta = {
        "encoder": args.encoder,
        "pca_path": str(pca_path),
        "n_components": int(n_comp) if isinstance(n_comp, (int, np.integer)) else str(n_comp),
        "vector_dim": int(sample_vec.shape[0]),
        "n_prompts": len(cache),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": [str(p) for p in data_paths],
    }
    meta_path = out_path.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
