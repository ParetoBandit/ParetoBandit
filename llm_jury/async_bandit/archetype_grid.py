#!/usr/bin/env python3
"""
Archetype Grid builder: cluster prompts -> select representative "golden prompts".

Workflow:
  - Sample N prompts from a proxy dataset (default: LMSYS Chatbot Arena).
  - Embed prompts with the same SentenceTransformer as the router.
  - Cluster into K archetypes (MiniBatchKMeans).
  - For each cluster, pick the single prompt closest to its centroid.
  - Write the K prompts to `data/priors/archetype_grid_prompts.jsonl`.

This is step A of the "Archetype Grid" strategy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llm_jury.async_bandit.bandit_router import DEFAULT_CONTEXT_MODEL


PROJECT_ROOT = Path(__file__).parent.parent.parent


def _extract_prompt_from_lmsys_row(ex: Dict[str, Any]) -> Optional[str]:
    """
    LMSYS Chatbot Arena schema: conversation_a / conversation_b is list[{'role','content'}]
    We take the first user message.
    """
    for key in ("conversation_a", "conversation_b"):
        msgs = ex.get(key)
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", "")).lower()
                content = m.get("content")
                if role == "user" and isinstance(content, str) and content.strip():
                    return content.strip()
    return None


def _extract_prompt(ex: Any) -> Optional[str]:
    if not isinstance(ex, dict):
        return None
    for k in ("prompt", "question", "instruction", "input", "query", "text"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    p = _extract_prompt_from_lmsys_row(ex)
    if p:
        return p
    return None


def _load_prompts(dataset: str, split: str, *, max_prompts: int, seed: int) -> List[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: datasets. Install with: pip install datasets") from e

    ds = load_dataset(dataset, split=split)
    rng = random.Random(int(seed))
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out: List[str] = []
    for i in idxs:
        p = _extract_prompt(ds[int(i)])
        if p:
            out.append(p)
        if len(out) >= int(max_prompts):
            break
    if not out:
        raise ValueError(f"No prompts extracted from dataset={dataset} split={split}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build archetype grid prompts via clustering")
    ap.add_argument("--dataset", type=str, default="lmsys/chatbot_arena_conversations")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max-prompts", type=int, default=50000, help="How many prompts to sample for clustering")
    ap.add_argument("--k", type=int, default=500, help="Number of archetypes/clusters")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-model", type=str, default=DEFAULT_CONTEXT_MODEL)
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "priors" / "archetype_grid_prompts.jsonl"))
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    prompts = _load_prompts(str(args.dataset), str(args.split), max_prompts=int(args.max_prompts), seed=int(args.seed))
    print(f"Loaded {len(prompts)} prompts for clustering.")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: sentence-transformers") from e

    enc = SentenceTransformer(str(args.context_model))
    X = enc.encode(prompts, batch_size=int(args.batch_size), show_progress_bar=True, normalize_embeddings=True)
    X = np.asarray(X, dtype=np.float32)

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise RuntimeError("Missing dependency: scikit-learn") from e

    k = int(args.k)
    km = MiniBatchKMeans(n_clusters=k, random_state=int(args.seed), batch_size=2048, n_init="auto")
    labels = km.fit_predict(X)
    centers = np.asarray(km.cluster_centers_, dtype=np.float32)
    # centers in embedding space; normalize for cosine distance
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)

    # Pick representative prompt per cluster: argmax cosine similarity to centroid
    best_idx: List[int] = [-1] * k
    best_sim: List[float] = [-1e9] * k
    for i, c in enumerate(labels):
        sim = float(X[i].dot(centers[int(c)]))
        if sim > best_sim[int(c)]:
            best_sim[int(c)] = sim
            best_idx[int(c)] = int(i)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    with out_path.open("w", encoding="utf-8") as f:
        for cid in range(k):
            i = best_idx[cid]
            if i < 0:
                continue
            row = {
                "run_id": run_id,
                "cluster_id": int(cid),
                "similarity": float(best_sim[cid]),
                "prompt": prompts[i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote archetype grid prompts: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


