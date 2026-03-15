#!/usr/bin/env python3
"""
Build a diverse, friction-stratified prompt set for BanditGPT.

Pipeline
--------
1. Load candidates from HuggingFace (LMSYS, WildFB, BBH) + local LMSYS file.
2. Quality filter + SentenceTransformer semantic deduplication.
3. **Friction scoring** via two local Ollama models (inter-model disagreement).
   Two small, architecturally distinct models generate short responses to each
   prompt.  Friction = 1 − cosine_similarity(response_A_emb, response_B_emb).
   High friction → models disagree → the prompt is likely to expose
   routing-relevant performance differences.
4. Friction-stratified KMeans centroid selection for difficulty diversity.
5. Output 5,000 prompts as JSONL.

Friction scoring is multi-threaded (``--friction-workers``) with one
ThreadPoolExecutor per model to avoid Ollama model-swap overhead.

Usage
-----
    # Full pipeline (embedding + friction + selection):
    python data_collection/scripts/build_diverse_prompt_set.py

    # Skip friction scoring (pure diversity clustering):
    python data_collection/scripts/build_diverse_prompt_set.py --no-friction

    # Tune concurrency and output:
    python data_collection/scripts/build_diverse_prompt_set.py \\
        --friction-workers 12 \\
        --output data_collection/prompts/diverse_5k.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config import (
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    LMSYS_BATTLES_PATH,
    PROMPTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Quality filter thresholds ────────────────────────────────────────────
MIN_PROMPT_LEN = 20
MAX_PROMPT_LEN = 5_000
MIN_ASCII_RATIO = 0.5

# ── Default Ollama friction models ───────────────────────────────────────
#   Two small, architecturally distinct models for response disagreement.
FRICTION_MODEL_A = "llama3.2:3b"
FRICTION_MODEL_B = "gemma:2b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MAX_TOKENS = 50
OLLAMA_TIMEOUT = 120


# =====================================================================
#  Stage 1 — Source Loading
# =====================================================================

def load_existing_prompts() -> set[str]:
    """Return prompts already in the reward dataset (to avoid overlap)."""
    existing: set[str] = set()
    for gz_path in [DEV_DATA_PATH_ALL_MODELS, HOLDOUT_DATA_PATH_ALL_MODELS]:
        if not gz_path.exists():
            continue
        with gzip.open(gz_path, "rt") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    existing.add(entry["prompt"])
    return existing


def load_local_lmsys(path: Path, limit: int = 15_000) -> list[str]:
    """Load prompts from the local LMSYS battles JSONL."""
    if not path.exists():
        logger.warning(f"  Local LMSYS file not found: {path}")
        return []
    seen: set[str] = set()
    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            if len(prompts) >= limit:
                break
            data = json.loads(line)
            text = data.get("prompt", "")
            if not text:
                try:
                    text = data["conversation"][0]["content"]
                except (KeyError, IndexError, TypeError):
                    continue
            text = text.strip()
            if text and text not in seen:
                seen.add(text)
                prompts.append(text)
    return prompts


def load_hf_lmsys(limit: int = 10_000) -> list[str]:
    """Load English prompts from LMSYS Chatbot Arena via HuggingFace."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "lmsys/chatbot_arena_conversations", split="train", streaming=True,
        )
        prompts: list[str] = []
        for row in ds:
            if len(prompts) >= limit:
                break
            if row.get("language") != "en":
                continue
            try:
                text = row["conversation_a"][0]["content"].strip()
            except (KeyError, IndexError, TypeError):
                continue
            if text:
                prompts.append(text)
        return prompts
    except Exception as exc:
        logger.warning(f"  HF LMSYS load failed ({exc}); falling back to local file")
        return []


def load_hf_wildchat(limit: int = 10_000) -> list[str]:
    """Load English user prompts from LMSYS WildChat-1M.

    Uses ``lmsys/lmsys-chat-1m`` — a large-scale corpus of real user
    conversations with diverse LLMs.  We extract only the first user
    turn of English conversations.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            if len(prompts) >= limit:
                break
            if row.get("language") != "English":
                continue
            try:
                text = row["conversation"][0]["content"].strip()
            except (KeyError, IndexError, TypeError):
                continue
            if text:
                prompts.append(text)
        return prompts
    except Exception as exc:
        logger.warning(f"  HF WildChat load failed ({exc})")
        return []


def load_hf_bbh() -> list[str]:
    """Load all questions from BIG-Bench Hard (28 sub-tasks)."""
    try:
        from datasets import load_dataset, get_dataset_config_names

        configs = get_dataset_config_names("Joschka/big_bench_hard")
        prompts: list[str] = []
        for cfg in configs:
            ds = load_dataset("Joschka/big_bench_hard", cfg, split=cfg)
            for row in ds:
                q = row.get("question", "").strip()
                if q:
                    prompts.append(q)
        return prompts
    except Exception as exc:
        logger.warning(f"  HF BBH load failed ({exc})")
        return []


# =====================================================================
#  Stage 2 — Quality Filtering
# =====================================================================

def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def quality_filter(prompts: list[str]) -> list[str]:
    """Apply length and ASCII-ratio quality gates."""
    return [
        p for p in prompts
        if MIN_PROMPT_LEN <= len(p) <= MAX_PROMPT_LEN
        and _ascii_ratio(p) >= MIN_ASCII_RATIO
    ]


# =====================================================================
#  Stage 3 — Embedding + Semantic Deduplication
# =====================================================================

def embed_and_dedup(
    prompts: list[str],
    embedder: SentenceTransformer,
    dedup_threshold: float = 0.85,
) -> tuple[list[str], np.ndarray]:
    """Embed prompts, then greedily remove near-duplicates.

    Returns the deduplicated prompts and their embeddings.
    """
    logger.info(f"  Encoding {len(prompts)} prompts ...")
    embeddings = embedder.encode(prompts, show_progress_bar=True, batch_size=256)

    keep_indices: list[int] = []
    covered = np.zeros(len(prompts), dtype=bool)

    for i in range(len(prompts)):
        if not covered[i]:
            keep_indices.append(i)
            sims = cosine_similarity([embeddings[i]], embeddings)[0]
            covered[sims > dedup_threshold] = True

    unique_prompts = [prompts[i] for i in keep_indices]
    unique_embeddings = embeddings[keep_indices]
    logger.info(
        f"  Deduplicated {len(prompts)} → {len(unique_prompts)} "
        f"(threshold={dedup_threshold})"
    )
    return unique_prompts, unique_embeddings


# =====================================================================
#  Stage 4 — Friction Scoring (Ollama, multi-threaded)
# =====================================================================

def _ollama_generate(
    prompt: str, model: str, base_url: str = OLLAMA_BASE_URL,
) -> str | None:
    """Generate a short response from a local Ollama model."""
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": OLLAMA_MAX_TOKENS,
                    "temperature": 0.7,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception:
        return None


def _generate_all(
    prompts: Sequence[str],
    model: str,
    workers: int,
    base_url: str = OLLAMA_BASE_URL,
) -> list[str | None]:
    """Generate responses for all prompts using one Ollama model.

    Processes prompts concurrently with a thread pool to saturate the
    Ollama server.  Model is held constant to avoid swap overhead.
    """
    responses: list[str | None] = [None] * len(prompts)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_ollama_generate, p, model, base_url): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            responses[idx] = future.result()
            done += 1
            if done % 200 == 0 or done == len(prompts):
                logger.info(
                    f"    [{model}] {done}/{len(prompts)} "
                    f"({100 * done / len(prompts):.0f}%)"
                )

    return responses


def compute_friction_scores(
    prompts: list[str],
    embedder: SentenceTransformer,
    model_a: str,
    model_b: str,
    workers: int,
    base_url: str = OLLAMA_BASE_URL,
) -> np.ndarray:
    """Score inter-model disagreement for each prompt.

    For every prompt, two Ollama models generate short responses.  Their
    embeddings are compared via cosine similarity.  Friction is defined as
    ``1 − cos_sim(emb_A, emb_B)``, ranging from 0 (identical responses)
    to ~2 (maximally opposed).  In practice values fall in [0, 1].

    Parameters
    ----------
    prompts : list[str]
        Prompt texts to score.
    embedder : SentenceTransformer
        Shared encoder for response embeddings.
    model_a, model_b : str
        Ollama model tags for the two friction probes.
    workers : int
        Concurrent HTTP requests per model.
    base_url : str
        Ollama server URL.

    Returns
    -------
    np.ndarray
        Friction score per prompt, shape ``(len(prompts),)``.
    """
    logger.info(f"  Generating responses from {model_a} ({workers} threads) ...")
    responses_a = _generate_all(prompts, model_a, workers, base_url)

    logger.info(f"  Generating responses from {model_b} ({workers} threads) ...")
    responses_b = _generate_all(prompts, model_b, workers, base_url)

    # Replace None responses with empty string so embedding doesn't crash
    safe_a = [r if r else "" for r in responses_a]
    safe_b = [r if r else "" for r in responses_b]

    logger.info("  Embedding responses ...")
    emb_a = embedder.encode(safe_a, show_progress_bar=True, batch_size=256)
    emb_b = embedder.encode(safe_b, show_progress_bar=True, batch_size=256)

    # Per-prompt cosine similarity → friction
    dot = np.sum(emb_a * emb_b, axis=1)
    norm_a = np.linalg.norm(emb_a, axis=1)
    norm_b = np.linalg.norm(emb_b, axis=1)
    cos_sim = dot / (norm_a * norm_b + 1e-12)
    friction = 1.0 - cos_sim

    # Mark prompts where either model failed as median friction
    failed = np.array(
        [(a is None or b is None) for a, b in zip(responses_a, responses_b)]
    )
    if failed.any():
        median = float(np.median(friction[~failed])) if (~failed).any() else 0.5
        friction[failed] = median
        logger.info(f"  {failed.sum()} prompts had generation failures → assigned median friction")

    return friction


# =====================================================================
#  Stage 5 — Friction-Stratified Centroid Selection
# =====================================================================

def select_stratified(
    prompts: list[str],
    embeddings: np.ndarray,
    friction: np.ndarray | None,
    n_to_select: int,
    n_strata: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Select prompts via KMeans centroids, stratified by friction tier.

    When friction scores are available, the pool is split into
    ``n_strata`` equal-frequency tiers.  Each tier is allocated a
    proportional share of the ``n_to_select`` budget, and KMeans centroid
    selection runs independently within each tier.  This prevents the
    final set from being dominated by easy or hard prompts.

    Without friction scores, falls back to global KMeans selection.

    Returns a list of dicts ``{"prompt": str, "friction": float, "tier": str}``.
    """
    if friction is None or len(friction) == 0:
        # Pure diversity clustering (no friction info)
        n_clusters = min(n_to_select, len(prompts))
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=1024, random_state=seed,
        )
        kmeans.fit(embeddings)
        closest, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, embeddings,
        )
        return [
            {"prompt": prompts[i], "friction": float("nan"), "tier": "unknown"}
            for i in closest
        ]

    # Assign strata based on friction percentiles
    boundaries = np.percentile(
        friction, np.linspace(0, 100, n_strata + 1),
    )
    tier_labels = ["very_easy", "easy", "medium", "hard", "very_hard"][:n_strata]

    selected: list[dict] = []
    per_tier = n_to_select // n_strata
    remainder = n_to_select - per_tier * n_strata

    for t in range(n_strata):
        lo, hi = boundaries[t], boundaries[t + 1]
        if t == n_strata - 1:
            mask = (friction >= lo) & (friction <= hi)
        else:
            mask = (friction >= lo) & (friction < hi)

        tier_idx = np.where(mask)[0]
        if len(tier_idx) == 0:
            continue

        budget = per_tier + (1 if t < remainder else 0)
        budget = min(budget, len(tier_idx))

        tier_emb = embeddings[tier_idx]
        kmeans = MiniBatchKMeans(
            n_clusters=budget, batch_size=max(256, budget), random_state=seed,
        )
        kmeans.fit(tier_emb)
        closest, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, tier_emb,
        )

        for local_i in closest:
            global_i = tier_idx[local_i]
            selected.append({
                "prompt": prompts[global_i],
                "friction": round(float(friction[global_i]), 6),
                "tier": tier_labels[t],
            })

    return selected


# =====================================================================
#  Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a diverse, friction-stratified prompt set.",
    )
    parser.add_argument(
        "--n-total", type=int, default=5000,
        help="Total prompts to select (default: 5000).",
    )
    parser.add_argument(
        "--n-lmsys", type=int, default=2000,
        help="Quota from LMSYS sources (default: 2000).",
    )
    parser.add_argument(
        "--n-wildchat", type=int, default=1500,
        help="Quota from WildChat (default: 1500).",
    )
    parser.add_argument(
        "--n-bbh", type=int, default=1500,
        help="Quota from BIG-Bench Hard (default: 1500).",
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.85,
        help="Cosine-similarity threshold for semantic dedup (default: 0.85).",
    )
    parser.add_argument(
        "--no-friction", action="store_true",
        help="Skip friction scoring (pure diversity clustering only).",
    )
    parser.add_argument(
        "--friction-model-a", type=str, default=FRICTION_MODEL_A,
        help=f"First Ollama friction probe model (default: {FRICTION_MODEL_A}).",
    )
    parser.add_argument(
        "--friction-model-b", type=str, default=FRICTION_MODEL_B,
        help=f"Second Ollama friction probe model (default: {FRICTION_MODEL_B}).",
    )
    parser.add_argument(
        "--friction-workers", type=int, default=8,
        help="Concurrent Ollama requests per model (default: 8).",
    )
    parser.add_argument(
        "--ollama-url", type=str, default=OLLAMA_BASE_URL,
        help=f"Ollama server URL (default: {OLLAMA_BASE_URL}).",
    )
    parser.add_argument(
        "--output", type=str,
        default="data_collection/prompts/diverse_5k.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    logger.info("=" * 65)
    logger.info("Build Diverse Prompt Set for BanditGPT")
    logger.info("=" * 65)

    # ── 1. Exclude existing reward prompts ────────────────────────────
    logger.info("\n1. Loading existing prompts to exclude ...")
    existing = load_existing_prompts()
    logger.info(f"   {len(existing)} prompts already in reward data")

    # ── 2. Load sources ──────────────────────────────────────────────
    logger.info("\n2. Loading source pools ...")

    logger.info("   [LMSYS] Local file ...")
    lmsys_local = load_local_lmsys(LMSYS_BATTLES_PATH, limit=15_000)
    logger.info(f"   [LMSYS] Local: {len(lmsys_local)}")

    logger.info("   [LMSYS] HuggingFace ...")
    lmsys_hf = load_hf_lmsys(limit=10_000)
    logger.info(f"   [LMSYS] HF: {len(lmsys_hf)}")

    lmsys_combined = list({p: None for p in lmsys_local + lmsys_hf}.keys())
    logger.info(f"   [LMSYS] Combined unique: {len(lmsys_combined)}")

    logger.info("   [WildChat] HuggingFace (lmsys/lmsys-chat-1m) ...")
    wildchat_raw = load_hf_wildchat(limit=10_000)
    logger.info(f"   [WildChat] Loaded: {len(wildchat_raw)}")

    logger.info("   [BBH] HuggingFace ...")
    bbh_raw = load_hf_bbh()
    logger.info(f"   [BBH] Loaded: {len(bbh_raw)}")

    # ── 3. Quality filter + exclude existing ─────────────────────────
    logger.info("\n3. Quality filtering + excluding existing prompts ...")
    sources: dict[str, list[str]] = {
        "lmsys": quality_filter([p for p in lmsys_combined if p not in existing]),
        "wildchat": quality_filter([p for p in wildchat_raw if p not in existing]),
        "bbh": quality_filter([p for p in bbh_raw if p not in existing]),
    }
    for name, pool in sources.items():
        logger.info(f"   [{name}] After filters: {len(pool)}")

    # ── 4. Embed + semantic dedup per source ─────────────────────────
    logger.info("\n4. Embedding + semantic deduplication ...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    deduped_prompts: dict[str, list[str]] = {}
    deduped_embeddings: dict[str, np.ndarray] = {}

    for name, pool in sources.items():
        if not pool:
            logger.warning(f"   [{name}] Empty pool — skipping")
            deduped_prompts[name] = []
            deduped_embeddings[name] = np.empty((0, 0))
            continue
        logger.info(f"   [{name}]")
        prompts, embs = embed_and_dedup(pool, embedder, args.dedup_threshold)
        deduped_prompts[name] = prompts
        deduped_embeddings[name] = embs

    # ── 5. Compute quotas and oversample centroids ─────────────────
    quotas: dict[str, int] = {
        "lmsys": args.n_lmsys,
        "wildchat": args.n_wildchat,
        "bbh": args.n_bbh,
    }

    # Redistribute quota from empty sources
    empty_surplus = sum(
        quotas[name] for name in sources if not deduped_prompts[name]
    )
    active_sources = [name for name in sources if deduped_prompts[name]]
    if empty_surplus > 0 and active_sources:
        per_active = empty_surplus // len(active_sources)
        for name in active_sources:
            quotas[name] += per_active
        quotas[active_sources[0]] += empty_surplus - per_active * len(active_sources)

    OVERSAMPLE = 2  # cluster into 2× the quota, then friction-refine
    logger.info(f"\n5. Oversampled centroid selection (quotas: {quotas}, {OVERSAMPLE}× oversample) ...")

    # Per-source: cluster into oversample×quota centroids
    candidate_prompts: dict[str, list[str]] = {}
    candidate_embeddings: dict[str, np.ndarray] = {}

    for name in sources:
        prompts = deduped_prompts[name]
        embs = deduped_embeddings[name]
        if not prompts:
            continue
        n_candidates = min(quotas[name] * OVERSAMPLE, len(prompts))
        logger.info(f"   [{name}] Clustering {len(prompts)} → {n_candidates} candidates ...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_candidates, batch_size=max(256, n_candidates),
            random_state=args.seed,
        )
        kmeans.fit(embs)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embs)
        candidate_prompts[name] = [prompts[i] for i in closest]
        candidate_embeddings[name] = embs[closest]

    total_candidates = sum(len(v) for v in candidate_prompts.values())
    logger.info(f"   Total candidates for friction scoring: {total_candidates}")

    # ── 6. Friction scoring on candidates only (optional) ────────────
    friction_scores: dict[str, np.ndarray | None] = {
        name: None for name in sources
    }

    if not args.no_friction:
        logger.info(f"\n6. Friction scoring {total_candidates} candidates via Ollama ...")
        logger.info(f"   Model A : {args.friction_model_a}")
        logger.info(f"   Model B : {args.friction_model_b}")
        logger.info(f"   Workers : {args.friction_workers}")

        for model_tag in [args.friction_model_a, args.friction_model_b]:
            logger.info(f"   Warming up {model_tag} ...")
            _ollama_generate("Say hello.", model_tag, args.ollama_url)

        for name, prompts in candidate_prompts.items():
            if not prompts:
                continue
            logger.info(f"   [{name}] Scoring {len(prompts)} candidates ...")
            t0 = time.perf_counter()
            friction = compute_friction_scores(
                prompts,
                embedder,
                args.friction_model_a,
                args.friction_model_b,
                args.friction_workers,
                args.ollama_url,
            )
            elapsed = time.perf_counter() - t0
            friction_scores[name] = friction
            logger.info(
                f"   [{name}] Done in {elapsed:.0f}s — "
                f"mean={friction.mean():.3f}  std={friction.std():.3f}  "
                f"[{friction.min():.3f}, {friction.max():.3f}]"
            )
    else:
        logger.info("\n6. Friction scoring SKIPPED (--no-friction)")

    # ── 7. Final friction-stratified selection ────────────────────────
    logger.info(f"\n7. Final selection from candidates ...")
    all_selected: list[dict] = []

    for name in sources:
        if name not in candidate_prompts:
            continue
        prompts = candidate_prompts[name]
        embs = candidate_embeddings[name]
        budget = min(quotas[name], len(prompts))
        logger.info(f"   [{name}] Selecting {budget} from {len(prompts)} candidates ...")
        selected = select_stratified(
            prompts, embs, friction_scores.get(name), budget, seed=args.seed,
        )
        for item in selected:
            item["source"] = name
        all_selected.extend(selected)

    logger.info(f"   Total selected: {len(all_selected)}")

    # ── 8. Summary statistics ────────────────────────────────────────
    logger.info("\n8. Summary:")
    for src in sources:
        items = [s for s in all_selected if s["source"] == src]
        if not items:
            continue
        frictions = [s["friction"] for s in items if not np.isnan(s["friction"])]
        lengths = [len(s["prompt"]) for s in items]
        logger.info(f"   [{src}] n={len(items)}")
        logger.info(f"     Prompt length — mean={np.mean(lengths):.0f}  "
                     f"median={np.median(lengths):.0f}")
        if frictions:
            arr = np.array(frictions)
            logger.info(f"     Friction      — mean={arr.mean():.3f}  "
                         f"std={arr.std():.3f}  "
                         f"[{arr.min():.3f}, {arr.max():.3f}]")
        tier_counts = {}
        for s in items:
            tier_counts[s["tier"]] = tier_counts.get(s["tier"], 0) + 1
        logger.info(f"     Tiers         — {tier_counts}")

    # ── 9. Write output ──────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in all_selected:
            f.write(json.dumps(item) + "\n")
    logger.info(f"\n9. Wrote {len(all_selected)} prompts to {output_path}")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
