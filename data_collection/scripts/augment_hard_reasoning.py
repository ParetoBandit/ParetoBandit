#!/usr/bin/env python3
"""
Augment the diverse prompt set with hard reasoning/math prompts.

Replaces formulaic BBH items (short boolean/sort/MCQ) with genuinely
challenging prompts from MATH, GPQA, TheoremQA, and GSM8K — the kind
of multi-step reasoning tasks that differentiate Gemini-2.5-Pro from
Gemini-2.5-Flash in a routing experiment.

Pipeline
--------
1. Load the existing ``diverse_5k.jsonl`` prompt set.
2. Identify the 500 shortest (most formulaic) BBH prompts for removal.
3. Load hard-reasoning candidates from four new sources:
   - **MATH** (Level 4+5 competition math)
   - **TheoremQA** (graduate-level theorem application)
   - **GPQA** (graduate-level science, open-ended format)
   - **GSM8K** (longer multi-step word problems)
4. Quality-filter, embed, and deduplicate against the surviving set.
5. Friction-score via Ollama (optional, ``--no-friction`` to skip).
6. Select 500 replacements stratified by friction tier.
7. Write the updated ``diverse_5k.jsonl``.

Usage
-----
    # Full pipeline (with friction scoring):
    python data_collection/scripts/augment_hard_reasoning.py

    # Skip friction scoring (faster, assign uniform tiers):
    python data_collection/scripts/augment_hard_reasoning.py --no-friction

    # Dry run — report what would change without writing:
    python data_collection/scripts/augment_hard_reasoning.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit.config import PROMPTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Replacement budget ────────────────────────────────────────────────
N_REPLACE = 500

# ── Source quotas for the 500 new prompts ─────────────────────────────
#   MATH dominates because competition math is the #1 differentiator
#   between Flash and Pro.  TheoremQA and GPQA add graduate-level
#   reasoning diversity.  GSM8K contributes the longer word problems
#   that require sustained chain-of-thought.
SOURCE_QUOTAS: Dict[str, int] = {
    "math": 200,
    "theoremqa": 120,
    "gpqa": 100,
    "gsm8k": 80,
}

# ── Quality filter thresholds (same as build_diverse_prompt_set.py) ──
MIN_PROMPT_LEN = 20
MAX_PROMPT_LEN = 5_000
MIN_ASCII_RATIO = 0.5

# ── Friction scoring (reuse constants from build script) ─────────────
FRICTION_MODEL_A = "llama3.2:3b"
FRICTION_MODEL_B = "gemma:2b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MAX_TOKENS = 50
OLLAMA_TIMEOUT = 120


# =====================================================================
#  Stage 1 — Identify BBH prompts to remove
# =====================================================================

def identify_bbh_to_remove(
    prompts: List[Dict],
    n_remove: int,
) -> set[int]:
    """Return indices of the ``n_remove`` most formulaic BBH prompts.

    Selection criteria (applied in order until budget is met):
    1. Shortest prompts first — short BBH items are almost always
       template-fill boolean/sort/MCQ tasks.
    2. Break ties by lowest friction (least model disagreement = least
       routing-informative).
    """
    bbh_with_idx = [
        (i, p) for i, p in enumerate(prompts) if p["source"] == "bbh"
    ]
    bbh_with_idx.sort(key=lambda x: (len(x[1]["prompt"]), x[1]["friction"]))
    return {i for i, _ in bbh_with_idx[:n_remove]}


# =====================================================================
#  Stage 2 — Load hard-reasoning sources
# =====================================================================

def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def _quality_ok(text: str) -> bool:
    return (
        MIN_PROMPT_LEN <= len(text) <= MAX_PROMPT_LEN
        and _ascii_ratio(text) >= MIN_ASCII_RATIO
    )


def _strip_mcq_options(question: str) -> str:
    """Convert an MCQ question to open-ended by removing answer options.

    GPQA questions embed options like ``(A) ... (B) ... (C) ... (D) ...``
    at the end.  Stripping them forces the model to reason from scratch
    rather than pattern-match on option labels.
    """
    cleaned = re.split(r"\n\s*\(?[A-D]\)", question)[0].strip()
    if len(cleaned) < MIN_PROMPT_LEN:
        return question.strip()
    return cleaned


def load_math_hard(limit: int = 3000) -> List[str]:
    """Load Level 4+5 problems from the MATH benchmark."""
    from datasets import load_dataset

    prompts: List[str] = []
    for row in load_dataset(
        "DigitalLearningGmbH/MATH-lighteval", split="test", streaming=True,
    ):
        if len(prompts) >= limit:
            break
        if row.get("level") not in ("Level 4", "Level 5"):
            continue
        text = row["problem"].strip()
        if _quality_ok(text):
            prompts.append(text)
    return prompts


def load_theoremqa(limit: int = 800) -> List[str]:
    """Load TheoremQA — graduate-level theorem application problems."""
    from datasets import load_dataset

    prompts: List[str] = []
    for row in load_dataset(
        "TIGER-Lab/TheoremQA", split="test", streaming=True,
    ):
        if len(prompts) >= limit:
            break
        text = row["Question"].strip()
        if row.get("Picture"):
            continue
        if _quality_ok(text):
            prompts.append(text)
    return prompts


def load_gpqa(limit: int = 500) -> List[str]:
    """Load GPQA-main — graduate-level science questions (open-ended)."""
    from datasets import load_dataset

    prompts: List[str] = []
    for row in load_dataset(
        "Idavidrein/gpqa", "gpqa_main", split="train", streaming=True,
    ):
        if len(prompts) >= limit:
            break
        raw = row["Question"].strip()
        text = _strip_mcq_options(raw)
        if _quality_ok(text):
            prompts.append(text)
    return prompts


def load_gsm8k_hard(limit: int = 500) -> List[str]:
    """Load the longest GSM8K problems (multi-step word problems).

    GSM8K problems vary in complexity.  Longer problems correlate with
    more reasoning steps, so we sort by length and take the top.
    """
    from datasets import load_dataset

    all_qs: List[str] = []
    for row in load_dataset(
        "openai/gsm8k", "main", split="test", streaming=True,
    ):
        text = row["question"].strip()
        if _quality_ok(text):
            all_qs.append(text)
    all_qs.sort(key=len, reverse=True)
    return all_qs[:limit]


# =====================================================================
#  Stage 3 — Embed + deduplicate against surviving set
# =====================================================================

def embed_and_dedup_against(
    candidates: List[str],
    existing_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    threshold: float = 0.85,
) -> tuple[List[str], np.ndarray]:
    """Embed candidates and remove any that are too similar to surviving prompts.

    Parameters
    ----------
    candidates:
        New candidate prompt texts.
    existing_embeddings:
        Pre-computed embeddings of the surviving (non-removed) prompt set.
    embedder:
        SentenceTransformer model.
    threshold:
        Cosine similarity above which a candidate is considered a duplicate.

    Returns
    -------
    Deduplicated candidate texts and their embeddings.
    """
    logger.info(f"  Encoding {len(candidates)} candidates ...")
    cand_embs = embedder.encode(candidates, show_progress_bar=True, batch_size=256)

    # Cross-similarity: each candidate vs every surviving prompt
    logger.info("  Computing cross-similarity ...")
    max_sim = np.zeros(len(candidates))
    batch_size = 512
    for start in range(0, len(existing_embeddings), batch_size):
        end = min(start + batch_size, len(existing_embeddings))
        sims = cosine_similarity(cand_embs, existing_embeddings[start:end])
        batch_max = sims.max(axis=1)
        max_sim = np.maximum(max_sim, batch_max)

    keep = max_sim < threshold
    # Also deduplicate among themselves
    kept_indices: List[int] = []
    covered = np.zeros(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        if not keep[i] or covered[i]:
            continue
        kept_indices.append(i)
        intra_sims = cosine_similarity([cand_embs[i]], cand_embs)[0]
        covered[intra_sims > threshold] = True

    unique = [candidates[i] for i in kept_indices]
    unique_embs = cand_embs[kept_indices]
    logger.info(
        f"  Deduplicated {len(candidates)} → {len(unique)} "
        f"(cross-threshold={threshold})"
    )
    return unique, unique_embs


# =====================================================================
#  Stage 4 — Friction scoring (reuse from build script)
# =====================================================================

def _ollama_generate(prompt: str, model: str, base_url: str) -> Optional[str]:
    import requests

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


def compute_friction_scores(
    prompts: List[str],
    embedder: SentenceTransformer,
    model_a: str,
    model_b: str,
    workers: int,
    base_url: str = OLLAMA_BASE_URL,
) -> np.ndarray:
    """Score inter-model disagreement for each prompt via Ollama."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _gen_all(texts: List[str], model: str) -> List[Optional[str]]:
        results: List[Optional[str]] = [None] * len(texts)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_ollama_generate, t, model, base_url): i
                for i, t in enumerate(texts)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                results[idx] = fut.result()
                done += 1
                if done % 50 == 0 or done == len(texts):
                    logger.info(f"    [{model}] {done}/{len(texts)}")
        return results

    logger.info(f"  Generating responses from {model_a} ...")
    resp_a = _gen_all(prompts, model_a)
    logger.info(f"  Generating responses from {model_b} ...")
    resp_b = _gen_all(prompts, model_b)

    safe_a = [r if r else "" for r in resp_a]
    safe_b = [r if r else "" for r in resp_b]

    logger.info("  Embedding responses ...")
    emb_a = embedder.encode(safe_a, show_progress_bar=True, batch_size=256)
    emb_b = embedder.encode(safe_b, show_progress_bar=True, batch_size=256)

    dot = np.sum(emb_a * emb_b, axis=1)
    norm_a = np.linalg.norm(emb_a, axis=1)
    norm_b = np.linalg.norm(emb_b, axis=1)
    cos_sim = dot / (norm_a * norm_b + 1e-12)
    friction = 1.0 - cos_sim

    failed = np.array(
        [(a is None or b is None) for a, b in zip(resp_a, resp_b)]
    )
    if failed.any():
        median = float(np.median(friction[~failed])) if (~failed).any() else 0.5
        friction[failed] = median
        logger.info(f"  {failed.sum()} failures → assigned median friction")

    return friction


# =====================================================================
#  Stage 5 — Friction-stratified selection
# =====================================================================

def select_stratified(
    prompts: List[str],
    embeddings: np.ndarray,
    friction: Optional[np.ndarray],
    n_select: int,
    source_label: str,
    n_strata: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """Select prompts via KMeans centroids, stratified by friction tier."""
    tier_labels = ["very_easy", "easy", "medium", "hard", "very_hard"]

    if friction is None or len(friction) == 0:
        n_clusters = min(n_select, len(prompts))
        if n_clusters == 0:
            return []
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=max(256, n_clusters),
            random_state=seed,
        )
        kmeans.fit(embeddings)
        closest, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, embeddings,
        )
        per_tier = n_clusters // n_strata
        selected = []
        for i, idx in enumerate(closest):
            tier = tier_labels[min(i // per_tier, n_strata - 1)]
            selected.append({
                "prompt": prompts[idx],
                "friction": float("nan"),
                "tier": tier,
                "source": source_label,
            })
        return selected

    boundaries = np.percentile(friction, np.linspace(0, 100, n_strata + 1))
    per_tier = n_select // n_strata
    remainder = n_select - per_tier * n_strata

    selected: List[Dict] = []
    for t in range(n_strata):
        lo, hi = boundaries[t], boundaries[t + 1]
        mask = (friction >= lo) & (friction <= hi) if t == n_strata - 1 else (friction >= lo) & (friction < hi)
        tier_idx = np.where(mask)[0]
        if len(tier_idx) == 0:
            continue

        budget = per_tier + (1 if t < remainder else 0)
        budget = min(budget, len(tier_idx))
        if budget == 0:
            continue

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
                "source": source_label,
            })

    return selected


# =====================================================================
#  Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment diverse_5k.jsonl with hard reasoning prompts.",
    )
    parser.add_argument(
        "--input", type=str,
        default=str(PROMPTS_DIR / "diverse_5k.jsonl"),
        help="Input JSONL path (default: data_collection/prompts/diverse_5k.jsonl).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: overwrite input).",
    )
    parser.add_argument(
        "--n-replace", type=int, default=N_REPLACE,
        help=f"Number of BBH prompts to replace (default: {N_REPLACE}).",
    )
    parser.add_argument(
        "--no-friction", action="store_true",
        help="Skip friction scoring (assign uniform tiers).",
    )
    parser.add_argument(
        "--friction-workers", type=int, default=8,
        help="Concurrent Ollama requests per model (default: 8).",
    )
    parser.add_argument(
        "--ollama-url", type=str, default=OLLAMA_BASE_URL,
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.85,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(args.input)

    logger.info("=" * 65)
    logger.info("Augment Diverse Prompt Set — Hard Reasoning")
    logger.info("=" * 65)

    # ── 1. Load existing set ──────────────────────────────────────────
    logger.info("\n1. Loading existing prompt set ...")
    existing: List[Dict] = []
    with open(args.input) as f:
        for line in f:
            existing.append(json.loads(line))
    logger.info(f"   Loaded {len(existing)} prompts")

    from collections import Counter
    src_dist = Counter(p["source"] for p in existing)
    tier_dist = Counter(p["tier"] for p in existing)
    logger.info(f"   Sources: {dict(src_dist)}")
    logger.info(f"   Tiers:   {dict(tier_dist)}")

    # ── 2. Identify BBH to remove ────────────────────────────────────
    logger.info(f"\n2. Identifying {args.n_replace} formulaic BBH prompts to remove ...")
    remove_idx = identify_bbh_to_remove(existing, args.n_replace)
    removed = [existing[i] for i in remove_idx]
    removed_lengths = [len(p["prompt"]) for p in removed]
    logger.info(
        f"   Removing {len(remove_idx)} BBH prompts "
        f"(length range: {min(removed_lengths)}-{max(removed_lengths)} chars)"
    )
    removed_tiers = Counter(p["tier"] for p in removed)
    logger.info(f"   Removed tier distribution: {dict(removed_tiers)}")

    survivors = [p for i, p in enumerate(existing) if i not in remove_idx]
    logger.info(f"   Survivors: {len(survivors)}")

    # ── 3. Load hard-reasoning sources ────────────────────────────────
    logger.info("\n3. Loading hard-reasoning datasets ...")
    sources: Dict[str, List[str]] = {}

    logger.info("   [MATH] Level 4+5 competition math ...")
    sources["math"] = load_math_hard()
    logger.info(f"   [MATH] Loaded: {len(sources['math'])}")

    logger.info("   [TheoremQA] Graduate-level theorems ...")
    sources["theoremqa"] = load_theoremqa()
    logger.info(f"   [TheoremQA] Loaded: {len(sources['theoremqa'])}")

    logger.info("   [GPQA] Graduate-level science (open-ended) ...")
    sources["gpqa"] = load_gpqa()
    logger.info(f"   [GPQA] Loaded: {len(sources['gpqa'])}")

    logger.info("   [GSM8K] Multi-step word problems (hardest) ...")
    sources["gsm8k"] = load_gsm8k_hard()
    logger.info(f"   [GSM8K] Loaded: {len(sources['gsm8k'])}")

    total_candidates = sum(len(v) for v in sources.values())
    logger.info(f"   Total candidates: {total_candidates}")

    # ── 4. Embed survivors + deduplicate candidates against them ──────
    logger.info("\n4. Embedding survivors and deduplicating candidates ...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    survivor_texts = [p["prompt"] for p in survivors]
    logger.info(f"   Encoding {len(survivor_texts)} survivors ...")
    survivor_embs = embedder.encode(
        survivor_texts, show_progress_bar=True, batch_size=256,
    )

    deduped: Dict[str, tuple[List[str], np.ndarray]] = {}
    for src_name, pool in sources.items():
        if not pool:
            continue
        logger.info(f"   [{src_name}] Deduplicating {len(pool)} against survivors ...")
        texts, embs = embed_and_dedup_against(
            pool, survivor_embs, embedder, args.dedup_threshold,
        )
        deduped[src_name] = (texts, embs)
        logger.info(f"   [{src_name}] {len(pool)} → {len(texts)} unique")

    # ── 5. Friction scoring (optional) ────────────────────────────────
    friction_map: Dict[str, Optional[np.ndarray]] = {k: None for k in deduped}

    if not args.no_friction:
        total_to_score = sum(len(v[0]) for v in deduped.values())
        logger.info(f"\n5. Friction scoring {total_to_score} candidates via Ollama ...")
        logger.info(f"   Model A: {FRICTION_MODEL_A}")
        logger.info(f"   Model B: {FRICTION_MODEL_B}")
        logger.info(f"   Workers: {args.friction_workers}")

        for model_tag in [FRICTION_MODEL_A, FRICTION_MODEL_B]:
            logger.info(f"   Warming up {model_tag} ...")
            _ollama_generate("Say hello.", model_tag, args.ollama_url)

        for src_name, (texts, embs) in deduped.items():
            if not texts:
                continue
            logger.info(f"   [{src_name}] Scoring {len(texts)} candidates ...")
            t0 = time.perf_counter()
            friction = compute_friction_scores(
                texts, embedder, FRICTION_MODEL_A, FRICTION_MODEL_B,
                args.friction_workers, args.ollama_url,
            )
            elapsed = time.perf_counter() - t0
            friction_map[src_name] = friction
            logger.info(
                f"   [{src_name}] Done in {elapsed:.0f}s — "
                f"mean={friction.mean():.3f}  std={friction.std():.3f}"
            )
    else:
        logger.info("\n5. Friction scoring SKIPPED (--no-friction)")

    # ── 6. Select replacements per source ─────────────────────────────
    logger.info(f"\n6. Selecting {args.n_replace} replacement prompts ...")

    # Adjust quotas if any source has fewer candidates than its quota
    quotas = dict(SOURCE_QUOTAS)
    deficit = 0
    for src_name, quota in quotas.items():
        if src_name not in deduped:
            deficit += quota
            quotas[src_name] = 0
        elif len(deduped[src_name][0]) < quota:
            deficit += quota - len(deduped[src_name][0])
            quotas[src_name] = len(deduped[src_name][0])

    # Redistribute deficit to sources with surplus
    if deficit > 0:
        surplus_sources = [
            s for s in quotas
            if s in deduped and len(deduped[s][0]) > quotas[s]
        ]
        if surplus_sources:
            per_src = deficit // len(surplus_sources)
            for s in surplus_sources:
                quotas[s] += per_src
            quotas[surplus_sources[0]] += deficit - per_src * len(surplus_sources)

    new_prompts: List[Dict] = []
    for src_name, (texts, embs) in deduped.items():
        budget = quotas.get(src_name, 0)
        if budget == 0 or not texts:
            continue
        logger.info(f"   [{src_name}] Selecting {budget} from {len(texts)} ...")
        selected = select_stratified(
            texts, embs, friction_map.get(src_name),
            budget, source_label=src_name, seed=args.seed,
        )
        new_prompts.extend(selected)

    logger.info(f"   Total new prompts: {len(new_prompts)}")

    # ── 7. Merge and write ────────────────────────────────────────────
    final = survivors + new_prompts
    logger.info(f"\n7. Final set: {len(final)} prompts")

    final_src = Counter(p["source"] for p in final)
    final_tier = Counter(p["tier"] for p in final)
    logger.info(f"   Sources: {dict(final_src)}")
    logger.info(f"   Tiers:   {dict(final_tier)}")

    # Power analysis summary
    _report_power_analysis(final)

    if args.dry_run:
        logger.info("\n   DRY RUN — no file written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in final:
            f.write(json.dumps(item) + "\n")
    logger.info(f"\n8. Wrote {len(final)} prompts to {output_path}")
    logger.info("Done.")


def _report_power_analysis(prompts: List[Dict]) -> None:
    """Log a statistical power estimate for the holdout split."""
    n_total = len(prompts)

    # Assume the same split ratio as the existing dataset (25/38/37)
    holdout_frac = 0.37
    n_holdout = int(n_total * holdout_frac)

    hard_reasoning_sources = {"math", "theoremqa", "gpqa", "gsm8k"}
    n_hard_new = sum(1 for p in prompts if p["source"] in hard_reasoning_sources)
    n_hard_bbh_remaining = sum(
        1 for p in prompts
        if p["source"] == "bbh" and p["tier"] in ("hard", "very_hard")
    )

    # Hard reasoning prompts that differentiate Pro from Flash
    text_keywords = {
        "calculate", "equation", "math", "solve", "proof", "probability",
        "derive", "analyze", "reasoning", "logic", "step by step",
        "algorithm", "implement", "debug", "optimize",
    }
    n_differentiator = sum(
        1 for p in prompts
        if (
            p["source"] in hard_reasoning_sources
            or (p["source"] == "bbh" and p["tier"] in ("hard", "very_hard"))
            or (
                p["tier"] in ("hard", "very_hard")
                and any(kw in p["prompt"].lower() for kw in text_keywords)
            )
        )
    )

    n_diff_holdout = int(n_differentiator * holdout_frac)

    # Two-sample power estimate (Pro vs Flash reward difference)
    # Assumptions: delta=0.10, sigma=0.30, alpha=0.05, two-sided
    delta, sigma, z_alpha, z_beta = 0.10, 0.30, 1.96, 0.84
    n_required = int(((z_alpha + z_beta) ** 2 * 2 * sigma ** 2) / delta ** 2)

    logger.info("\n   ── Statistical Power Estimate ──")
    logger.info(f"   Total prompts:           {n_total}")
    logger.info(f"   Est. holdout (37%):      {n_holdout}")
    logger.info(f"   New hard-reasoning:      {n_hard_new}")
    logger.info(f"   Remaining hard BBH:      {n_hard_bbh_remaining}")
    logger.info(f"   Pro-differentiator total:{n_differentiator}")
    logger.info(f"   Est. in holdout:         {n_diff_holdout}")
    logger.info(f"   Required (δ=0.10, σ=0.30, α=0.05): {n_required}")
    if n_diff_holdout >= n_required:
        logger.info(f"   ✓ Sufficient for statistical significance")
    else:
        logger.info(
            f"   ⚠ Marginal — consider increasing total prompts "
            f"or enriching hard-reasoning fraction"
        )


if __name__ == "__main__":
    main()
