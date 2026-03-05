"""
Model-to-model transfer utilities for BanditGPT experiments.

Provides functions for computing reward-based neighbor similarity
(tetrachoric correlation), selecting transfer donors, and building
filtered warmup priors for leave-one-out experiments.

Used by:
    - experiments/06_figure/run_semantic_transfer.py  (Figure 6)
    - experiments/appendix/D_semantic_transfer_ablation/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from bandit_gpt.router import tetrachoric_corr


def get_provider(model_id: str) -> str:
    """Extract provider prefix from a model ID.

    Args:
        model_id: Slash-delimited model identifier
            (e.g. ``"openai/gpt-4.1"``).

    Returns:
        Provider string (e.g. ``"openai"``).  If *model_id* contains
        no ``/``, the full string is returned.
    """
    return model_id.split("/")[0] if "/" in model_id else model_id


def build_reward_vectors(
    data: List[Dict],
    models: List[str],
) -> Dict[str, np.ndarray]:
    """Binarize per-prompt rewards for tetrachoric correlation.

    For each prompt the model receives 1 if its reward is at or above
    the prompt-level median across all *models*, 0 otherwise.

    Args:
        data: List of dicts with ``"rewards"`` mapping model_id to
            float reward.
        models: Portfolio model IDs (defines the median reference set).

    Returns:
        Mapping from model_id to a binary ``np.ndarray`` of shape
        ``(len(data),)``.
    """
    vectors: Dict[str, list] = {m: [] for m in models}
    for item in data:
        rewards = [item["rewards"][m] for m in models]
        median_r = float(np.median(rewards))
        for m in models:
            vectors[m].append(1 if item["rewards"][m] >= median_r else 0)
    return {m: np.array(v, dtype=float) for m, v in vectors.items()}


def find_tetrachoric_neighbor(
    target: str,
    candidates: List[str],
    reward_vectors: Dict[str, np.ndarray],
    within_provider_only: bool = True,
) -> Tuple[Optional[str], float]:
    """Select the best transfer donor by tetrachoric correlation.

    When *within_provider_only* is ``True`` (the default), only
    candidates sharing the same provider prefix are eligible --
    matching the library's ``compute_correlation_families()`` design.

    Args:
        target: Model ID of the newcomer.
        candidates: Pool of potential donors (typically K-1 base
            models).
        reward_vectors: Output of :func:`build_reward_vectors`.
        within_provider_only: Restrict search to same-provider models.

    Returns:
        ``(best_neighbor, tetrachoric_sim)`` or ``(None, -1.0)`` when
        no eligible neighbor exists.
    """
    target_prov = get_provider(target)
    best_neighbor: Optional[str] = None
    best_corr = -1.0
    for cand in candidates:
        if cand == target:
            continue
        if within_provider_only and get_provider(cand) != target_prov:
            continue
        r_tet = tetrachoric_corr(reward_vectors[target], reward_vectors[cand])
        if not np.isnan(r_tet) and r_tet > best_corr:
            best_corr = r_tet
            best_neighbor = cand
    return (best_neighbor, float(best_corr))


def build_filtered_warmup(
    base_models: List[str],
    warmup_path: Path,
) -> Dict:
    """Load warmup priors and subset to *base_models*.

    The returned dict contains only the ``A`` and ``b`` matrices for
    models present in both *base_models* and the prior file, plus the
    observation count ``n``.

    Args:
        base_models: Model IDs to retain.
        warmup_path: Path to a ``.joblib`` warmup-prior artifact.

    Returns:
        Dict with keys ``"A"``, ``"b"``, ``"n"``.
    """
    raw = joblib.load(warmup_path)
    n = raw.get("n", raw.get("n_prompts", 20_000))
    A = {m: raw["A"][m].copy() for m in base_models if m in raw.get("A", {})}
    b = {m: raw["b"][m].copy() for m in base_models if m in raw.get("b", {})}
    return {"A": A, "b": b, "n": max(n, 1)}
