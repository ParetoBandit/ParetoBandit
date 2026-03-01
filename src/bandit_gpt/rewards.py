"""
Canonical reward extraction for BanditGPT.

Every data-loading path must derive per-response rewards through
:func:`extract_reward` so that the reward definition is consistent
across the entire codebase.

Reward signal
-------------
``mean(vote × confidence)`` across the multi-judge panel.

Each judge emits a binary *vote* (1 = pass, 0 = fail) and a scalar
*confidence* ∈ [0, 1].  The product ``vote × confidence`` maps to:

- High-confidence pass → ~0.95
- Low-confidence pass  → ~0.60
- Any fail             → 0.0  (vote = 0 zeroes the product)

Averaging across judges produces a continuous reward in [0, 1] that
preserves evaluative signal lost by the binary majority vote.

Under binary ``raw_score``, 58-66 % of prompts give identical rewards
for all K models.  Under ``mean(vote × confidence)``, this drops to
< 1 %, and best-arm entropy rises from ~1.1 bits to ~2.7 bits (K = 10).

Fallback
--------
If ``judge_details`` is absent (e.g. legacy RouteLLM battle data),
the function falls back to ``raw_score``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def extract_reward(entry: Dict[str, Any]) -> float:
    """Derive a continuous reward from a single data-file entry.

    Parameters
    ----------
    entry : dict
        A parsed JSONL record.  Expected fields:

        - ``judge_details`` (list[dict]): per-judge ``vote`` and
          ``confidence`` (preferred path).
        - ``raw_score`` (float): binary majority-vote fallback.

    Returns
    -------
    float
        Reward in [0, 1].  ``NaN`` when neither field is usable.
    """
    judges: List[Dict] | None = entry.get("judge_details")
    if judges:
        products = []
        for j in judges:
            vote = j.get("vote")
            conf = j.get("confidence")
            if vote is not None and conf is not None:
                products.append(float(vote) * float(conf))
        if products:
            return float(np.mean(products))

    raw = entry.get("raw_score")
    if raw is not None:
        return float(raw)

    return float("nan")
