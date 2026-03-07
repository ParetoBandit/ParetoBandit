"""
Canonical reward extraction for BanditGPT.

Every data-loading path must derive per-response rewards through
:func:`extract_reward` so that the reward definition is consistent
across the entire codebase.

Reward signal (v3 — All-continuous Discriminative Router Judge rubric)
----------------------------------------------------------------------
Each judge scores three continuous factors:

- **Reasoning Quality** (continuous 0–1, weight 0.40)
- **Instruction Following** (continuous 0–1, weight 0.30)
- **Communication Quality** (continuous 0–1, weight 0.30)

Per-judge composite:
    ``reward = reasoning × 0.4 + instruction × 0.3 + communication × 0.3``

The final reward is ``mean(per-judge composite)`` across the panel.

Backward compatibility
----------------------
v2 records (binary ``logic`` / ``constraint`` + continuous ``utility``)
and v1 records (``vote`` × ``confidence``) are still supported
transparently.  The v3 extractor is tried first; if the v3 keys are
absent the function falls through to v2, then v1.

If ``judge_details`` is absent entirely (e.g. legacy battle data),
the function falls back to ``raw_score``.

Fallback
--------
If no usable field is present, returns ``NaN``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# v3 rubric weights — must match the judge system prompt in rejudge_cot.py.
# ---------------------------------------------------------------------------
_W_REASONING: float = 0.40
_W_INSTRUCTION: float = 0.30
_W_COMMUNICATION: float = 0.30

# ---------------------------------------------------------------------------
# v2 rubric weights (kept for backward-compatible extraction of old data).
# ---------------------------------------------------------------------------
_W_LOGIC: float = 0.5
_W_CONSTRAINT: float = 0.3
_W_UTILITY: float = 0.2


def _extract_v3_reward(judge: Dict[str, Any]) -> float | None:
    """Compute composite reward from a v3 all-continuous rubric record.

    Returns ``None`` if the required v3 fields are missing.
    """
    reasoning = judge.get("reasoning_quality")
    instruction = judge.get("instruction_following")
    communication = judge.get("communication_quality")
    if reasoning is not None and instruction is not None and communication is not None:
        return (
            float(reasoning) * _W_REASONING
            + float(instruction) * _W_INSTRUCTION
            + float(communication) * _W_COMMUNICATION
        )
    return None


def _extract_rubric_reward(judge: Dict[str, Any]) -> float | None:
    """Compute composite reward from a v2 rubric judge record.

    Returns ``None`` if the required rubric fields are missing.
    """
    logic = judge.get("logic")
    constraint = judge.get("constraint")
    utility = judge.get("utility")
    if logic is not None and constraint is not None and utility is not None:
        return (
            float(logic) * _W_LOGIC
            + float(constraint) * _W_CONSTRAINT
            + float(utility) * _W_UTILITY
        )
    return None


def _extract_legacy_reward(judge: Dict[str, Any]) -> float | None:
    """Compute reward from a v1 vote×confidence judge record.

    Returns ``None`` if the required fields are missing.
    """
    vote = judge.get("vote")
    conf = judge.get("confidence")
    if vote is not None and conf is not None:
        return float(vote) * float(conf)
    return None


def extract_reward(entry: Dict[str, Any]) -> float:
    """Derive a continuous reward from a single data-file entry.

    Parameters
    ----------
    entry : dict
        A parsed JSONL record.  Supports three judge-detail formats
        (tried in order):

        **v3 (all-continuous)** — each judge dict has
        ``reasoning_quality``, ``instruction_following``,
        ``communication_quality`` (all float 0–1).  Composite:
        ``reasoning × 0.4 + instruction × 0.3 + communication × 0.3``.

        **v2 (rubric)** — each judge dict has ``logic``, ``constraint``,
        ``utility`` fields.  Composite:
        ``logic × 0.5 + constraint × 0.3 + utility × 0.2``.

        **v1 (legacy)** — each judge dict has ``vote`` and ``confidence``.
        Composite: ``vote × confidence``.

        Falls back to ``raw_score`` when ``judge_details`` is absent.

    Returns
    -------
    float
        Reward in [0, 1].  ``NaN`` when no usable field is present.
    """
    judges: List[Dict] | None = entry.get("judge_details")
    if judges:
        rewards: list[float] = []
        for j in judges:
            r = _extract_v3_reward(j)
            if r is None:
                r = _extract_rubric_reward(j)
            if r is None:
                r = _extract_legacy_reward(j)
            if r is not None:
                rewards.append(r)
        if rewards:
            return float(np.mean(rewards))

    raw = entry.get("raw_score")
    if raw is not None:
        return float(raw)

    return float("nan")
