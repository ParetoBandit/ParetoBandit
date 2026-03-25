"""Shared utilities for the judge-robustness appendix.

Centralises data loading, matrix construction, judge-name canonicalisation,
Lin's CCC, and matplotlib configuration so that diagnostic scripts do not
silently diverge.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pareto_bandit.config import CALIBRATION_DIR, PARETO_REWARDS_PATH

# ── Paths ──────────────────────────────────────────────────────────────────
SUBSET_PROMPTS_PATH = CALIBRATION_DIR / "judge_robustness_prompts.jsonl"
SUPPLEMENTARY_REWARDS_PATH = CALIBRATION_DIR / "judge_robustness_rewards.jsonl"

# ── Canonical model / judge mappings ───────────────────────────────────────

MODELS: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-pro",
]

MODEL_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

JUDGE_ID_TO_SHORT: Dict[str, str] = {
    "openai/gpt-4.1-mini": "GPT-4.1-mini",
    "anthropic/claude-3.7-sonnet": "Claude-3.7-Sonnet",
}
"""Exact judge-ID → display-name mapping.  Used for loading supplementary
scores so that a substring match cannot misfire on longer model IDs."""

# ── Colour-blind-safe palette ──────────────────────────────────────────────
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#7B2D8E"
CB_GRAY = "#999999"

JUDGE_PLOT_META: Dict[str, Dict[str, str]] = {
    "openai/gpt-4.1-mini": {"short": "GPT-4.1-mini", "color": CB_ORANGE},
    "anthropic/claude-3.7-sonnet": {"short": "Claude-3.7-Sonnet", "color": CB_GREEN},
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_all_scores(
    subset_path: Path = SUBSET_PROMPTS_PATH,
    r1_path: Path = PARETO_REWARDS_PATH,
    supp_path: Path = SUPPLEMENTARY_REWARDS_PATH,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """Load R1 + supplementary scores keyed by short judge name.

    Supplementary judge IDs are resolved via :data:`JUDGE_ID_TO_SHORT`
    using **exact** equality, avoiding fragile substring matching.

    Parameters
    ----------
    subset_path:
        Path to ``judge_robustness_prompts.jsonl``.
    r1_path:
        Path to the full ``pareto_rewards.jsonl``.
    supp_path:
        Path to ``judge_robustness_rewards.jsonl``.

    Returns
    -------
    Dict[str, Dict[Tuple[str, str], float]]
        ``{"R1": {...}, "GPT-4.1-mini": {...}, "Claude-3.7-Sonnet": {...}}``.
    """
    prompts: Set[str] = set()
    with open(subset_path) as f:
        for line in f:
            prompts.add(json.loads(line)["prompt"])

    r1: Dict[Tuple[str, str], float] = {}
    with open(r1_path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok") or rec["prompt"] not in prompts:
                continue
            r1[(rec["prompt"], rec["model_id"])] = rec["raw_score"]

    supp: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    with open(supp_path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            key = (rec["prompt"], rec["model_id"])
            for jd in rec.get("judge_details", []):
                short = JUDGE_ID_TO_SHORT.get(jd["judge"])
                if short is not None:
                    supp[short][key] = jd["reward"]

    return {"R1": r1, **dict(supp)}


def build_prompt_matrices(
    all_scores: Dict[str, Dict[Tuple[str, str], float]],
    models: List[str] = MODELS,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Build ``{judge: [n_prompts x n_models]}`` matrices on common keys.

    Only prompts with scores from **every** judge for **every** model are
    retained, so the returned matrices are fully aligned.

    Parameters
    ----------
    all_scores:
        Output of :func:`load_all_scores`.
    models:
        Ordered list of model IDs (columns of each matrix).

    Returns
    -------
    Tuple[List[str], Dict[str, np.ndarray]]
        ``(sorted_prompt_list, {judge_name: score_matrix})``.
    """
    common_keys = set.intersection(
        *[set(s.keys()) for s in all_scores.values()]
    )
    prompts_with_all = sorted({
        p for p, _ in common_keys
        if all((p, m) in common_keys for m in models)
    })

    matrices: Dict[str, np.ndarray] = {}
    for judge, scores in all_scores.items():
        matrices[judge] = np.array([
            [scores[(p, m)] for m in models]
            for p in prompts_with_all
        ])
    return prompts_with_all, matrices


# ═══════════════════════════════════════════════════════════════════════════
# Agreement metrics
# ═══════════════════════════════════════════════════════════════════════════


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient.

    Measures agreement on the identity line, combining precision (Pearson r)
    with accuracy (how far the best-fit line deviates from y = x).  Unlike
    Pearson, CCC is penalised by both scale shift and location shift.

    Parameters
    ----------
    x, y:
        Paired measurements of equal length.

    Returns
    -------
    float
        CCC in [-1, 1].  Values near 1 indicate near-perfect agreement.

    References
    ----------
    Lin, L.I. (1989). A concordance correlation coefficient to evaluate
    reproducibility. *Biometrics*, 45(1), 255-268.
    """
    mx, my = np.mean(x), np.mean(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return float(2.0 * sxy / (sx2 + sy2 + (mx - my) ** 2))


# ═══════════════════════════════════════════════════════════════════════════
# Matplotlib setup
# ═══════════════════════════════════════════════════════════════════════════


def setup_matplotlib() -> None:
    """Configure matplotlib for publication-quality output.

    Safe to call multiple times; idempotent.
    """
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
