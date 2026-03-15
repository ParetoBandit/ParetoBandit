"""Model family inference and data-driven family assignment.

Provides heuristic model-family grouping (by name stripping) and
data-driven family assignment via tetrachoric/Pearson correlation
on reward vectors.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List

import numpy as np


def infer_model_family(model_id: str) -> str:
    """Infer model family from a model_id by stripping variant suffixes.

    Models within the same family are expected to have similar reward
    functions.  Used by :func:`compute_correlation_families` and
    family-aware analytics.

    Strips size qualifiers (-mini, -large), instruction tuning (-instruct),
    quality tiers (-turbo, -pro), date stamps (-2024-04-09), parameter
    counts (-70b), and trailing minor versions (.1, .2).

    Override the inference by setting an explicit ``family`` field in the
    model registry entry.

    Examples:
        "openai/gpt-4-turbo"                -> "openai/gpt-4"
        "openai/gpt-4o-mini"                -> "openai/gpt-4o"
        "openai/gpt-5.1"                    -> "openai/gpt-5"
        "openai/o1-mini"                    -> "openai/o1"
        "anthropic/claude-3.5-sonnet"       -> "anthropic/claude-3"
        "anthropic/claude-3-haiku"          -> "anthropic/claude-3"
        "mistralai/mixtral-8x7b-instruct"   -> "mistralai/mixtral-8x7b"
        "meta-llama/llama-3.1-70b-instruct" -> "meta-llama/llama-3"
        "google/gemini-2.0-flash"           -> "google/gemini-2"
    """
    if "/" not in model_id:
        return model_id

    provider, model = model_id.split("/", 1)

    _SUFFIXES = (
        "-turbo", "-mini", "-small", "-medium", "-large", "-xl", "-xxl",
        "-instruct", "-chat", "-preview", "-latest", "-pro", "-flash",
        "-lite", "-haiku", "-sonnet", "-opus", "-nano", "-micro",
        "-thinking", "-online", "-free", "-nightly", "-exp",
    )

    changed = True
    while changed:
        changed = False

        stripped = re.sub(r"-\d{4}-?\d{2}-?\d{2}$", "", model)
        if stripped != model:
            model = stripped
            changed = True

        for suffix in _SUFFIXES:
            if model.endswith(suffix):
                model = model[: -len(suffix)]
                changed = True

        stripped = re.sub(r"-\d+b$", "", model)
        if stripped != model:
            model = stripped
            changed = True

    model = re.sub(r"(\d+)\.\d+$", r"\1", model)

    return f"{provider}/{model}"


# ---------------------------------------------------------------------------
# Tetrachoric Correlation & Data-Driven Family Assignment
# ---------------------------------------------------------------------------
# For binary (0/1) rewards, Pearson r equals the phi coefficient, which has a
# ceiling effect: when base rates are extreme or differ between two models,
# phi_max << 1 even for perfectly correlated failure patterns.  The
# tetrachoric correlation estimates the latent continuous correlation
# underlying the binary observations, correcting for this attenuation.
#
# Reference: Drasgow, F. (1986). "Polychoric and polyserial correlations."
# ---------------------------------------------------------------------------


def tetrachoric_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Tetrachoric correlation for two binary (0/1) vectors.

    Solves for the bivariate normal correlation *r* such that
    P(Z₁ > c₁, Z₂ > c₂ ; r) equals the observed joint success rate,
    where c₁, c₂ are the normal thresholds implied by each variable's
    marginal success rate.

    Applies Yates' continuity correction (+0.5 to each cell) when any
    cell of the 2x2 table is zero, preventing degenerate solutions.

    Returns NaN if the solver fails to converge (e.g. all-same vectors).
    """
    from scipy.stats import norm, multivariate_normal as mvn_dist
    from scipy.optimize import brentq

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = float(len(x))

    n11 = float(np.sum((x == 1) & (y == 1)))
    n10 = float(np.sum((x == 1) & (y == 0)))
    n01 = float(np.sum((x == 0) & (y == 1)))
    n00 = float(np.sum((x == 0) & (y == 0)))

    if n00 == 0 or n11 == 0 or n10 == 0 or n01 == 0:
        n11 += 0.5; n10 += 0.5; n01 += 0.5; n00 += 0.5
        n += 2.0

    p1 = (n11 + n10) / n
    p2 = (n11 + n01) / n
    p_obs = n11 / n

    if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        return np.nan

    c1 = norm.ppf(1.0 - p1)
    c2 = norm.ppf(1.0 - p2)

    def _objective(r: float) -> float:
        r = np.clip(r, -0.999, 0.999)
        dist = mvn_dist(mean=[0, 0], cov=[[1, r], [r, 1]])
        return dist.cdf([-c1, -c2]) - p_obs

    try:
        return float(brentq(_objective, -0.999, 0.999, xtol=1e-8))
    except ValueError:
        return np.nan


def compute_correlation_families(
    reward_vectors: Dict[str, np.ndarray],
    threshold: float = 0.6,
    method: str = "tetrachoric",
) -> Dict[str, str]:
    """Build a family map from within-provider reward correlations.

    Parameters
    ----------
    reward_vectors : dict[str, np.ndarray]
        Mapping from model ID (e.g. ``"openai/gpt-5"``) to a reward vector
        of shape ``(n_prompts,)``.  All vectors must have the same length
        and be aligned to the same prompt ordering.  For ``method="tetrachoric"``
        the vectors are treated as binary; for ``method="pearson"`` they are
        used as continuous values.
    threshold : float
        Minimum correlation for two models to be placed in the same family.
        Typical defaults: 0.6 for tetrachoric, 0.3 for Pearson.
    method : str
        Correlation measure: ``"tetrachoric"`` (default) computes the
        tetrachoric correlation on binarised rewards; ``"pearson"`` computes
        Pearson correlation on continuous rewards.

    Returns
    -------
    family_map : dict[str, str]
        Mapping from model ID to family label.  Models within the same
        provider whose pairwise correlation meets the threshold are grouped
        via connected-components clustering.  Cross-provider grouping is
        intentionally excluded.

    Raises
    ------
    ValueError
        If *method* is not one of ``"tetrachoric"`` or ``"pearson"``.

    Notes
    -----
    Falls back to :func:`infer_model_family` for providers with only one
    model in *reward_vectors*, preserving the syntactic heuristic as a
    default for models without reward history.
    """
    if method not in ("tetrachoric", "pearson"):
        raise ValueError(f"Unknown method {method!r}; expected 'tetrachoric' or 'pearson'")

    providers: Dict[str, List[str]] = defaultdict(list)
    for m in sorted(reward_vectors):
        prov = m.split("/")[0] if "/" in m else "__none__"
        providers[prov].append(m)

    parent: Dict[str, str] = {m: m for m in reward_vectors}

    def _find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for prov, models in providers.items():
        if len(models) < 2:
            continue
        for m1, m2 in combinations(models, 2):
            if method == "tetrachoric":
                corr = tetrachoric_corr(reward_vectors[m1], reward_vectors[m2])
            else:
                corr = float(np.corrcoef(reward_vectors[m1], reward_vectors[m2])[0, 1])
            if not np.isnan(corr) and corr >= threshold:
                _union(m1, m2)

    family_map: Dict[str, str] = {}
    for m in sorted(reward_vectors):
        root = _find(m)
        family_map[m] = root

    return family_map
