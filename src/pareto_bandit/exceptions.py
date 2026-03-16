"""Exception classes for BanditRouter."""

from __future__ import annotations

from typing import Dict, List


class MissingCostError(ValueError):
    """Raised when a model has no blended cost and it cannot be derived.

    The router requires every registered model to have a known
    ``blended_cost_per_m`` ($/M tokens).  This can be provided directly,
    via ``price_1m_blended``, or derived from both ``input_cost_per_m``
    and ``output_cost_per_m``.  If none of these are available (or only
    one of input/output is present), this error is raised at init time
    so the user can fix the registry entry.
    """


class NoEligibleModelsError(Exception):
    """Raised when no models pass the hard cost/latency/quality constraints.

    Attributes:
        reasons: Mapping of model_id -> list of human-readable exclusion strings.
    """

    def __init__(
        self,
        reasons: Dict[str, List[str]],
        max_cost: float | None = None,
        max_latency: float | None = None,
        quality_floor: Dict[str, float | None] | None = None,
    ):
        self.reasons = reasons
        lines = ["No models meet the specified constraints:"]
        if max_cost is not None:
            lines.append(f"  max_cost: ${max_cost:.6f}/1k tokens")
        if max_latency is not None:
            lines.append(f"  max_latency: {max_latency:.3f}s")
        if quality_floor:
            lines.append(f"  quality_floor: {quality_floor}")
        lines.append("")
        for model, model_reasons in reasons.items():
            lines.append(f"  {model}: {', '.join(model_reasons)}")
        super().__init__("\n".join(lines))


class NoModelScoredError(ValueError):
    """Raised when model scoring receives no eligible/scorable candidates.

    This error is used by lower-level selectors (e.g. cost-aware expert
    adapters) to provide a strict API contract for open-source consumers:
    selection methods that are typed to return ``str`` never return ``None``.
    """
