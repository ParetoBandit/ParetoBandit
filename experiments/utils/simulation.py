"""Shared simulation primitives for ParetoBandit experiments.

Provides the canonical data-loading, cost-normalization, and plotting
constants used across multiple experiment scripts (01_figure, 03_figure,
etc.).  All functions accept an explicit ``arm_order`` parameter so the
same code path serves K=2, K=3, or any other portfolio size.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pareto_bandit.config import K3_MODELS_CONFIG_PATH

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Colorblind-safe palette (Wong, Nature Methods 2011)
# ═══════════════════════════════════════════════════════════════════════════

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GRAY = "#999999"
CB_RED = "#D55E00"
CB_GREEN = "#009E73"
CB_PURPLE = "#CC79A7"
CB_TEAL = "#56B4E9"
CB_YELLOW = "#F0E442"


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SplitData:
    """Pre-processed split with embeddings ready for bandit simulation.

    Attributes:
        prompts: Raw prompt strings.
        rewards: ``{model_id: np.ndarray}`` of per-prompt rewards.
        costs: ``{model_id: np.ndarray}`` of per-prompt costs.
        embeddings: ``(n_prompts, feature_dim)`` feature matrix.
    """

    prompts: List[str]
    rewards: Dict[str, np.ndarray]
    costs: Dict[str, np.ndarray]
    embeddings: np.ndarray

    @property
    def n(self) -> int:
        """Number of prompts in this split."""
        return len(self.prompts)


def load_split(
    path: Path,
    fs: "FeatureService",
    arm_order: List[str],
) -> SplitData:
    """Load a JSONL split and encode prompts into feature vectors.

    Args:
        path: Path to a JSONL file where each line contains ``prompt``
            and ``arms`` with per-model ``reward`` and ``cost``.
        fs: Feature service for encoding prompts.
        arm_order: Model identifiers to extract from the ``arms`` dict.

    Returns:
        Fully loaded and embedded split data.
    """
    prompts: List[str] = []
    per_arm_rewards: Dict[str, List[float]] = {a: [] for a in arm_order}
    per_arm_costs: Dict[str, List[float]] = {a: [] for a in arm_order}

    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["prompt"])
            for arm_id in arm_order:
                info = r["arms"][arm_id]
                per_arm_rewards[arm_id].append(info["reward"])
                per_arm_costs[arm_id].append(info["cost"])

    rewards = {a: np.array(v) for a, v in per_arm_rewards.items()}
    costs = {a: np.array(v) for a, v in per_arm_costs.items()}

    logger.info("  Encoding %d prompts from %s ...", len(prompts), path.name)
    embeddings = fs.extract_features_batch(prompts)

    return SplitData(
        prompts=prompts, rewards=rewards, costs=costs, embeddings=embeddings,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Reward Manipulation
# ═══════════════════════════════════════════════════════════════════════════


def apply_reward_swap(
    split: SplitData,
    arm_a: str,
    arm_b: str,
) -> SplitData:
    """Return a new ``SplitData`` with rewards swapped between two arms.

    Only reward columns are exchanged; cost columns are left unchanged.
    This models a pure quality shift (e.g. a provider upgrades one model
    and degrades another) while API pricing remains fixed.  Cost-level
    shifts are tested separately in Experiment 03.

    Args:
        split: Original data.
        arm_a: First arm ID whose reward column is exchanged.
        arm_b: Second arm ID whose reward column is exchanged.

    Returns:
        New ``SplitData`` with swapped reward arrays (costs and embeddings
        are shallow-copied unchanged).
    """
    new_rewards = dict(split.rewards)
    new_rewards[arm_a], new_rewards[arm_b] = (
        split.rewards[arm_b].copy(),
        split.rewards[arm_a].copy(),
    )
    return SplitData(
        prompts=split.prompts,
        rewards=new_rewards,
        costs=split.costs,
        embeddings=split.embeddings,
    )


def apply_quality_degradation(
    split: SplitData,
    degraded_arm: str,
    degraded_reward: float = 0.75,
) -> SplitData:
    """Return a new ``SplitData`` with one arm's rewards degraded but costs unchanged.

    Models a silent quality regression (e.g., a model update ships a
    regression, the API continues to respond and charge normally, but
    response quality drops).  Only the reward signal reveals the problem;
    the cost signal provides no warning.

    Args:
        split: Original data.
        degraded_arm: Model ID whose quality has regressed.
        degraded_reward: Fixed degraded reward assigned to every prompt
            for the affected arm (default 0.75).

    Returns:
        New ``SplitData`` with degraded rewards for ``degraded_arm``;
        costs, other arms, and embeddings unchanged.
    """
    new_rewards = dict(split.rewards)
    new_rewards[degraded_arm] = np.full_like(
        split.rewards[degraded_arm], degraded_reward,
    )
    return SplitData(
        prompts=split.prompts,
        rewards=new_rewards,
        costs=split.costs,
        embeddings=split.embeddings,
    )


def apply_catastrophic_failure(
    split: SplitData,
    failed_arm: str,
    failure_reward: float = 0.05,
) -> SplitData:
    """Return a new ``SplitData`` with one arm's rewards near-zero and costs zero.

    Models a catastrophic API failure where requests fail (no useful
    response, no charge).  Rewards are set to a small positive value
    (rather than exact zero) to reflect that the model may still return
    *something* before timing out.  Costs are zeroed because cloud
    providers do not bill for failed requests (5xx, timeout).

    Args:
        split: Original data.
        failed_arm: Model ID whose quality has collapsed.
        failure_reward: Fixed low reward assigned to every prompt
            for the failed arm (default 0.05).

    Returns:
        New ``SplitData`` with degraded rewards and zeroed costs for
        ``failed_arm``; all other arms and embeddings unchanged.
    """
    new_rewards = dict(split.rewards)
    new_rewards[failed_arm] = np.full_like(
        split.rewards[failed_arm], failure_reward,
    )
    new_costs = dict(split.costs)
    new_costs[failed_arm] = np.zeros_like(split.costs[failed_arm])
    return SplitData(
        prompts=split.prompts,
        rewards=new_rewards,
        costs=new_costs,
        embeddings=split.embeddings,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════


def build_model_registry(
    arm_order: List[str],
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build model registry filtered to the requested arm set.

    Args:
        arm_order: Model identifiers to include.
        config_path: Path to the JSON config file listing all models.
            Defaults to ``data_collection/config/models_k3.json``.

    Returns:
        ``{model_id: config_dict}`` with keys ``model_id``,
        ``display_name``, ``input_cost_per_m``, ``output_cost_per_m``.
    """
    if config_path is None:
        config_path = K3_MODELS_CONFIG_PATH
    with open(config_path) as f:
        data = json.load(f)
    arm_set = set(arm_order)
    registry: Dict[str, Any] = {}
    for m in data["models"]:
        if m["model_id"] in arm_set:
            registry[m["model_id"]] = {
                "model_id": m["model_id"],
                "display_name": m.get("display", m["model_id"]),
                "input_cost_per_m": m["input_cost_per_m"],
                "output_cost_per_m": m["output_cost_per_m"],
            }
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# Cost Normalization
# ═══════════════════════════════════════════════════════════════════════════


def compute_normalized_costs(
    registry: Dict[str, Any],
    arm_order: List[str],
) -> Dict[str, float]:
    """Compute per-model normalized costs from registry pricing.

    Uses :func:`pareto_bandit.costs.log_normalize_cost` — the same canonical
    normalization as ``BanditRouter._calculate_absolute_penalty`` — so that
    a given ``cost_penalty`` value has identical semantics across all
    methods.

    Market anchors and the ``(input + output) / 2 / 1000`` blending formula
    are read from ``RouterConfig`` (single source of truth).

    Args:
        registry: Model registry with ``input_cost_per_m`` and
            ``output_cost_per_m`` entries ($/1M tokens).
        arm_order: Model identifiers to normalise.

    Returns:
        ``{model_id: normalized_cost}`` in [0, 1].
    """
    from pareto_bandit.costs import log_normalize_cost
    from pareto_bandit.router import RouterConfig

    cfg = RouterConfig()
    norm: Dict[str, float] = {}
    for arm in arm_order:
        meta = registry[arm]
        input_cost = meta["input_cost_per_m"]
        output_cost = meta["output_cost_per_m"]
        avg_cost_per_1k = ((input_cost + output_cost) / 2.0) / 1000.0
        norm[arm] = log_normalize_cost(
            avg_cost_per_1k,
            floor=cfg.market_cost_floor,
            ceiling=cfg.market_cost_ceiling,
        )
    return norm
