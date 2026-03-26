"""
Tests for normalized AUCPC (Appendix H tuning metric).

The key requirement is interpretability under endpoint normalization:
- cheap baseline maps to (0, 0)
- frontier baseline maps to (1, 1)
so the diagonal baseline has area ~0.5 and near-oracle curves approach 1.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _import_pareto_utils():
    experiments_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(experiments_root))
    from utils.pareto import pareto_aucpc_normalized  # type: ignore

    return pareto_aucpc_normalized


def test_aucpc_normalized_diagonal_is_half():
    pareto_aucpc_normalized = _import_pareto_utils()

    cheap_cost, frontier_cost = 1.0, 3.0
    cheap_reward, frontier_reward = 0.2, 0.8

    # A straight line from (cheap, cheap_reward) to (frontier, frontier_reward)
    # becomes y=x after normalization, whose area is 0.5.
    costs = [cheap_cost, frontier_cost]
    rewards = [cheap_reward, frontier_reward]
    auc = pareto_aucpc_normalized(
        costs,
        rewards,
        cheap_cost=cheap_cost,
        frontier_cost=frontier_cost,
        cheap_reward=cheap_reward,
        frontier_reward=frontier_reward,
        clip_quality_to_unit=True,
    )
    assert auc == pytest.approx(0.5, abs=1e-9)


def test_aucpc_normalized_oracle_like_is_one():
    pareto_aucpc_normalized = _import_pareto_utils()

    cheap_cost, frontier_cost = 1.0, 3.0
    cheap_reward, frontier_reward = 0.2, 0.8

    # Frontier-quality everywhere -> normalized quality is 1 across x ∈ [0,1].
    costs = [cheap_cost, frontier_cost]
    rewards = [frontier_reward, frontier_reward]
    auc = pareto_aucpc_normalized(
        costs,
        rewards,
        cheap_cost=cheap_cost,
        frontier_cost=frontier_cost,
        cheap_reward=cheap_reward,
        frontier_reward=frontier_reward,
        clip_quality_to_unit=True,
    )
    assert auc == pytest.approx(1.0, abs=1e-9)


def test_aucpc_normalized_cheapest_everywhere_is_zero():
    pareto_aucpc_normalized = _import_pareto_utils()

    cheap_cost, frontier_cost = 1.0, 3.0
    cheap_reward, frontier_reward = 0.2, 0.8

    costs = [cheap_cost, frontier_cost]
    rewards = [cheap_reward, cheap_reward]
    auc = pareto_aucpc_normalized(
        costs,
        rewards,
        cheap_cost=cheap_cost,
        frontier_cost=frontier_cost,
        cheap_reward=cheap_reward,
        frontier_reward=frontier_reward,
        clip_quality_to_unit=True,
    )
    assert auc == pytest.approx(0.0, abs=1e-9)


def test_aucpc_normalized_clips_above_frontier():
    pareto_aucpc_normalized = _import_pareto_utils()

    cheap_cost, frontier_cost = 1.0, 3.0
    cheap_reward, frontier_reward = 0.2, 0.8

    # If rewards exceed the frontier baseline, clipping keeps AUCPC <= 1.
    costs = [cheap_cost, frontier_cost]
    rewards = [0.95, 0.95]
    auc = pareto_aucpc_normalized(
        costs,
        rewards,
        cheap_cost=cheap_cost,
        frontier_cost=frontier_cost,
        cheap_reward=cheap_reward,
        frontier_reward=frontier_reward,
        clip_quality_to_unit=True,
    )
    assert auc == pytest.approx(1.0, abs=1e-9)


def test_aucpc_is_partial_over_practical_cost_region():
    pareto_aucpc_normalized = _import_pareto_utils()

    cheap_cost, frontier_cost = 1.0, 3.0
    cheap_reward, frontier_reward = 0.2, 0.8

    # Inside-range points define the diagonal -> AUC = 0.5.
    costs_in = [cheap_cost, frontier_cost]
    rewards_in = [cheap_reward, frontier_reward]
    auc_in = pareto_aucpc_normalized(
        costs_in,
        rewards_in,
        cheap_cost=cheap_cost,
        frontier_cost=frontier_cost,
        cheap_reward=cheap_reward,
        frontier_reward=frontier_reward,
        clip_quality_to_unit=True,
    )
    assert auc_in == pytest.approx(0.5, abs=1e-9)

    # Add an out-of-range point that would otherwise dominate and inflate the
    # frontier if not excluded by the pAUCPC cost bound.
    costs = [cheap_cost - 0.25, cheap_cost, frontier_cost]
    rewards = [0.99, cheap_reward, frontier_reward]
    auc = pareto_aucpc_normalized(
        costs,
        rewards,
        cheap_cost=cheap_cost,
        frontier_cost=frontier_cost,
        cheap_reward=cheap_reward,
        frontier_reward=frontier_reward,
        clip_quality_to_unit=True,
    )
    assert auc == pytest.approx(0.5, abs=1e-9)

