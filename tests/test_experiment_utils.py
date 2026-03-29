import json
from unittest.mock import MagicMock

import pytest

from src.pareto_bandit.utils.experiment import ExperimentBurnIn


@pytest.fixture
def mock_registry():
    return {
        "model_a": {"hle": 0.1, "cost_per_1m": 0.5},
        "model_b": {"hle": 0.3, "cost_per_1m": 2.0},
    }


@pytest.fixture
def mock_rewards():
    return {
        "p1": {"model_a": 0.8, "model_b": 0.2},  # Contentious (Var > 0.05)
        "p2": {"model_a": 0.5, "model_b": 0.5},  # Not contentious
        "p3": {"model_a": 0.9, "model_b": 0.9},  # Not contentious
    }


def test_generate_curriculum(mock_registry, mock_rewards):
    # p1 is hard (Var(0.8, 0.2) = 0.09 > 0.05)
    # p2 is easy (Var(0.5, 0.5) = 0.0)
    burner = ExperimentBurnIn(mock_registry, mock_rewards)
    curriculum = burner.generate_curriculum(["p1", "p2"])

    # p1 should be present 3 times (hard-prompt oversampling)
    assert curriculum.count("p1") == 3
    assert "p2" in curriculum
    assert len(curriculum) > 3


def test_perform_burn_in(mock_registry, mock_rewards):
    burner = ExperimentBurnIn(mock_registry, mock_rewards)

    mock_router = MagicMock()
    mock_router.route.return_value = ("model_a", {})

    burner.perform_burn_in(mock_router, ["p1", "p2"])

    assert mock_router.route.call_count == 2
    assert mock_router.update.call_count == 2
    # For p1, model_a reward is 0.8
    mock_router.update.assert_any_call("model_a", "p1", 0.8)


# ============================================================================
# Three-way split generation
# ============================================================================

def test_create_three_way_splits(mock_rewards, tmp_path):
    """create_three_way_splits() writes the canonical JSON and returns disjoint lists."""
    splits_path = tmp_path / "splits_three_way.json"

    prior_train, online_learn = ExperimentBurnIn.create_three_way_splits(
        oracle_rewards=mock_rewards,
        splits_path=splits_path,
        prior_ratio=0.5,
        random_state=42,
        min_models=1,
    )

    assert splits_path.exists()
    assert set(prior_train).isdisjoint(set(online_learn))
    assert len(prior_train) + len(online_learn) == len(mock_rewards)

    with open(splits_path) as f:
        data = json.load(f)
    assert "prior_train_pool" in data
    assert "online_learn_pool" in data
    assert data["prior_train_pool"] == prior_train
    assert data["online_learn_pool"] == online_learn


def test_create_three_way_splits_reproducible(mock_rewards, tmp_path):
    """Same seed produces identical splits."""
    p1 = tmp_path / "s1.json"
    p2 = tmp_path / "s2.json"

    tr1, ol1 = ExperimentBurnIn.create_three_way_splits(
        mock_rewards, p1, prior_ratio=0.5, random_state=42, min_models=1
    )
    tr2, ol2 = ExperimentBurnIn.create_three_way_splits(
        mock_rewards, p2, prior_ratio=0.5, random_state=42, min_models=1
    )

    assert tr1 == tr2
    assert ol1 == ol2


def test_create_three_way_splits_min_models_filter(tmp_path):
    """Prompts with fewer models than min_models are excluded from the split."""
    oracle_rewards = {
        "full_a":  {"m1": 0.8, "m2": 0.6},
        "full_b":  {"m1": 0.7, "m2": 0.9},
        "full_c":  {"m1": 0.5, "m2": 0.4},
        "full_d":  {"m1": 0.6, "m2": 0.8},
        "partial": {"m1": 0.5},
    }
    splits_path = tmp_path / "splits.json"

    prior_train, online_learn = ExperimentBurnIn.create_three_way_splits(
        oracle_rewards=oracle_rewards,
        splits_path=splits_path,
        prior_ratio=0.5,
        random_state=42,
        min_models=2,
    )

    all_split = set(prior_train) | set(online_learn)
    assert "partial" not in all_split
    assert all(p in all_split for p in ["full_a", "full_b", "full_c", "full_d"])


def test_create_three_way_splits_data_leakage_detection(tmp_path):
    """No overlap between prior_train and online_learn for a large pool."""
    oracle_rewards = {f"p{i}": {"m1": 0.5, "m2": 0.5} for i in range(100)}
    splits_path = tmp_path / "splits.json"

    prior_train, online_learn = ExperimentBurnIn.create_three_way_splits(
        oracle_rewards, splits_path, prior_ratio=0.4, random_state=42, min_models=2
    )

    overlap = set(prior_train) & set(online_learn)
    assert len(overlap) == 0, f"Leakage: {len(overlap)} shared prompts"
