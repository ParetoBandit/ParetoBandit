"""Test state save/load dimension validation."""


import numpy as np
import pytest

from pareto_bandit.router import DisjointLinUCBPolicy


def test_state_save_includes_metadata(tmp_path):
    """Test that save_state includes dimension metadata."""
    policy = DisjointLinUCBPolicy(model_names=["model1"], dim=10, alpha=0.5)

    save_path = tmp_path / "state.npz"
    policy.save_state(save_path)

    # Load and check metadata
    data = np.load(save_path)
    assert '_metadata_dim' in data
    assert int(data['_metadata_dim']) == 10
    assert '_metadata_models' in data


def test_dimension_mismatch_raises_error(tmp_path):
    """Test that loading state with wrong dimension raises clear error."""
    # Create and save state with dim=10
    policy1 = DisjointLinUCBPolicy(model_names=["model1"], dim=10, alpha=0.5)
    save_path = tmp_path / "state.npz"
    policy1.save_state(save_path)

    # Try to load into policy with dim=20
    policy2 = DisjointLinUCBPolicy(model_names=["model1"], dim=20, alpha=0.5)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        policy2.load_state(save_path)


def test_legacy_state_warning(tmp_path, caplog):
    """Test that loading legacy state (no metadata) warns but proceeds."""
    # Manually create legacy state file without metadata
    policy = DisjointLinUCBPolicy(model_names=["model1"], dim=10, alpha=0.5)

    save_path = tmp_path / "legacy_state.npz"
    # Save without metadata (simulate old version)
    data = {}
    for m in policy.models:
        data[f"{m}_A"] = policy.A[m]
        data[f"{m}_b"] = policy.b[m]
    np.savez_compressed(save_path, **data)

    # Load should warn
    policy2 = DisjointLinUCBPolicy(model_names=["model1"], dim=10, alpha=0.5)
    policy2.load_state(save_path)

    assert "without dimension metadata" in caplog.text


def test_matrix_shape_validation(tmp_path):
    """Test that incorrect matrix shapes are caught."""
    DisjointLinUCBPolicy(model_names=["model1"], dim=10, alpha=0.5)

    save_path = tmp_path / "state.npz"

    # Manually create corrupted state
    data = {
        '_metadata_dim': 10,
        '_metadata_models': ['model1'],
        'model1_A': np.random.randn(5, 5),  # Wrong shape!
        'model1_b': np.random.randn(10)
    }
    np.savez_compressed(save_path, **data)

    policy2 = DisjointLinUCBPolicy(model_names=["model1"], dim=10, alpha=0.5)

    with pytest.raises(ValueError, match="wrong shape"):
        policy2.load_state(save_path)


def test_successful_load_after_save(tmp_path):
    """Test that save/load cycle works correctly with validation."""
    policy1 = DisjointLinUCBPolicy(model_names=["model1"], dim=10, alpha=0.5)

    # Add some data
    x = np.random.randn(10)
    policy1.update("model1", x, reward=0.8, weight=1.0)

    save_path = tmp_path / "state.npz"
    policy1.save_state(save_path)

    # Load into new policy
    policy2 = DisjointLinUCBPolicy(model_names=["model1"], dim=10, alpha=0.5)
    policy2.load_state(save_path)

    # Verify matrices match
    assert np.allclose(policy1.A["model1"], policy2.A["model1"])
    assert np.allclose(policy1.b["model1"], policy2.b["model1"])
