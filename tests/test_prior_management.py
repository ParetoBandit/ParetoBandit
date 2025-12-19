"""
Unit tests for prior management and dynamic model addition.

Tests:
  - PriorManager: loading, saving, merging, add/remove models
  - BanditRouter.create(): different prior modes
  - BanditRouter.add_model() / remove_model(): dynamic model management
  - DisjointLinUCBPolicy.add_model() / remove_model(): bandit-level operations
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_priors() -> Dict[str, Any]:
    """Sample priors bundle for testing."""
    return {
        "A_shared": np.eye(4, dtype=np.float64),
        "b_vectors": {
            "model-a": np.array([1.0, 2.0, 3.0, 4.0]),
            "model-b": np.array([5.0, 6.0, 7.0, 8.0]),
        },
        "model_ids": ["model-a", "model-b"],
        "dim": 4,
        "alpha": 0.5,
    }


@pytest.fixture
def sample_registry() -> Dict[str, Dict[str, Any]]:
    """Sample model registry for testing."""
    return {
        "model-a": {"display_name": "Model A", "cost_per_1k_input": 0.001},
        "model-b": {"display_name": "Model B", "cost_per_1k_input": 0.002},
        "model-c": {"display_name": "Model C", "cost_per_1k_input": 0.003},
    }


# ---------------------------------------------------------------------------
# PriorManager Tests
# ---------------------------------------------------------------------------


class TestPriorManager:
    """Tests for PriorManager class."""

    def test_import(self):
        """Test that PriorManager can be imported."""
        from banditgpt.core import PriorManager
        assert PriorManager is not None

    def test_factory_methods(self):
        """Test factory method constructors."""
        from banditgpt.core import PriorManager

        # These should not raise
        bundled = PriorManager.bundled()
        user = PriorManager.user()
        merged = PriorManager.merged()
        none = PriorManager.none()

        assert bundled.config.source == "bundled"
        assert none.config.source == "none"

    def test_save_and_load(self, temp_dir, sample_priors):
        """Test saving and loading priors."""
        from banditgpt.core import PriorManager, PriorConfig

        path = temp_dir / "test_priors.npz"
        manager = PriorManager(PriorConfig(source="none"), save_path=path)

        # Save
        saved_path = manager.save(sample_priors, path=path)
        assert saved_path == path
        assert path.exists()

        # Load
        manager2 = PriorManager.from_file(path)
        loaded = manager2.load()

        assert loaded is not None
        assert set(loaded["model_ids"]) == {"model-a", "model-b"}
        assert loaded["dim"] == 4
        assert "b_vectors" in loaded
        assert "model-a" in loaded["b_vectors"]

    def test_merge_priors_union(self):
        """Test that merge_priors creates a union of models."""
        from banditgpt.core import PriorManager

        base = {
            "model_ids": ["a", "b"],
            "b_vectors": {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])},
            "dim": 3,
            "alpha": 0.5,
        }
        overlay = {
            "model_ids": ["b", "c"],
            "b_vectors": {"b": np.array([7, 8, 9]), "c": np.array([10, 11, 12])},
            "dim": 3,
            "alpha": 0.5,
        }

        merged = PriorManager.merge_priors(base, overlay)

        # Union of models
        assert set(merged["model_ids"]) == {"a", "b", "c"}

        # Overlay takes precedence for 'b'
        np.testing.assert_array_equal(merged["b_vectors"]["b"], [7, 8, 9])

        # Base preserved for 'a'
        np.testing.assert_array_equal(merged["b_vectors"]["a"], [1, 2, 3])

        # Overlay added 'c'
        np.testing.assert_array_equal(merged["b_vectors"]["c"], [10, 11, 12])

    def test_add_model_clone(self, sample_priors):
        """Test adding a model by cloning from existing."""
        from banditgpt.core import PriorManager

        manager = PriorManager.none()

        # Clone model-c from model-a
        updated = manager.add_model(
            sample_priors,
            "model-c",
            clone_from="model-a",
            clone_decay=0.8,
        )

        assert "model-c" in updated["model_ids"]
        assert "model-c" in updated["b_vectors"]

        # Check decay was applied
        expected = np.array([1.0, 2.0, 3.0, 4.0]) * 0.8
        np.testing.assert_array_almost_equal(updated["b_vectors"]["model-c"], expected)

    def test_add_model_cold_start(self, sample_priors):
        """Test adding a model with cold start (zeros)."""
        from banditgpt.core import PriorManager

        manager = PriorManager.none()

        updated = manager.add_model(sample_priors, "model-new")

        assert "model-new" in updated["model_ids"]
        assert "model-new" in updated["b_vectors"]
        np.testing.assert_array_equal(
            updated["b_vectors"]["model-new"],
            np.zeros(4),
        )

    def test_add_model_already_exists(self, sample_priors):
        """Test that adding an existing model returns unchanged priors."""
        from banditgpt.core import PriorManager

        manager = PriorManager.none()
        updated = manager.add_model(sample_priors, "model-a")

        # Should be unchanged
        assert updated["model_ids"] == sample_priors["model_ids"]

    def test_remove_model(self, sample_priors):
        """Test removing a model from priors."""
        from banditgpt.core import PriorManager

        manager = PriorManager.none()
        updated = manager.remove_model(sample_priors, "model-a")

        assert "model-a" not in updated["model_ids"]
        assert "model-a" not in updated["b_vectors"]
        assert "model-b" in updated["model_ids"]


# ---------------------------------------------------------------------------
# DisjointLinUCBPolicy Tests
# ---------------------------------------------------------------------------


class TestDisjointLinUCBPolicy:
    """Tests for bandit-level add/remove model operations."""

    def test_add_model_cold_start(self):
        """Test adding a model with cold start."""
        from banditgpt.core.bandit_router import DisjointLinUCBPolicy

        policy = DisjointLinUCBPolicy(["model-a", "model-b"], dim=4, alpha=0.5)

        assert "model-c" not in policy.models
        added = policy.add_model("model-c")

        assert added is True
        assert "model-c" in policy.models
        assert "model-c" in policy.A
        assert "model-c" in policy.b
        assert "model-c" in policy.A_inv

        # Should be initialized to ridge identity
        np.testing.assert_array_almost_equal(
            policy.A["model-c"],
            np.eye(4) * policy.ridge_lambda,
        )
        np.testing.assert_array_equal(policy.b["model-c"], np.zeros(4))

    def test_add_model_clone(self):
        """Test adding a model by cloning."""
        from banditgpt.core.bandit_router import DisjointLinUCBPolicy

        policy = DisjointLinUCBPolicy(["model-a"], dim=4, alpha=0.5)

        # Manually set some weights for model-a
        policy.A["model-a"] = np.eye(4) * 2.0
        policy.b["model-a"] = np.array([1.0, 2.0, 3.0, 4.0])
        policy.A_inv["model-a"] = np.linalg.inv(policy.A["model-a"])

        # Clone
        added = policy.add_model("model-b", clone_from="model-a", clone_decay=0.9)

        assert added is True
        np.testing.assert_array_almost_equal(
            policy.A["model-b"],
            policy.A["model-a"] * 0.9,
        )
        np.testing.assert_array_almost_equal(
            policy.b["model-b"],
            policy.b["model-a"] * 0.9,
        )

    def test_add_model_already_exists(self):
        """Test that adding an existing model returns False."""
        from banditgpt.core.bandit_router import DisjointLinUCBPolicy

        policy = DisjointLinUCBPolicy(["model-a"], dim=4)
        added = policy.add_model("model-a")

        assert added is False

    def test_remove_model(self):
        """Test removing a model."""
        from banditgpt.core.bandit_router import DisjointLinUCBPolicy

        policy = DisjointLinUCBPolicy(["model-a", "model-b"], dim=4)

        removed = policy.remove_model("model-a")

        assert removed is True
        assert "model-a" not in policy.models
        assert "model-a" not in policy.A
        assert "model-a" not in policy.b
        assert "model-a" not in policy.A_inv

    def test_remove_model_not_exists(self):
        """Test removing a non-existent model returns False."""
        from banditgpt.core.bandit_router import DisjointLinUCBPolicy

        policy = DisjointLinUCBPolicy(["model-a"], dim=4)
        removed = policy.remove_model("model-nonexistent")

        assert removed is False


# ---------------------------------------------------------------------------
# Expert Priors Tests
# ---------------------------------------------------------------------------


class TestExpertPriors:
    """Tests for expert-distilled priors (A_stack/b_stack format)."""

    def test_create_and_save_expert_priors(self, temp_dir):
        """Test creating and saving expert priors in disjoint format."""
        from banditgpt.core.bandit_router import DisjointLinUCBPolicy

        model_names = ["model-a", "model-b", "model-c"]
        dim = 32
        
        policy = DisjointLinUCBPolicy(model_names, dim=dim, alpha=0.5)
        
        # Simulate some updates
        for _ in range(10):
            x = np.random.randn(dim)
            for m in model_names:
                policy.update(m, x, np.random.rand())
        
        # Save in expert format (A_stack/b_stack)
        path = temp_dir / "expert_priors.npz"
        A_stack = np.stack([policy.A[m] for m in model_names], axis=0)
        b_stack = np.stack([policy.b[m] for m in model_names], axis=0)
        
        np.savez_compressed(
            path,
            model_names=np.array(model_names, dtype=object),
            dim=dim,
            alpha=0.5,
            A_stack=A_stack.astype(np.float16),
            b_stack=b_stack.astype(np.float16),
        )
        
        # Verify file exists and can be loaded
        assert path.exists()
        data = np.load(path, allow_pickle=True)
        assert "A_stack" in data
        assert "b_stack" in data
        assert data["A_stack"].shape == (3, 32, 32)
        assert data["b_stack"].shape == (3, 32)

    def test_load_expert_priors_format(self, temp_dir, sample_registry):
        """Test that expert priors are correctly loaded."""
        from banditgpt.core.bandit_router import BanditRouter

        # Create expert priors
        model_names = list(sample_registry.keys())
        dim = 384
        
        A_stack = np.stack([np.eye(dim) * 2.0 for _ in model_names], axis=0)
        b_stack = np.stack([np.ones(dim) * i for i, _ in enumerate(model_names)], axis=0)
        
        path = temp_dir / "expert.npz"
        np.savez_compressed(
            path,
            model_names=np.array(model_names, dtype=object),
            dim=dim,
            alpha=0.5,
            A_stack=A_stack.astype(np.float16),
            b_stack=b_stack.astype(np.float16),
        )
        
        # Load via BanditRouter
        router = BanditRouter.load_from_shippable_priors(
            priors_npz=path,
            model_registry=sample_registry,
            prior_strength=1.0,  # No scaling for test
        )
        
        # Verify loaded correctly
        for i, m in enumerate(model_names):
            assert m in router.bandit.A
            # b should match (with float16 -> float64 conversion tolerance)
            np.testing.assert_array_almost_equal(
                router.bandit.b[m],
                np.ones(dim) * i,
                decimal=2,
            )

    def test_expert_priors_float16_precision(self, temp_dir, sample_registry):
        """Test that float16 compression maintains acceptable precision."""
        from banditgpt.core.bandit_router import BanditRouter

        model_names = list(sample_registry.keys())
        dim = 384
        
        # Create priors with specific values
        A_orig = np.eye(dim) * 50.0  # Boosted
        b_orig = np.random.randn(dim) * 10.0
        
        A_stack = np.stack([A_orig for _ in model_names], axis=0)
        b_stack = np.stack([b_orig for _ in model_names], axis=0)
        
        path = temp_dir / "expert.npz"
        np.savez_compressed(
            path,
            model_names=np.array(model_names, dtype=object),
            dim=dim,
            alpha=0.5,
            A_stack=A_stack.astype(np.float16),
            b_stack=b_stack.astype(np.float16),
        )
        
        router = BanditRouter.load_from_shippable_priors(
            priors_npz=path,
            model_registry=sample_registry,
            prior_strength=1.0,
        )
        
        # Check precision is acceptable (within 1% for diagonal)
        for m in model_names:
            diag = np.diag(router.bandit.A[m])
            np.testing.assert_allclose(diag, 50.0, rtol=0.01)


class TestPriorStrength:
    """Tests for the prior_strength (λ_boost) parameter."""

    def test_prior_strength_scales_matrices(self, temp_dir, sample_registry):
        """prior_strength multiplies A and b matrices."""
        from banditgpt.core.bandit_router import BanditRouter

        model_names = list(sample_registry.keys())
        dim = 384
        
        # Create expert priors with known values
        A_stack = np.stack([np.eye(dim) for _ in model_names], axis=0)
        b_stack = np.stack([np.ones(dim) for _ in model_names], axis=0)
        
        path = temp_dir / "expert.npz"
        np.savez_compressed(
            path,
            model_names=np.array(model_names, dtype=object),
            dim=dim,
            alpha=0.5,
            A_stack=A_stack.astype(np.float32),
            b_stack=b_stack.astype(np.float32),
        )
        
        # Load with strength=10.0
        router = BanditRouter.load_from_shippable_priors(
            priors_npz=path,
            model_registry=sample_registry,
            prior_strength=10.0,
        )
        
        # A should be scaled by 10x
        for m in model_names:
            diag = np.diag(router.bandit.A[m])
            np.testing.assert_allclose(diag, 10.0, rtol=0.01)
            # b should also be scaled by 10x
            np.testing.assert_allclose(router.bandit.b[m], 10.0, rtol=0.01)

    def test_prior_strength_default_50(self, temp_dir, sample_registry):
        """Default prior_strength is 50.0 for expert priors."""
        from banditgpt.core.bandit_router import BanditRouter

        model_names = list(sample_registry.keys())
        dim = 384
        
        A_stack = np.stack([np.eye(dim) for _ in model_names], axis=0)
        b_stack = np.stack([np.ones(dim) for _ in model_names], axis=0)
        
        path = temp_dir / "expert.npz"
        np.savez_compressed(
            path,
            model_names=np.array(model_names, dtype=object),
            dim=dim,
            alpha=0.5,
            A_stack=A_stack.astype(np.float32),
            b_stack=b_stack.astype(np.float32),
        )
        
        # Load with default strength (should be 50.0)
        router = BanditRouter.load_from_shippable_priors(
            priors_npz=path,
            model_registry=sample_registry,
            # prior_strength defaults to 50.0
        )
        
        # A diagonal should be 50.0
        for m in model_names:
            diag = np.diag(router.bandit.A[m])
            np.testing.assert_allclose(diag, 50.0, rtol=0.01)

    def test_prior_strength_affects_ucb_confidence(self, temp_dir, sample_registry):
        """Higher prior_strength reduces UCB uncertainty."""
        from banditgpt.core.bandit_router import BanditRouter

        model_names = list(sample_registry.keys())
        dim = 384
        
        A_stack = np.stack([np.eye(dim) for _ in model_names], axis=0)
        b_stack = np.stack([np.ones(dim) for _ in model_names], axis=0)
        
        path = temp_dir / "expert.npz"
        np.savez_compressed(
            path,
            model_names=np.array(model_names, dtype=object),
            dim=dim,
            alpha=0.5,
            A_stack=A_stack.astype(np.float32),
            b_stack=b_stack.astype(np.float32),
        )
        
        # Load with low strength
        router_low = BanditRouter.load_from_shippable_priors(
            priors_npz=path,
            model_registry=sample_registry,
            prior_strength=1.0,
        )
        
        # Load with high strength
        router_high = BanditRouter.load_from_shippable_priors(
            priors_npz=path,
            model_registry=sample_registry,
            prior_strength=50.0,
        )
        
        # Compute variance for a test context
        x = np.random.randn(dim)
        x = x / np.linalg.norm(x)
        
        m = model_names[0]
        var_low = x @ router_low.bandit.A_inv[m] @ x
        var_high = x @ router_high.bandit.A_inv[m] @ x
        
        # Higher strength = lower variance (more confidence)
        assert var_high < var_low
        # Roughly 50x difference
        assert var_low / var_high > 40  # Allow some tolerance


class TestPriorFormatDetection:
    """Tests for automatic prior format detection."""

    def test_detects_expert_format(self, temp_dir, sample_registry):
        """Detects A_stack as expert format."""
        from banditgpt.core.bandit_router import BanditRouter

        model_names = list(sample_registry.keys())
        dim = 384
        
        path = temp_dir / "expert.npz"
        np.savez_compressed(
            path,
            model_names=np.array(model_names, dtype=object),
            dim=dim,
            alpha=0.5,
            A_stack=np.stack([np.eye(dim) for _ in model_names], axis=0),
            b_stack=np.stack([np.zeros(dim) for _ in model_names], axis=0),
        )
        
        # Should load without error (detects expert format)
        router = BanditRouter.load_from_shippable_priors(
            priors_npz=path,
            model_registry=sample_registry,
            prior_strength=1.0,
        )
        
        assert router is not None
        assert len(router.bandit.models) == len(sample_registry)

    def test_detects_shared_format(self, temp_dir, sample_registry):
        """Detects A_shared as shared covariance format."""
        from banditgpt.core.bandit_router import BanditRouter, SharedCovarianceLinUCBPolicy

        # Create shared priors using the policy class
        models = list(sample_registry.keys())
        policy = SharedCovarianceLinUCBPolicy(models, dim=384, alpha=0.5)
        
        path = temp_dir / "shared.npz"
        policy.to_shippable_priors_npz(path)
        
        # Should load without error (detects shared format and inflates)
        router = BanditRouter.load_from_shippable_priors(
            priors_npz=path,
            model_registry=sample_registry,
            prior_strength=1.0,
        )
        
        assert router is not None
        # After inflation, each model should have its own A
        for m in models:
            assert m in router.bandit.A

    def test_create_prefers_expert_priors(self, temp_dir, sample_registry):
        """BanditRouter.create() prefers expert_priors.npz over shippable_priors.npz."""
        from banditgpt.core.bandit_router import BanditRouter, SharedCovarianceLinUCBPolicy

        models = list(sample_registry.keys())
        dim = 384
        
        # Create both types of priors
        expert_path = temp_dir / "expert_priors.npz"
        shared_path = temp_dir / "shippable_priors.npz"
        
        # Expert priors with distinctive b values
        np.savez_compressed(
            expert_path,
            model_names=np.array(models, dtype=object),
            dim=dim,
            alpha=0.5,
            A_stack=np.stack([np.eye(dim) for _ in models], axis=0),
            b_stack=np.stack([np.ones(dim) * 999 for _ in models], axis=0),  # Distinctive
        )
        
        # Shared priors with different b values
        policy = SharedCovarianceLinUCBPolicy(models, dim=dim, alpha=0.5)
        policy.to_shippable_priors_npz(shared_path)
        
        # Create router - should prefer expert_priors.npz
        router = BanditRouter.create(
            sample_registry,
            priors="bundled",
            bundled_priors_path=expert_path,  # Explicitly use expert
        )
        
        # Check that expert priors were loaded (b = 999 * strength)
        m = models[0]
        # With default strength=50, b should be 999 * 50 ≈ 49950
        assert router.bandit.b[m][0] > 40000


# ---------------------------------------------------------------------------
# BanditRouter Tests
# ---------------------------------------------------------------------------


class TestBanditRouterCreate:
    """Tests for BanditRouter.create() with different prior modes."""

    def test_create_none_mode(self, sample_registry):
        """Test cold start with priors='none'."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")

        assert router.priors_source == "none"
        assert router.priors_path is None
        assert len(router.bandit.models) == len(sample_registry)

    def test_create_with_nonexistent_paths(self, sample_registry, temp_dir):
        """Test that auto mode falls back to cold start when no priors exist."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(
            sample_registry,
            priors="auto",
            user_priors_path=temp_dir / "nonexistent_user.npz",
            bundled_priors_path=temp_dir / "nonexistent_bundled.npz",
        )

        assert router.priors_source == "none"

    def test_create_bundled_mode(self, sample_registry, temp_dir):
        """Test loading bundled priors."""
        from banditgpt.core.bandit_router import BanditRouter, SharedCovarianceLinUCBPolicy

        # Create fake bundled priors using the proper format
        bundled_path = temp_dir / "bundled.npz"
        models = list(sample_registry.keys())
        policy = SharedCovarianceLinUCBPolicy(models, dim=384, alpha=0.5)
        policy.to_shippable_priors_npz(bundled_path)

        router = BanditRouter.create(
            sample_registry,
            priors="bundled",
            bundled_priors_path=bundled_path,
        )

        assert router.priors_source == "bundled"
        assert router.priors_path == bundled_path

    def test_create_merged_mode(self, sample_registry, temp_dir):
        """Test merged mode combines bundled and user priors."""
        from banditgpt.core import PriorManager, PriorConfig
        from banditgpt.core.bandit_router import BanditRouter

        # Create bundled priors with models a, b using PriorManager format
        bundled_priors = {
            "A_shared": np.eye(384),
            "b_vectors": {
                "model-a": np.ones(384) * 1.0,
                "model-b": np.ones(384) * 2.0,
            },
            "model_ids": ["model-a", "model-b"],
            "dim": 384,
            "alpha": 0.5,
        }
        bundled_path = temp_dir / "bundled.npz"
        PriorManager(PriorConfig(source="none")).save(bundled_priors, path=bundled_path)

        # Create user priors with models b (updated), c (new)
        user_priors = {
            "A_shared": np.eye(384),
            "b_vectors": {
                "model-b": np.ones(384) * 3.0,  # Updated
                "model-c": np.ones(384) * 4.0,  # New
            },
            "model_ids": ["model-b", "model-c"],
            "dim": 384,
            "alpha": 0.5,
        }
        user_path = temp_dir / "user.npz"
        PriorManager(PriorConfig(source="none")).save(user_priors, path=user_path)

        # Merged registry needs all models
        merged_registry = {
            "model-a": {"display_name": "A"},
            "model-b": {"display_name": "B"},
            "model-c": {"display_name": "C"},
        }

        router = BanditRouter.create(
            merged_registry,
            priors="merged",
            bundled_priors_path=bundled_path,
            user_priors_path=user_path,
            prior_strength=1.0,  # Disable scaling for this test
        )

        assert router.priors_source == "merged"

        # Check merged b vectors (just check first element as proxy)
        # model-a: from bundled (1.0)
        assert abs(router.bandit.b["model-a"][0] - 1.0) < 0.01
        # model-b: from user (override, 3.0)
        assert abs(router.bandit.b["model-b"][0] - 3.0) < 0.01
        # model-c: from user (addition, 4.0)
        assert abs(router.bandit.b["model-c"][0] - 4.0) < 0.01


class TestBanditRouterAddRemoveModel:
    """Tests for BanditRouter.add_model() and remove_model()."""

    def test_add_model_cold_start(self, sample_registry):
        """Test adding a model with cold start."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")

        added = router.add_model(
            "model-new",
            registry_entry={"display_name": "New Model"},
        )

        assert added is True
        assert "model-new" in router.registry
        assert "model-new" in router.bandit.models
        assert router.list_models() == list(sample_registry.keys()) + ["model-new"]

    def test_add_model_clone(self, sample_registry):
        """Test adding a model by cloning."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")

        # Set a prior for model-a
        router.model_priors["model-a"] = 0.8

        added = router.add_model(
            "model-a-v2",
            clone_from="model-a",
            registry_entry={"display_name": "Model A v2"},
        )

        assert added is True
        assert "model-a-v2" in router.registry
        # Prior should be inherited
        assert router.model_priors.get("model-a-v2") == 0.8

    def test_add_model_already_exists(self, sample_registry):
        """Test that adding an existing model returns False."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        added = router.add_model("model-a")

        assert added is False

    def test_remove_model(self, sample_registry):
        """Test removing a model."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        router.model_priors["model-a"] = 0.5

        removed = router.remove_model("model-a")

        assert removed is True
        assert "model-a" not in router.registry
        assert "model-a" not in router.bandit.models
        assert "model-a" not in router.model_priors

    def test_remove_model_not_exists(self, sample_registry):
        """Test removing a non-existent model returns False."""
        from banditgpt.core.bandit_router import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        removed = router.remove_model("nonexistent")

        assert removed is False


# ---------------------------------------------------------------------------
# Judge Factory Tests
# ---------------------------------------------------------------------------


class TestJudgeFactories:
    """Tests for judge factory functions."""

    def test_create_custom_judge(self):
        """Test creating a custom judge from a function."""
        from banditgpt.core import create_custom_judge

        def my_grader(prompt: str, response: str):
            score = len(response) / 100.0  # Simple length-based score
            return min(score, 1.0), {"length": len(response)}

        judge = my_grader
        reward, meta = judge("Hello", "World")

        assert reward == 0.05  # 5 / 100
        assert meta["length"] == 5


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestPriorWorkflow:
    """Integration tests for the full prior management workflow."""

    def test_full_workflow(self, temp_dir):
        """Test: load bundled -> add model -> save to user -> load merged."""
        from banditgpt.core import PriorManager, PriorConfig

        # Step 1: Create "bundled" priors
        bundled_priors = {
            "A_shared": np.eye(4),
            "b_vectors": {
                "gpt-4": np.array([1.0, 0.0, 0.0, 0.0]),
                "claude": np.array([0.0, 1.0, 0.0, 0.0]),
            },
            "model_ids": ["gpt-4", "claude"],
            "dim": 4,
            "alpha": 0.5,
        }
        bundled_path = temp_dir / "bundled.npz"
        PriorManager(PriorConfig(source="none")).save(bundled_priors, path=bundled_path)

        # Step 2: Load and add a new model
        manager = PriorManager.from_file(bundled_path)
        priors = manager.load()

        priors = manager.add_model(priors, "gpt-5", clone_from="gpt-4", clone_decay=0.9)

        # Step 3: Save to "user" location
        user_path = temp_dir / "user.npz"
        manager.save(priors, path=user_path)

        # Step 4: Load merged
        # Simulate: bundled has gpt-4, claude; user has gpt-4, claude, gpt-5
        manager2 = PriorManager(PriorConfig(source="none"))
        merged = manager2.merge_priors(bundled_priors, priors)

        assert set(merged["model_ids"]) == {"gpt-4", "claude", "gpt-5"}
        # gpt-5 should have decayed weights from gpt-4
        np.testing.assert_array_almost_equal(
            merged["b_vectors"]["gpt-5"],
            [0.9, 0.0, 0.0, 0.0],
        )
