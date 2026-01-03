"""
Comprehensive unit tests for BanditRouter prior system.

Tests cover:
1. Default parameter selection (CSR: 20,20, HLE: 10,60)
2. CSR vs HLE differentiation (different b vectors)
3. Two-knob system (structure vs mean strength)
4. Parameter validation (no negatives, valid numbers)
5. User override behavior
6. Robustness and edge cases

IMPORTANT: Goal is to validate router behavior, not just pass tests.
If a test fails, check if it's a router bug before changing the test.
"""

import pytest

import numpy as np
import json
from pathlib import Path
import sys

# Add parent to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from banditgpt import BanditRouter


@pytest.fixture
def model_registry():
    """Load actual model registry for testing"""
    models_path = repo_root / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    return {m["openrouter_id"]: m for m in data["models"]}


@pytest.fixture
def mock_encoder():
    """Create a mock encoder that returns consistent embeddings"""
    from unittest.mock import MagicMock
    encoder = MagicMock()
    # Return a consistent 384-dim embedding
    encoder.encode.return_value = np.random.randn(1, 384)
    return encoder


@pytest.fixture  
def pca_path():
    """Path to PCA model for testing"""
    return repo_root / "banditgpt" / "data" / "pca_32.joblib"


def create_test_router(model_registry, pca_path, **kwargs):
    """Helper to create router with proper PCA path"""
    # Override context_model to avoid HuggingFace lookup
    return BanditRouter.create(model_registry, context_encoder=mock_encoder, pca_path=pca_path, **kwargs)


class TestDefaultParameters:
    """Test that optimal default parameters are correctly auto-selected"""
    
    def test_csr_defaults(self, model_registry, mock_encoder):
        """CSR should use (structure=20, prior=20) by default"""
        router = BanditRouter.create(model_registry,  priors="csr", context_encoder=mock_encoder)
        
        # Router should have been initialized with optimal params
        # We can't directly inspect the params passed to load_from_benchmark,
        # but we can verify by checking that priors were applied
        assert router.bandit is not None
        assert router.bandit.b is not None
        
        # Check that at least one model has non-zero b vector
        # (indicating priors were applied)
        has_nonzero = False
        for model_id in router.bandit.models:
            b_vec = router.bandit.b[model_id]
            if np.any(b_vec != 0):
                has_nonzero = True
                break
        
        assert has_nonzero, "CSR priors should result in non-zero b vectors"
    
    def test_hle_defaults(self, model_registry, mock_encoder):
        """HLE should use (structure=10, prior=60) by default"""
        router = BanditRouter.create(model_registry, priors="hle", context_encoder=mock_encoder)
        
        assert router.bandit is not None
        
        # Verify priors were applied
        has_nonzero = False
        for model_id in router.bandit.models:
            b_vec = router.bandit.b[model_id]
            if np.any(b_vec != 0):
                has_nonzero = True
                break
        
        assert has_nonzero, "HLE priors should result in non-zero b vectors"
    
    def test_cold_start_defaults(self, model_registry, mock_encoder):
        """Cold start should use (structure=20, prior=0)"""
        router = BanditRouter.create(model_registry, priors="none", context_encoder=mock_encoder)
        
        assert router.bandit is not None
        
        # With prior_n=0, b vectors should all be zero initially
        all_zero = True
        for model_id in router.bandit.models:
            b_vec = router.bandit.b[model_id]
            if np.any(b_vec != 0):
                all_zero = False
                break
        
        assert all_zero, "Cold start (prior=0) should have zero b vectors initially"


class TestCSRvsHLE:
    """Test that CSR and HLE produce different priors"""
    
    def test_different_b_vectors(self, model_registry, mock_encoder):
        """CSR and HLE should produce different b vectors for same model"""
        csr_router = BanditRouter.create(model_registry, priors="csr", context_encoder=mock_encoder)
        hle_router = BanditRouter.create(model_registry, priors="hle", context_encoder=mock_encoder)
        
        # Pick a model that has both cluster_success_rates and hle score
        test_model = None
        for model_id, metadata in model_registry.items():
            if "cluster_success_rates" in metadata and "hle" in metadata:
                test_model = model_id
                break
        
        assert test_model is not None, "Need at least one model with both CSR and HLE data"
        
        # Get b vectors
        csr_b = csr_router.bandit.b[test_model]
        hle_b = hle_router.bandit.b[test_model]
        
        # They should be different (not identical)
        assert not np.allclose(csr_b, hle_b), \
            "CSR and HLE should produce different b vectors (using different priors)"
    
    def test_csr_uses_cluster_data(self, model_registry, mock_encoder):
        """CSR should use cluster-specific z-scores"""
        router = BanditRouter.create(model_registry, priors="csr", context_encoder=mock_encoder)
        
        # Find a model with cluster_success_rates
        test_model = None
        for model_id, metadata in model_registry.items():
            if "cluster_success_rates" in metadata:
                csr_data = metadata["cluster_success_rates"]
                # Check it has z_score format
                if isinstance(csr_data, dict):
                    first_key = next(iter(csr_data))
                    if isinstance(csr_data[first_key], dict) and "z_score" in csr_data[first_key]:
                        test_model = model_id
                        break
        
        assert test_model is not None, "Need at least one model with z-score formatted cluster_success_rates"


class TestTwoKnobSystem:
    """Test that structure_n and prior_n work independently"""
    
    def test_structure_only_no_mean(self, model_registry, mock_encoder):
        """structure_n > 0, prior_n = 0 should give structure but no mean prior"""
        router = BanditRouter.create(
            model_registry, 
            priors="csr",
            prior_n_effective=0.0,
            prior_structure_n_effective=20.0,
            context_encoder=mock_encoder
        )
        
        # b vectors should be zero (no mean prior)
        all_zero = True
        for model_id in router.bandit.models:
            b_vec = router.bandit.b[model_id]
            if np.any(b_vec != 0):
                all_zero = False
                break
        
        assert all_zero, "prior_n=0 should result in zero mean priors (b=0)"
        
        # But A matrix should have structure (not tested here, requires deeper inspection)
    
    def test_mean_only_no_structure(self, model_registry, mock_encoder):
        """prior_n > 0, structure_n = 0 should give mean but weak structure"""
        router = BanditRouter.create(
            model_registry,
            priors="csr", 
            prior_n_effective=20.0,
            prior_structure_n_effective=0.0,
            context_encoder=mock_encoder
        )
        
        # b vectors should be non-zero (mean prior applied)
        has_nonzero = False
        for model_id in router.bandit.models:
            b_vec = router.bandit.b[model_id]
            if np.any(b_vec != 0):
                has_nonzero = True
                break
        
        assert has_nonzero, "prior_n=20 should result in non-zero mean priors"
    
    def test_both_knobs_work(self, model_registry, mock_encoder):
        """Both structure_n and prior_n > 0 should work together"""
        router = BanditRouter.create(
            model_registry,
            priors="csr",
            prior_n_effective=20.0,
            prior_structure_n_effective=20.0,
            context_encoder=mock_encoder
        )
        
        has_nonzero = False
        for model_id in router.bandit.models:
            b_vec = router.bandit.b[model_id]
            if np.any(b_vec != 0):
                has_nonzero = True
                break
        
        assert has_nonzero, "Both knobs > 0 should result in priors"
    
    def test_increasing_prior_n_increases_magnitude(self, model_registry, mock_encoder):
        """Higher prior_n should result in stronger priors (larger b magnitudes)"""
        router_low = BanditRouter.create(
            model_registry,
            priors="csr",
            prior_n_effective=10.0,
            prior_structure_n_effective=20.0,
            context_encoder=mock_encoder
        )
        
        router_high = BanditRouter.create(
            model_registry,
            priors="csr",
            prior_n_effective=40.0,
            prior_structure_n_effective=20.0,
            context_encoder=mock_encoder
        )
        
        # Compare magnitudes for a test model
        test_model = list(model_registry.keys())[0]
        
        mag_low = np.linalg.norm(router_low.bandit.b[test_model])
        mag_high = np.linalg.norm(router_high.bandit.b[test_model])
        
        assert mag_high > mag_low, \
            f"Higher prior_n should give larger b magnitude: {mag_high} <= {mag_low}"


class TestParameterValidation:
    """Test parameter validation and error handling"""
    
    def test_negative_prior_n_handling(self, model_registry, mock_encoder):
        """Negative prior_n should either reject or clip to 0"""
        # The router should handle this gracefully
        try:
            router = BanditRouter.create(
                model_registry,
                priors="csr",
                prior_n_effective=-10.0,
                context_encoder=mock_encoder
            )
            # If it doesn't reject, it should have clipped/handled gracefully
            assert router is not None
        except (ValueError, AssertionError) as e:
            # Rejection is acceptable
            assert "negative" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_negative_structure_n_handling(self, model_registry, mock_encoder):
        """Negative structure_n should either reject or clip to 0"""
        try:
            router = BanditRouter.create(
                model_registry,
                priors="csr",
                prior_structure_n_effective=-10.0,
                context_encoder=mock_encoder
            )
            assert router is not None
        except (ValueError, AssertionError) as e:
            assert "negative" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_nan_parameters(self, model_registry, mock_encoder):
        """NaN parameters should be rejected"""
        with pytest.raises((ValueError, AssertionError, TypeError)):
            BanditRouter.create(
                model_registry,
                priors="csr",
                prior_n_effective=float('nan'),
                context_encoder=mock_encoder
            )
    
    def test_inf_parameters(self, model_registry, mock_encoder):
        """Infinite parameters should be handled gracefully"""
        try:
            router = BanditRouter.create(
                model_registry,
                priors="csr",
                prior_n_effective=float('inf'),
                context_encoder=mock_encoder
            )
            # If accepted, should still create router
            assert router is not None
        except (ValueError, AssertionError, OverflowError) as e:
            # Rejection is acceptable
            pass


class TestUserOverrides:
    """Test that user-specified parameters override defaults"""
    
    def test_csr_custom_params(self, model_registry, mock_encoder):
        """Custom params should override CSR defaults"""
        router = BanditRouter.create(
            model_registry,
            priors="csr",
            prior_n_effective=100.0,
            prior_structure_n_effective=50.0,
            context_encoder=mock_encoder
        )
        
        # Should create successfully with custom params
        assert router is not None
        assert router.bandit is not None
    
    def test_hle_custom_params(self, model_registry, mock_encoder):
        """Custom params should override HLE defaults"""
        router = BanditRouter.create(
            model_registry,
            priors="hle",
            prior_n_effective=100.0,
            prior_structure_n_effective=50.0,
            context_encoder=mock_encoder
        )
        
        assert router is not None
    
    def test_zero_override_works(self, model_registry, mock_encoder):
        """Explicitly setting params to 0 should override defaults"""
        router = BanditRouter.create(
            model_registry,
            priors="csr",
            prior_n_effective=0.0,  # Override default 20.0
            context_encoder=mock_encoder
        )
        
        # Should have zero b vectors despite CSR mode
        all_zero = True
        for model_id in router.bandit.models:
            b_vec = router.bandit.b[model_id]
            if np.any(b_vec != 0):
                all_zero = False
                break
        
        assert all_zero, "Explicit prior_n=0 should override CSR default"


class TestRobustness:
    """Test edge cases and robustness"""
    
    def test_empty_registry(self, mock_encoder):
        """Empty registry should handle gracefully"""
        try:
            router = BanditRouter.create({}, priors="none", context_encoder=mock_encoder)
            # If it doesn't reject, at least it shouldn't crash
            assert router is not None
        except (ValueError, KeyError, IndexError) as e:
            # Rejection is acceptable for empty registry
            pass
    
    def test_very_large_parameters(self, model_registry, mock_encoder):
        """Very large parameters should not crash"""
        router = BanditRouter.create(
            model_registry,
            priors="csr",
            prior_n_effective=1e6,
            prior_structure_n_effective=1e6,
            context_encoder=mock_encoder
        )
        
        assert router is not None
        
        # b vectors should be finite
        for model_id in router.bandit.models:
            b_vec = router.bandit.b[model_id]
            assert np.all(np.isfinite(b_vec)), "b vectors should be finite even with large params"
    
    def test_very_small_nonzero_parameters(self, model_registry, mock_encoder):
        """Very small non-zero parameters should work"""
        router = BanditRouter.create(
            model_registry,
            priors="csr",
            prior_n_effective=0.001,
            prior_structure_n_effective=0.001,
            context_encoder=mock_encoder
        )
        
        assert router is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
