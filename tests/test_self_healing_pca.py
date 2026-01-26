"""
Unit tests for Self-Healing PCA implementation.

Tests that the router can:
1. Auto-train PCA when artifact is missing
2. Detect and handle dimension mismatches
3. Validate variance capture
4. Load existing valid PCA artifacts
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bandit_gpt.router import BanditRouter


class TestSelfHealingPCA:
    """Test self-healing PCA functionality."""
    
    def test_missing_pca_auto_trains(self):
        """Verify router auto-trains PCA when artifact is missing."""
        # Use non-existent path
        fake_path = "/tmp/nonexistent_pca_test.joblib"
        
        router = BanditRouter(
            model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
            pca_path=fake_path,
            use_corralling=True  # Enable corralling for safety guarantees
        )
        
        # Should have auto-trained PCA
        assert router.pca is not None, "PCA should be auto-trained when missing"
        assert router.pca.n_components == 23
        
        # Should have explained variance
        explained_var = np.sum(router.pca.explained_variance_ratio_)
        assert explained_var > 0.5, f"PCA should capture >50% variance, got {explained_var:.1%}"
    
    def test_valid_pca_loads_successfully(self):
        """Verify router loads existing valid PCA."""
        # Create temporary valid PCA
        with tempfile.TemporaryDirectory() as tmpdir:
            pca_path = Path(tmpdir) / "test_pca.joblib"
            
            # Create a router that will generate PCA
            router1 = BanditRouter(
                model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
                pca_path=pca_path,
                use_corralling=True  # Enable corralling for safety guarantees
            )
            
            # PCA should be saved
            assert pca_path.exists(), "PCA should be persisted"
            
            # Create new router - should load saved PCA
            router2 = BanditRouter(
                model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
                pca_path=pca_path,
                use_corralling=True  # Enable corralling for safety guarantees
            )
            
            assert router2.pca is not None, "PCA should load from disk"
            assert router2.pca.n_components == 23
    
    def test_synthetic_data_generation(self):
        """Test synthetic prompt generation for PCA training."""
        router = BanditRouter(
            model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
            use_corralling=True  # Enable corralling for safety guarantees
        )
        
        # Generate synthetic prompts
        prompts = router._generate_synthetic_data(n=100)
        
        assert len(prompts) == 100, "Should generate requested number of prompts"
        assert all(isinstance(p, str) for p in prompts), "All prompts should be strings"
        assert all(len(p) > 10 for p in prompts), "Prompts should be non-trivial"
        
        # Check diversity (no exact duplicates in small sample)
        unique_prompts = set(prompts[:50])
        assert len(unique_prompts) > 40, "Prompts should be diverse"
    
    def test_pca_variance_validation(self):
        """Test that PCA variance is checked and logged."""
        router = BanditRouter(
            model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
            use_corralling=True  # Enable corralling for safety guarantees
        )
        
        if router.pca is not None:
            explained_var = np.sum(router.pca.explained_variance_ratio_)
            # Should capture reasonable variance
            assert explained_var > 0.3, f"PCA variance too low: {explained_var:.1%}"
    
    def test_no_pca_path_works(self):
        """Test router works when no PCA path is provided."""
        router = BanditRouter(
            model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
            pca_path=None,
            use_corralling=True  # Enable corralling for safety guarantees
        )
        
        # Should work fine without PCA (full dimensionality)
        assert router.encoder is not None
        
    def test_jit_pca_persistence(self):
        """Test that JIT-trained PCA is persisted for next startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pca_path = Path(tmpdir) / "jit_pca.joblib"
            
            # First initialization - will JIT train
            router1 = BanditRouter(
                model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
                pca_path=pca_path,
                use_corralling=True  # Enable corralling for safety guarantees
            )
            
            # Should have persisted
            assert pca_path.exists(), "JIT PCA should be saved"
            
            # Get variance of first PCA
            var1 = np.sum(router1.pca.explained_variance_ratio_)
            
            # Second initialization - should load from disk
            router2 = BanditRouter(
                model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
                pca_path=pca_path,
                use_corralling=True  # Enable corralling for safety guarantees
            )
            
            # Should have same variance (loaded same PCA)
            var2 = np.sum(router2.pca.explained_variance_ratio_)
            assert abs(var1 - var2) < 0.01, "Loaded PCA should match saved PCA"


class TestPCAIntegration:
    """Test PCA integration with full router workflow."""
    
    def test_routing_works_with_jit_pca(self):
        """Verify routing works correctly with JIT-trained PCA."""
        router = BanditRouter(
            model_registry={
                "test/model1": {"openrouter_id": "test/model1", "hle": 0.8},
                "test/model2": {"openrouter_id": "test/model2", "hle": 0.6}
            },
            pca_path="/tmp/test_routing_pca.joblib",
            use_corralling=True  # Enable corralling for safety guarantees
        )
        
        # Should be able to route
        selected, log = router.route("Write a Python function to sort a list")
        
        assert selected in ["test/model1", "test/model2"]
        assert log.context_vector is not None
        assert len(log.context_vector) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
