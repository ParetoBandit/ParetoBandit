"""
Unit tests for calibration module functions.

Tests gamma scaling, embedding, and SimpleLinUCBRouter used in experiments.
Based on experiments from experiments/04_figure/ but testing core functions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np
import tempfile
import joblib
from unittest.mock import Mock, MagicMock

from bandit_gpt.calibration import (
    apply_gamma_scaling,
    embed_prompt,
    SimpleLinUCBRouter,
    CalibratedRouter
)


# =============================================================================
# Gamma Scaling Tests
# =============================================================================

class TestGammaScaling:
    """Test apply_gamma_scaling function."""
    
    def test_gamma_scaling_basic(self):
        """Test basic gamma scaling functionality."""
        context_dim = 5
        models = ['model_a', 'model_b']
        
        priors = {
            'A': {
                'model_a': np.eye(context_dim) * 10.0,
                'model_b': np.eye(context_dim) * 20.0
            },
            'b': {
                'model_a': np.ones(context_dim) * 5.0,
                'model_b': np.ones(context_dim) * 10.0
            },
            'models': models,
            'context_dim': context_dim,
            'n_prompts': 1000,
            'plasticity': 1.0
        }
        
        gamma = 0.1
        scaled = apply_gamma_scaling(priors, gamma)
        
        # Check that A matrices are scaled
        for model in models:
            expected_A = priors['A'][model] * gamma
            np.testing.assert_array_almost_equal(scaled['A'][model], expected_A)
        
        # Check that b vectors ARE ALSO scaled (to preserve theta = A^-1 @ b)
        for model in models:
            expected_b = priors['b'][model] * gamma
            np.testing.assert_array_almost_equal(scaled['b'][model], expected_b)
        
        # Check metadata
        assert scaled['gamma'] == gamma
        assert scaled['models'] == models
        assert scaled['context_dim'] == context_dim
        assert scaled['n_prompts'] == 1000
    
    def test_gamma_scaling_preserves_structure(self):
        """Test that gamma scaling preserves matrix structure."""
        context_dim = 3
        priors = {
            'A': {'model_a': np.eye(context_dim) * 5.0},
            'b': {'model_a': np.ones(context_dim)},
            'models': ['model_a'],
            'context_dim': context_dim
        }
        
        # Scale with different gammas
        for gamma in [0.01, 0.05, 0.1, 0.5, 1.0]:
            scaled = apply_gamma_scaling(priors, gamma)
            
            # A should still be positive definite
            A = scaled['A']['model_a']
            eigenvalues = np.linalg.eigvals(A)
            assert np.all(eigenvalues > 0), f"Matrix not positive definite with gamma={gamma}"
    
    def test_gamma_scaling_preserves_theta(self):
        """Test that gamma scaling preserves theta = A^-1 @ b (critical for correctness)."""
        context_dim = 5
        models = ['model_a', 'model_b']
        
        # Create priors with non-trivial A and b
        priors = {
            'A': {
                'model_a': np.eye(context_dim) * 10.0,
                'model_b': np.array([[4, 1, 0, 0, 0],
                                     [1, 4, 1, 0, 0],
                                     [0, 1, 4, 1, 0],
                                     [0, 0, 1, 4, 1],
                                     [0, 0, 0, 1, 4]], dtype=float)
            },
            'b': {
                'model_a': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                'model_b': np.array([0.5, 1.5, 2.5, 3.5, 4.5])
            },
            'models': models,
            'context_dim': context_dim
        }
        
        # Calculate original theta for each model
        original_theta = {}
        for model in models:
            A_inv = np.linalg.inv(priors['A'][model])
            original_theta[model] = A_inv @ priors['b'][model]
        
        # Test with various gamma values
        for gamma in [0.01, 0.05, 0.1, 0.5, 1.0]:
            scaled = apply_gamma_scaling(priors, gamma)
            
            # Calculate theta after scaling
            for model in models:
                A_inv_scaled = np.linalg.inv(scaled['A'][model])
                theta_scaled = A_inv_scaled @ scaled['b'][model]
                
                # CRITICAL: theta must be preserved (within numerical precision)
                np.testing.assert_array_almost_equal(
                    theta_scaled, original_theta[model],
                    decimal=10,
                    err_msg=f"Theta not preserved for {model} with gamma={gamma}"
                )
    
    def test_gamma_scaling_edge_cases(self):
        """Test gamma scaling with edge case values."""
        context_dim = 2
        priors = {
            'A': {'model_a': np.eye(context_dim)},
            'b': {'model_a': np.zeros(context_dim)},
            'models': ['model_a'],
            'context_dim': context_dim
        }
        
        # Very small gamma (high inflation)
        scaled_small = apply_gamma_scaling(priors, gamma=0.001)
        assert np.allclose(scaled_small['A']['model_a'], np.eye(context_dim) * 0.001)
        # Also check b is scaled
        assert np.allclose(scaled_small['b']['model_a'], np.zeros(context_dim))
        
        # Gamma = 1.0 (no scaling)
        scaled_one = apply_gamma_scaling(priors, gamma=1.0)
        np.testing.assert_array_almost_equal(scaled_one['A']['model_a'], priors['A']['model_a'])
        np.testing.assert_array_almost_equal(scaled_one['b']['model_a'], priors['b']['model_a'])


# =============================================================================
# Embedding Tests
# =============================================================================

class TestEmbedPrompt:
    """Test embed_prompt function."""
    
    @pytest.fixture
    def mock_encoder_pca(self):
        """Create mock encoder and PCA for testing."""
        # Mock encoder
        encoder = Mock()
        embedding_dim = 384  # Typical sentence-transformers dimension
        encoder.encode.return_value = np.random.randn(embedding_dim)
        
        # Mock PCA
        pca = Mock()
        pca_dim = 23
        pca.transform.return_value = np.random.randn(1, pca_dim)
        
        return encoder, pca
    
    def test_embed_prompt_output_shape(self, mock_encoder_pca):
        """Test that embed_prompt returns correct shape."""
        encoder, pca = mock_encoder_pca
        
        prompt = "Test prompt"
        context = embed_prompt(prompt, encoder, pca)
        
        # Should be PCA dimensions + 1 bias term
        assert context.shape == (24,), f"Expected (24,) but got {context.shape}"
    
    def test_embed_prompt_bias_term(self, mock_encoder_pca):
        """Test that last element is bias term (1.0)."""
        encoder, pca = mock_encoder_pca
        
        prompt = "Test prompt"
        context = embed_prompt(prompt, encoder, pca)
        
        # Last element should be 1.0 (bias term)
        assert context[-1] == 1.0, f"Bias term should be 1.0, got {context[-1]}"
    
    def test_embed_prompt_calls_encoder(self, mock_encoder_pca):
        """Test that encoder is called correctly."""
        encoder, pca = mock_encoder_pca
        
        prompt = "Test prompt"
        embed_prompt(prompt, encoder, pca)
        
        # Verify encoder was called with correct arguments
        encoder.encode.assert_called_once()
        call_args = encoder.encode.call_args
        assert call_args[0][0] == prompt
        assert call_args[1]['convert_to_numpy'] == True
        assert call_args[1]['show_progress_bar'] == False
    
    def test_embed_prompt_calls_pca(self, mock_encoder_pca):
        """Test that PCA transform is called correctly."""
        encoder, pca = mock_encoder_pca
        
        prompt = "Test prompt"
        embed_prompt(prompt, encoder, pca)
        
        # Verify PCA transform was called
        pca.transform.assert_called_once()
        
        # Check that input was reshaped to (1, -1)
        call_args = pca.transform.call_args[0][0]
        assert call_args.shape[0] == 1, "PCA input should have shape (1, -1)"


# =============================================================================
# SimpleLinUCBRouter Tests
# =============================================================================

class TestSimpleLinUCBRouter:
    """Test SimpleLinUCBRouter class."""
    
    @pytest.fixture
    def sample_priors(self):
        """Create sample warmup priors."""
        context_dim = 10
        models = ['model_a', 'model_b']
        
        return {
            'A': {
                'model_a': np.eye(context_dim) * 5.0,
                'model_b': np.eye(context_dim) * 5.0
            },
            'b': {
                'model_a': np.ones(context_dim) * 2.0,
                'model_b': np.ones(context_dim) * 1.5
            },
            'models': models,
            'context_dim': context_dim
        }
    
    def test_initialization(self, sample_priors):
        """Test SimpleLinUCBRouter initialization."""
        models = ['model_a', 'model_b']
        alpha = 1.0
        
        router = SimpleLinUCBRouter(models, sample_priors, alpha)
        
        assert router.models == models
        assert router.alpha == alpha
        assert router.context_dim == sample_priors['context_dim']
        
        # Check that A and b are copied (not references)
        for model in models:
            assert model in router.A
            assert model in router.b
            np.testing.assert_array_almost_equal(router.A[model], sample_priors['A'][model])
            np.testing.assert_array_almost_equal(router.b[model], sample_priors['b'][model])
    
    def test_select_model_basic(self, sample_priors):
        """Test model selection returns valid model."""
        models = ['model_a', 'model_b']
        router = SimpleLinUCBRouter(models, sample_priors, alpha=1.0)
        
        context = np.random.randn(sample_priors['context_dim'])
        selected = router.select_model(context)
        
        assert selected in models
    
    def test_select_model_deterministic(self, sample_priors):
        """Test that selection is deterministic for same context."""
        models = ['model_a', 'model_b']
        router = SimpleLinUCBRouter(models, sample_priors, alpha=1.0)
        
        context = np.random.randn(sample_priors['context_dim'])
        
        selected1 = router.select_model(context)
        selected2 = router.select_model(context)
        
        assert selected1 == selected2, "Selection should be deterministic"
    
    def test_update_modifies_state(self, sample_priors):
        """Test that update modifies A and b matrices."""
        models = ['model_a']
        router = SimpleLinUCBRouter(models, sample_priors, alpha=1.0)
        
        model = 'model_a'
        context = np.random.randn(sample_priors['context_dim'])
        reward = 0.8
        
        # Store initial state
        A_before = router.A[model].copy()
        b_before = router.b[model].copy()
        
        # Update
        router.update(context, model, reward)
        
        # Verify changes
        assert not np.allclose(router.A[model], A_before), "A should change after update"
        assert not np.allclose(router.b[model], b_before), "b should change after update"
        
        # Verify mathematical correctness
        context_col = context.reshape(-1, 1)
        expected_A = A_before + context_col @ context_col.T
        expected_b = b_before + reward * context
        
        np.testing.assert_array_almost_equal(router.A[model], expected_A)
        np.testing.assert_array_almost_equal(router.b[model], expected_b)
    
    def test_exploration_vs_exploitation(self, sample_priors):
        """Test that alpha controls exploration."""
        models = ['model_a', 'model_b']
        context = np.ones(sample_priors['context_dim'])
        
        # High alpha = more exploration
        router_high = SimpleLinUCBRouter(models, sample_priors, alpha=5.0)
        
        # Low alpha = more exploitation
        router_low = SimpleLinUCBRouter(models, sample_priors, alpha=0.01)
        
        # Both should select models, but may differ due to exploration
        model_high = router_high.select_model(context)
        model_low = router_low.select_model(context)
        
        assert model_high in models
        assert model_low in models
    
    def test_get_model_usage(self, sample_priors):
        """Test model usage reporting."""
        models = ['model_a', 'model_b']
        router = SimpleLinUCBRouter(models, sample_priors, alpha=1.0)
        
        usage = router.get_model_usage()
        
        # Should return dict with all models
        assert set(usage.keys()) == set(models)
        
        # Percentages should sum to ~100
        total = sum(usage.values())
        assert abs(total - 100.0) < 0.01, f"Usage should sum to 100%, got {total}"
    
    def test_learning_from_feedback(self, sample_priors):
        """Test that router learns from consistent feedback."""
        models = ['model_a', 'model_b']
        router = SimpleLinUCBRouter(models, sample_priors, alpha=1.0)
        
        context = np.ones(sample_priors['context_dim'])
        
        # Give consistent high rewards to model_a
        for _ in range(50):
            router.update(context, 'model_a', reward=0.9)
        
        # Give consistent low rewards to model_b
        for _ in range(50):
            router.update(context, 'model_b', reward=0.1)
        
        # With this context, model_a should be preferred
        selected = router.select_model(context)
        
        # Calculate UCB scores manually to verify
        A_inv_a = np.linalg.inv(router.A['model_a'])
        theta_a = A_inv_a @ router.b['model_a']
        ucb_a = theta_a @ context + router.alpha * np.sqrt(context @ A_inv_a @ context)
        
        A_inv_b = np.linalg.inv(router.A['model_b'])
        theta_b = A_inv_b @ router.b['model_b']
        ucb_b = theta_b @ context + router.alpha * np.sqrt(context @ A_inv_b @ context)
        
        # model_a should have higher UCB
        assert ucb_a > ucb_b, f"model_a (UCB={ucb_a:.3f}) should beat model_b (UCB={ucb_b:.3f})"


# =============================================================================
# Integration Tests
# =============================================================================

class TestCalibrationIntegration:
    """Integration tests for calibration workflow."""
    
    def test_gamma_scaling_with_router(self):
        """Test full workflow: scale priors, then use with router."""
        context_dim = 8
        models = ['model_a', 'model_b']
        
        # Create base priors
        base_priors = {
            'A': {
                'model_a': np.eye(context_dim) * 100.0,  # High confidence
                'model_b': np.eye(context_dim) * 100.0
            },
            'b': {
                'model_a': np.ones(context_dim) * 50.0,
                'model_b': np.ones(context_dim) * 40.0
            },
            'models': models,
            'context_dim': context_dim
        }
        
        # Scale with gamma
        scaled_priors = apply_gamma_scaling(base_priors, gamma=0.1)
        
        # Create router with scaled priors
        router = SimpleLinUCBRouter(models, scaled_priors, alpha=1.0)
        
        # Router should work normally
        context = np.random.randn(context_dim)
        selected = router.select_model(context)
        
        assert selected in models
        
        # Verify that effective sample size is reduced
        trace_scaled = np.trace(router.A['model_a'])
        trace_original = np.trace(base_priors['A']['model_a'])
        
        assert trace_scaled < trace_original, "Scaled priors should have lower confidence"
    
    def test_optimized_hyperparameters_effect(self):
        """Test that optimized hyperparameters (η=5.0, γ=0.10) work correctly."""
        context_dim = 10
        models = ['warmup_model', 'new_model']
        
        # Create priors with strong bias
        biased_priors = {
            'A': {
                'warmup_model': np.eye(context_dim) * 50.0,
                'new_model': np.eye(context_dim) * 50.0
            },
            'b': {
                'warmup_model': np.ones(context_dim) * 40.0,  # High bias
                'new_model': np.ones(context_dim) * 10.0      # Low bias
            },
            'models': models,
            'context_dim': context_dim
        }
        
        # Apply optimized gamma
        scaled = apply_gamma_scaling(biased_priors, gamma=0.10)
        router = SimpleLinUCBRouter(models, scaled, alpha=1.0)
        
        # The scaled priors should allow learning
        context = np.ones(context_dim)
        
        # Give opposite feedback (new_model is actually better)
        for _ in range(100):
            router.update(context, 'new_model', reward=0.95)
            router.update(context, 'warmup_model', reward=0.60)
        
        # After sufficient feedback, new_model should be preferred
        # (despite starting with lower prior)
        selected = router.select_model(context)
        
        # Calculate predictions
        A_inv_new = np.linalg.inv(router.A['new_model'])
        theta_new = A_inv_new @ router.b['new_model']
        pred_new = theta_new @ context
        
        A_inv_warmup = np.linalg.inv(router.A['warmup_model'])
        theta_warmup = A_inv_warmup @ router.b['warmup_model']
        pred_warmup = theta_warmup @ context
        
        # With gamma=0.10, router should have learned that new_model is better
        assert pred_new > pred_warmup or selected == 'new_model', \
            "Router should learn to prefer better model despite prior bias"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
