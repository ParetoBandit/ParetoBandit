#!/usr/bin/env python3
import pytest
import numpy as np
import joblib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from pareto_bandit.router import BanditRouter
from pareto_bandit.utils.warmup import get_heuristic_prior

class TestHeuristicPrior:
    def test_get_heuristic_prior_shapes(self):
        """Verify that get_heuristic_prior returns matrices of correct shape."""
        dim = 24
        model_data = {"quality_score": 0.8}
        A, b = get_heuristic_prior(model_data, dim)
        
        assert A.shape == (dim, dim)
        assert b.shape == (dim,)
        assert np.allclose(A, np.eye(dim))
        
    def test_get_heuristic_prior_values(self):
        """Verify that quality and n_effective correctly set the bias term."""
        dim = 24
        quality = 0.8
        n_eff = 10.0
        model_data = {"initial_quality": quality}
        
        A, b = get_heuristic_prior(model_data, dim, n_effective=n_eff)
        
        # Bias should be the last element
        assert b[-1] == quality * n_eff
        assert np.all(b[:-1] == 0)

class TestCascadingWarmup:
    @patch('joblib.load')
    @patch('pathlib.Path.exists')
    def test_cascading_initialization_flow(self, mock_exists, mock_load):
        """Verify that BanditRouter.create correctly handles hits and misses in joblib."""
        mock_exists.return_value = True
        
        # Mock joblib data: only 'model_a' exists
        mock_load.return_value = {
            "A": {"model_a": np.eye(24) * 100},
            "b": {"model_a": np.ones(24) * 10},
            "n": 20000
        }
        
        # Mock feature service instance
        mock_fs = MagicMock()
        mock_fs.dimension = 24
        mock_fs.pca_components = 23
        
        # Registry with two models
        registry = {
            # get_heuristic_prior expects "initial_quality" (not legacy "quality_score")
            "model_a": {"initial_quality": 0.9},
            "model_b": {"initial_quality": 0.5}  # Missing from joblib
        }
        
        # Create router with explicit priors file (mock_exists makes it "exist")
        # We inject the mock_fs to avoid real PCA/Encoder initialization
        router = BanditRouter.create(
            model_registry=registry,
            priors="mock_priors.joblib",
            prior_n_effective=20.0,
            feature_service=mock_fs
        )
        
        # model_a should be from joblib (scaled)
        # scale = 20 / 20000 = 0.001
        # scaled_A = 100 * 0.001 = 0.1
        # b = 10 * 0.001 = 0.01
        
        # Note: router.bandit.A["model_a"] will also have init_lambda adding np.eye later in router.py
        # router.bandit.A[model_id] += np.eye(router.bandit.dim) * router.bandit.init_lambda (default lambda=1.0)
        # So A_final = 0.1*I + 1.0*I = 1.1*I
        assert np.allclose(router.bandit.A["model_a"][0, 0], 1.1)
        assert np.allclose(router.bandit.b["model_a"][0], 0.01)
        
        # model_b should be heuristic-warmup
        # A_heuristic is initialized with init_lambda*I = 1.0*I
        # THEN BanditRouter.create adds lambda*I again:
        # So model_b A should be 2.0*I
        assert np.allclose(router.bandit.A["model_b"][0, 0], 2.0)
        
        # b[-1] initially encodes quality * prior_n_effective = 0.5 * 20.0 = 10.0.
        # However BanditRouter.create() runs calibrate_priors(), which clamps
        # extreme bias-only predictions to target_max_pred=0.9 via theta
        # reconstruction. With A = 2I after post-warmup regularization,
        # clamped theta_bias=0.9 implies b_bias = 2 * 0.9 = 1.8.
        assert router.bandit.b["model_b"][-1] == pytest.approx(1.8)
        assert np.all(router.bandit.b["model_b"][:-1] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
