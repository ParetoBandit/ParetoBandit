#!/usr/bin/env python3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
        # A = n_effective * I (default n_effective=5.0)
        assert np.allclose(A, np.eye(dim) * 5.0)

    def test_get_heuristic_prior_values(self):
        """Verify that quality and n_effective correctly set the bias term."""
        dim = 24
        quality = 0.8
        n_eff = 10.0
        init_lambda = 1.0
        model_data = {"initial_quality": quality}

        A, b = get_heuristic_prior(model_data, dim, init_lambda=init_lambda, n_effective=n_eff)

        # b[-1] = quality * (n_effective + init_lambda) so that
        # theta_bias = b[-1] / A[-1,-1] = quality after post-warmup A += lambda*I
        assert b[-1] == quality * (n_eff + init_lambda)
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
            "model_a": {"initial_quality": 0.9, "input_cost_per_m": 1.0, "output_cost_per_m": 3.0},
            "model_b": {"initial_quality": 0.5, "input_cost_per_m": 1.0, "output_cost_per_m": 3.0},
        }

        # Create router with explicit priors file (mock_exists makes it "exist")
        # We inject the mock_fs to avoid real PCA/Encoder initialization
        router = BanditRouter.create(
            model_registry=registry,
            priors="mock_priors.joblib",
            prior_n_effective=20.0,
            feature_service=mock_fs
        )

        # model_a should be from joblib (scaled by n_effective / A[-1,-1])
        # A[-1,-1] = 100.0, so scale = 20.0 / 100.0 = 0.2
        # A_scaled = 100 * 0.2 = 20.0, then + lambda*I → 21.0
        init_lambda = router.bandit.init_lambda
        a_diag = 100.0
        scale_a = 20.0 / a_diag
        assert np.allclose(router.bandit.A["model_a"][0, 0], a_diag * scale_a + init_lambda)
        # b_final = b_orig * scale + init_lambda * theta_true
        # theta_true[0] = 10.0 / 100.0 = 0.1
        expected_b0 = 10.0 * scale_a + init_lambda * (10.0 / a_diag)
        assert np.allclose(router.bandit.b["model_a"][0], expected_b0)

        # model_b should be heuristic-warmup
        # get_heuristic_prior: A = n_effective * I = 20.0 * I
        # Post-warmup: A += lambda * I → (20.0 + 1.0) * I = 21.0 * I
        assert np.allclose(router.bandit.A["model_b"][0, 0], 20.0 + init_lambda)

        # b[-1] = quality * (n_effective + init_lambda) = 0.5 * 21.0 = 10.5
        # calibrate_priors may rescale if bias-only prediction > 1.5;
        # with A_bias = 21.0, theta_bias = 10.5/21.0 = 0.5, which is safe.
        theta_b = router.bandit.A_inv["model_b"] @ router.bandit.b["model_b"]
        assert theta_b[-1] == pytest.approx(0.5, abs=0.05)
        assert np.allclose(router.bandit.b["model_b"][:-1], 0)

if __name__ == "__main__":
    pytest.main([__file__])
