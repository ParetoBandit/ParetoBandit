"""
Tests for the Model Manager module.

Tests model cache operations (no API calls required).
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_cache():
    """Create a temporary models cache for testing."""
    with tempfile.TemporaryDirectory() as d:
        cache_path = Path(d) / "models_cache.json"
        cache_path.write_text(json.dumps({
            "models": [
                {
                    "openrouter_id": "test/model-a",
                    "display_name": "Model A",
                    "name": "model-a",
                    "input_cost_per_m": 1.0,
                    "output_cost_per_m": 2.0,
                    "price_1m_blended": 1.25,
                    "time_to_first_token_seconds": 0.5,
                    "output_tokens_per_second": 50,
                },
                {
                    "openrouter_id": "test/model-b",
                    "display_name": "Model B",
                    "name": "model-b",
                    "input_cost_per_m": 0.5,
                    "output_cost_per_m": 1.0,
                    "price_1m_blended": 0.625,
                    "time_to_first_token_seconds": 0.3,
                    "output_tokens_per_second": 100,
                },
            ]
        }))
        yield cache_path


class TestModelCacheOperations:
    """Test cache load/save operations."""

    def test_load_models_cache(self, temp_cache):
        from banditgpt.core.model_manager import load_models_cache

        cache = load_models_cache(temp_cache)
        
        assert "models" in cache
        assert len(cache["models"]) == 2
        assert cache["models"][0]["openrouter_id"] == "test/model-a"

    def test_save_models_cache(self, temp_cache):
        from banditgpt.core.model_manager import load_models_cache, save_models_cache

        cache = load_models_cache(temp_cache)
        cache["models"].append({
            "openrouter_id": "test/model-c",
            "display_name": "Model C",
        })
        
        save_models_cache(cache, temp_cache)
        
        # Reload and verify
        reloaded = load_models_cache(temp_cache)
        assert len(reloaded["models"]) == 3

    def test_list_models(self, temp_cache):
        from banditgpt.core.model_manager import list_models

        models = list_models(temp_cache)
        
        assert len(models) == 2
        assert models[0]["openrouter_id"] == "test/model-a"
        assert models[1]["openrouter_id"] == "test/model-b"

    def test_remove_model_from_cache(self, temp_cache):
        from banditgpt.core.model_manager import remove_model_from_cache, list_models

        result = remove_model_from_cache("test/model-a", cache_path=temp_cache)
        
        assert result is True
        models = list_models(temp_cache)
        assert len(models) == 1
        assert models[0]["openrouter_id"] == "test/model-b"

    def test_remove_nonexistent_model(self, temp_cache):
        from banditgpt.core.model_manager import remove_model_from_cache

        result = remove_model_from_cache("test/nonexistent", cache_path=temp_cache)
        
        assert result is False


class TestTTFTEstimates:
    """Test TTFT estimate initialization."""

    def test_initialize_ttft_estimates(self, temp_cache):
        from banditgpt.core.model_manager import initialize_ttft_estimates, load_models_cache

        count = initialize_ttft_estimates(temp_cache)
        
        assert count == 2
        
        cache = load_models_cache(temp_cache)
        model = cache["models"][0]
        
        # Check new fields were added
        assert "ttft_mean" in model
        assert "ttft_std" in model
        assert "ttft_ci_95_lower" in model
        assert "ttft_ci_95_upper" in model
        assert "ttft_p50" in model
        assert "ttft_p95" in model
        assert "ttft_p99" in model
        assert model["ttft_samples"] == 0  # Estimated, not measured

    def test_ttft_estimates_values(self, temp_cache):
        from banditgpt.core.model_manager import initialize_ttft_estimates, load_models_cache

        initialize_ttft_estimates(temp_cache)
        
        cache = load_models_cache(temp_cache)
        model = cache["models"][0]
        
        ttft = model["time_to_first_token_seconds"]
        
        # Check estimates are reasonable
        assert model["ttft_mean"] == ttft
        assert model["ttft_std"] == ttft * 0.20  # 20% of mean
        assert model["ttft_ci_95_lower"] == ttft * 0.90  # mean - 10%
        assert model["ttft_ci_95_upper"] == ttft * 1.10  # mean + 10%
        assert model["ttft_p95"] == ttft * 1.3
        assert model["ttft_p99"] == ttft * 1.5

    def test_skip_already_measured(self, temp_cache):
        from banditgpt.core.model_manager import initialize_ttft_estimates, load_models_cache, save_models_cache

        # Pre-set one model as already measured
        cache = load_models_cache(temp_cache)
        cache["models"][0]["ttft_samples"] = 100
        save_models_cache(cache, temp_cache)
        
        count = initialize_ttft_estimates(temp_cache)
        
        # Should only update the second model
        assert count == 1


class TestModelCacheFields:
    """Test that cache has required fields for router."""

    def test_required_fields_present(self, temp_cache):
        from banditgpt.core.model_manager import load_models_cache

        cache = load_models_cache(temp_cache)
        model = cache["models"][0]
        
        # Required for router
        assert "openrouter_id" in model
        assert "display_name" in model or "name" in model
        
        # Cost fields
        assert "input_cost_per_m" in model or "price_1m_input" in model
        assert "output_cost_per_m" in model or "price_1m_output" in model
        
        # Latency fields
        assert "time_to_first_token_seconds" in model

    def test_build_registry_compatibility(self, temp_cache):
        """Test that cache works with build_registry_from_models_cache."""
        from banditgpt.core.bandit_router import build_registry_from_models_cache

        registry = build_registry_from_models_cache(temp_cache)
        
        assert "test/model-a" in registry
        assert "test/model-b" in registry
        
        # Check derived fields
        assert "cost" in registry["test/model-a"]
        assert "latency_s" in registry["test/model-a"]
        assert registry["test/model-a"]["cost"] > 0
        assert registry["test/model-a"]["latency_s"] > 0


class TestCallOpenRouter:
    """Test OpenRouter call function (mocked)."""

    def test_call_requires_api_key(self, monkeypatch):
        from banditgpt.core.model_manager import call_openrouter

        # Ensure no API key
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            call_openrouter("test/model", "Hello")

    def test_call_requires_openai_package(self, monkeypatch):
        import sys
        
        # This test would require mocking the import, skip if openai is installed
        try:
            import openai
            pytest.skip("openai package is installed")
        except ImportError:
            from banditgpt.core.model_manager import call_openrouter
            
            monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
            
            with pytest.raises(ImportError, match="openai"):
                call_openrouter("test/model", "Hello")
