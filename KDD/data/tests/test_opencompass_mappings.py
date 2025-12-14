"""
Unit tests for opencompass_name_mappings.py

Tests the model name mapping functionality between OpenCompass and cache names.
"""

import sys
import os
import pytest

# Add core_scripts directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'core_scripts'))

from opencompass_name_mappings import OPENCOMPASS_TO_CACHE


class TestOpenCompassMappings:
    """Test model name mapping dictionary."""
    
    def test_mapping_exists(self):
        """Test that mapping dictionary exists and is not empty."""
        assert OPENCOMPASS_TO_CACHE is not None
        assert len(OPENCOMPASS_TO_CACHE) > 0
        assert isinstance(OPENCOMPASS_TO_CACHE, dict)
    
    def test_common_models_mapped(self):
        """Test that common models are properly mapped."""
        # Test some known critical mappings (use actual current mappings)
        critical_mappings = {
            'gpt-4o-mini-2024-07-18': 'GPT-4o mini',
            'gpt4o-20240806': 'GPT-4o',
            'claude-3-7-sonnet-20250219': 'Claude 3.7 Sonnet (Reasoning)',
            'qwen2.5-72b-instruct-turbomind': 'Qwen2.5 Instruct 72B'
        }
        
        for opencompass_name, expected_cache_name in critical_mappings.items():
            assert opencompass_name in OPENCOMPASS_TO_CACHE, \
                f"Missing mapping for {opencompass_name}"
            assert OPENCOMPASS_TO_CACHE[opencompass_name] == expected_cache_name, \
                f"Wrong mapping: {opencompass_name} -> {OPENCOMPASS_TO_CACHE[opencompass_name]}, expected {expected_cache_name}"
    
    def test_all_values_are_strings(self):
        """Test that all dictionary values are strings."""
        for key, value in OPENCOMPASS_TO_CACHE.items():
            assert isinstance(key, str), f"Key {key} is not a string"
            assert isinstance(value, str), f"Value {value} is not a string"
    
    def test_no_empty_mappings(self):
        """Test that no mappings are empty strings."""
        for key, value in OPENCOMPASS_TO_CACHE.items():
            assert len(key) > 0, f"Empty key found"
            assert len(value) > 0, f"Empty value for key {key}"
    
    def test_unique_mappings(self):
        """Test that OpenCompass names are unique (no duplicates)."""
        keys = list(OPENCOMPASS_TO_CACHE.keys())
        assert len(keys) == len(set(keys)), "Duplicate OpenCompass names found"
    
    def test_models_for_each_intent(self):
        """Test that we have mappings for models used in each intent."""
        # RAG models (from validate_rag_with_mmlu_pro.py)
        rag_models = [
            'gpt-4o-mini-2024-07-18',
            'qwen2.5-72b-instruct-turbomind',
            'deepseek-chat-v3'
        ]
        
        for model in rag_models:
            assert model in OPENCOMPASS_TO_CACHE, \
                f"Missing RAG model: {model}"
    
    def test_mapping_count(self):
        """Test that we have a reasonable number of mappings."""
        # Should have at least 40 models mapped
        assert len(OPENCOMPASS_TO_CACHE) >= 40, \
            f"Too few mappings: {len(OPENCOMPASS_TO_CACHE)}"


class TestMappingUsage:
    """Test typical usage patterns of the mappings."""
    
    def test_lookup_existing_model(self):
        """Test looking up an existing model."""
        result = OPENCOMPASS_TO_CACHE.get('gpt-4o-mini-2024-07-18')
        assert result == 'GPT-4o mini'
    
    def test_lookup_missing_model(self):
        """Test looking up a non-existent model returns None."""
        result = OPENCOMPASS_TO_CACHE.get('non-existent-model-12345')
        assert result is None
    
    def test_fallback_pattern(self):
        """Test the typical fallback pattern used in code."""
        # This is how the code typically uses the mapping
        opencompass_name = 'gpt-4o-mini-2024-07-18'
        cache_name = OPENCOMPASS_TO_CACHE.get(opencompass_name, opencompass_name)
        assert cache_name == 'GPT-4o mini'
        
        # Test fallback for missing model
        missing_name = 'missing-model'
        cache_name = OPENCOMPASS_TO_CACHE.get(missing_name, missing_name)
        assert cache_name == 'missing-model'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
