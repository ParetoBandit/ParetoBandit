"""
Unit tests for train_final_xgboost_models.py

Tests the model training pipeline functions.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataLoading:
    """Test data loading functions."""
    
    def test_training_data_exists(self):
        """Test that training data file exists."""
        data_path = Path(__file__).parent.parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
        assert data_path.exists(), "Training data file not found"
    
    def test_training_data_structure(self):
        """Test that training data has expected structure."""
        data_path = Path(__file__).parent.parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
        
        # Load a sample
        df = pd.read_csv(data_path, nrows=100, low_memory=False)
        
        # Check required columns
        required_columns = [
            'intent', 'model', 'prompt', 'success',
            'nvidia_creativity', 'nvidia_reasoning', 'nvidia_constraint',
            'nvidia_domain_knowledge', 'nvidia_contextual_knowledge', 'nvidia_few_shots'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_intent_distribution(self):
        """Test that all 4 intents are present in data."""
        data_path = Path(__file__).parent.parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
        df = pd.read_csv(data_path, low_memory=False)
        
        intents = set(df['intent'].unique())
        expected_intents = {'reasoning', 'coding', 'summarization', 'rag'}
        
        assert intents == expected_intents, f"Expected {expected_intents}, got {intents}"
    
    def test_success_is_binary(self):
        """Test that success column is binary (0 or 1)."""
        data_path = Path(__file__).parent.parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
        df = pd.read_csv(data_path, nrows=1000, low_memory=False)
        
        unique_values = set(df['success'].unique())
        assert unique_values.issubset({0, 1, 0.0, 1.0}), \
            f"Success column should be binary, got: {unique_values}"


class TestModelsCache:
    """Test models cache structure."""
    
    def test_cache_exists(self):
        """Test that models_cache.json exists."""
        cache_path = Path(__file__).parent.parent.parent.parent / 'data' / 'models_cache.json'
        assert cache_path.exists(), "models_cache.json not found"
    
    def test_cache_structure(self):
        """Test models cache has expected structure."""
        cache_path = Path(__file__).parent.parent.parent.parent / 'data' / 'models_cache.json'
        
        with open(cache_path) as f:
            cache_data = json.load(f)
        
        assert 'models' in cache_data, "Cache should have 'models' key"
        assert isinstance(cache_data['models'], list), "Models should be a list"
        assert len(cache_data['models']) > 0, "Should have at least some models"
    
    def test_model_has_required_fields(self):
        """Test that models have required fields."""
        cache_path = Path(__file__).parent.parent.parent.parent / 'data' / 'models_cache.json'
        
        with open(cache_path) as f:
            cache_data = json.load(f)
        
        # Check first model
        model = cache_data['models'][0]
        assert 'name' in model, "Model should have 'name'"
        assert 'slug' in model, "Model should have 'slug'"
    
    def test_mmlu_pro_coverage(self):
        """Test MMLU-Pro coverage for RAG intent."""
        cache_path = Path(__file__).parent.parent.parent.parent / 'data' / 'models_cache.json'
        
        with open(cache_path) as f:
            cache_data = json.load(f)
        
        models_with_mmlu = [m for m in cache_data['models'] 
                           if m.get('mmlu_pro') and m['mmlu_pro'] != 'N/A']
        
        # Should have good coverage
        assert len(models_with_mmlu) >= 50, \
            f"Expected at least 50 models with MMLU-Pro, got {len(models_with_mmlu)}"


class TestProductionModels:
    """Test that production models exist and are loadable."""
    
    def test_all_models_exist(self):
        """Test that all 4 production models exist."""
        # Models now in llm_jury/models/production/
        models_dir = Path(__file__).parent.parent.parent.parent / 'llm_jury' / 'models' / 'production'
        
        expected_models = [
            'reasoning_xgboost_model.joblib',
            'coding_xgboost_model.joblib',
            'summarization_xgboost_model.joblib',
            'rag_xgboost_model.joblib'
        ]
        
        for model_file in expected_models:
            model_path = models_dir / model_file
            assert model_path.exists(), f"Model not found: {model_file}"
    
    def test_model_cards_exist(self):
        """Test that model cards exist for all models."""
        # Models now in llm_jury/models/production/
        models_dir = Path(__file__).parent.parent.parent.parent / 'llm_jury' / 'models' / 'production'
        
        expected_cards = [
            'reasoning_model_card.json',
            'coding_model_card.json',
            'summarization_model_card.json',
            'rag_model_card.json'
        ]
        
        for card_file in expected_cards:
            card_path = models_dir / card_file
            assert card_path.exists(), f"Model card not found: {card_file}"
    
    def test_model_cards_structure(self):
        """Test that model cards have expected structure."""
        # Models now in llm_jury/models/production/
        models_dir = Path(__file__).parent.parent.parent.parent / 'llm_jury' / 'models' / 'production'
        card_path = models_dir / 'rag_model_card.json'
        
        with open(card_path) as f:
            card = json.load(f)
        
        required_fields = [
            'intent', 'capability_proxy', 'n_total_examples',
            'n_train_examples', 'n_test_examples',
            'test_accuracy', 'test_auc', 'feature_names', 'feature_importance'
        ]
        
        for field in required_fields:
            assert field in card, f"Missing field in model card: {field}"
    
    @pytest.mark.skipif(not sys.modules.get('joblib'), reason="joblib not installed")
    def test_model_loadable(self):
        """Test that a model can be loaded."""
        try:
            import joblib
            # Models now in llm_jury/models/production/
            models_dir = Path(__file__).parent.parent.parent.parent / 'llm_jury' / 'models' / 'production'
            model_path = models_dir / 'rag_xgboost_model.joblib'
            
            model = joblib.load(model_path)
            
            # Test basic properties
            assert hasattr(model, 'predict'), "Model should have predict method"
            assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"
        except ImportError:
            pytest.skip("joblib not available")


class TestFeatureEngineering:
    """Test feature engineering logic."""
    
    def test_nvidia_features_range(self):
        """Test that NVIDIA features are in expected range [0, 1] or small integers."""
        data_path = Path(__file__).parent.parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
        df = pd.read_csv(data_path, nrows=1000, low_memory=False)
        
        # These should be in [0, 1]
        bounded_features = ['nvidia_creativity', 'nvidia_reasoning', 
                           'nvidia_domain_knowledge', 'nvidia_contextual_knowledge']
        
        for feat in bounded_features:
            assert df[feat].min() >= 0, f"{feat} has values < 0"
            assert df[feat].max() <= 1, f"{feat} has values > 1"
        
        # These should be small integers
        assert df['nvidia_constraint'].min() >= 0, "constraint should be >= 0"
        assert df['nvidia_few_shots'].min() >= 0, "few_shots should be >= 0"
    
    def test_no_null_features(self):
        """Test that there are no null values in key features."""
        data_path = Path(__file__).parent.parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
        df = pd.read_csv(data_path, nrows=1000, low_memory=False)
        
        key_features = [
            'nvidia_creativity', 'nvidia_reasoning', 'nvidia_constraint',
            'nvidia_domain_knowledge', 'nvidia_contextual_knowledge', 
            'nvidia_few_shots', 'success'
        ]
        
        for feat in key_features:
            null_count = df[feat].isnull().sum()
            assert null_count == 0, f"{feat} has {null_count} null values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
