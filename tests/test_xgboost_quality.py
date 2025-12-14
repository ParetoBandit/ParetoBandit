"""
Unit tests for XGBoost quality predictor with intent fallback chain.

Tests the updated xgboost_quality.py that uses intent-level fallback
instead of IntentQualityScorer fallback.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from llm_jury.optimization.xgboost_quality import (
    XGBoostQualityPredictor,
    predict_quality_xgboost,
    create_quality_predictor,
    INTENT_MAPPING,
    INTENT_FALLBACK_CHAIN,
    CAPABILITY_FIELDS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_xgb_models():
    """Mock XGBoost models for testing."""
    models = {}
    for intent in ['coding', 'reasoning', 'rag', 'summarization']:
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        
        mock_card = {
            'intent': intent,
            'test_auc': 0.85,
            'n_train_examples': 10000,
        }
        models[intent] = (mock_model, mock_card)
    
    return models


@pytest.fixture
def sample_model_data():
    """Sample model data with all capability scores."""
    return {
        'name': 'GPT-4o mini',
        'livecodebench': 0.855,   # coding (0-1 scale, converted to 85.5)
        'gpqa': 78.2,             # reasoning
        'summedits_score': 82.3,  # summarization
        'mmlu_pro': 73.4,         # rag
    }


@pytest.fixture
def sample_model_data_partial():
    """Sample model data with only some capability scores."""
    return {
        'name': 'Test Model',
        'livecodebench': 0.65,  # coding available (0-1 scale)
        'gpqa': None,           # reasoning missing
        'summedits_score': 'N/A',  # summarization invalid
        # mmlu_pro missing entirely
    }


@pytest.fixture
def mock_nvidia_features():
    """Mock NVIDIA feature extraction."""
    return {
        'nvidia_creativity': 0.5,
        'nvidia_reasoning': 0.7,
        'nvidia_constraint': 0.3,
        'nvidia_domain_knowledge': 0.6,
        'nvidia_contextual_knowledge': 0.4,
        'nvidia_few_shots': 0.2,
    }


# =============================================================================
# Test Intent Mapping
# =============================================================================

class TestIntentMapping:
    """Test intent mapping from optimization intents to XGBoost intents."""
    
    def test_intent_mapping_coverage(self):
        """All common intents should map to XGBoost intents."""
        assert 'coding' in INTENT_MAPPING
        assert 'reasoning' in INTENT_MAPPING
        assert 'rag' in INTENT_MAPPING
        assert 'summarization' in INTENT_MAPPING
        assert 'general' in INTENT_MAPPING
        assert 'agentic' in INTENT_MAPPING
    
    def test_intent_mapping_values(self):
        """Intent mappings should point to valid XGBoost intents."""
        valid_intents = {'coding', 'reasoning', 'rag', 'summarization'}
        for intent, mapped in INTENT_MAPPING.items():
            assert mapped in valid_intents, f"'{intent}' maps to invalid intent '{mapped}'"
    
    def test_direct_mappings(self):
        """Direct mappings should be identity."""
        assert INTENT_MAPPING['coding'] == 'coding'
        assert INTENT_MAPPING['reasoning'] == 'reasoning'
        assert INTENT_MAPPING['rag'] == 'rag'
        assert INTENT_MAPPING['summarization'] == 'summarization'
    
    def test_fallback_chain_exists(self):
        """All XGBoost intents should have fallback chains."""
        for intent in ['coding', 'reasoning', 'rag', 'summarization']:
            assert intent in INTENT_FALLBACK_CHAIN
            assert isinstance(INTENT_FALLBACK_CHAIN[intent], list)
            assert len(INTENT_FALLBACK_CHAIN[intent]) >= 1


# =============================================================================
# Test Capability Field Mapping
# =============================================================================

class TestCapabilityFields:
    """Test capability field mappings."""
    
    def test_all_intents_have_capabilities(self):
        """All XGBoost intents should map to capability fields."""
        for intent in ['coding', 'reasoning', 'rag', 'summarization']:
            assert intent in CAPABILITY_FIELDS
    
    def test_capability_field_names(self):
        """Capability fields should be valid benchmark names with good uniqueness."""
        # Using fields with high uniqueness across models for better differentiation
        assert CAPABILITY_FIELDS['coding'] == 'livecodebench'  # 80/81 unique
        assert CAPABILITY_FIELDS['reasoning'] == 'gpqa'         # 78/81 unique
        assert CAPABILITY_FIELDS['summarization'] == 'summedits_score'  # 77/81 unique
        assert CAPABILITY_FIELDS['rag'] == 'mmlu_pro'           # 75/81 unique


# =============================================================================
# Test XGBoostQualityPredictor Initialization
# =============================================================================

class TestXGBoostQualityPredictorInit:
    """Test XGBoostQualityPredictor initialization."""
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_initialization_success(self, mock_get_info, mock_load_models, mock_xgb_models):
        """Predictor should load models successfully."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {intent: card for intent, (_, card) in mock_xgb_models.items()}
        
        predictor = XGBoostQualityPredictor()
        
        assert len(predictor.models) == 4
        assert 'coding' in predictor.models
        assert 'reasoning' in predictor.models
        mock_load_models.assert_called_once()
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_initialization_failure_graceful(self, mock_get_info, mock_load_models):
        """Predictor should handle model loading failures gracefully."""
        mock_load_models.side_effect = Exception("Model not found")
        
        with pytest.warns(UserWarning):
            predictor = XGBoostQualityPredictor()
        
        assert predictor.models == {}
        assert predictor.model_info == {}


# =============================================================================
# Test Capability Score Extraction
# =============================================================================

class TestCapabilityScoreExtraction:
    """Test _get_capability_score method."""
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_extract_valid_capability(self, mock_get_info, mock_load_models, 
                                     mock_xgb_models, sample_model_data):
        """Should extract valid capability scores."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        
        # Test coding capability
        score = predictor._get_capability_score(sample_model_data, 'coding')
        assert score == 85.5
        
        # Test reasoning capability
        score = predictor._get_capability_score(sample_model_data, 'reasoning')
        assert score == 78.2
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_extract_missing_capability(self, mock_get_info, mock_load_models, 
                                       mock_xgb_models, sample_model_data_partial):
        """Should return None for missing capabilities."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        
        # Test missing MMLU-Pro (rag)
        score = predictor._get_capability_score(sample_model_data_partial, 'rag')
        assert score is None
        
        # Test None value (reasoning)
        score = predictor._get_capability_score(sample_model_data_partial, 'reasoning')
        assert score is None
        
        # Test 'N/A' value (summarization)
        score = predictor._get_capability_score(sample_model_data_partial, 'summarization')
        assert score is None
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_convert_decimal_to_percentage(self, mock_get_info, mock_load_models, mock_xgb_models):
        """Should convert 0-1 scores to 0-100 scale."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        
        # Test with decimal value (livecodebench is 0-1 scale)
        model_data = {'name': 'Test', 'livecodebench': 0.85}
        score = predictor._get_capability_score(model_data, 'coding')
        assert score == 85.0


# =============================================================================
# Test Intent Fallback Chain
# =============================================================================

class TestIntentFallbackChain:
    """Test intent-level fallback chain functionality."""
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_primary_intent_success(self, mock_get_info, mock_load_models, 
                                    mock_xgb_models, sample_model_data, mock_nvidia_features):
        """Should use primary intent when features available."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        
        # Mock NVIDIA features extraction
        predictor._extract_nvidia_features = Mock(return_value=mock_nvidia_features)
        
        quality = predictor.predict_quality(
            prompt="Write a function",
            model_data=sample_model_data,
            intent='coding'
        )
        
        # Should use coding model (primary)
        # Expected: blend of XGBoost (0.7) and capability (0.855)
        # adjusted = 0.7 * 0.7 + 0.3 * 0.855 = 0.7465
        assert 0.7 <= quality <= 0.8  # With capability adjustment
        assert mock_xgb_models['coding'][0].predict_proba.called
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_fallback_to_next_intent(self, mock_get_info, mock_load_models, 
                                     mock_xgb_models, sample_model_data_partial, mock_nvidia_features):
        """Should fallback to next intent when primary capability missing."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        predictor._extract_nvidia_features = Mock(return_value=mock_nvidia_features)
        
        # Try reasoning intent, but model has no GPQA score
        # Should fallback to rag (which has no mmlu_pro), then coding (which has livecodebench)
        quality = predictor.predict_quality(
            prompt="Solve this problem",
            model_data=sample_model_data_partial,
            intent='reasoning'
        )
        
        # Should eventually use coding model (has livecodebench=0.65)
        # adjusted = 0.7 * 0.7 + 0.3 * 0.65 = 0.685
        assert 0.6 <= quality <= 0.75
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_all_fallbacks_fail(self, mock_get_info, mock_load_models, mock_xgb_models):
        """Should return neutral score when all fallbacks fail."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        
        # No NVIDIA features available
        predictor._extract_nvidia_features = Mock(return_value=None)
        
        with pytest.warns(UserWarning, match="All XGBoost predictions failed"):
            quality = predictor.predict_quality(
                prompt="Test",
                model_data={'name': 'Test'},
                intent='coding'
            )
        
        assert quality == 0.5  # Neutral score


# =============================================================================
# Test Predict Quality
# =============================================================================

class TestPredictQuality:
    """Test predict_quality method."""
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_successful_prediction(self, mock_get_info, mock_load_models, 
                                   mock_xgb_models, sample_model_data, mock_nvidia_features):
        """Should return success probability from XGBoost model."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        predictor._extract_nvidia_features = Mock(return_value=mock_nvidia_features)
        
        quality = predictor.predict_quality(
            prompt="Write a function",
            model_data=sample_model_data,
            intent='coding'
        )
        
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
        # With capability adjustment: blend of XGBoost (0.7) and capability (0.855)
        assert 0.7 <= quality <= 0.8
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_nvidia_features_unavailable(self, mock_get_info, mock_load_models, 
                                         mock_xgb_models, sample_model_data):
        """Should return neutral score when NVIDIA features unavailable."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        predictor._extract_nvidia_features = Mock(return_value=None)
        
        with pytest.warns(UserWarning):
            quality = predictor.predict_quality(
                prompt="Test",
                model_data=sample_model_data,
                intent='coding'
            )
        
        assert quality == 0.5


# =============================================================================
# Test Batch Prediction
# =============================================================================

class TestBatchPrediction:
    """Test predict_batch method."""
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_batch_prediction(self, mock_get_info, mock_load_models, 
                             mock_xgb_models, mock_nvidia_features):
        """Should predict quality for multiple models."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        predictor._extract_nvidia_features = Mock(return_value=mock_nvidia_features)
        
        models_data = [
            {'name': 'Model 1', 'livecodebench': 0.80},
            {'name': 'Model 2', 'livecodebench': 0.70},
            {'name': 'Model 3', 'livecodebench': 0.90},
        ]
        
        qualities = predictor.predict_batch(
            prompt="Write code",
            models_data=models_data,
            intent='coding'
        )
        
        assert len(qualities) == 3
        assert all(isinstance(q, float) for q in qualities)
        assert all(0.0 <= q <= 1.0 for q in qualities)


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_predict_quality_xgboost(self, mock_get_info, mock_load_models, 
                                     mock_xgb_models, sample_model_data, mock_nvidia_features):
        """Test predict_quality_xgboost convenience function."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        with patch.object(XGBoostQualityPredictor, '_extract_nvidia_features', 
                         return_value=mock_nvidia_features):
            quality = predict_quality_xgboost(
                prompt="Test",
                model_data=sample_model_data,
                intent='coding'
            )
        
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_create_quality_predictor(self, mock_get_info, mock_load_models, mock_xgb_models):
        """Test create_quality_predictor factory function."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = create_quality_predictor()
        
        assert isinstance(predictor, XGBoostQualityPredictor)
        assert len(predictor.models) == 4


# =============================================================================
# Test Model Info Summary
# =============================================================================

class TestModelInfoSummary:
    """Test get_model_info_summary method."""
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    @patch('llm_jury.optimization.xgboost_quality.get_all_model_info')
    def test_model_info_summary(self, mock_get_info, mock_load_models, mock_xgb_models):
        """Should return summary of loaded models."""
        mock_load_models.return_value = mock_xgb_models
        mock_get_info.return_value = {}
        
        predictor = XGBoostQualityPredictor()
        summary = predictor.get_model_info_summary()
        
        assert isinstance(summary, str)
        assert 'coding' in summary
        assert 'reasoning' in summary
        assert 'Test AUC' in summary
    
    @patch('llm_jury.optimization.xgboost_quality.load_all_models')
    def test_model_info_summary_no_models(self, mock_load_models):
        """Should handle case with no models loaded."""
        mock_load_models.side_effect = Exception("No models")
        
        with pytest.warns(UserWarning):
            predictor = XGBoostQualityPredictor()
        
        summary = predictor.get_model_info_summary()
        assert "No XGBoost models loaded" in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
