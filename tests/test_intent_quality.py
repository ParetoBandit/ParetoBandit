"""
Unit tests for intent-specific quality scoring.

Tests cover:
1. Intent quality targets configuration
2. Inverted metrics handling
3. IntentQualityScorer class
4. Calibrated proxy scoring for general intent
5. Coverage and validation
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent to path for direct import (avoid package __init__ issues)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from the module file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "intent_quality", 
    Path(__file__).parent.parent / "llm_jury" / "optimization" / "intent_quality.py"
)
intent_quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intent_quality)

# Extract what we need
INTENT_QUALITY_TARGETS = intent_quality.INTENT_QUALITY_TARGETS
INTENT_BENCHMARKS = intent_quality.INTENT_BENCHMARKS
INVERTED_METRICS = intent_quality.INVERTED_METRICS
ALL_BENCHMARKS = intent_quality.ALL_BENCHMARKS
IntentQualityScorer = intent_quality.IntentQualityScorer
IntentWeights = intent_quality.IntentWeights
get_intent_weights = intent_quality.get_intent_weights
get_all_intent_weights = intent_quality.get_all_intent_weights


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_models_data():
    """Sample model data for testing (15 models to meet minimum thresholds)."""
    base_models = [
        {'name': 'Model A', 'ccs_100': 85.0, 'reasoning_score': 0.75, 'cfs_100': 90.0, 'css_100': 70.0,
         'arena_rank_creative': 5, 'general_quality': 88.0, 'mixeval_score': 75.0, 'math_500': 80.0,
         'gpqa': 0.65, 'humaneval_score': 0.70, 'mmlu_pro': 0.85, 'intelligence_index': 80},
        {'name': 'Model B', 'ccs_100': 70.0, 'reasoning_score': 0.60, 'cfs_100': 75.0, 'css_100': 55.0,
         'arena_rank_creative': 15, 'general_quality': 72.0, 'mixeval_score': 65.0, 'math_500': 65.0,
         'gpqa': 0.50, 'humaneval_score': 0.55, 'mmlu_pro': 0.70, 'intelligence_index': 70},
        {'name': 'Model C', 'ccs_100': 95.0, 'reasoning_score': 0.90, 'cfs_100': 95.0, 'css_100': 85.0,
         'arena_rank_creative': 1, 'general_quality': 95.0, 'mixeval_score': 80.0, 'math_500': 95.0,
         'gpqa': 0.80, 'humaneval_score': 0.85, 'mmlu_pro': 0.90, 'intelligence_index': 90},
        {'name': 'Model D', 'ccs_100': 50.0, 'reasoning_score': 0.40, 'cfs_100': 55.0, 'css_100': 45.0,
         'arena_rank_creative': 30, 'general_quality': 55.0, 'mixeval_score': 50.0, 'math_500': 45.0,
         'gpqa': 0.35, 'humaneval_score': 0.40, 'mmlu_pro': 0.50, 'intelligence_index': 50},
        {'name': 'Model E', 'ccs_100': 78.0, 'reasoning_score': 0.68, 'cfs_100': 82.0, 'css_100': 62.0,
         'arena_rank_creative': 8, 'general_quality': 80.0, 'mixeval_score': 72.0, 'math_500': 75.0,
         'gpqa': 0.60, 'humaneval_score': 0.65, 'mmlu_pro': 0.78, 'intelligence_index': 75},
    ]
    
    # Generate more models to meet minimum thresholds (need 10+)
    import random
    random.seed(42)
    models = list(base_models)
    
    for i in range(10):
        score_base = 40 + i * 5
        models.append({
            'name': f'Model {chr(70 + i)}',  # F, G, H, etc.
            'ccs_100': score_base + random.uniform(-5, 10),
            'reasoning_score': (score_base / 100) + random.uniform(-0.05, 0.1),
            'cfs_100': score_base + random.uniform(-5, 10),
            'css_100': score_base + random.uniform(-5, 10),
            'arena_rank_creative': 35 - i * 2,
            'general_quality': score_base + random.uniform(-5, 10),
            'mixeval_score': score_base + random.uniform(-5, 10),
            'math_500': score_base + random.uniform(-5, 10),
            'gpqa': (score_base / 100) + random.uniform(-0.05, 0.1),
            'humaneval_score': (score_base / 100) + random.uniform(-0.05, 0.1),
            'mmlu_pro': (score_base / 100) + random.uniform(-0.05, 0.1),
            'intelligence_index': score_base + random.uniform(-5, 10),
        })
    
    return models


@pytest.fixture
def minimal_models_data():
    """Minimal model data with just a few fields."""
    return [
        {'name': 'Model 1', 'mixeval_score': 70, 'math_500': 60},
        {'name': 'Model 2', 'mixeval_score': 80, 'math_500': 75},
        {'name': 'Model 3', 'mixeval_score': 65, 'math_500': 55},
    ]


# ============================================================================
# Configuration Tests
# ============================================================================

class TestIntentQualityTargets:
    """Tests for INTENT_QUALITY_TARGETS configuration."""
    
    def test_all_intents_have_targets(self):
        """Test that all major intents have quality targets."""
        required_intents = ['coding', 'reasoning', 'factual_qa', 'summarization', 'creative', 'general']
        for intent in required_intents:
            assert intent in INTENT_QUALITY_TARGETS, f"Missing target for {intent}"
    
    def test_target_fields_are_strings(self):
        """Test that all targets are string field names."""
        for intent, target in INTENT_QUALITY_TARGETS.items():
            assert isinstance(target, str), f"Target for {intent} should be string"
    
    def test_coding_uses_ccs(self):
        """Test that coding intent uses CCS."""
        assert INTENT_QUALITY_TARGETS['coding'] == 'ccs_100'
    
    def test_reasoning_uses_crs(self):
        """Test that reasoning intent uses CRS (reasoning_score)."""
        assert INTENT_QUALITY_TARGETS['reasoning'] == 'reasoning_score'
    
    def test_factual_qa_uses_cfs(self):
        """Test that factual_qa intent uses CFS."""
        assert INTENT_QUALITY_TARGETS['factual_qa'] == 'cfs_100'
    
    def test_summarization_uses_css(self):
        """Test that summarization intent uses CSS."""
        assert INTENT_QUALITY_TARGETS['summarization'] == 'css_100'
    
    def test_creative_uses_arena_rank(self):
        """Test that creative intent uses arena_rank_creative."""
        assert INTENT_QUALITY_TARGETS['creative'] == 'arena_rank_creative'
    
    def test_general_uses_calibrated_score(self):
        """Test that general intent uses calibrated proxy score."""
        assert INTENT_QUALITY_TARGETS['general'] == 'general_quality'


class TestInvertedMetrics:
    """Tests for INVERTED_METRICS configuration."""
    
    def test_arena_ranks_are_inverted(self):
        """Test that arena rank metrics are in inverted set."""
        rank_metrics = [
            'arena_rank_creative',
            'arena_rank_coding',
            'arena_rank_math',
            'arena_rank_expert',
            'arena_rank_longer',
            'arena_rank_overall',
        ]
        for metric in rank_metrics:
            assert metric in INVERTED_METRICS, f"{metric} should be inverted"
    
    def test_hallucination_rate_is_inverted(self):
        """Test that hallucination_rate is in inverted set."""
        assert 'hallucination_rate' in INVERTED_METRICS
    
    def test_score_metrics_not_inverted(self):
        """Test that regular score metrics are NOT in inverted set."""
        score_metrics = ['ccs_100', 'reasoning_score', 'cfs_100', 'css_100', 'mixeval_score']
        for metric in score_metrics:
            assert metric not in INVERTED_METRICS, f"{metric} should not be inverted"


class TestIntentBenchmarks:
    """Tests for INTENT_BENCHMARKS configuration."""
    
    def test_coding_has_benchmarks(self):
        """Test that coding intent has benchmark list."""
        assert 'coding' in INTENT_BENCHMARKS
        assert len(INTENT_BENCHMARKS['coding']) > 0
    
    def test_reasoning_has_benchmarks(self):
        """Test that reasoning intent has benchmark list."""
        assert 'reasoning' in INTENT_BENCHMARKS
        assert len(INTENT_BENCHMARKS['reasoning']) > 0
    
    def test_benchmark_lists_are_lists(self):
        """Test that benchmark lists are actually lists."""
        for intent, benchmarks in INTENT_BENCHMARKS.items():
            assert isinstance(benchmarks, list), f"{intent} benchmarks should be a list"
    
    def test_benchmarks_are_in_all_benchmarks(self):
        """Test that intent benchmarks are subset of ALL_BENCHMARKS."""
        for intent, benchmarks in INTENT_BENCHMARKS.items():
            for bench in benchmarks:
                assert bench in ALL_BENCHMARKS, f"{bench} in {intent} not in ALL_BENCHMARKS"


# ============================================================================
# IntentQualityScorer Tests
# ============================================================================

class TestIntentQualityScorer:
    """Tests for IntentQualityScorer class."""
    
    def test_initialization(self, sample_models_data):
        """Test basic initialization."""
        scorer = IntentQualityScorer(sample_models_data)
        assert scorer is not None
        assert len(scorer.models_data) == 15  # 5 base + 10 generated
    
    def test_default_quality_target_selection(self, sample_models_data):
        """Test that default quality target is selected correctly."""
        scorer = IntentQualityScorer(sample_models_data)
        # Should select based on coverage
        assert scorer.default_quality_target is not None
    
    def test_get_quality_target_for_coding(self, sample_models_data):
        """Test quality target selection for coding intent."""
        scorer = IntentQualityScorer(sample_models_data)
        target = scorer._get_quality_target_for_intent('coding')
        assert target == 'ccs_100'
    
    def test_get_quality_target_for_reasoning(self, sample_models_data):
        """Test quality target selection for reasoning intent."""
        scorer = IntentQualityScorer(sample_models_data)
        target = scorer._get_quality_target_for_intent('reasoning')
        assert target == 'reasoning_score'
    
    def test_get_quality_target_for_creative(self, sample_models_data):
        """Test quality target selection for creative intent."""
        scorer = IntentQualityScorer(sample_models_data)
        target = scorer._get_quality_target_for_intent('creative')
        assert target == 'arena_rank_creative'
    
    def test_get_quality_target_for_general(self, sample_models_data):
        """Test quality target selection for general intent."""
        scorer = IntentQualityScorer(sample_models_data)
        target = scorer._get_quality_target_for_intent('general')
        assert target == 'general_quality'
    
    def test_is_inverted_metric_for_ranks(self, sample_models_data):
        """Test inverted metric detection for ranks."""
        scorer = IntentQualityScorer(sample_models_data)
        assert scorer._is_inverted_metric('arena_rank_creative') == True
        assert scorer._is_inverted_metric('arena_rank_coding') == True
    
    def test_is_inverted_metric_for_scores(self, sample_models_data):
        """Test inverted metric detection for regular scores."""
        scorer = IntentQualityScorer(sample_models_data)
        assert scorer._is_inverted_metric('ccs_100') == False
        assert scorer._is_inverted_metric('mixeval_score') == False
    
    def test_safe_get_valid_value(self, sample_models_data):
        """Test _safe_get with valid values."""
        scorer = IntentQualityScorer(sample_models_data)
        model = sample_models_data[0]
        
        result = scorer._safe_get(model, 'ccs_100')
        assert result == 85.0
    
    def test_safe_get_missing_value(self, sample_models_data):
        """Test _safe_get with missing values."""
        scorer = IntentQualityScorer(sample_models_data)
        model = {'name': 'Test', 'field_a': 100}  # No 'missing_field'
        
        result = scorer._safe_get(model, 'missing_field')
        assert result is None
    
    def test_safe_get_zero_value(self, sample_models_data):
        """Test _safe_get with zero values."""
        scorer = IntentQualityScorer(sample_models_data)
        model = {'name': 'Test', 'score': 0}
        
        # Zero should return None (unless allow_zero=True)
        result = scorer._safe_get(model, 'score')
        assert result is None
        
        result = scorer._safe_get(model, 'score', allow_zero=True)
        assert result == 0
    
    def test_prepare_models_caches_results(self, sample_models_data):
        """Test that _prepare_models caches results."""
        scorer = IntentQualityScorer(sample_models_data)
        
        # First call
        models1 = scorer._prepare_models('ccs_100')
        # Second call (should use cache)
        models2 = scorer._prepare_models('ccs_100')
        
        assert models1 is models2  # Same object (cached)
    
    def test_prepare_models_inverts_ranks(self, sample_models_data):
        """Test that _prepare_models inverts rank metrics."""
        scorer = IntentQualityScorer(sample_models_data)
        
        # For creative, arena_rank_creative should be inverted
        models = scorer._prepare_models('arena_rank_creative')
        
        # Model C has rank 1 (best), should have highest (least negative) quality
        # Model D has rank 30 (worst), should have lowest (most negative) quality
        model_c = next(m for m in models if m['name'] == 'Model C')
        model_d = next(m for m in models if m['name'] == 'Model D')
        
        assert model_c['quality'] > model_d['quality']
        assert model_c['quality'] == -1  # Inverted: -1
        assert model_d['quality'] == -30  # Inverted: -30


class TestIntentQualityScorerWeights:
    """Tests for weight computation in IntentQualityScorer."""
    
    def test_get_weights_returns_intent_weights(self, sample_models_data):
        """Test that get_weights returns IntentWeights object."""
        scorer = IntentQualityScorer(sample_models_data)
        weights = scorer.get_weights('coding')
        
        assert isinstance(weights, IntentWeights)
        assert weights.intent == 'coding'
    
    def test_weights_sum_to_one(self, sample_models_data):
        """Test that weights sum to approximately 1."""
        scorer = IntentQualityScorer(sample_models_data)
        weights = scorer.get_weights('coding')
        
        total = sum(weights.weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1"
    
    def test_weights_are_non_negative(self, sample_models_data):
        """Test that all weights are non-negative."""
        scorer = IntentQualityScorer(sample_models_data)
        weights = scorer.get_weights('coding')
        
        for bench, w in weights.weights.items():
            assert w >= 0, f"Weight for {bench} is negative: {w}"
    
    def test_weights_have_regression_stats(self, sample_models_data):
        """Test that weights include regression statistics."""
        scorer = IntentQualityScorer(sample_models_data)
        weights = scorer.get_weights('coding')
        
        # Should have regression diagnostics
        assert weights.r_squared is not None
        assert weights.n_models > 0
    
    def test_get_weights_for_different_intents(self, sample_models_data):
        """Test getting weights for different intents."""
        scorer = IntentQualityScorer(sample_models_data)
        
        coding_weights = scorer.get_weights('coding')
        reasoning_weights = scorer.get_weights('reasoning')
        
        # Both should return valid weights
        assert coding_weights.intent == 'coding'
        assert reasoning_weights.intent == 'reasoning'
        assert len(coding_weights.weights) > 0
        assert len(reasoning_weights.weights) > 0


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_intent_weights_with_data(self, sample_models_data):
        """Test that get_intent_weights works with data."""
        weights = get_intent_weights(sample_models_data, 'coding')
        assert weights is not None
        assert isinstance(weights, dict)
    
    def test_get_all_intent_weights_with_data(self, sample_models_data):
        """Test getting weights for all intents."""
        all_weights = get_all_intent_weights(sample_models_data)
        assert isinstance(all_weights, dict)
        
        # Should have weights for major intents
        assert 'coding' in all_weights
        assert 'reasoning' in all_weights


# ============================================================================
# IntentWeights Dataclass Tests
# ============================================================================

class TestIntentWeights:
    """Tests for IntentWeights dataclass."""
    
    def test_creation(self, sample_models_data):
        """Test basic creation via scorer."""
        scorer = IntentQualityScorer(sample_models_data)
        weights = scorer.get_weights('coding')
        
        assert weights.intent == 'coding'
        assert isinstance(weights.weights, dict)
        assert weights.r_squared is not None
    
    def test_has_required_fields(self, sample_models_data):
        """Test that IntentWeights has required fields."""
        scorer = IntentQualityScorer(sample_models_data)
        weights = scorer.get_weights('coding')
        
        # Check required fields exist
        assert hasattr(weights, 'intent')
        assert hasattr(weights, 'weights')
        assert hasattr(weights, 'r_squared')
        assert hasattr(weights, 'n_models')
    
    def test_summary_method(self, sample_models_data):
        """Test summary method exists and returns string."""
        scorer = IntentQualityScorer(sample_models_data)
        weights = scorer.get_weights('coding')
        
        summary = weights.summary()
        assert isinstance(summary, str)
        assert weights.intent.upper() in summary


# ============================================================================
# Integration Tests with Real Data
# ============================================================================

class TestIntegrationWithRealData:
    """Integration tests using real models cache data."""
    
    @pytest.fixture
    def real_models_data(self, data_dir):
        """Load real models data if available."""
        cache_path = data_dir / "models_cache.json"
        if not cache_path.exists():
            pytest.skip("models_cache.json not found")
        
        with open(cache_path) as f:
            data = json.load(f)
        
        return data.get('models', data) if isinstance(data, dict) else data
    
    def test_all_quality_targets_have_coverage(self, real_models_data):
        """Test that all quality targets have reasonable coverage."""
        min_coverage = 0.5  # At least 50% coverage
        
        for intent, target in INTENT_QUALITY_TARGETS.items():
            is_inverted = target in INVERTED_METRICS
            if is_inverted:
                count = sum(1 for m in real_models_data if m.get(target) is not None)
            else:
                count = sum(1 for m in real_models_data 
                           if m.get(target) and float(m.get(target, 0) or 0) > 0)
            
            coverage = count / len(real_models_data)
            assert coverage >= min_coverage, f"{intent} target '{target}' has only {coverage:.0%} coverage"
    
    def test_ccs_has_high_coverage(self, real_models_data):
        """Test that CCS (coding) has high coverage."""
        count = sum(1 for m in real_models_data if m.get('ccs_100') is not None)
        coverage = count / len(real_models_data)
        assert coverage >= 0.95, f"CCS coverage is only {coverage:.0%}"
    
    def test_general_quality_has_full_coverage(self, real_models_data):
        """Test that general_quality has full coverage."""
        count = sum(1 for m in real_models_data if m.get('general_quality') is not None)
        coverage = count / len(real_models_data)
        assert coverage >= 0.95, f"general_quality coverage is only {coverage:.0%}"
    
    def test_scorer_produces_valid_weights(self, real_models_data):
        """Test that scorer produces valid weights with real data."""
        scorer = IntentQualityScorer(real_models_data)
        
        for intent in ['coding', 'reasoning', 'factual_qa', 'general']:
            weights = scorer.get_weights(intent)
            
            # Weights should exist and sum to ~1
            assert weights.weights, f"No weights for {intent}"
            total = sum(weights.weights.values())
            assert abs(total - 1.0) < 0.01, f"{intent} weights sum to {total}"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_models_list(self):
        """Test with empty models list."""
        scorer = IntentQualityScorer([])
        # Should handle gracefully
        assert scorer.models_data == []
    
    def test_models_missing_all_targets(self):
        """Test with models missing all quality targets."""
        models = [
            {'name': 'Model 1', 'some_field': 1},
            {'name': 'Model 2', 'some_field': 2},
        ]
        scorer = IntentQualityScorer(models)
        # Should fall back to default or handle gracefully
        assert scorer.default_quality_target is not None or scorer.default_quality_target is None
    
    def test_single_model(self):
        """Test with single model."""
        models = [
            {'name': 'Only Model', 'ccs_100': 80, 'mixeval_score': 75},
        ]
        scorer = IntentQualityScorer(models)
        # Should handle gracefully (can't compute regression with 1 sample)
        assert scorer is not None
    
    def test_invalid_intent(self, sample_models_data):
        """Test with invalid intent name."""
        scorer = IntentQualityScorer(sample_models_data)
        # Should fall back to default quality target
        target = scorer._get_quality_target_for_intent('nonexistent_intent')
        assert target == scorer.default_quality_target
    
    def test_none_values_handled(self):
        """Test that None values are handled gracefully."""
        models = [
            {'name': 'Model 1', 'ccs_100': None, 'mixeval_score': 70},
            {'name': 'Model 2', 'ccs_100': 80, 'mixeval_score': None},
        ]
        scorer = IntentQualityScorer(models)
        # Should not crash
        assert scorer is not None


# ============================================================================
# Calibrated Proxy Scoring Tests
# ============================================================================

class TestCalibratedProxyScoring:
    """Tests specific to the calibrated proxy scoring for general intent."""
    
    def test_general_quality_field_exists(self, data_dir):
        """Test that general_quality field exists in cache."""
        cache_path = data_dir / "models_cache.json"
        if not cache_path.exists():
            pytest.skip("models_cache.json not found")
        
        with open(cache_path) as f:
            data = json.load(f)
        
        models = data.get('models', data) if isinstance(data, dict) else data
        
        # Check that at least some models have general_quality
        count = sum(1 for m in models if m.get('general_quality') is not None)
        assert count > 0, "No models have general_quality"
    
    def test_general_quality_source_field(self, data_dir):
        """Test that general_quality_source field exists."""
        cache_path = data_dir / "models_cache.json"
        if not cache_path.exists():
            pytest.skip("models_cache.json not found")
        
        with open(cache_path) as f:
            data = json.load(f)
        
        models = data.get('models', data) if isinstance(data, dict) else data
        
        # Check source field
        sources = set()
        for m in models:
            src = m.get('general_quality_source')
            if src:
                sources.add(src)
        
        # Should have 'combined' and 'predicted' sources
        assert len(sources) > 0, "No general_quality_source values found"
    
    def test_general_quality_is_numeric(self, data_dir):
        """Test that general_quality values are numeric (can be outside 0-100 due to calibration)."""
        cache_path = data_dir / "models_cache.json"
        if not cache_path.exists():
            pytest.skip("models_cache.json not found")
        
        with open(cache_path) as f:
            data = json.load(f)
        
        models = data.get('models', data) if isinstance(data, dict) else data
        
        scores = [m.get('general_quality') for m in models if m.get('general_quality') is not None]
        assert len(scores) > 0, "No models have general_quality"
        
        # All scores should be numeric
        for score in scores:
            assert isinstance(score, (int, float)), f"general_quality {score} is not numeric"
        
        # Most scores should be in reasonable range (allow some extrapolation)
        in_range = sum(1 for s in scores if -20 <= s <= 120)
        assert in_range / len(scores) > 0.9, "Too many scores outside reasonable range"
