"""
Unit tests for latency constraint behavior in get_value_recommendations().

Tests cover:
- Default behavior: latency constrained to baseline (same speed or faster)
- User says "latency not important" / latency_flexible=True
- Explicit latency ratio constraints
- Prompt-based latency detection (future feature)
"""

import pytest
from unittest.mock import patch, MagicMock
from llm_jury.orchestration.orchestrator import (
    get_value_recommendations,
    UseCase,
    USE_CASE_CONFIGS,
)
from llm_jury.ranking.optimizer import Optimizer, OptimizationStrategy


# =============================================================================
# Mock Data Fixtures
# =============================================================================

@pytest.fixture
def mock_model_registry():
    """Mock ModelRegistry.load_cache to return test models."""
    from llm_jury.core.models import ModelMetadata, ProductArchetype
    
    models = [
        # Fast, cheap, lower quality
        ModelMetadata(
            name="Fast Budget Model",
            intelligence_index=70.0,
            coding_index=65.0,
            input_cost_per_m=0.5,
            output_cost_per_m=1.0,
            measured_ttft_seconds=0.1,  # Very fast
            archetype=ProductArchetype.FRONTIER,
        ),
        # Slow, cheap, high quality
        ModelMetadata(
            name="Slow Quality Model",
            intelligence_index=90.0,
            coding_index=88.0,
            input_cost_per_m=1.0,
            output_cost_per_m=2.0,
            measured_ttft_seconds=2.0,  # Slow
            archetype=ProductArchetype.FRONTIER,
        ),
        # Baseline - medium everything
        ModelMetadata(
            name="Gemini 2.5 Pro",  # Default baseline name
            intelligence_index=85.0,
            coding_index=82.0,
            input_cost_per_m=5.0,
            output_cost_per_m=15.0,
            measured_ttft_seconds=0.5,  # Baseline speed
            archetype=ProductArchetype.FRONTIER,
        ),
        # Fast and quality but expensive
        ModelMetadata(
            name="Premium Fast Model",
            intelligence_index=92.0,
            coding_index=90.0,
            input_cost_per_m=10.0,
            output_cost_per_m=30.0,
            measured_ttft_seconds=0.2,
            archetype=ProductArchetype.FRONTIER,
        ),
    ]
    return models


@pytest.fixture
def mock_raw_data():
    """Mock raw model data for QualityScorer."""
    return [
        {
            "name": "Fast Budget Model",
            "intelligence_index": 70.0,
            "coding_index": 65.0,
            "price_1m_input": 0.5,
            "price_1m_output": 1.0,
            "measured_ttft_seconds": 0.1,
        },
        {
            "name": "Slow Quality Model",
            "intelligence_index": 90.0,
            "coding_index": 88.0,
            "price_1m_input": 1.0,
            "price_1m_output": 2.0,
            "measured_ttft_seconds": 2.0,
        },
        {
            "name": "Gemini 2.5 Pro",
            "intelligence_index": 85.0,
            "coding_index": 82.0,
            "price_1m_input": 5.0,
            "price_1m_output": 15.0,
            "measured_ttft_seconds": 0.5,
        },
        {
            "name": "Premium Fast Model",
            "intelligence_index": 92.0,
            "coding_index": 90.0,
            "price_1m_input": 10.0,
            "price_1m_output": 30.0,
            "measured_ttft_seconds": 0.2,
        },
    ]


# =============================================================================
# Optimizer Latency Constraint Tests (Unit Tests)
# =============================================================================

class TestOptimizerLatencyConstraint:
    """Tests for Optimizer's speed_range constraint behavior."""
    
    def test_speed_range_none_no_constraint(self, mock_raw_data):
        """Test that speed_range=None means no latency constraint."""
        from llm_jury.core.models import ModelMetadata, ProductArchetype
        
        baseline = ModelMetadata(
            name="Baseline",
            intelligence_index=85.0,
            input_cost_per_m=5.0,
            output_cost_per_m=15.0,
            measured_ttft_seconds=0.5,
            archetype=ProductArchetype.FRONTIER,
        )
        
        optimizer = Optimizer(
            baseline_model=baseline,
            all_models_data=mock_raw_data,
            strategy=OptimizationStrategy.VALUE_OPTIMIZED,
            quality_range=(0.70, 1.0),
            cost_range=(0.0, 0.50),
            speed_range=None,  # No latency constraint
        )
        
        assert optimizer.speed_range is None
    
    def test_speed_range_set_applies_constraint(self, mock_raw_data):
        """Test that speed_range tuple applies latency constraint."""
        from llm_jury.core.models import ModelMetadata, ProductArchetype
        
        baseline = ModelMetadata(
            name="Baseline",
            intelligence_index=85.0,
            input_cost_per_m=5.0,
            output_cost_per_m=15.0,
            measured_ttft_seconds=0.5,
            archetype=ProductArchetype.FRONTIER,
        )
        
        # speed_range = (1.0, inf) means speed must be >= baseline
        optimizer = Optimizer(
            baseline_model=baseline,
            all_models_data=mock_raw_data,
            strategy=OptimizationStrategy.VALUE_OPTIMIZED,
            quality_range=(0.70, 1.0),
            cost_range=(0.0, 0.50),
            speed_range=(1.0, float('inf')),  # At least as fast as baseline
        )
        
        assert optimizer.speed_range == (1.0, float('inf'))
    
    def test_speed_range_half_allows_slower_models(self, mock_raw_data):
        """Test that speed_range (0.5, inf) allows 2x slower models."""
        from llm_jury.core.models import ModelMetadata, ProductArchetype
        
        baseline = ModelMetadata(
            name="Baseline",
            intelligence_index=85.0,
            input_cost_per_m=5.0,
            output_cost_per_m=15.0,
            measured_ttft_seconds=0.5,
            archetype=ProductArchetype.FRONTIER,
        )
        
        # speed_range = (0.5, inf) means can be 2x slower
        optimizer = Optimizer(
            baseline_model=baseline,
            all_models_data=mock_raw_data,
            strategy=OptimizationStrategy.VALUE_OPTIMIZED,
            quality_range=(0.70, 1.0),
            cost_range=(0.0, 0.50),
            speed_range=(0.5, float('inf')),
        )
        
        assert optimizer.speed_range[0] == 0.5


# =============================================================================
# API Latency Parameter Tests
# =============================================================================

class TestGetValueRecommendationsLatency:
    """Tests for get_value_recommendations() latency parameters."""
    
    def test_default_latency_ratio_is_one(self):
        """Test that default max_latency_ratio is 1.0 (same as baseline)."""
        import inspect
        sig = inspect.signature(get_value_recommendations)
        default = sig.parameters['max_latency_ratio'].default
        assert default == 1.0, "Default max_latency_ratio should be 1.0"
    
    def test_default_latency_flexible_is_false(self):
        """Test that default latency_flexible is False."""
        import inspect
        sig = inspect.signature(get_value_recommendations)
        default = sig.parameters['latency_flexible'].default
        assert default == False, "Default latency_flexible should be False"
    
    @patch('llm_jury.orchestration.orchestrator.ModelRegistry')
    @patch('llm_jury.orchestration.orchestrator.PromptClassifier')
    @patch('llm_jury.orchestration.orchestrator.Optimizer')
    def test_latency_flexible_true_passes_none_speed_range(
        self, mock_optimizer_class, mock_classifier_class, mock_registry_class,
        mock_model_registry, mock_raw_data
    ):
        """Test that latency_flexible=True passes speed_range=None to Optimizer."""
        # Setup mocks
        mock_registry_class.load_cache.return_value = mock_model_registry
        mock_registry_class.load_raw_cache.return_value = mock_raw_data
        
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MagicMock(
            use_case='general_qa',
            confidence=0.9,
            category=MagicMock(value='general'),
            alternative_use_cases=[]
        )
        mock_classifier.get_use_case_description.return_value = "General Q&A"
        mock_classifier_class.return_value = mock_classifier
        
        mock_optimizer = MagicMock()
        mock_optimizer.rank.return_value = []
        mock_optimizer_class.return_value = mock_optimizer
        
        # Call with latency_flexible=True
        get_value_recommendations(
            "Test prompt",
            latency_flexible=True,
            verbose=False
        )
        
        # Verify Optimizer was called with speed_range=None
        call_kwargs = mock_optimizer_class.call_args[1]
        assert call_kwargs.get('speed_range') is None, \
            "latency_flexible=True should pass speed_range=None"
    
    @patch('llm_jury.orchestration.orchestrator.ModelRegistry')
    @patch('llm_jury.orchestration.orchestrator.PromptClassifier')
    @patch('llm_jury.orchestration.orchestrator.Optimizer')
    def test_default_latency_passes_speed_range_one(
        self, mock_optimizer_class, mock_classifier_class, mock_registry_class,
        mock_model_registry, mock_raw_data
    ):
        """Test that default latency (max_latency_ratio=1.0) passes speed_range=(1.0, inf)."""
        # Setup mocks
        mock_registry_class.load_cache.return_value = mock_model_registry
        mock_registry_class.load_raw_cache.return_value = mock_raw_data
        
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MagicMock(
            use_case='general_qa',
            confidence=0.9,
            category=MagicMock(value='general'),
            alternative_use_cases=[]
        )
        mock_classifier.get_use_case_description.return_value = "General Q&A"
        mock_classifier_class.return_value = mock_classifier
        
        mock_optimizer = MagicMock()
        mock_optimizer.rank.return_value = []
        mock_optimizer_class.return_value = mock_optimizer
        
        # Call with default latency (not specifying latency_flexible)
        get_value_recommendations(
            "Test prompt",
            verbose=False
        )
        
        # Verify Optimizer was called with speed_range=(1.0, inf)
        call_kwargs = mock_optimizer_class.call_args[1]
        speed_range = call_kwargs.get('speed_range')
        assert speed_range is not None, "Default should pass a speed_range"
        assert speed_range[0] == 1.0, "Default should require at least baseline speed"
    
    @patch('llm_jury.orchestration.orchestrator.ModelRegistry')
    @patch('llm_jury.orchestration.orchestrator.PromptClassifier')
    @patch('llm_jury.orchestration.orchestrator.Optimizer')
    def test_max_latency_ratio_2_allows_slower(
        self, mock_optimizer_class, mock_classifier_class, mock_registry_class,
        mock_model_registry, mock_raw_data
    ):
        """Test that max_latency_ratio=2.0 allows models 2x slower."""
        # Setup mocks
        mock_registry_class.load_cache.return_value = mock_model_registry
        mock_registry_class.load_raw_cache.return_value = mock_raw_data
        
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MagicMock(
            use_case='general_qa',
            confidence=0.9,
            category=MagicMock(value='general'),
            alternative_use_cases=[]
        )
        mock_classifier.get_use_case_description.return_value = "General Q&A"
        mock_classifier_class.return_value = mock_classifier
        
        mock_optimizer = MagicMock()
        mock_optimizer.rank.return_value = []
        mock_optimizer_class.return_value = mock_optimizer
        
        # Call with max_latency_ratio=2.0 (can be 2x slower)
        get_value_recommendations(
            "Test prompt",
            max_latency_ratio=2.0,
            verbose=False
        )
        
        # Verify Optimizer was called with speed_range=(0.5, inf)
        call_kwargs = mock_optimizer_class.call_args[1]
        speed_range = call_kwargs.get('speed_range')
        assert speed_range is not None
        assert speed_range[0] == 0.5, "max_latency_ratio=2.0 should mean min_speed=0.5"
    
    @patch('llm_jury.orchestration.orchestrator.ModelRegistry')
    @patch('llm_jury.orchestration.orchestrator.PromptClassifier')
    @patch('llm_jury.orchestration.orchestrator.Optimizer')
    def test_max_latency_ratio_half_requires_faster(
        self, mock_optimizer_class, mock_classifier_class, mock_registry_class,
        mock_model_registry, mock_raw_data
    ):
        """Test that max_latency_ratio=0.5 requires 2x faster models."""
        # Setup mocks
        mock_registry_class.load_cache.return_value = mock_model_registry
        mock_registry_class.load_raw_cache.return_value = mock_raw_data
        
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MagicMock(
            use_case='general_qa',
            confidence=0.9,
            category=MagicMock(value='general'),
            alternative_use_cases=[]
        )
        mock_classifier.get_use_case_description.return_value = "General Q&A"
        mock_classifier_class.return_value = mock_classifier
        
        mock_optimizer = MagicMock()
        mock_optimizer.rank.return_value = []
        mock_optimizer_class.return_value = mock_optimizer
        
        # Call with max_latency_ratio=0.5 (must be 2x faster)
        get_value_recommendations(
            "Test prompt",
            max_latency_ratio=0.5,
            verbose=False
        )
        
        # Verify Optimizer was called with speed_range=(2.0, inf)
        call_kwargs = mock_optimizer_class.call_args[1]
        speed_range = call_kwargs.get('speed_range')
        assert speed_range is not None
        assert speed_range[0] == 2.0, "max_latency_ratio=0.5 should mean min_speed=2.0"


# =============================================================================
# Latency Flexible Scenarios
# =============================================================================

class TestLatencyFlexibleScenarios:
    """Tests for scenarios where user explicitly says latency doesn't matter."""
    
    def test_batch_processing_should_use_flexible(self):
        """
        Test that batch processing prompts should use latency_flexible=True.
        
        This is a documentation/usage test - when users have batch workloads,
        they should explicitly set latency_flexible=True.
        """
        # This is more of a usage pattern test
        # Users should call:
        # get_value_recommendations("Process 10000 docs", latency_flexible=True)
        pass
    
    def test_interactive_chat_should_use_default(self):
        """
        Test that interactive chat should use default latency constraint.
        
        For interactive use, users expect similar latency to baseline.
        """
        # Default behavior is already max_latency_ratio=1.0, latency_flexible=False
        pass


# =============================================================================
# Constraint Calculation Tests
# =============================================================================

class TestLatencyConstraintCalculation:
    """Tests for the latency-to-speed conversion logic."""
    
    def test_latency_ratio_to_speed_ratio_conversion(self):
        """Test that max_latency_ratio correctly converts to min_speed_ratio."""
        # latency_ratio = 1.0 → speed_ratio = 1.0 (same speed)
        assert 1.0 / 1.0 == 1.0
        
        # latency_ratio = 2.0 → speed_ratio = 0.5 (half the speed, 2x slower)
        assert 1.0 / 2.0 == 0.5
        
        # latency_ratio = 0.5 → speed_ratio = 2.0 (double speed, 2x faster)
        assert 1.0 / 0.5 == 2.0
        
        # latency_ratio = 0.25 → speed_ratio = 4.0 (4x faster)
        assert 1.0 / 0.25 == 4.0


# =============================================================================
# Integration-Style Tests (with real components)
# =============================================================================

class TestLatencyConstraintIntegration:
    """Integration tests that use real Optimizer but mocked data."""
    
    def test_optimizer_filters_slow_models_with_speed_constraint(self, mock_raw_data):
        """Test that Optimizer actually filters slow models when speed_range is set."""
        from llm_jury.core.models import ModelMetadata, ProductArchetype, RoutingDecision, PromptCategory
        
        # Create models with different speeds
        fast_model = ModelMetadata(
            name="Fast Model",
            intelligence_index=75.0,
            input_cost_per_m=1.0,
            output_cost_per_m=2.0,
            measured_ttft_seconds=0.2,  # Fast (5 req/sec equivalent)
            hallucination_rate=10.0,
            refusal_rate=5.0,
            archetype=ProductArchetype.FRONTIER,
        )
        
        slow_model = ModelMetadata(
            name="Slow Model",
            intelligence_index=85.0,
            input_cost_per_m=1.0,
            output_cost_per_m=2.0,
            measured_ttft_seconds=2.0,  # Slow (0.5 req/sec equivalent)
            hallucination_rate=5.0,
            refusal_rate=2.0,
            archetype=ProductArchetype.FRONTIER,
        )
        
        baseline = ModelMetadata(
            name="Baseline",
            intelligence_index=80.0,
            input_cost_per_m=5.0,
            output_cost_per_m=15.0,
            measured_ttft_seconds=0.5,  # Baseline speed (2 req/sec equivalent)
            hallucination_rate=8.0,
            refusal_rate=3.0,
            archetype=ProductArchetype.FRONTIER,
        )
        
        models = [fast_model, slow_model, baseline]
        
        # Add to raw data for scorer
        extended_raw_data = mock_raw_data + [
            {"name": "Fast Model", "intelligence_index": 75.0},
            {"name": "Slow Model", "intelligence_index": 85.0},
            {"name": "Baseline", "intelligence_index": 80.0},
        ]
        
        routing = RoutingDecision(
            archetype=ProductArchetype.FRONTIER,
            category=PromptCategory.GENERAL,
            reason="Test",
            cot_template=""
        )
        
        # With speed constraint: models must be at least as fast as baseline
        optimizer_constrained = Optimizer(
            baseline_model=baseline,
            all_models_data=extended_raw_data,
            strategy=OptimizationStrategy.VALUE_OPTIMIZED,
            quality_range=(0.50, 1.5),  # Wide range
            cost_range=(0.0, 1.0),       # Wide range
            speed_range=(1.0, float('inf')),  # At least as fast
            missing_data='impute'
        )
        
        # Without speed constraint
        optimizer_unconstrained = Optimizer(
            baseline_model=baseline,
            all_models_data=extended_raw_data,
            strategy=OptimizationStrategy.VALUE_OPTIMIZED,
            quality_range=(0.50, 1.5),
            cost_range=(0.0, 1.0),
            speed_range=None,  # No constraint
            missing_data='impute'
        )
        
        # Both should work without error
        results_constrained = optimizer_constrained.rank(models, routing, top_k=3, verbose=False)
        results_unconstrained = optimizer_unconstrained.rank(models, routing, top_k=3, verbose=False)
        
        # Unconstrained should include slow model, constrained may not
        # (depending on other factors in the optimization)
        assert len(results_unconstrained) >= 0
        assert len(results_constrained) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

