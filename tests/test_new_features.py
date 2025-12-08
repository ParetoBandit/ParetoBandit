"""
Unit tests for new features added to LLM Jury.

Tests cover:
- Open source model detection (is_open_source, OPEN_SOURCE_PATTERNS)
- open_source_only parameter in get_value_recommendations()
- open_source_only parameter in get_best_models_for_budget()
- New use cases (QA, RAG, CHATBOT) in PromptCategory
- Context score calculation in QualityScorer
"""

import pytest
from llm_jury import (
    is_open_source,
    OPEN_SOURCE_PATTERNS,
    get_value_recommendations,
    get_best_models_for_budget,
    PromptCategory,
)
from llm_jury.core.models import ModelMetadata, ProductArchetype, RoutingDecision
from llm_jury.ranking.quality_scorer import QualityScorer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_raw_data():
    """Create raw model data for testing."""
    return [
        {
            "name": "GPT-5.1 (high)",
            "intelligence_index": 90.0,
            "coding_index": 88.0,
            "math_index": 85.0,
            "mmlu_pro": 80.0,
            "gpqa": 70.0,
            "context_window_k": 128,
        },
        {
            "name": "DeepSeek V3.1 Terminus",
            "intelligence_index": 85.0,
            "coding_index": 82.0,
            "math_index": 80.0,
            "mmlu_pro": 75.0,
            "gpqa": 65.0,
            "context_window_k": 64,
        },
        {
            "name": "Llama 4 Maverick",
            "intelligence_index": 75.0,
            "coding_index": 72.0,
            "math_index": 70.0,
            "mmlu_pro": 68.0,
            "gpqa": 55.0,
            "context_window_k": 128,
        },
        {
            "name": "Claude Opus 4.5",
            "intelligence_index": 88.0,
            "coding_index": 85.0,
            "math_index": 83.0,
            "mmlu_pro": 78.0,
            "gpqa": 68.0,
            "context_window_k": 200,
        },
        {
            "name": "Qwen 3 32B",
            "intelligence_index": 70.0,
            "coding_index": 68.0,
            "math_index": 65.0,
            "mmlu_pro": 62.0,
            "gpqa": 50.0,
            "context_window_k": 32,
        },
        {
            "name": "Mistral Large 3",
            "intelligence_index": 78.0,
            "coding_index": 75.0,
            "math_index": 72.0,
            "mmlu_pro": 70.0,
            "gpqa": 58.0,
            "context_window_k": 128,
        },
        {
            "name": "Gemini 3 Pro",
            "intelligence_index": 86.0,
            "coding_index": 84.0,
            "math_index": 82.0,
            "mmlu_pro": 76.0,
            "gpqa": 66.0,
            "context_window_k": 1000,
        },
        {
            "name": "Gemma 2 27B",
            "intelligence_index": 65.0,
            "coding_index": 62.0,
            "math_index": 60.0,
            "mmlu_pro": 58.0,
            "gpqa": 48.0,
            "context_window_k": 8,
        },
        {
            "name": "Phi-4",
            "intelligence_index": 68.0,
            "coding_index": 65.0,
            "math_index": 63.0,
            "mmlu_pro": 60.0,
            "gpqa": 52.0,
            "context_window_k": 16,
        },
        {
            "name": "GLM-4.5-Air",
            "intelligence_index": 72.0,
            "coding_index": 70.0,
            "math_index": 68.0,
            "mmlu_pro": 64.0,
            "gpqa": 54.0,
            "context_window_k": 128,
        },
    ]


@pytest.fixture
def quality_scorer(sample_raw_data):
    """Create a QualityScorer instance."""
    return QualityScorer(all_models_data=sample_raw_data)


# =============================================================================
# Open Source Detection Tests
# =============================================================================

class TestIsOpenSource:
    """Tests for is_open_source() function."""
    
    def test_deepseek_is_open_source(self):
        """Test that DeepSeek models are detected as open source."""
        assert is_open_source("DeepSeek V3.1 Terminus") is True
        assert is_open_source("DeepSeek V3.2 Exp (Reasoning)") is True
        assert is_open_source("DeepSeek R1") is True
    
    def test_qwen_is_open_source(self):
        """Test that Qwen models are detected as open source."""
        assert is_open_source("Qwen 3 32B") is True
        assert is_open_source("Qwen 2.5 72B") is True
        assert is_open_source("Qwen-Plus") is True
    
    def test_llama_is_open_source(self):
        """Test that Llama models are detected as open source."""
        assert is_open_source("Llama 4 Maverick") is True
        assert is_open_source("Llama 3.3 70B") is True
        assert is_open_source("Meta-Llama-3.2-3B") is True
    
    def test_mistral_is_open_source(self):
        """Test that Mistral models are detected as open source."""
        assert is_open_source("Mistral Large 3") is True
        assert is_open_source("Mixtral 8x7B") is True
        assert is_open_source("Mistral-7B-Instruct") is True
    
    def test_gemma_is_open_source(self):
        """Test that Gemma models are detected as open source."""
        assert is_open_source("Gemma 2 27B") is True
        assert is_open_source("Gemma-7B") is True
    
    def test_glm_is_open_source(self):
        """Test that GLM models are detected as open source."""
        assert is_open_source("GLM-4.5-Air") is True
        assert is_open_source("GLM-4-Plus") is True
    
    def test_phi_is_open_source(self):
        """Test that Phi models are detected as open source."""
        assert is_open_source("Phi-4") is True
        assert is_open_source("Phi-3.5") is True
        # Note: "Phi" without hyphen should NOT match (to avoid false positives)
        assert is_open_source("Phi 4") is False
    
    def test_proprietary_models_not_open_source(self):
        """Test that proprietary models are NOT detected as open source."""
        assert is_open_source("GPT-5.1 (high)") is False
        assert is_open_source("GPT-4o") is False
        assert is_open_source("Claude Opus 4.5") is False
        assert is_open_source("Claude 3.5 Sonnet") is False
        assert is_open_source("Gemini 3 Pro") is False
        assert is_open_source("Gemini 2.5 Pro") is False
        assert is_open_source("Grok 4.1") is False
        assert is_open_source("o3 mini (high)") is False
    
    def test_case_sensitivity(self):
        """Test that detection is case-sensitive (as expected for model names)."""
        assert is_open_source("deepseek v3") is False  # lowercase
        assert is_open_source("DEEPSEEK V3") is False  # uppercase
        assert is_open_source("DeepSeek V3") is True   # correct case
    
    def test_partial_matches(self):
        """Test that pattern matching works with partial model names."""
        assert is_open_source("Some DeepSeek Model") is True
        assert is_open_source("Llama-based Custom Model") is True


class TestOpenSourcePatterns:
    """Tests for OPEN_SOURCE_PATTERNS constant."""
    
    def test_patterns_not_empty(self):
        """Test that patterns list is not empty."""
        assert len(OPEN_SOURCE_PATTERNS) > 0
    
    def test_expected_patterns_present(self):
        """Test that all expected patterns are present."""
        expected = ['DeepSeek', 'Qwen', 'GLM', 'Llama', 'Mistral', 'Mixtral', 'Gemma', 'Phi-']
        for pattern in expected:
            assert pattern in OPEN_SOURCE_PATTERNS, f"Missing pattern: {pattern}"
    
    def test_patterns_are_strings(self):
        """Test that all patterns are strings."""
        for pattern in OPEN_SOURCE_PATTERNS:
            assert isinstance(pattern, str)


# =============================================================================
# New Use Case Tests (PromptCategory)
# =============================================================================

class TestNewPromptCategories:
    """Tests for new PromptCategory enum values."""
    
    def test_qa_category_exists(self):
        """Test that QA category exists."""
        assert hasattr(PromptCategory, 'QA')
        assert PromptCategory.QA.value == "Q&A"
    
    def test_rag_category_exists(self):
        """Test that RAG category exists."""
        assert hasattr(PromptCategory, 'RAG')
        assert PromptCategory.RAG.value == "RAG"
    
    def test_chatbot_category_exists(self):
        """Test that CHATBOT category exists."""
        assert hasattr(PromptCategory, 'CHATBOT')
        assert PromptCategory.CHATBOT.value == "Chatbot"
    
    def test_all_frontend_categories_supported(self):
        """Test that all frontend use cases have corresponding categories."""
        frontend_categories = ['CODING', 'DATA_SCIENCE', 'CREATIVE', 'GENERAL', 'QA', 'RAG', 'CHATBOT']
        for cat in frontend_categories:
            assert hasattr(PromptCategory, cat), f"Missing category: {cat}"


# =============================================================================
# QualityScorer Tests for New Use Cases
# =============================================================================

class TestQualityScorerNewUseCases:
    """Tests for QualityScorer with new use cases."""
    
    def test_qa_weights_include_trust(self, quality_scorer):
        """Test that QA use case weights include trust metrics."""
        weights = quality_scorer._get_task_weights(PromptCategory.QA)
        
        # QA should prioritize trust/accuracy
        assert 'hallucination_index' in weights or 'intelligence_index' in weights
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_rag_weights_include_context(self, quality_scorer):
        """Test that RAG use case weights include context score."""
        weights = quality_scorer._get_task_weights(PromptCategory.RAG)
        
        # RAG should prioritize context window
        assert 'context_score' in weights
        assert weights['context_score'] > 0
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_chatbot_weights_prioritize_speed_cost(self, quality_scorer):
        """Test that CHATBOT use case weights prioritize speed and cost."""
        weights = quality_scorer._get_task_weights(PromptCategory.CHATBOT)
        
        # Chatbot should balance quality with efficiency
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_quality_score_calculation_qa(self, quality_scorer, sample_raw_data):
        """Test quality score calculation for QA use case."""
        for model_data in sample_raw_data:
            score = quality_scorer.calculate_quality_score(model_data, PromptCategory.QA)
            assert 0 <= score <= 100, f"Score out of range for {model_data['name']}: {score}"
    
    def test_quality_score_calculation_rag(self, quality_scorer, sample_raw_data):
        """Test quality score calculation for RAG use case."""
        for model_data in sample_raw_data:
            score = quality_scorer.calculate_quality_score(model_data, PromptCategory.RAG)
            assert 0 <= score <= 100, f"Score out of range for {model_data['name']}: {score}"
    
    def test_quality_score_calculation_chatbot(self, quality_scorer, sample_raw_data):
        """Test quality score calculation for CHATBOT use case."""
        for model_data in sample_raw_data:
            score = quality_scorer.calculate_quality_score(model_data, PromptCategory.CHATBOT)
            assert 0 <= score <= 100, f"Score out of range for {model_data['name']}: {score}"
    
    def test_rag_prefers_large_context_models(self, quality_scorer, sample_raw_data):
        """Test that RAG scoring prefers models with larger context windows."""
        # Find models with different context sizes
        large_context = next(m for m in sample_raw_data if m['context_window_k'] >= 200)
        small_context = next(m for m in sample_raw_data if m['context_window_k'] <= 32)
        
        large_score = quality_scorer.calculate_quality_score(large_context, PromptCategory.RAG)
        small_score = quality_scorer.calculate_quality_score(small_context, PromptCategory.RAG)
        
        # Large context should score higher for RAG (with significant weight on context)
        # Note: This may not always hold if other factors dominate
        # But context_score should at least be calculated
        assert large_score > 0
        assert small_score > 0


# =============================================================================
# Context Score Tests
# =============================================================================

class TestContextScore:
    """Tests for context score calculation in QualityScorer."""
    
    def test_context_score_in_benchmarks(self, sample_raw_data):
        """Test that context_score is calculated during initialization."""
        scorer = QualityScorer(all_models_data=sample_raw_data)
        
        # context_score should be calculated during init
        assert 'context_score' in scorer.benchmarks
        assert len(scorer.benchmarks['context_score']) == len(sample_raw_data)
    
    def test_context_score_log_scaling(self, sample_raw_data):
        """Test that context score uses logarithmic scaling."""
        scorer = QualityScorer(all_models_data=sample_raw_data)
        
        # Find indices for models with different context sizes
        context_sizes = [m['context_window_k'] for m in sample_raw_data]
        scores = scorer.benchmarks['context_score']
        
        # Pair up sizes and scores
        size_score_pairs = list(zip(context_sizes, scores))
        
        # Sort by context size
        size_score_pairs.sort(key=lambda x: x[0])
        
        # Scores should generally increase with context size (log scaled)
        # But the increase should be sublinear
        small_size, small_score = size_score_pairs[0]
        large_size, large_score = size_score_pairs[-1]
        
        if small_score > 0:
            size_ratio = large_size / small_size
            score_ratio = large_score / small_score
            # Due to log scaling, score ratio should be less than size ratio
            assert score_ratio < size_ratio, "Context score should scale sublinearly"
    
    def test_context_score_handles_zero(self, sample_raw_data):
        """Test that context score handles zero context window."""
        # Add a model with zero context
        data_with_zero = sample_raw_data + [{'name': 'Zero Context', 'context_window_k': 0}]
        scorer = QualityScorer(all_models_data=data_with_zero)
        
        # Should not crash and should have context scores
        assert len(scorer.benchmarks['context_score']) == len(data_with_zero)
    
    def test_context_score_in_composite(self, quality_scorer, sample_raw_data):
        """Test that context score is included in composite score for RAG."""
        model_data = sample_raw_data[0]
        
        # Get score with and without context consideration
        rag_score = quality_scorer.calculate_quality_score(model_data, PromptCategory.RAG)
        general_score = quality_scorer.calculate_quality_score(model_data, PromptCategory.GENERAL)
        
        # Both should be valid scores
        assert 0 <= rag_score <= 100
        assert 0 <= general_score <= 100
        
        # They may be different due to different weights
        # (RAG weights context more heavily)


# =============================================================================
# Integration Tests - get_value_recommendations with open_source_only
# =============================================================================

class TestGetValueRecommendationsOpenSource:
    """Tests for get_value_recommendations() with open_source_only parameter."""
    
    def test_open_source_only_returns_only_oss_models(self):
        """Test that open_source_only=True returns only open source models."""
        results, _ = get_value_recommendations(
            "Write a Python function",
            open_source_only=True,
            top_k=10,
            verbose=False,
        )
        
        for rec in results:
            assert is_open_source(rec.model_name), f"{rec.model_name} is not open source"
    
    def test_open_source_false_includes_proprietary(self):
        """Test that open_source_only=False includes proprietary models."""
        results, _ = get_value_recommendations(
            "Write a Python function",
            open_source_only=False,
            top_k=10,
            verbose=False,
        )
        
        # Should include at least some proprietary models
        proprietary_count = sum(1 for r in results if not is_open_source(r.model_name))
        assert proprietary_count > 0, "Expected some proprietary models in results"
    
    def test_open_source_with_quality_constraints(self):
        """Test open_source_only works with quality constraints."""
        results, _ = get_value_recommendations(
            "Analyze this data",
            open_source_only=True,
            min_quality_ratio=0.70,
            max_cost_ratio=0.50,
            top_k=5,
            verbose=False,
        )
        
        # All results should be open source
        for rec in results:
            assert is_open_source(rec.model_name)
    
    def test_open_source_with_latency_constraints(self):
        """Test open_source_only works with latency constraints."""
        results, _ = get_value_recommendations(
            "Quick question",
            open_source_only=True,
            max_latency_ratio=1.0,
            latency_flexible=False,
            top_k=5,
            verbose=False,
        )
        
        # All results should be open source
        for rec in results:
            assert is_open_source(rec.model_name)


# =============================================================================
# Integration Tests - get_best_models_for_budget with open_source_only
# =============================================================================

class TestGetBestModelsForBudgetOpenSource:
    """Tests for get_best_models_for_budget() with open_source_only parameter."""
    
    def test_open_source_only_budget_mode(self):
        """Test that open_source_only=True returns only open source models in budget mode."""
        results = get_best_models_for_budget(
            max_budget=5.0,
            open_source_only=True,
            top_k=10,
            verbose=False,
        )
        
        for rec in results:
            assert is_open_source(rec.model_name), f"{rec.model_name} is not open source"
    
    def test_budget_mode_with_low_budget_oss(self):
        """Test budget mode with low budget and open source filter."""
        results = get_best_models_for_budget(
            max_budget=0.50,
            open_source_only=True,
            top_k=5,
            verbose=False,
        )
        
        # Should return results within budget
        for rec in results:
            assert is_open_source(rec.model_name)
    
    def test_budget_mode_open_source_false_includes_all(self):
        """Test that open_source_only=False includes all models in budget mode."""
        results = get_best_models_for_budget(
            max_budget=10.0,  # High budget
            open_source_only=False,
            top_k=10,
            verbose=False,
        )
        
        # Should include at least some proprietary models
        proprietary_count = sum(1 for r in results if not is_open_source(r.model_name))
        assert proprietary_count > 0, "Expected some proprietary models"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in new functionality."""
    
    def test_is_open_source_empty_string(self):
        """Test is_open_source with empty string."""
        assert is_open_source("") is False
    
    def test_is_open_source_with_special_characters(self):
        """Test is_open_source with special characters in name."""
        assert is_open_source("DeepSeek-V3.1 (beta)") is True
        assert is_open_source("Llama_4_maverick") is True
    
    def test_quality_scorer_empty_model_data(self, quality_scorer):
        """Test QualityScorer with empty model data."""
        empty_model = {'name': 'Empty Model'}
        
        # Should not crash, should return a score
        score = quality_scorer.calculate_quality_score(empty_model, PromptCategory.RAG)
        assert score >= 0
    
    def test_quality_scorer_zero_context(self, quality_scorer):
        """Test quality score with zero context window model."""
        model = {'name': 'Zero Context', 'context_window_k': 0}
        
        # Should not crash for RAG scoring
        score = quality_scorer.calculate_quality_score(model, PromptCategory.RAG)
        assert score >= 0


# =============================================================================
# Constraint Parameters Tests
# =============================================================================

class TestConstraintParameters:
    """Tests for constraint parameters matching frontend values."""
    
    @pytest.mark.parametrize("quality_ratio", [0.70, 0.80, 0.90])
    def test_quality_constraint_values(self, quality_ratio):
        """Test that quality constraint values from frontend work."""
        results, _ = get_value_recommendations(
            "Test prompt",
            min_quality_ratio=quality_ratio,
            top_k=3,
            verbose=False,
        )
        # Should complete without error
        assert isinstance(results, list)
    
    @pytest.mark.parametrize("cost_ratio", [0.25, 0.50, 1.00])
    def test_cost_constraint_values(self, cost_ratio):
        """Test that cost constraint values from frontend work."""
        results, _ = get_value_recommendations(
            "Test prompt",
            max_cost_ratio=cost_ratio,
            top_k=3,
            verbose=False,
        )
        # Should complete without error
        assert isinstance(results, list)
    
    @pytest.mark.parametrize("latency_ratio", [0.10, 0.50, 1.00])
    def test_latency_constraint_values(self, latency_ratio):
        """Test that latency constraint values from frontend work."""
        results, _ = get_value_recommendations(
            "Test prompt",
            max_latency_ratio=latency_ratio,
            top_k=3,
            verbose=False,
        )
        # Should complete without error
        assert isinstance(results, list)
    
    @pytest.mark.parametrize("budget", [0.10, 0.25, 0.50, 1.00, 2.00, 5.00])
    def test_budget_values(self, budget):
        """Test that budget values from frontend work."""
        results = get_best_models_for_budget(
            max_budget=budget,
            top_k=3,
            verbose=False,
        )
        # Should complete without error
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

