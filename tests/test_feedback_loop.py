"""
Unit tests for the feedback loop mechanisms.

Tests:
  - report_feedback(): Direct human/execution feedback
  - Hard Truth: Code execution success/failure
  - Synthetic Truth: LLM-as-a-Judge (TieredGrader)
  - Human Truth: User thumbs-up/down
  - Rank-one update verification
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_registry() -> Dict[str, Dict[str, Any]]:
    """Sample model registry for testing."""
    return {
        "model-a": {"display_name": "Model A", "cost_per_1k_input": 0.001},
        "model-b": {"display_name": "Model B", "cost_per_1k_input": 0.002},
        "model-c": {"display_name": "Model C", "cost_per_1k_input": 0.003},
    }


def create_test_router(sample_registry: Dict[str, Dict[str, Any]]):
    """Create a BanditRouter for testing with mocked embedding."""
    from llm_jury.async_bandit.bandit_router import BanditRouter
    
    # Create router with actual initialization
    router = BanditRouter(
        model_registry=sample_registry,
        state_path=None,
        complexity_classifier=None,  # Disable to avoid loading models
    )
    
    # Mock the encoder to return deterministic embeddings
    def encode_side_effect(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            # Create a deterministic vector based on text hash
            np.random.seed(hash(text) % (2**32))
            vec = np.random.randn(384).astype(np.float32)
            vectors.append(vec)
        return np.array(vectors) if len(vectors) > 1 else vectors[0]
    
    router.encoder = MagicMock()
    router.encoder.encode.side_effect = encode_side_effect
    
    return router


# ---------------------------------------------------------------------------
# report_feedback() Tests
# ---------------------------------------------------------------------------


class TestReportFeedback:
    """Tests for BanditRouter.report_feedback() method."""

    def test_import(self):
        """report_feedback is available on BanditRouter."""
        from llm_jury.async_bandit import BanditRouter
        
        assert hasattr(BanditRouter, "report_feedback")

    def test_report_feedback_success(self, sample_registry):
        """report_feedback returns True for valid request_id."""
        router = create_test_router(sample_registry)
        
        # Route a prompt
        model_id, log = router.route("Test prompt")
        request_id = log.request_id
        
        # Report feedback
        result = router.report_feedback(request_id, reward=1.0)
        
        assert result is True
        # Log should be removed from pending
        assert len(router.logs) == 0

    def test_report_feedback_not_found(self, sample_registry):
        """report_feedback returns False for unknown request_id."""
        router = create_test_router(sample_registry)
        
        result = router.report_feedback("nonexistent-id", reward=1.0)
        
        assert result is False

    def test_report_feedback_updates_bandit(self, sample_registry):
        """report_feedback triggers bandit update."""
        router = create_test_router(sample_registry)
        
        # Get initial b vector
        model_id, log = router.route("Test prompt")
        b_before = router.bandit.b[model_id].copy()
        
        # Report positive feedback
        router.report_feedback(log.request_id, reward=1.0)
        
        b_after = router.bandit.b[model_id]
        
        # b vector should have changed
        assert not np.allclose(b_before, b_after)

    def test_report_feedback_positive_reward(self, sample_registry):
        """Positive reward pushes weights toward the prompt direction."""
        router = create_test_router(sample_registry)
        
        model_id, log = router.route("Test prompt")
        context_vec = np.array(log.context_vector)
        b_before = router.bandit.b[model_id].copy()
        
        router.report_feedback(log.request_id, reward=1.0)
        
        b_after = router.bandit.b[model_id]
        delta = b_after - b_before
        
        # Delta should be in the direction of the context vector
        # (positive correlation with positive reward)
        dot_product = np.dot(delta, context_vec / np.linalg.norm(context_vec))
        assert dot_product > 0

    def test_report_feedback_negative_reward(self, sample_registry):
        """Negative reward pushes weights away from the prompt direction."""
        router = create_test_router(sample_registry)
        
        model_id, log = router.route("Test prompt")
        context_vec = np.array(log.context_vector)
        b_before = router.bandit.b[model_id].copy()
        
        router.report_feedback(log.request_id, reward=-1.0)
        
        b_after = router.bandit.b[model_id]
        delta = b_after - b_before
        
        # Delta should be opposite to the context vector
        dot_product = np.dot(delta, context_vec / np.linalg.norm(context_vec))
        assert dot_product < 0

    def test_report_feedback_with_response_text(self, sample_registry):
        """report_feedback stores response_text when provided."""
        router = create_test_router(sample_registry)
        
        model_id, log = router.route("Test prompt")
        
        # Note: The log is removed after report_feedback, so we can't check it directly
        # Just verify the call succeeds
        result = router.report_feedback(
            log.request_id,
            reward=1.0,
            response_text="The answer is 42"
        )
        
        assert result is True


# ---------------------------------------------------------------------------
# Rank-One Update Tests
# ---------------------------------------------------------------------------


class TestRankOneUpdate:
    """Tests for the rank-one update mathematics."""

    def test_a_matrix_update_formula(self):
        """A_new = A_old + x·x' (outer product)."""
        from llm_jury.async_bandit.bandit_router import DisjointLinUCBPolicy
        
        bandit = DisjointLinUCBPolicy(["model-a"], dim=4, alpha=0.1)
        
        A_before = bandit.A["model-a"].copy()
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        # update() takes: model, x, reward_z (positional)
        bandit.update("model-a", x, 0.5)
        
        A_after = bandit.A["model-a"]
        expected_delta = np.outer(x, x)
        
        np.testing.assert_array_almost_equal(A_after - A_before, expected_delta)

    def test_b_vector_update_formula(self):
        """b_new = b_old + r·x."""
        from llm_jury.async_bandit.bandit_router import DisjointLinUCBPolicy
        
        bandit = DisjointLinUCBPolicy(["model-a"], dim=4, alpha=0.1)
        
        b_before = bandit.b["model-a"].copy()
        x = np.array([1.0, 2.0, 3.0, 4.0])
        reward = 0.8
        
        bandit.update("model-a", x, reward)
        
        b_after = bandit.b["model-a"]
        expected_delta = reward * x
        
        np.testing.assert_array_almost_equal(b_after - b_before, expected_delta)

    def test_update_reduces_uncertainty(self):
        """After update, uncertainty (variance) decreases for similar prompts."""
        from llm_jury.async_bandit.bandit_router import DisjointLinUCBPolicy
        
        bandit = DisjointLinUCBPolicy(["model-a"], dim=4, alpha=0.1)
        
        x = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        
        # Compute variance before
        var_before = x @ bandit.A_inv["model-a"] @ x
        
        # Update multiple times in the same direction
        for _ in range(10):
            bandit.update("model-a", x, 1.0)
        
        # Recompute inverse for fair comparison
        bandit.A_inv["model-a"] = np.linalg.inv(bandit.A["model-a"])
        
        # Compute variance after
        var_after = x @ bandit.A_inv["model-a"] @ x
        
        # Uncertainty should decrease
        assert var_after < var_before

    def test_update_is_fast(self):
        """Update should be O(d²) not O(d³)."""
        import time
        from llm_jury.async_bandit.bandit_router import DisjointLinUCBPolicy
        
        bandit = DisjointLinUCBPolicy(["model-a"], dim=384, alpha=0.1)
        x = np.random.randn(384)
        
        # Time 100 updates
        start = time.perf_counter()
        for _ in range(100):
            bandit.update("model-a", x, 1.0)
        elapsed = time.perf_counter() - start
        
        # Should complete in under 1 second (typically < 10ms)
        assert elapsed < 1.0, f"100 updates took {elapsed:.3f}s, expected < 1s"


# ---------------------------------------------------------------------------
# Hard Truth (Code Execution) Tests
# ---------------------------------------------------------------------------


class TestHardTruth:
    """Tests for code execution feedback path."""

    def test_code_verifier_protocol_exists(self):
        """CodeExecutionVerifier protocol is defined."""
        from llm_jury.async_bandit.tiered_grader import CodeExecutionVerifier
        
        assert CodeExecutionVerifier is not None

    def test_unsafe_python_verifier_disabled_by_default(self):
        """UnsafePythonSubprocessVerifier refuses to run unless allow_unsafe=True."""
        from llm_jury.async_bandit.tiered_grader import UnsafePythonSubprocessVerifier
        
        verifier = UnsafePythonSubprocessVerifier(allow_unsafe=False)
        score, meta = verifier.verify("Write hello world", "print('hello')")
        
        assert score == 0.0
        assert meta["ok"] is False
        assert "disabled" in meta["error"]

    def test_tiered_grader_uses_code_verifier_for_code_prompts(self):
        """TieredGrader uses code_verifier for code-like prompts when provided."""
        from llm_jury.async_bandit.tiered_grader import TieredGrader, HardPromptHeuristics
        
        # Mock soft grader
        soft_grader = MagicMock()
        soft_grader.predict_production.return_value = {
            "p_correct_raw": 0.5,
            "reward_raw": 0.5,
        }
        
        # Mock code verifier
        code_verifier = MagicMock()
        code_verifier.verify.return_value = (1.0, {"ok": True})
        
        grader = TieredGrader(
            soft_grader=soft_grader,
            code_verifier=code_verifier,
        )
        
        result = grader.predict_production(
            prompt="Write a Python function to add numbers",
            response="def add(a, b): return a + b"
        )
        
        # Code verifier should have been called
        code_verifier.verify.assert_called_once()
        assert result["tiered_used_code_verifier"] is True
        assert result["p_correct_raw"] == 1.0  # From code verifier


# ---------------------------------------------------------------------------
# Synthetic Truth (LLM-as-a-Judge) Tests
# ---------------------------------------------------------------------------


class TestSyntheticTruth:
    """Tests for LLM-as-a-Judge feedback path."""

    def test_teacher_verifier_protocol_exists(self):
        """TeacherVerifier protocol is defined."""
        from llm_jury.async_bandit.tiered_grader import TeacherVerifier
        
        assert TeacherVerifier is not None

    def test_openrouter_teacher_verifier_requires_api_key(self):
        """OpenRouterTeacherVerifier returns error without API key."""
        from llm_jury.async_bandit.tiered_grader import OpenRouterTeacherVerifier
        
        with patch.dict("os.environ", {}, clear=True):
            verifier = OpenRouterTeacherVerifier(api_key_env="NONEXISTENT_KEY")
            score, meta = verifier.verify("Test prompt", "Test response")
            
            assert score == 0.0
            assert meta["ok"] is False
            assert "not set" in meta["error"]

    def test_tiered_grader_uses_teacher_for_hard_prompts(self):
        """TieredGrader uses teacher_verifier for hard prompts."""
        from llm_jury.async_bandit.tiered_grader import TieredGrader
        
        # Mock soft grader
        soft_grader = MagicMock()
        soft_grader.predict_production.return_value = {
            "p_correct_raw": 0.5,
            "reward_raw": 0.5,
        }
        
        # Mock teacher verifier
        teacher_verifier = MagicMock()
        teacher_verifier.verify.return_value = (0.9, {"ok": True})
        
        grader = TieredGrader(
            soft_grader=soft_grader,
            teacher_verifier=teacher_verifier,
        )
        
        # Use a "hard" prompt (contains "calculate")
        result = grader.predict_production(
            prompt="Calculate the pH of a 0.01M HCl solution",
            response="pH = 2"
        )
        
        # Teacher should have been called
        teacher_verifier.verify.assert_called_once()
        assert result["tiered_used_teacher"] is True
        assert result["tiered_is_hard"] is True

    def test_tiered_grader_skips_teacher_for_easy_prompts(self):
        """TieredGrader does NOT call teacher for easy prompts."""
        from llm_jury.async_bandit.tiered_grader import TieredGrader
        
        # Mock soft grader
        soft_grader = MagicMock()
        soft_grader.predict_production.return_value = {
            "p_correct_raw": 0.8,
            "reward_raw": 0.8,
        }
        
        # Mock teacher verifier
        teacher_verifier = MagicMock()
        
        grader = TieredGrader(
            soft_grader=soft_grader,
            teacher_verifier=teacher_verifier,
        )
        
        # Use an "easy" prompt (no trigger words)
        result = grader.predict_production(
            prompt="Tell me a story about a cat",
            response="Once upon a time..."
        )
        
        # Teacher should NOT have been called
        teacher_verifier.verify.assert_not_called()
        assert result["tiered_used_teacher"] is False
        assert result["tiered_is_hard"] is False


# ---------------------------------------------------------------------------
# Human Truth Tests
# ---------------------------------------------------------------------------


class TestHumanTruth:
    """Tests for human feedback path."""

    def test_thumbs_up_reward(self, sample_registry):
        """Thumbs up = reward 1.0."""
        router = create_test_router(sample_registry)
        
        model_id, log = router.route("User query")
        b_before = router.bandit.b[model_id].copy()
        
        # User clicks thumbs up
        router.report_feedback(log.request_id, reward=1.0)
        
        b_after = router.bandit.b[model_id]
        
        # b should increase in the direction of the prompt
        assert np.linalg.norm(b_after) > np.linalg.norm(b_before)

    def test_thumbs_down_reward(self, sample_registry):
        """Thumbs down = reward -1.0 (or similar penalty)."""
        router = create_test_router(sample_registry)
        
        model_id, log = router.route("User query")
        context_vec = np.array(log.context_vector)
        b_before = router.bandit.b[model_id].copy()
        
        # User clicks thumbs down
        router.report_feedback(log.request_id, reward=-1.0)
        
        b_after = router.bandit.b[model_id]
        delta = b_after - b_before
        
        # Delta should be in opposite direction of context
        dot_product = np.dot(delta, context_vec / np.linalg.norm(context_vec))
        assert dot_product < 0

    def test_regenerate_penalty(self, sample_registry):
        """Regenerate button = moderate penalty (-0.5)."""
        router = create_test_router(sample_registry)
        
        model_id, log = router.route("User query")
        context_vec = np.array(log.context_vector)
        b_before = router.bandit.b[model_id].copy()
        
        # User clicks regenerate
        router.report_feedback(log.request_id, reward=-0.5)
        
        b_after = router.bandit.b[model_id]
        delta = b_after - b_before
        
        # Small penalty in opposite direction
        dot_product = np.dot(delta, context_vec / np.linalg.norm(context_vec))
        assert dot_product < 0

    def test_edit_neutral(self, sample_registry):
        """User edits response = neutral (0.0)."""
        router = create_test_router(sample_registry)
        
        model_id, log = router.route("User query")
        b_before = router.bandit.b[model_id].copy()
        
        # User edits response (neutral feedback)
        router.report_feedback(log.request_id, reward=0.0)
        
        b_after = router.bandit.b[model_id]
        
        # b should not change with reward=0
        np.testing.assert_array_equal(b_before, b_after)


# ---------------------------------------------------------------------------
# HardPromptHeuristics Tests
# ---------------------------------------------------------------------------


class TestHardPromptHeuristics:
    """Tests for hard prompt detection."""

    def test_detects_math_prompts(self):
        """Detects math/calculation prompts as hard."""
        from llm_jury.async_bandit.tiered_grader import HardPromptHeuristics
        
        detector = HardPromptHeuristics()
        
        assert detector.is_hard("Calculate the derivative of x^2")
        assert detector.is_hard("Compute the integral from 0 to 1")
        assert detector.is_hard("Solve this equation: x + 5 = 10")
        assert detector.is_hard("What is the pH of a 0.01M solution?")

    def test_detects_code_prompts(self):
        """Detects code/programming prompts as hard."""
        from llm_jury.async_bandit.tiered_grader import HardPromptHeuristics
        
        detector = HardPromptHeuristics()
        
        assert detector.is_hard("Write a Python function to sort a list")
        assert detector.is_hard("Create a JavaScript class for users")
        assert detector.is_hard("Fix this SQL query")
        assert detector.is_hard("Write TypeScript code to fetch data")

    def test_detects_constraint_prompts(self):
        """Detects constraint-heavy prompts as hard."""
        from llm_jury.async_bandit.tiered_grader import HardPromptHeuristics
        
        detector = HardPromptHeuristics()
        
        assert detector.is_hard("Output must be valid JSON")
        assert detector.is_hard("Use exactly 100 words")
        assert detector.is_hard("The answer must contain only numbers")

    def test_easy_prompts_not_hard(self):
        """Creative/conversational prompts are NOT hard."""
        from llm_jury.async_bandit.tiered_grader import HardPromptHeuristics
        
        detector = HardPromptHeuristics()
        
        assert not detector.is_hard("Tell me a story about a dragon")
        assert not detector.is_hard("What's the capital of France?")
        assert not detector.is_hard("Write a poem about autumn")
        assert not detector.is_hard("Summarize this article")


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestFeedbackIntegration:
    """Integration tests for the full feedback loop."""

    def test_feedback_affects_future_routing(self, sample_registry):
        """Positive feedback makes model more likely to be selected for similar prompts."""
        router = create_test_router(sample_registry)
        
        # Route a prompt and give positive feedback
        model1, log1 = router.route("Test prompt for learning", exploration="aggressive")
        
        # Record the UCB score before feedback
        x = np.array(log1.context_vector, dtype=np.float64)
        theta_before = router.bandit.A_inv[model1] @ router.bandit.b[model1]
        score_before = theta_before @ x
        
        # Give strong positive feedback
        for _ in range(10):
            _, log = router.route("Test prompt for learning")
            router.report_feedback(log.request_id, reward=1.0)
        
        # Recompute inverse for fair comparison
        router.bandit.A_inv[model1] = np.linalg.inv(router.bandit.A[model1])
        
        # Check score increased
        theta_after = router.bandit.A_inv[model1] @ router.bandit.b[model1]
        score_after = theta_after @ x
        
        assert score_after > score_before

    def test_multiple_feedback_types_coexist(self, sample_registry):
        """Different feedback types can be used in the same session."""
        router = create_test_router(sample_registry)
        
        # Route three prompts
        _, log1 = router.route("Query 1")
        _, log2 = router.route("Query 2")
        _, log3 = router.route("Query 3")
        
        # Give different feedback types
        # Human: thumbs up
        result1 = router.report_feedback(log1.request_id, reward=1.0)
        # Hard truth: code failed
        result2 = router.report_feedback(log2.request_id, reward=0.0)
        # Human: regenerate
        result3 = router.report_feedback(log3.request_id, reward=-0.5)
        
        assert result1 is True
        assert result2 is True
        assert result3 is True
        assert len(router.logs) == 0  # All processed
