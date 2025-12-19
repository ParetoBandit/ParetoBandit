"""
Tests for async_bandit package entrypoints.
"""


def test_async_bandit_grader_entrypoints_import():
    """Test that core grader components can be imported."""
    from banditgpt.async_bandit.quality_cost_predictor import QualityCostPredictor
    from banditgpt.async_bandit.tiered_grader import (
        HardPromptHeuristics,
        OpenRouterTeacherVerifier,
        TieredGrader,
        UnsafePythonSubprocessVerifier,
    )

    assert QualityCostPredictor is not None
    assert TieredGrader is not None
    assert OpenRouterTeacherVerifier is not None
    assert HardPromptHeuristics is not None
    assert UnsafePythonSubprocessVerifier is not None


def test_async_bandit_package_import_does_not_crash():
    """Package should import even if optional dependencies are missing."""
    import banditgpt.async_bandit as ab  # noqa: F401

    # Check core exports exist
    assert hasattr(ab, "TieredGrader")
    assert hasattr(ab, "QualityCostPredictor")
    assert hasattr(ab, "PriorManager")


def test_complexity_module_import():
    """Test that complexity classifiers can be imported."""
    from banditgpt.async_bandit.complexity import (
        LocalComplexityClassifier,
        NvidiaComplexityClassifier,
        get_complexity_classifier,
    )

    assert LocalComplexityClassifier is not None
    assert NvidiaComplexityClassifier is not None
    assert get_complexity_classifier is not None


def test_judge_module_import():
    """Test that judge abstraction can be imported."""
    from banditgpt.async_bandit.judge import (
        Judge,
        PriorConfig,
        PriorManager,
        create_custom_judge,
    )

    assert Judge is not None
    assert PriorConfig is not None
    assert PriorManager is not None
    assert create_custom_judge is not None


def test_main_package_import():
    """Test that main banditgpt package imports correctly."""
    import banditgpt

    assert banditgpt.__version__ is not None
    assert hasattr(banditgpt, "TieredGrader")
    assert hasattr(banditgpt, "QualityCostPredictor")
    assert hasattr(banditgpt, "PriorManager")
