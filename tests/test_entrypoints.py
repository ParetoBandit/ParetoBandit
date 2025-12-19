"""
Tests for banditgpt.core package entrypoints.
"""


def test_core_grader_entrypoints_import():
    """Test that core grader components can be imported."""
    from banditgpt.core.quality_cost_predictor import QualityCostPredictor
    from banditgpt.core.tiered_grader import (
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


def test_core_package_import_does_not_crash():
    """Package should import even if optional dependencies are missing."""
    import banditgpt.core as core  # noqa: F401

    # Check core exports exist
    assert hasattr(core, "TieredGrader")
    assert hasattr(core, "QualityCostPredictor")
    assert hasattr(core, "PriorManager")


def test_judge_module_import():
    """Test that judge abstraction can be imported."""
    from banditgpt.core.judge import (
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
