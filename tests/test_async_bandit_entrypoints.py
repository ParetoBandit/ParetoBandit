def test_async_bandit_grader_entrypoints_import():
    from llm_jury.async_bandit.grader import (
        QualityCostPredictor,
        TieredGrader,
        OpenRouterTeacherVerifier,
        HardPromptHeuristics,
        UnsafePythonSubprocessVerifier,
    )

    # Just sanity-check symbols exist
    assert QualityCostPredictor is not None
    assert TieredGrader is not None
    assert OpenRouterTeacherVerifier is not None
    assert HardPromptHeuristics is not None
    assert UnsafePythonSubprocessVerifier is not None


def test_async_bandit_package_import_does_not_crash():
    # Should import even if bandit dependencies are missing (lazy import).
    import llm_jury.async_bandit as ab  # noqa: F401

