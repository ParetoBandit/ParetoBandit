"""
Async Bandit Router for LLM Model Selection.

This package provides a contextual bandit-based router for selecting the optimal
LLM model for each prompt, balancing quality, cost, and latency.

Core Components:
    BanditRouter        - Main router with LinUCB policy
    TieredGrader        - Tiered grading (soft + hard verifier)
    QualityCostPredictor - Local quality prediction model
    PriorManager        - Prior loading/saving/merging

Prior Storage Locations:
    BUNDLED (read-only):  <package>/data/priors/shippable_priors.npz
    USER (read-write):    ~/.llm_jury/priors/user_priors.npz

Quick Start:
    from llm_jury.async_bandit import BanditRouter, PriorManager

    # Create router with automatic prior detection
    router = BanditRouter.create(model_registry, priors="merged")

    # Get recommendations
    model, log = router.route("Write a Python function...")

    # Add a new model dynamically
    router.add_model("openai/gpt-5", clone_from="openai/gpt-4o")
"""

from __future__ import annotations

# Core graders
from llm_jury.async_bandit.quality_cost_predictor import (
    LogitReward,
    QualityCostPredictor,
    RunningZScoreNormalizer,
    get_device,
)
from llm_jury.async_bandit.tiered_grader import (
    HardPromptHeuristics,
    OpenRouterTeacherVerifier,
    TieredGrader,
    UnsafePythonSubprocessVerifier,
)

# Judge abstraction and prior management
from llm_jury.async_bandit.judge import (
    Judge,
    JudgeWithComplexity,
    PriorConfig,
    PriorManager,
    create_custom_judge,
    create_soft_judge,
    create_tiered_judge,
)

__all__ = [
    # Judge abstraction
    "Judge",
    "JudgeWithComplexity",
    "PriorConfig",
    "PriorManager",
    "create_soft_judge",
    "create_tiered_judge",
    "create_custom_judge",
    # Graders
    "TieredGrader",
    "OpenRouterTeacherVerifier",
    "HardPromptHeuristics",
    "UnsafePythonSubprocessVerifier",
    "QualityCostPredictor",
    "RunningZScoreNormalizer",
    "LogitReward",
    "get_device",
]

# Optional: Bandit router (requires sentence-transformers)
try:
    from llm_jury.async_bandit.bandit_router import (
        BanditRouter,
        DisjointLinUCBPolicy,
        ExplorationRate,
        OptimizationProfile,
        RoutingLog,
        SharedCovarianceLinUCBPolicy,
        build_cost_proportional_priors,
        build_registry_from_models_cache,
    )

    __all__ += [
        "BanditRouter",
        "DisjointLinUCBPolicy",
        "ExplorationRate",
        "OptimizationProfile",
        "SharedCovarianceLinUCBPolicy",
        "RoutingLog",
        "build_registry_from_models_cache",
        "build_cost_proportional_priors",
    ]
except ImportError:
    pass  # sentence-transformers not installed

# Optional: Complexity classifiers
try:
    from llm_jury.async_bandit.complexity import (
        LocalComplexityClassifier,
        LocalComplexityResult,
        NvidiaComplexityClassifier,
        NvidiaComplexityResult,
        get_complexity_classifier,
    )

    __all__ += [
        "LocalComplexityClassifier",
        "LocalComplexityResult",
        "NvidiaComplexityClassifier",
        "NvidiaComplexityResult",
        "get_complexity_classifier",
    ]
except ImportError:
    pass  # transformers not installed
