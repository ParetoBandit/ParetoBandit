"""
Core module for BanditGPT - Contextual Bandit Router for LLM Model Selection.

This package provides a contextual bandit-based router for selecting the optimal
LLM model for each prompt, balancing quality, cost, and latency.

Core Components:
    BanditRouter        - Main router with LinUCB policy
    TieredGrader        - Tiered grading (soft + hard verifier)
    QualityCostPredictor - Local quality prediction model
    PriorManager        - Prior loading/saving/merging

Prior Storage Locations:
    BUNDLED (read-only):  <package>/data/priors/shippable_priors.npz
    USER (read-write):    ~/.banditgpt/priors/user_priors.npz

Quick Start:
    from banditgpt.core import BanditRouter, PriorManager

    # Create router with automatic prior detection
    router = BanditRouter.create(model_registry, priors="merged")

    # Get recommendations
    model, log = router.route("Write a Python function...")

    # Add a new model dynamically
    router.add_model("openai/gpt-5", clone_from="openai/gpt-4o")
"""

from __future__ import annotations

# Core graders
from banditgpt.core.quality_cost_predictor import (
    LogitReward,
    QualityCostPredictor,
    RunningZScoreNormalizer,
    get_device,
)
from banditgpt.core.tiered_grader import (
    HardPromptHeuristics,
    OpenRouterTeacherVerifier,
    TieredGrader,
    UnsafePythonSubprocessVerifier,
)

# Judge abstraction and prior management
from banditgpt.core.judge import (
    Judge,
    JudgeWithComplexity,
    PriorConfig,
    PriorManager,
    create_custom_judge,
    create_soft_judge,
    create_tiered_judge,
)
from banditgpt.core.prior_downloader import ensure_priors, PriorDownloadError
from banditgpt.core.prior_manifest import (
    PriorFileInfo,
    PriorIntegrityError,
    PriorsManifest,
    load_priors_manifest,
)
from banditgpt.settings import Settings, load_settings
from banditgpt.logging_utils import configure_logging

__all__ = [
    # Judge abstraction
    "Judge",
    "JudgeWithComplexity",
    "PriorConfig",
    "PriorManager",
    "create_soft_judge",
    "create_tiered_judge",
    "create_custom_judge",
    "ensure_priors",
    "PriorDownloadError",
    "PriorsManifest",
    "PriorIntegrityError",
    "PriorFileInfo",
    "load_priors_manifest",
    "Settings",
    "load_settings",
    "configure_logging",
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
    from banditgpt.core.bandit_router import (
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
