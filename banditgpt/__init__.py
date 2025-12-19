"""
BanditGPT: Contextual Bandit Router for LLM Model Selection.

This package provides a contextual bandit-based router for selecting the optimal
LLM model for each prompt, balancing quality, cost, and latency.

Core Components:
    BanditRouter        - Main router with LinUCB policy
    TieredGrader        - Tiered grading (soft + hard verifier)
    QualityCostPredictor - Local quality prediction model
    PriorManager        - Prior loading/saving/merging

Quick Start:
    from banditgpt import BanditRouter, PriorManager

    router = BanditRouter.create(model_registry, priors="merged")
    model, log = router.route("Write a Python function...")
"""

__version__ = "0.1.0"

# Core graders (always available)
from banditgpt.core.quality_cost_predictor import QualityCostPredictor  # noqa: F401
from banditgpt.core.tiered_grader import (  # noqa: F401
    HardPromptHeuristics,
    OpenRouterTeacherVerifier,
    TieredGrader,
    UnsafePythonSubprocessVerifier,
)

# Prior management
from banditgpt.core.judge import PriorManager  # noqa: F401
from banditgpt.settings import Settings, load_settings  # noqa: F401

# Optional: Demo
try:
    from banditgpt.core.demo_quality_grader import run_demo  # noqa: F401
except Exception:  # pragma: no cover
    run_demo = None

# Optional: Bandit router (requires sentence-transformers)
try:
    from banditgpt.core.bandit_router import (  # noqa: F401
        BanditRouter,
        DisjointLinUCBPolicy,
        RoutingLog,
    )
except ImportError:  # pragma: no cover
    BanditRouter = None
    DisjointLinUCBPolicy = None
    RoutingLog = None

__all__ = [
    "__version__",
    # Graders
    "QualityCostPredictor",
    "TieredGrader",
    "OpenRouterTeacherVerifier",
    "HardPromptHeuristics",
    "UnsafePythonSubprocessVerifier",
    # Prior management
    "PriorManager",
    # Settings
    "Settings",
    "load_settings",
    # Demo
    "run_demo",
    # Router
    "BanditRouter",
    "DisjointLinUCBPolicy",
    "RoutingLog",
]
