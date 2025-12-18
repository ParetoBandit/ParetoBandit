"""
Async bandit routing (organized entrypoints).

This package groups:
  - Judge abstraction (abstract grading interface + prior management)
  - Grader code (tiered grader + soft grader types)
  - Bandit code (BanditRouter)
  - Demo entrypoints

Prior Storage Locations:
  - BUNDLED (read-only):  <package>/data/priors/shippable_priors.npz
  - USER (read-write):    ~/.llm_jury/priors/user_priors.npz
  - CUSTOM:               User-specified path
"""

from llm_jury.async_bandit.grader import (
    TieredGrader,
    OpenRouterTeacherVerifier,
    HardPromptHeuristics,
    UnsafePythonSubprocessVerifier,
)
from llm_jury.async_bandit.quality_cost_predictor import (
    QualityCostPredictor,
    RunningZScoreNormalizer,
    LogitReward,
    get_device,
)
from llm_jury.async_bandit.judge import (
    Judge,
    JudgeWithComplexity,
    PriorConfig,
    PriorManager,
    create_soft_judge,
    create_tiered_judge,
    create_custom_judge,
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

# Optional: bandit router (requires sentence-transformers)
try:
    from llm_jury.async_bandit.bandit import BanditRouter, DisjointLinUCBPolicy, RoutingLog

    __all__ += ["BanditRouter", "DisjointLinUCBPolicy", "RoutingLog"]
except Exception:
    pass

