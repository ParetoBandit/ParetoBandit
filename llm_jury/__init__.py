"""
Lightweight async bandit-focused exports for llm_jury.

This package now centers on the async bandit entrypoints and their supporting
grader utilities. Other legacy modules were removed to slim the codebase.
"""

__version__ = "0.1.0"

from llm_jury.async_bandit.grader import (  # noqa: F401
    QualityCostPredictor,
    TieredGrader,
    OpenRouterTeacherVerifier,
    HardPromptHeuristics,
    UnsafePythonSubprocessVerifier,
)
from llm_jury.async_bandit.demo import run_demo  # noqa: F401

try:  # Optional: bandit routing extras
    from llm_jury.async_bandit.bandit import (  # noqa: F401
        BanditRouter,
        DisjointLinUCBPolicy,
        RoutingLog,
    )
except Exception:  # pragma: no cover
    BanditRouter = DisjointLinUCBPolicy = RoutingLog = None

__all__ = [
    "__version__",
    "QualityCostPredictor",
    "TieredGrader",
    "OpenRouterTeacherVerifier",
    "HardPromptHeuristics",
    "UnsafePythonSubprocessVerifier",
    "run_demo",
    "BanditRouter",
    "DisjointLinUCBPolicy",
    "RoutingLog",
]
