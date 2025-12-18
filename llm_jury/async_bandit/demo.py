"""
Demo entrypoints for async bandit routing.

This keeps a stable import path, while the "real" demo lives in
`llm_jury.async_bandit.demo_quality_grader`.
"""

from llm_jury.async_bandit.demo_quality_grader import run_demo

__all__ = ["run_demo"]

