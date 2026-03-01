"""
Re-export :func:`extract_reward` from the canonical ``src`` location.

All experiment code should ``from utils.rewards import extract_reward``.
The implementation lives in ``src/bandit_gpt/rewards.py`` so that both
library code (``src/``) and experiment scripts can share it.
"""

from bandit_gpt.rewards import extract_reward  # noqa: F401

__all__ = ["extract_reward"]
