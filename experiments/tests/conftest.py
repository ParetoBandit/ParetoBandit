"""Shared fixtures for experiment regression tests.

These tests live outside the default ``testpaths = ["tests"]`` and are
**not** auto-discovered by CI.  Run them explicitly::

    pytest experiments/tests/ -v
    pytest experiments/tests/ -v -m "not slow"   # config-only checks
    pytest experiments/tests/ -v -k exp01         # single experiment
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
# Allow ``from helpers import ...`` inside test modules.
sys.path.insert(0, str(Path(__file__).parent))

from bandit_gpt.config import (
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    VAL_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from utils.simulation import SplitData, build_model_registry, load_split


def pytest_configure(config: Any) -> None:
    """Register the ``experiment`` marker."""
    config.addinivalue_line(
        "markers",
        "experiment: marks experiment regression tests "
        "(not auto-discovered by CI)",
    )


# ── Session-scoped heavy fixtures ────────────────────────────────────────


@pytest.fixture(scope="session")
def feature_service() -> FeatureService:
    """Load the sentence-transformer + PCA pipeline once per session."""
    return FeatureService()


@pytest.fixture(scope="session")
def feature_dim(feature_service: FeatureService) -> int:
    return feature_service.dimension


@pytest.fixture(scope="session")
def val_split(feature_service: FeatureService) -> SplitData:
    """Val split (used as online-learning stream in experiments)."""
    return load_split(VAL_DATA_PATH, feature_service, K3_ARM_ORDER)


@pytest.fixture(scope="session")
def test_split(feature_service: FeatureService) -> SplitData:
    """Holdout/test split (used for evaluation)."""
    return load_split(HOLDOUT_DATA_PATH, feature_service, K3_ARM_ORDER)


@pytest.fixture(scope="session")
def model_registry() -> Dict[str, Any]:
    return build_model_registry(K3_ARM_ORDER)
