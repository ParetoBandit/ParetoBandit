"""
Pytest configuration and shared fixtures.
"""

import sys
import pytest
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "stress: marks stress and soak-style validation tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmark and regression tests"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def models_dir(project_root):
    """Return the models directory."""
    return project_root / "models"


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def trained_model_path(models_dir):
    """Return path to trained XGBoost model if it exists."""
    model_path = models_dir / "xgboost_intent_classifier.json"
    if model_path.exists():
        return model_path
    return None

