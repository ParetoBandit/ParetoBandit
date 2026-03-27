"""Pytest configuration for the integration test suite."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "stress: marks stress and soak-style validation tests"
    )
