"""
Resource utilities for the standalone bandit package.
"""
from pathlib import Path

def get_package_data_dir() -> Path:
    """Get the directory containing this file."""
    return Path(__file__).parent

def get_data_path(filename: str) -> Path:
    """Get path to a data file."""
    return get_package_data_dir() / filename

def get_priors_path(filename: str = "shippable_priors.npz") -> Path:
    """Get path to a priors file."""
    return get_package_data_dir() / filename

def get_quality_predictor_path(filename: str = "best_quality_predictor.pt") -> Path:
    """Get path to quality predictor model."""
    return get_package_data_dir() / filename
