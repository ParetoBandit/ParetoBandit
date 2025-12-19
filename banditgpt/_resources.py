"""
Package resource utilities for accessing bundled data files.

This module provides a consistent API for accessing data files that are
bundled with the package, using importlib.resources for proper pip install support.

Usage:
    from banditgpt._resources import get_data_path, get_priors_path

    # Get path to bundled priors
    priors_path = get_priors_path("shippable_priors.npz")

    # Get path to any data file
    cache_path = get_data_path("models_cache.json")
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

# Use importlib.resources for Python 3.9+
if sys.version_info >= (3, 9):
    from importlib.resources import files, as_file
else:
    from importlib_resources import files, as_file  # type: ignore[import-not-found]


_PKG_DATA_DIR: Optional[Path] = None


def get_package_data_dir() -> Path:
    """
    Get the path to the package's data directory.
    
    Returns:
        Path to banditgpt/data/ directory
    """
    global _PKG_DATA_DIR
    if _PKG_DATA_DIR is not None:
        return _PKG_DATA_DIR

    data_files = files("banditgpt.data")
    try:
        with as_file(data_files) as data_path:
            _PKG_DATA_DIR = Path(data_path)
            return _PKG_DATA_DIR
    except (TypeError, AttributeError):
        _PKG_DATA_DIR = Path(__file__).parent / "data"
        return _PKG_DATA_DIR


def get_data_path(filename: str) -> Path:
    """
    Get the path to a file in the package's data directory.
    
    Args:
        filename: Name of the file (e.g., "models_cache.json")
        
    Returns:
        Path to the file
    """
    return get_package_data_dir() / filename


def get_priors_path(filename: str = "shippable_priors.npz") -> Path:
    """
    Get the path to a priors file.
    
    Args:
        filename: Name of the priors file (default: "shippable_priors.npz")
        
    Returns:
        Path to the priors file in banditgpt/data/priors/
    """
    return get_package_data_dir() / "priors" / filename


def get_priors_manifest_path() -> Path:
    """
    Get the path to the priors manifest file.

    Returns:
        Path to banditgpt/data/priors/manifest.json
    """
    return get_priors_path("manifest.json")


def get_quality_predictor_path(filename: str = "best_quality_predictor.pt") -> Path:
    """
    Get the path to a quality predictor model file.
    
    Args:
        filename: Name of the model file
        
    Returns:
        Path to the model file in banditgpt/data/quality_predictor/
    """
    return get_package_data_dir() / "quality_predictor" / filename


def get_user_data_dir() -> Path:
    """
    Get the user-specific data directory (~/.banditgpt/).
    
    This directory is used for user-specific priors and cached data.
    Creates the directory if it doesn't exist. Falls back to the system
    temp directory if the home directory is not writable.
    
    Returns:
        Path to ~/.banditgpt/
    """
    user_dir = Path.home() / ".banditgpt"
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    except OSError:
        tmp_dir = Path(tempfile.gettempdir()) / "banditgpt"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir


def get_user_priors_dir() -> Path:
    """
    Get the user-specific priors directory (~/.banditgpt/priors/).
    
    Creates the directory if it doesn't exist.
    
    Returns:
        Path to ~/.banditgpt/priors/
    """
    priors_dir = get_user_data_dir() / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)
    return priors_dir


def get_user_priors_path(filename: str = "user_priors.npz") -> Path:
    """
    Get the path to a user priors file.
    
    Args:
        filename: Name of the priors file (default: "user_priors.npz")
        
    Returns:
        Path to the user priors file
    """
    return get_user_priors_dir() / filename


# For backwards compatibility and ease of use
def get_bundled_priors_path() -> Path:
    """Get the default bundled priors path (shippable_priors.npz)."""
    return get_priors_path("shippable_priors.npz")


def get_expert_priors_path() -> Path:
    """Get the expert priors path (expert_priors.npz)."""
    return get_priors_path("expert_priors.npz")


def get_models_cache_path() -> Path:
    """Get the models cache JSON path."""
    return get_data_path("models_cache.json")
