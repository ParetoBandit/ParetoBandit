from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from banditgpt._resources import get_models_cache_path, get_user_priors_dir, get_user_priors_path

_SETTINGS_CACHE: Optional["Settings"] = None


ENV_USER_PRIORS_DIR = "BANDITGPT_USER_PRIORS_DIR"
ENV_MODELS_CACHE = "BANDITGPT_MODELS_CACHE"
ENV_DEFAULT_PRIOR_STRENGTH = "BANDITGPT_DEFAULT_PRIOR_STRENGTH"
ENV_DEFAULT_EXPLORATION = "BANDITGPT_DEFAULT_EXPLORATION"


@dataclass(frozen=True)
class Settings:
    """Runtime settings with env overrides for common paths and tunables."""

    user_priors_dir: Path
    models_cache_path: Path
    default_prior_strength: float = 50.0
    default_exploration: str = "safe"

    @property
    def user_priors_path(self) -> Path:
        return Path(self.user_priors_dir) / "user_priors.npz"

    def ensure_writable_user_dir(self) -> None:
        """
        Best-effort creation of the user priors directory.

        Designed to be safe in read-only environments: failures are swallowed so
        callers can choose alternative paths if needed.
        """
        try:
            self.user_priors_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Read-only or permission errors; caller may switch to a temp dir.
            pass


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return fallback


def _env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def _env_str(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    return value if value else fallback


def load_settings() -> Settings:
    """Load settings using environment overrides with sensible defaults."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None:
        return _SETTINGS_CACHE

    user_dir = _env_path(ENV_USER_PRIORS_DIR, get_user_priors_dir())
    models_cache = _env_path(ENV_MODELS_CACHE, get_models_cache_path())
    prior_strength = _env_float(ENV_DEFAULT_PRIOR_STRENGTH, 50.0)
    exploration = _env_str(ENV_DEFAULT_EXPLORATION, "safe")

    _SETTINGS_CACHE = Settings(
        user_priors_dir=user_dir,
        models_cache_path=models_cache,
        default_prior_strength=prior_strength,
        default_exploration=exploration,
    )
    return _SETTINGS_CACHE


def reset_settings_cache() -> None:
    """Reset cached settings (useful for tests)."""
    global _SETTINGS_CACHE
    _SETTINGS_CACHE = None
