import os
from pathlib import Path

from banditgpt import load_settings, Settings
from banditgpt.settings import reset_settings_cache
from banditgpt._resources import get_models_cache_path, get_user_priors_dir


def test_settings_defaults():
    reset_settings_cache()
    settings = load_settings()
    assert settings.user_priors_dir == get_user_priors_dir()
    assert settings.models_cache_path == get_models_cache_path()
    assert settings.default_prior_strength == 50.0
    assert settings.default_exploration == "safe"
    assert settings.user_priors_path == Path(settings.user_priors_dir) / "user_priors.npz"
    # ensure_writable_user_dir should not raise even if path is default
    settings.ensure_writable_user_dir()


def test_settings_env_overrides(monkeypatch, tmp_path):
    reset_settings_cache()
    user_dir = tmp_path / "priors"
    cache_path = tmp_path / "models_cache.json"
    cache_path.write_text("{}")
    monkeypatch.setenv("BANDITGPT_USER_PRIORS_DIR", str(user_dir))
    monkeypatch.setenv("BANDITGPT_MODELS_CACHE", str(cache_path))
    monkeypatch.setenv("BANDITGPT_DEFAULT_PRIOR_STRENGTH", "12.5")
    monkeypatch.setenv("BANDITGPT_DEFAULT_EXPLORATION", "balanced")

    settings = load_settings()
    assert settings.user_priors_dir == user_dir
    assert settings.models_cache_path == cache_path
    assert settings.default_prior_strength == 12.5
    assert settings.default_exploration == "balanced"
    settings.ensure_writable_user_dir()
