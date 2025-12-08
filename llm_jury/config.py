"""Configuration management for LLM Jury."""

import os
from pathlib import Path
from typing import Optional
import json

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    import os as _os
    
    # Try multiple locations for .env file
    possible_env_files = [
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent.parent.parent / ".env",  # Project root (3 levels up from this file)
        Path.home() / ".env",  # Home directory
    ]
    
    for env_path in possible_env_files:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
            
except ImportError:
    pass


class Config:
    """Configuration manager for LLM Jury."""

    def __init__(self):
        """Initialize configuration from environment and config files."""
        self.config_dir = Path.home() / ".llm_jury"
        self.config_file = self.config_dir / "config.json"
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from file if it exists."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}

    def save_config(self):
        """Save current configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self._config, f, indent=2)

    @property
    def artificial_analysis_api_key(self) -> Optional[str]:
        """Get Artificial Analysis API key from environment or config."""
        # Priority: environment variable > config file
        return os.getenv("ARTIFICIAL_ANALYSIS_API_KEY") or self._config.get("artificial_analysis_api_key")

    @artificial_analysis_api_key.setter
    def artificial_analysis_api_key(self, value: str):
        """Set Artificial Analysis API key in config file."""
        self._config["artificial_analysis_api_key"] = value
        self.save_config()

    @property
    def data_dir(self) -> Path:
        """Get data directory for caching."""
        # Default to project directory if not configured
        default_dir = Path.cwd() / "data"
        data_dir = Path(self._config.get("data_dir", default_dir))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @data_dir.setter
    def data_dir(self, value: str):
        """Set data directory."""
        self._config["data_dir"] = str(value)
        self.save_config()

    @property
    def cache_file(self) -> Path:
        """Get path to models cache file."""
        # Check if custom path is set
        custom_path = self._config.get("cache_file_path")
        if custom_path:
            return Path(custom_path)
        return self.data_dir / "models_cache.json"
    
    @property
    def cache_file_path(self) -> Path:
        """Alias for cache_file for backwards compatibility."""
        return self.cache_file
    
    @cache_file_path.setter
    def cache_file_path(self, value: str):
        """Set custom cache file path."""
        self._config["cache_file_path"] = str(value)
        self.save_config()

    @property
    def auto_update(self) -> bool:
        """Check if automatic updates are enabled."""
        return self._config.get("auto_update", False)

    @auto_update.setter
    def auto_update(self, value: bool):
        """Enable/disable automatic updates."""
        self._config["auto_update"] = value
        self.save_config()

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value):
        """Set configuration value."""
        self._config[key] = value
        self.save_config()

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration. Returns (is_valid, error_messages)."""
        errors = []

        if not self.artificial_analysis_api_key:
            errors.append(
                "Artificial Analysis API key not set. "
                "Set it with: export ARTIFICIAL_ANALYSIS_API_KEY='your-key' or add to .env file"
            )

        return len(errors) == 0, errors


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

