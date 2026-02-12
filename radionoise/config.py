"""
RadioNoise persistent configuration.

Loads/saves settings from ~/.radionoise/config.json.
"""

import json
from pathlib import Path
from typing import Any, Optional


DEFAULTS = {
    "capture": {
        "frequency_mhz": 100.0,
        "samples": 500000,
        "allow_fallback": True,
        "use_rdseed": False,
        "capture_raw": True,
    },
    "nist": {
        "fast_mode": True,
    },
    "generator": {
        "type": "password",
        "length": 16,
        "count": 5,
        "charset": "safe",
    },
    "gui": {
        "theme": "dark",
    },
}

CONFIG_DIR = Path.home() / ".radionoise"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Persistent configuration with deep-merge defaults."""

    def __init__(self, config_file: Optional[Path] = None):
        self._file = config_file or CONFIG_FILE
        self._data = self._load()

    def _load(self) -> dict:
        """Load config from file, deep-merged with defaults."""
        if self._file.exists():
            try:
                with open(self._file, 'r') as f:
                    user_data = json.load(f)
                return _deep_merge(DEFAULTS, user_data)
            except (json.JSONDecodeError, OSError):
                pass
        return DEFAULTS.copy()

    def get(self, section: str, key: str) -> Any:
        """Get a config value."""
        return self._data.get(section, {}).get(key)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a config value."""
        if section not in self._data:
            self._data[section] = {}
        self._data[section][key] = value

    def save(self) -> None:
        """Save config to file."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file, 'w') as f:
            json.dump(self._data, f, indent=2)
