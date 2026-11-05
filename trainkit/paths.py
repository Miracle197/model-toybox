"""Where TrainKit keeps its environments, runs, datasets and weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


def _needs_safe_windows_home(home: Path) -> bool:
    """Non-ASCII user folders and OneDrive-synced folders break many ML tools on Windows."""
    text = str(home)
    return (not text.isascii()) or ("onedrive" in text.lower())


def home() -> Path:
    override = os.environ.get("TRAINKIT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    user_home = Path.home()
    if IS_WINDOWS and _needs_safe_windows_home(user_home):
        drive = os.environ.get("SystemDrive", "C:")
        return Path(f"{drive}\\trainkit")
    return user_home / ".trainkit"


def envs_dir() -> Path:
    return home() / "envs"


def runs_dir() -> Path:
    return home() / "runs"


def datasets_dir() -> Path:
    return home() / "datasets"


def weights_dir() -> Path:
    return home() / "weights"


def models_dir() -> Path:
    """Unpacked model bundles and exported models."""
    return home() / "models"


def ultralytics_config_dir() -> Path:
    return home() / "ultralytics-config"


def config_file() -> Path:
    return home() / "config.json"


def ensure_dirs() -> None:
    for d in (envs_dir(), runs_dir(), datasets_dir(), weights_dir(), models_dir(), ultralytics_config_dir()):
        d.mkdir(parents=True, exist_ok=True)


def package_dir() -> Path:
    return Path(__file__).resolve().parent
