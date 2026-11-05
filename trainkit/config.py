"""User config stored in <trainkit home>/config.json."""

from __future__ import annotations

import json
from typing import Any

from .paths import config_file, home

DEFAULTS: dict[str, Any] = {
    "lang": "",  # "zh" | "en" | "" (auto)
    "pypi_index": "",  # e.g. https://pypi.tuna.tsinghua.edu.cn/simple
    "roboflow_api_key": "",
    "dashboard_port": 7720,
    "playground_port": 7721,
}


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    path = config_file()
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    home().mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in cfg.items() if k in DEFAULTS}
    config_file().write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_value(key: str, value: str) -> None:
    if key not in DEFAULTS:
        raise KeyError(key)
    cfg = load_config()
    default = DEFAULTS[key]
    cfg[key] = int(value) if isinstance(default, int) else value
    save_config(cfg)
