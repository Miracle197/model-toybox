"""Helpers shared by scripts that run *inside* a recipe environment.

These files are executed by path with the environment's Python, so they may only import the standard library,
each other, and packages installed in that environment (torch, ultralytics, ...). Never import `trainkit` here.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

JSON_MARK = "TRAINKIT_JSON="


def emit_result(obj: dict) -> None:
    sys.stdout.flush()
    print(JSON_MARK + json.dumps(obj, ensure_ascii=False, default=str), flush=True)


def configure_ultralytics() -> None:
    """Point Ultralytics at TrainKit's folders and turn off analytics."""
    try:
        from ultralytics import settings

        updates = {"sync": False}
        for key, env in (("datasets_dir", "TRAINKIT_DATASETS_DIR"), ("weights_dir", "TRAINKIT_WEIGHTS_DIR"),
                         ("runs_dir", "TRAINKIT_RUNS_DIR")):
            if os.environ.get(env):
                updates[key] = os.environ[env]
        settings.update(updates)
    except Exception as e:  # never block on settings
        print(f"[trainkit] could not update Ultralytics settings: {e}", file=sys.stderr)


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def model_class(name: str | None, weights: str = ""):
    from ultralytics import RTDETR, YOLO

    if (name or "").upper() == "RTDETR" or "rtdetr" in Path(weights).name.lower():
        return RTDETR
    return YOLO


def load_model(path: str, class_name: str | None = None):
    return model_class(class_name, path)(path)


class EventWriter:
    """Append-only JSON-lines event log. Readers tolerate a partial last line."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def emit(self, type_: str, **data) -> None:
        rec = {"t": round(time.time(), 3), "type": type_, **data}
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class ControlReader:
    """Polls control.json written by the CLI or dashboard: {"cmd": "pause|continue|save_stop|stop", "seq": n}."""

    def __init__(self, path: Path):
        self.path = path
        self._mtime = None
        self.last_seq = self._current_seq()

    def _read(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _current_seq(self) -> int:
        data = self._read()
        return int(data.get("seq", 0)) if data else 0

    def poll(self) -> dict | None:
        try:
            mtime = self.path.stat().st_mtime_ns
        except OSError:
            return None
        if mtime == self._mtime:
            return None
        self._mtime = mtime
        data = self._read()
        if not data:
            return None
        seq = int(data.get("seq", 0))
        if seq <= self.last_seq:
            return None
        self.last_seq = seq
        return data
