"""Dataset registry and the single entry point `resolve()` that turns any --data value into a trainable path."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import paths
from ..config import load_config
from ..i18n import tr
from .prepare import (DatasetError, download_roboflow, extract_zip, parse_roboflow, prepare_folder, safe_name)
from .validate import Report, validate_classify, validate_yolo

CLASSIFY_BUILTINS = {"imagenet10", "imagenette", "imagenette160", "imagenette320", "imagewoof", "imagewoof160",
                     "imagewoof320", "mnist", "mnist160", "cifar10", "cifar100", "caltech101", "caltech256",
                     "fashion-mnist", "imagenet100", "imagenet1000"}

__all__ = ["Resolved", "resolve", "register", "registry", "check", "DatasetError"]


@dataclass
class Resolved:
    data: str  # value passed to Ultralytics `data=`
    kind: str  # builtin | yaml | folder | zip | roboflow | registry
    label: str  # short name for display
    local: bool  # True when TrainKit can validate it before training


def _registry_file() -> Path:
    return paths.datasets_dir() / "registry.json"


def registry() -> dict[str, dict]:
    f = _registry_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def register(name: str, data: str, task: str, source: str) -> None:
    reg = registry()
    reg[name] = {"data": data, "task": task, "source": source, "added": time.strftime("%Y-%m-%d %H:%M")}
    _registry_file().parent.mkdir(parents=True, exist_ok=True)
    _registry_file().write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def unregister(name: str) -> bool:
    reg = registry()
    if name not in reg:
        return False
    del reg[name]
    _registry_file().write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def is_builtin(spec: str, task: str) -> bool:
    if Path(spec).expanduser().exists() or "/" in spec or "\\" in spec:
        return False
    if task == "classify":
        return spec in CLASSIFY_BUILTINS
    return spec.endswith((".yaml", ".yml"))


def resolve(spec: str, task: str, log: Callable[[str], None] = lambda _: None, name: str | None = None) -> Resolved:
    spec = spec.strip()
    reg = registry()
    if spec in reg:
        entry = reg[spec]
        return Resolved(entry["data"], "registry", spec, Path(entry["data"]).exists())
    if is_builtin(spec, task):
        return Resolved(spec, "builtin", spec, False)
    rf = parse_roboflow(spec)
    if rf:
        ws, project, version = rf
        folder = download_roboflow(ws, project, version, task, load_config().get("roboflow_api_key", ""), log)
        data = prepare_folder(folder, task, name or f"rf-{project}-{version}", log)
        return Resolved(data, "roboflow", f"{project} v{version}", True)
    p = Path(spec).expanduser()
    if not p.exists():
        raise DatasetError(tr(f"找不到数据集：{spec}", f"Dataset not found: {spec}"))
    p = p.resolve()
    if p.suffix.lower() == ".zip":
        folder = extract_zip(p, paths.datasets_dir() / safe_name(name or p.stem), log)
        return Resolved(prepare_folder(folder, task, name or p.stem, log), "zip", p.stem, True)
    if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
        return Resolved(prepare_folder_from_yaml(p, task, name, log), "yaml", p.stem, True)
    if p.is_dir():
        return Resolved(prepare_folder(p, task, name, log), "folder", p.name, True)
    raise DatasetError(tr(f"不支持的数据集：{spec}", f"Unsupported dataset: {spec}"))


def prepare_folder_from_yaml(p: Path, task: str, name: str | None, log) -> str:
    from .prepare import _fix_yaml_paths

    out = paths.datasets_dir() / "_prepared" / safe_name(name or p.parent.name)
    return _fix_yaml_paths(p, out, log)


def check(resolved: Resolved, task: str) -> Report | None:
    """Validate a local dataset. Built-in datasets are downloaded and checked by Ultralytics instead."""
    if not resolved.local:
        return None
    path = Path(resolved.data)
    if task == "classify":
        return validate_classify(path)
    return validate_yolo(path, task)
