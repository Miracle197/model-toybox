"""Find, package and export trained models."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

from . import paths
from .i18n import tr
from .runner import runs

BUNDLE_META = "trainkit-model.json"


class ModelError(RuntimeError):
    pass


def resolve(spec: str) -> Path:
    """A run id, a model bundle (.zip), or a path to .pt/.onnx/OpenVINO folder → local model path."""
    rd = paths.runs_dir() / spec
    if rd.is_dir():
        w = runs.best_weights(rd)
        if not w:
            raise ModelError(tr("这个训练记录还没有权重文件", "This run has no weights yet"))
        return w
    p = Path(spec).expanduser()
    if p.suffix.lower() == ".zip" and p.is_file():
        return unpack(p)
    if p.exists():
        return p.resolve()
    w = paths.weights_dir() / spec
    if w.exists():
        return w
    if spec.endswith(".pt") and "/" not in spec and "\\" not in spec:
        return Path(spec)  # official weights name: downloaded on first use
    raise ModelError(tr(f"找不到模型：{spec}", f"Model not found: {spec}"))


def unpack(bundle: Path) -> Path:
    dest = paths.models_dir() / bundle.stem
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(bundle) as zf:
        for m in zf.infolist():
            if not str((dest / m.filename).resolve()).startswith(str(root)):
                raise ModelError(f"Unsafe path in bundle: {m.filename}")
        zf.extractall(dest)
    meta_file = dest / BUNDLE_META
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return dest / meta.get("model", "model.pt")
    pts = sorted(dest.rglob("*.pt")) or sorted(dest.rglob("*.onnx"))
    if not pts:
        raise ModelError(tr("压缩包里没有模型文件", "No model file in the bundle"))
    return pts[0]


def pack(run_id: str, out: Path | None = None, include: list[Path] | None = None) -> Path:
    rd = runs.run_dir(run_id)
    meta = runs.read_meta(rd)
    weights = runs.best_weights(rd)
    if not weights:
        raise ModelError(tr("这个训练记录还没有权重文件", "This run has no weights yet"))
    s, _ = runs.summarize(rd)
    info: dict[str, Any] = {
        "format": "trainkit-model/1",
        "model": "model.pt",
        "name": meta.get("name", run_id),
        "recipe": meta.get("recipe"),
        "task": meta.get("task"),
        "model_class": meta.get("model_class", "YOLO"),
        "names": s.names,
        "imgsz": meta.get("overrides", {}).get("imgsz"),
        "epochs_trained": s.epoch,
        "metrics": s.final or s.metrics,
        "dataset": meta.get("data_label"),
        "packed": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = out or Path.cwd() / f"{meta.get('name', run_id)}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(weights, "model.pt")
        zf.writestr(BUNDLE_META, json.dumps(info, indent=2, ensure_ascii=False))
        for extra in include or []:
            if extra.is_file():
                zf.write(extra, extra.name)
            elif extra.is_dir():
                for f in extra.rglob("*"):
                    if f.is_file():
                        zf.write(f, str(Path(extra.name) / f.relative_to(extra)))
        for plot in ("results.png", "confusion_matrix.png"):
            p = rd / "train" / plot
            if p.exists():
                zf.write(p, f"plots/{plot}")
    return out


def model_class_for(path: Path) -> str:
    return "RTDETR" if "rtdetr" in path.name.lower() else "YOLO"
