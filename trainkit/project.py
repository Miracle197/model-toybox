"""trainkit.yaml: a shareable project file (recipe + dataset + training settings) to commit next to your code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TEMPLATE = """\
# TrainKit project file. Teammates run:
#   trainkit setup trainkit.yaml
#   trainkit train trainkit.yaml
name: {name}

# Model recipe (see `trainkit recipes`)
recipe: {recipe}

# Starting weights: an official name, or a path to your own .pt
weights: {weights}

# Dataset: built-in name, folder, .zip, data.yaml, or roboflow:workspace/project/version
dataset: {dataset}

# Any Ultralytics train() argument. batch: -1 picks the batch size from GPU memory.
train:
  epochs: {epochs}
  imgsz: {imgsz}
  batch: -1
  patience: 50
"""


def is_project_file(target: str | None) -> bool:
    return bool(target) and target.endswith((".yaml", ".yml")) and Path(target).is_file() and \
        "recipe" in (yaml.safe_load(Path(target).read_text(encoding="utf-8")) or {})


def load(path: str) -> dict[str, Any]:
    p = Path(path).resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    ds = data.get("dataset")
    if isinstance(ds, dict):
        ds = ds.get("source")
    # Relative dataset/weights paths are relative to the project file, not the current directory
    for key, val in (("dataset", ds), ("weights", data.get("weights"))):
        if isinstance(val, str) and (val.startswith(".") or "/" in val or "\\" in val) and not val.startswith("roboflow:") \
                and not val.startswith("http"):
            cand = (p.parent / val).resolve()
            if cand.exists():
                val = str(cand)
        data[key] = val
    data.setdefault("train", {})
    data["_path"] = str(p)
    return data


def write_template(path: Path, name: str, recipe: str, weights: str, dataset: str, epochs: int, imgsz: int) -> None:
    path.write_text(TEMPLATE.format(name=name, recipe=recipe, weights=weights, dataset=dataset, epochs=epochs,
                                    imgsz=imgsz), encoding="utf-8")
