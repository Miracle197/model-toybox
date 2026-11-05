"""Recipes describe one trainable model family: dependencies, hardware needs, weights and datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..i18n import lang

RECIPES_DIR = Path(__file__).resolve().parent

# Environments are shared by recipes with the same dependencies.
ENVS: dict[str, dict[str, Any]] = {
    "ultralytics": {
        "python": "3.12",
        "packages": ["torch", "torchvision", "ultralytics", "gradio", "onnx", "onnxslim", "onnxruntime"],
        "constraints": "constraints/ultralytics.txt",
    }
}


def localized(value: Any) -> str:
    if isinstance(value, dict):
        return value.get(lang()) or value.get("en") or next(iter(value.values()), "")
    return str(value or "")


@dataclass
class Recipe:
    id: str
    name: Any
    description: Any
    family: str
    model_class: str
    task: str
    env: str
    weights: dict[str, Any]
    dataset: dict[str, Any]
    requirements: dict[str, dict[str, float]]
    defaults: dict[str, Any] = field(default_factory=dict)
    order: int = 100

    @property
    def title(self) -> str:
        return localized(self.name)

    @property
    def summary(self) -> str:
        return localized(self.description)

    @property
    def default_weights(self) -> str:
        return self.weights.get("default", "")

    @property
    def smoke_dataset(self) -> str:
        return self.dataset.get("smoke", "")

    @property
    def builtin_datasets(self) -> list[dict[str, Any]]:
        return list(self.dataset.get("builtin", []))

    def needs(self, profile: str) -> dict[str, float]:
        return dict(self.requirements.get(profile, {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "task": self.task,
            "model_class": self.model_class,
            "env": self.env,
            "weights": self.weights,
            "default_weights": self.default_weights,
            "i18n": {"title": self.name, "summary": self.description},
            "builtin_datasets": [
                {"name": d["name"], "note": localized(d.get("note", "")), "i18n_note": d.get("note", "")}
                for d in self.builtin_datasets
            ],
            "requirements": self.requirements,
            "defaults": self.defaults,
        }


def _load(path: Path) -> Recipe:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Recipe(**data)


def all_recipes() -> list[Recipe]:
    recipes = [_load(p) for p in RECIPES_DIR.glob("*.yaml")]
    return sorted(recipes, key=lambda r: (r.order, r.id))


def get_recipe(recipe_id: str) -> Recipe:
    for r in all_recipes():
        if r.id == recipe_id:
            return r
    known = ", ".join(r.id for r in all_recipes())
    raise KeyError(f"Unknown recipe '{recipe_id}'. Known: {known}")


def recipe_for_task(task: str) -> Recipe:
    """Best recipe for a task name found in a model file (detect/segment/pose/classify/obb)."""
    for r in all_recipes():
        if r.task == task and r.model_class == "YOLO":
            return r
    return get_recipe("yolo-detect")


def env_spec(name: str) -> dict[str, Any]:
    spec = dict(ENVS[name])
    spec["constraints_path"] = RECIPES_DIR / spec["constraints"]
    return spec
