"""Check a YOLO-format dataset before training, so problems show up in seconds instead of mid-epoch."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..i18n import tr

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dng", ".mpo", ".pfm"}


@dataclass
class Report:
    ok: bool = True
    task: str = "detect"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "task": self.task, "errors": self.errors, "warnings": self.warnings,
                "stats": self.stats}


def load_data_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, list):
        data["names"] = dict(enumerate(names))
    elif isinstance(names, dict):
        data["names"] = {int(k): v for k, v in names.items()}
    elif "nc" in data:
        data["names"] = {i: f"class{i}" for i in range(int(data["nc"]))}
    return data


def dataset_root(yaml_path: Path, data: dict[str, Any]) -> Path:
    root = Path(str(data.get("path", "") or ""))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def images_in(entry: Any, root: Path) -> list[Path]:
    """Resolve a train/val entry (dir, txt list, or list of those) to image paths."""
    if entry is None:
        return []
    if isinstance(entry, (list, tuple)):
        out: list[Path] = []
        for e in entry:
            out += images_in(e, root)
        return out
    p = Path(str(entry))
    if not p.is_absolute():
        p = (root / p).resolve()
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
    if p.suffix == ".txt" and p.exists():
        files = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            f = Path(line)
            if not f.is_absolute():
                f = (p.parent / line).resolve() if line.startswith(".") else (root / line).resolve()
            files.append(f)
        return files
    return []


def label_path(img: Path) -> Path:
    """Same rule as Ultralytics: last '/images/' becomes '/labels/', extension becomes .txt."""
    s = str(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa in s:
        head, tail = s.rsplit(sa, 1)
        s = head + sb + tail
    return Path(s).with_suffix(".txt")


def _check_label_file(path: Path, task: str, nc: int, kpt: tuple[int, int] | None, counts: Counter,
                      problems: list[str]) -> int:
    n = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        problems.append(f"{path.name}: {e}")
        return 0
    for ln, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            problems.append(f"{path.name}:{ln} " + tr("不是数字", "not numeric"))
            continue
        if cls < 0 or (nc and cls >= nc):
            problems.append(f"{path.name}:{ln} " + tr(f"类别 {cls} 超出范围 0..{nc - 1}", f"class {cls} outside 0..{nc - 1}"))
        if task == "pose" and kpt:
            want = 4 + kpt[0] * kpt[1]
            if len(vals) != want:
                problems.append(f"{path.name}:{ln} " + tr(f"应有 {want + 1} 个数，实际 {len(vals) + 1}", f"expected {want + 1} values, got {len(vals) + 1}"))
            coords = vals[:4] + [v for i, v in enumerate(vals[4:]) if kpt[1] != 3 or i % 3 != 2]
        elif task == "obb":
            if len(vals) != 8:
                problems.append(f"{path.name}:{ln} " + tr("旋转框应有 9 个数", "OBB rows need 9 values"))
            coords = vals
        else:
            if len(vals) < 4 or (len(vals) > 4 and len(vals) % 2):
                problems.append(f"{path.name}:{ln} " + tr("坐标数量不对", "wrong number of coordinates"))
            if task == "segment" and len(vals) == 4:
                problems.append(f"{path.name}:{ln} " + tr("分割任务需要多边形，这里只有方框", "segmentation needs polygons, found a box"))
            coords = vals
        if any(v < -0.01 or v > 1.01 for v in coords):
            problems.append(f"{path.name}:{ln} " + tr("坐标未归一化到 0–1", "coordinates not normalized to 0–1"))
        counts[cls] += 1
        n += 1
    return n


def validate_yolo(yaml_path: Path, task: str = "detect", max_problems: int = 12) -> Report:
    rep = Report(task=task)
    if not yaml_path.exists():
        rep.error(tr(f"找不到 {yaml_path}", f"{yaml_path} not found"))
        return rep
    try:
        data = load_data_yaml(yaml_path)
    except yaml.YAMLError as e:
        rep.error(tr(f"data.yaml 格式错误：{e}", f"data.yaml is not valid YAML: {e}"))
        return rep
    names = data.get("names") or {}
    nc = len(names)
    if not nc:
        rep.error(tr("data.yaml 缺少 names（类别名）", "data.yaml is missing names"))
    kpt = tuple(data["kpt_shape"]) if task == "pose" and data.get("kpt_shape") else None
    if task == "pose" and not kpt:
        rep.error(tr("姿态数据集需要 kpt_shape", "Pose datasets need kpt_shape"))
    root = dataset_root(yaml_path, data)
    counts: Counter = Counter()
    problems: list[str] = []
    for split in ("train", "val"):
        imgs = images_in(data.get(split), root)
        missing = empty = 0
        for img in imgs:
            lp = label_path(img)
            if not lp.exists():
                missing += 1
                continue
            if _check_label_file(lp, task, nc, kpt, counts, problems) == 0:
                empty += 1
        rep.stats[f"{split}_images"] = len(imgs)
        rep.stats[f"{split}_no_label"] = missing
        rep.stats[f"{split}_empty"] = empty
        if not imgs:
            rep.error(tr(f"{split} 集没有图片（检查 data.yaml 里的路径）", f"No {split} images (check paths in data.yaml)"))
        elif missing + empty == len(imgs):
            rep.error(tr(f"{split} 集的图片全部没有标注", f"No {split} image has labels"))
        elif (missing + empty) / len(imgs) > 0.5:
            rep.warnings.append(tr(f"{split} 集有 {missing + empty}/{len(imgs)} 张图没有标注（会被当作背景）",
                                   f"{missing + empty}/{len(imgs)} {split} images have no labels (treated as background)"))
    rep.stats["instances"] = {names.get(k, str(k)): v for k, v in sorted(counts.items())}
    rep.stats["nc"] = nc
    if problems:
        rep.error(tr(f"发现 {len(problems)} 处标注问题", f"Found {len(problems)} label problems"))
        rep.errors += problems[:max_problems]
    if rep.stats.get("train_images", 0) and rep.stats["train_images"] < 50:
        rep.warnings.append(tr("训练图片少于 50 张，效果可能较差", "Fewer than 50 training images; results may be poor"))
    unused = [names[i] for i in names if counts.get(i, 0) == 0]
    if unused and nc:
        rep.warnings.append(tr("这些类别没有任何标注：", "These classes have no labels: ") + ", ".join(map(str, unused[:10])))
    return rep


def validate_classify(root: Path) -> Report:
    rep = Report(task="classify")
    if not root.is_dir():
        rep.error(tr(f"找不到文件夹 {root}", f"Folder {root} not found"))
        return rep
    train = root / "train"
    val = root / "val" if (root / "val").is_dir() else root / "test"
    if not train.is_dir():
        rep.error(tr("分类数据集需要 train/ 文件夹（每个类别一个子文件夹）", "Classification datasets need train/ with one sub-folder per class"))
        return rep
    if not val.is_dir():
        rep.error(tr("缺少 val/ 或 test/ 文件夹", "Missing val/ or test/ folder"))
    for split, d in (("train", train), ("val", val)):
        if not d.is_dir():
            continue
        per_class = {c.name: sum(1 for f in c.rglob("*") if f.suffix.lower() in IMG_EXT)
                     for c in sorted(d.iterdir()) if c.is_dir()}
        rep.stats[f"{split}_classes"] = per_class
        rep.stats[f"{split}_images"] = sum(per_class.values())
        empty = [c for c, n in per_class.items() if n == 0]
        if empty:
            rep.warnings.append(tr(f"{split} 中这些类别没有图片：", f"Empty classes in {split}: ") + ", ".join(empty))
    tr_classes = set(rep.stats.get("train_classes", {}))
    va_classes = set(rep.stats.get("val_classes", {}))
    if va_classes and tr_classes != va_classes:
        rep.error(tr("train 和 val 的类别文件夹不一致", "train and val have different class folders"))
    rep.stats["nc"] = len(tr_classes)
    return rep
