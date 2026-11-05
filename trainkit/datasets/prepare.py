"""Turn whatever the user has (folder, zip, VOC, COCO, Roboflow export) into something Ultralytics can train on.

Never modifies the user's files: generated YAML, split lists and converted labels go to <trainkit home>/datasets.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable

import yaml

from .. import paths
from ..i18n import tr
from .validate import IMG_EXT, label_path

Log = Callable[[str], None]


class DatasetError(RuntimeError):
    pass


def _noop(_: str) -> None:
    pass


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "dataset"


def _images(folder: Path) -> list[Path]:
    return sorted(f for f in folder.rglob("*") if f.suffix.lower() in IMG_EXT)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_list(items: list[Path], val_ratio: float = 0.2, seed: int = 0) -> tuple[list[Path], list[Path]]:
    items = sorted(items)
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(items) * val_ratio)) if len(items) > 1 else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:]) or val
    return train, val


def read_class_names(folder: Path) -> list[str] | None:
    for name in ("classes.txt", "names.txt", "obj.names", "labels.txt"):
        for p in (folder / name, *folder.glob(f"*/{name}")):
            if p.exists():
                names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if names:
                    return names
    notes = folder / "notes.json"  # Label Studio YOLO export
    if notes.exists():
        try:
            cats = json.loads(notes.read_text(encoding="utf-8")).get("categories", [])
            return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
        except (ValueError, KeyError):
            pass
    return None


def _infer_nc(label_files: list[Path]) -> int:
    top = -1
    for lf in label_files:
        try:
            for line in lf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    top = max(top, int(float(line.split()[0])))
        except (OSError, ValueError):
            continue
    return top + 1


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def extract_zip(zip_path: Path, dest: Path, log: Log = _noop) -> Path:
    log(tr(f"解压 {zip_path.name} …", f"Extracting {zip_path.name} …"))
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(root)):  # zip-slip guard
                raise DatasetError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dest)
    # Unwrap a single top-level folder
    entries = [e for e in dest.iterdir() if not e.name.startswith(("__MACOSX", "."))]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


# ---------------------------------------------------------------- format converters ---------------------------
def convert_voc(folder: Path, out: Path, log: Log = _noop) -> Path:
    """PASCAL VOC (Annotations/*.xml + JPEGImages/) → YOLO detect dataset."""
    ann_dir = next((d for d in (folder / "Annotations", *folder.glob("*/Annotations")) if d.is_dir()), None)
    if not ann_dir:
        raise DatasetError("Annotations/ not found")
    img_dir = next((d for d in (ann_dir.parent / "JPEGImages", ann_dir.parent / "images") if d.is_dir()), ann_dir.parent)
    xmls = sorted(ann_dir.glob("*.xml"))
    log(tr(f"转换 {len(xmls)} 个 VOC 标注…", f"Converting {len(xmls)} VOC annotations…"))
    names: list[str] = []
    samples: list[tuple[Path, list[str]]] = []
    for x in xmls:
        root = ET.parse(x).getroot()
        fname = root.findtext("filename") or (x.stem + ".jpg")
        img = img_dir / fname
        if not img.exists():
            cands = [img_dir / (x.stem + e) for e in IMG_EXT]
            img = next((c for c in cands if c.exists()), img)
        if not img.exists():
            continue
        size = root.find("size")
        w = float(size.findtext("width") or 0) if size is not None else 0
        h = float(size.findtext("height") or 0) if size is not None else 0
        if not (w and h):
            continue
        rows = []
        for obj in root.iter("object"):
            name = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if not name or bb is None:
                continue
            if name not in names:
                names.append(name)
            x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
            rows.append(f"{names.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        samples.append((img, rows))
    return _write_converted(samples, names, out, log)


def convert_coco(json_path: Path, out: Path, task: str = "detect", log: Log = _noop) -> Path:
    """COCO JSON → YOLO detect/segment dataset (polygons only; crowd RLE masks are skipped)."""
    coco = json.loads(json_path.read_text(encoding="utf-8"))
    cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    images = {im["id"]: im for im in coco.get("images", [])}
    per_image: dict[int, list[str]] = {i: [] for i in images}
    for a in coco.get("annotations", []):
        im = images.get(a["image_id"])
        if not im or a.get("iscrowd") or a["category_id"] not in cat_index:
            continue
        w, h = float(im["width"]), float(im["height"])
        c = cat_index[a["category_id"]]
        if task == "segment":
            segs = a.get("segmentation")
            if not isinstance(segs, list) or not segs:
                continue
            poly = max(segs, key=len)
            pts = " ".join(f"{poly[i] / w:.6f} {poly[i + 1] / h:.6f}" for i in range(0, len(poly) - 1, 2))
            per_image[a["image_id"]].append(f"{c} {pts}")
        else:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            per_image[a["image_id"]].append(f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    base = json_path.parent
    img_roots = [base, base / "images", base.parent / "images", base.parent]
    samples = []
    for iid, im in images.items():
        fname = Path(im["file_name"])
        img = next((r / fname for r in img_roots if (r / fname).exists()),
                   next((r / fname.name for r in img_roots if (r / fname.name).exists()), None))
        if img:
            samples.append((img, per_image[iid]))
    log(tr(f"转换 COCO：{len(samples)} 张图，{len(names)} 个类别", f"Converting COCO: {len(samples)} images, {len(names)} classes"))
    return _write_converted(samples, names, out, log)


def _write_converted(samples: list[tuple[Path, list[str]]], names: list[str], out: Path, log: Log) -> Path:
    if not samples:
        raise DatasetError(tr("没有找到可用的图片和标注", "No usable images with annotations found"))
    imgs = [s[0] for s in samples]
    train, val = split_list(imgs)
    rows = {s[0]: s[1] for s in samples}
    for split, items in (("train", train), ("val", val)):
        for img in items:
            dst = out / "images" / split / img.name
            _link_or_copy(img, dst)
            lbl = out / "labels" / split / (img.stem + ".txt")
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(rows[img]) + ("\n" if rows[img] else ""), encoding="utf-8")
    return _write_yaml(out / "data.yaml", {"path": str(out), "train": "images/train", "val": "images/val",
                                           "names": dict(enumerate(names))})


# ---------------------------------------------------------------- folder detection -----------------------------
def prepare_folder(folder: Path, task: str, name: str | None = None, log: Log = _noop) -> str:
    """Return a `data=` value for Ultralytics (data.yaml path, or a folder for classification)."""
    folder = folder.resolve()
    name = safe_name(name or folder.name)
    out = paths.datasets_dir() / "_prepared" / name
    if task == "classify":
        return _prepare_classify(folder, out, log)

    # 1) The folder already has a data.yaml
    for cand in ("data.yaml", "dataset.yaml", "data.yml"):
        for y in (folder / cand, *folder.glob(f"*/{cand}")):
            if y.exists():
                return _fix_yaml_paths(y, out, log)

    # 2) COCO / VOC
    coco = [*folder.glob("annotations/*.json"), *folder.glob("*.json"), *folder.glob("*/_annotations.coco.json")]
    coco = [c for c in coco if c.name != "notes.json"]
    if coco:
        return str(convert_coco(max(coco, key=lambda p: p.stat().st_size), out, task, log))
    if (folder / "Annotations").is_dir() or any(folder.glob("*/Annotations")):
        return str(convert_voc(folder, out, log))

    # 3) Split folders: images/{train,val} or {train,valid,val}/images
    names = read_class_names(folder)
    split_dirs = {}
    for split, cands in (("train", ["images/train", "train/images", "train"]),
                         ("val", ["images/val", "images/valid", "valid/images", "val/images", "valid", "val"]),
                         ("test", ["images/test", "test/images", "test"])):
        for c in cands:
            if (folder / c).is_dir() and _images(folder / c):
                split_dirs[split] = c
                break
    if "train" in split_dirs and "val" in split_dirs:
        lbls = [label_path(i) for i in _images(folder / split_dirs["train"])[:2000]]
        names = names or [f"class{i}" for i in range(_infer_nc([lp for lp in lbls if lp.exists()]))]
        data = {"path": str(folder), **split_dirs, "names": dict(enumerate(names))}
        return str(_write_yaml(out / "data.yaml", data))

    # 4) Flat: all images with labels next to them or in a parallel labels/ folder → random 80/20 split
    imgs = _images(folder)
    if not imgs:
        raise DatasetError(tr(f"{folder} 里没有图片", f"No images in {folder}"))
    labelled = [i for i in imgs if label_path(i).exists()]
    if not labelled:
        raise DatasetError(tr("没有找到 YOLO 格式的 .txt 标注（应与图片同名，放在同一目录或 labels/ 目录）",
                              "No YOLO .txt labels found (same name as the image, next to it or in labels/)"))
    if not names:
        nc = _infer_nc([label_path(i) for i in labelled])
        names = [f"class{i}" for i in range(nc)]
        log(tr(f"没有 classes.txt，自动命名为 class0..class{nc - 1}", f"No classes.txt; naming classes class0..class{nc - 1}"))
    train, val = split_list(imgs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(map(str, train)) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(map(str, val)) + "\n", encoding="utf-8")
    log(tr(f"自动划分：训练 {len(train)} 张 / 验证 {len(val)} 张", f"Auto split: {len(train)} train / {len(val)} val"))
    return str(_write_yaml(out / "data.yaml", {"path": str(out), "train": str(out / "train.txt"),
                                               "val": str(out / "val.txt"), "names": dict(enumerate(names))}))


def _fix_yaml_paths(y: Path, out: Path, log: Log) -> str:
    """Rewrite a data.yaml with an absolute `path` so it works from any working directory.

    Roboflow exports use '../train/images' relative to the yaml; Ultralytics resolves those inconsistently.
    """
    data = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
    base = y.parent
    root = Path(str(data.get("path") or ""))
    root = root if root.is_absolute() else (base / root).resolve()
    fixed = dict(data)
    fixed["path"] = str(root)
    for split in ("train", "val", "test"):
        v = data.get(split)
        if not v:
            continue
        vals = v if isinstance(v, list) else [v]
        resolved = []
        for item in vals:
            p = Path(str(item))
            if p.is_absolute():
                resolved.append(str(p))
                continue
            for cand in (root / p, base / p, base / str(item).lstrip("./").removeprefix("../")):
                if cand.resolve().exists():
                    resolved.append(str(cand.resolve()))
                    break
            else:
                resolved.append(str((root / p).resolve()))
        fixed[split] = resolved if isinstance(v, list) else resolved[0]
    if "val" not in fixed and "valid" in fixed:
        fixed["val"] = fixed.pop("valid")
    return str(_write_yaml(out / "data.yaml", fixed))


def _prepare_classify(folder: Path, out: Path, log: Log) -> str:
    if (folder / "train").is_dir() and ((folder / "val").is_dir() or (folder / "test").is_dir()):
        return str(folder)
    classes = [d for d in sorted(folder.iterdir()) if d.is_dir() and _images(d)]
    if not classes:
        raise DatasetError(tr("分类数据集应为每个类别一个文件夹", "Classification data needs one folder per class"))
    log(tr(f"按类别文件夹自动划分：{len(classes)} 类", f"Splitting class folders: {len(classes)} classes"))
    for c in classes:
        train, val = split_list(_images(c))
        for split, items in (("train", train), ("val", val)):
            for img in items:
                _link_or_copy(img, out / split / c.name / img.name)
    return str(out)


# ---------------------------------------------------------------- Roboflow -------------------------------------
_RF_PATTERNS = [
    re.compile(r"^roboflow:([\w.-]+)/([\w.-]+)/(\d+)$"),
    re.compile(r"^https?://(?:universe|app)\.roboflow\.com/([\w.-]+)/([\w.-]+)/(?:dataset/|model/)?(\d+)"),
]


def parse_roboflow(spec: str) -> tuple[str, str, int] | None:
    for pat in _RF_PATTERNS:
        m = pat.match(spec.strip())
        if m:
            return m.group(1), m.group(2), int(m.group(3))
    return None


ROBOFLOW_FORMATS = {"detect": "yolov8", "segment": "yolov8", "pose": "yolov8", "obb": "yolov8-obb", "classify": "folder"}


def download_roboflow(ws: str, project: str, version: int, task: str, api_key: str, log: Log = _noop) -> Path:
    import httpx

    if not api_key:
        raise DatasetError(tr("需要 Roboflow API key：trainkit config roboflow_api_key <key>（在 roboflow.com 设置页免费获取）",
                              "A Roboflow API key is needed: trainkit config roboflow_api_key <key> (free in your roboflow.com settings)"))
    fmt = ROBOFLOW_FORMATS.get(task, "yolov8")
    dest = paths.datasets_dir() / safe_name(f"rf-{ws}-{project}-{version}")
    if dest.exists() and any(dest.iterdir()):
        log(tr("已下载过，直接使用", "Already downloaded"))
        return _unwrap(dest)
    url = f"https://api.roboflow.com/{ws}/{project}/{version}/{fmt}"
    link = None
    for attempt in range(100):  # Roboflow generates exports on demand
        r = httpx.get(url, params={"api_key": api_key}, timeout=60)
        if r.status_code in (401, 403):
            raise DatasetError(tr("Roboflow 拒绝访问：检查 API key 和数据集权限", "Roboflow refused access: check the API key and dataset permissions"))
        if r.status_code == 404:
            raise DatasetError(tr("Roboflow 找不到这个数据集版本", "Roboflow could not find that dataset version"))
        r.raise_for_status()
        link = (r.json().get("export") or {}).get("link")
        if link:
            break
        if attempt == 0:
            log(tr("Roboflow 正在生成导出包…", "Roboflow is generating the export…"))
        time.sleep(3)
    if not link:
        raise DatasetError("Roboflow export timed out")
    zip_path = dest.with_suffix(".zip")
    log(tr("下载数据集…", "Downloading dataset…"))
    with httpx.stream("GET", link, timeout=None, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
    root = extract_zip(zip_path, dest, log)
    zip_path.unlink(missing_ok=True)
    return root


def _unwrap(dest: Path) -> Path:
    entries = [e for e in dest.iterdir() if not e.name.startswith(".")]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else dest
