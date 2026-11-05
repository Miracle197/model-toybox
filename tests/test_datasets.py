import json
from pathlib import Path

import pytest
import yaml

from trainkit import datasets as ds
from trainkit.datasets.prepare import convert_coco, parse_roboflow, prepare_folder, split_list
from trainkit.datasets.validate import label_path, validate_classify, validate_yolo

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # validation never decodes images


def make_flat(root: Path, n=10, bad=False):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i in range(n):
        (root / "images" / f"im{i}.jpg").write_bytes(PNG)
        line = "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"
        if bad and i == 0:
            line = "5 1.7 0.5 0.2\n"
        (root / "labels" / f"im{i}.txt").write_text(line)
    (root / "classes.txt").write_text("cone\nbox\n")


def test_split_is_deterministic():
    items = [Path(f"{i}.jpg") for i in range(10)]
    a, b = split_list(items), split_list(items)
    assert a == b and len(a[1]) == 2 and not set(a[0]) & set(a[1])


def test_flat_folder_is_split_and_valid(tmp_path):
    make_flat(tmp_path / "obstacles")
    data = Path(prepare_folder(tmp_path / "obstacles", "detect"))
    y = yaml.safe_load(data.read_text())
    assert y["names"] == {0: "cone", 1: "box"}
    rep = validate_yolo(data, "detect")
    assert rep.ok, rep.errors
    assert rep.stats["train_images"] == 8 and rep.stats["val_images"] == 2
    assert rep.stats["instances"] == {"cone": 10, "box": 10}


def test_bad_labels_are_reported(tmp_path):
    make_flat(tmp_path / "bad", bad=True)
    rep = validate_yolo(Path(prepare_folder(tmp_path / "bad", "detect")), "detect")
    assert not rep.ok
    text = " ".join(rep.errors)
    assert "outside" in text and "normalized" in text


def test_label_path_rule(tmp_path):
    img = tmp_path / "set" / "images" / "train" / "a.png"
    assert label_path(img) == tmp_path / "set" / "labels" / "train" / "a.txt"
    assert label_path(tmp_path / "flat" / "b.jpg") == tmp_path / "flat" / "b.txt"


def test_voc_conversion(tmp_path):
    root = tmp_path / "voc"
    (root / "Annotations").mkdir(parents=True)
    (root / "JPEGImages").mkdir()
    for i in range(4):
        (root / "JPEGImages" / f"{i}.jpg").write_bytes(PNG)
        (root / "Annotations" / f"{i}.xml").write_text(
            f"<annotation><filename>{i}.jpg</filename><size><width>100</width><height>50</height></size>"
            "<object><name>chair</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>"
            "</annotation>")
    data = Path(prepare_folder(root, "detect"))
    rep = validate_yolo(data, "detect")
    assert rep.ok, rep.errors
    assert rep.stats["instances"] == {"chair": 4}
    lbl = next((data.parent / "labels").rglob("*.txt")).read_text().split()
    assert lbl[0] == "0" and abs(float(lbl[1]) - 0.2) < 1e-6 and abs(float(lbl[2]) - 0.5) < 1e-6


def test_coco_conversion_segment(tmp_path):
    root = tmp_path / "coco"
    (root / "images").mkdir(parents=True)
    for i in range(3):
        (root / "images" / f"{i}.jpg").write_bytes(PNG)
    coco = {"images": [{"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 100} for i in range(3)],
            "categories": [{"id": 7, "name": "person"}],
            "annotations": [{"id": i, "image_id": i, "category_id": 7, "bbox": [20, 10, 40, 50], "iscrowd": 0,
                             "segmentation": [[20, 10, 60, 10, 60, 60, 20, 60]]} for i in range(3)]}
    (root / "annotations").mkdir()
    (root / "annotations" / "instances.json").write_text(json.dumps(coco))
    data = convert_coco(root / "annotations" / "instances.json", tmp_path / "out", "segment")
    rep = validate_yolo(data, "segment")
    assert rep.ok, rep.errors
    assert rep.stats["instances"] == {"person": 3}


def test_classify_folders(tmp_path):
    for c in ("clear", "blocked"):
        (tmp_path / "cls" / c).mkdir(parents=True)
        for i in range(5):
            (tmp_path / "cls" / c / f"{i}.jpg").write_bytes(PNG)
    root = Path(prepare_folder(tmp_path / "cls", "classify"))
    rep = validate_classify(root)
    assert rep.ok, rep.errors
    assert rep.stats["train_images"] == 8 and rep.stats["val_images"] == 2


def test_roboflow_parsing():
    assert parse_roboflow("roboflow:me/cones/3") == ("me", "cones", 3)
    assert parse_roboflow("https://universe.roboflow.com/me/cones/dataset/4") == ("me", "cones", 4)
    assert parse_roboflow("coco8.yaml") is None


def test_resolve_and_registry(tmp_path):
    assert ds.resolve("coco8.yaml", "detect").kind == "builtin"
    assert ds.resolve("imagenet10", "classify").kind == "builtin"
    make_flat(tmp_path / "mine")
    r = ds.resolve(str(tmp_path / "mine"), "detect")
    ds.register("mine", r.data, "detect", str(tmp_path / "mine"))
    assert ds.resolve("mine", "detect").kind == "registry"
    with pytest.raises(ds.DatasetError):
        ds.resolve(str(tmp_path / "missing"), "detect")
