"""TrainKit playground: try any YOLO model live. Runs inside a recipe environment (needs ultralytics + gradio).

Independent of the training part: point it at any .pt / .onnx / OpenVINO model, from TrainKit or elsewhere.
Usage: python app.py [--model PATH] [--source 0|rtsp://...|video.mp4] [--port 7721] [--host 127.0.0.1] [--open]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

HOME = Path(os.environ.get("TRAINKIT_HOME", Path.home() / ".trainkit"))
RUNS = Path(os.environ.get("TRAINKIT_RUNS_DIR", HOME / "runs"))
WEIGHTS = Path(os.environ.get("TRAINKIT_WEIGHTS_DIR", HOME / "weights"))
MODELS = HOME / "models"
RECORDINGS = HOME / "recordings"
NONE = "—"
DEFAULT_ZONE = "0.25,1.0; 0.40,0.55; 0.60,0.55; 0.75,1.0"  # trapezoid in front of a forward-facing robot camera

TEXT = {
    "zh": {
        "title": "TrainKit 试玩区", "subtitle": "实时检查模型效果：摄像头、视频、图片。支持任何来源的 YOLO 模型。",
        "model": "模型 A", "model_b": "模型 B（可选，并排对比）", "upload": "上传模型文件（.pt / .onnx）",
        "conf": "置信度阈值", "iou": "IoU 阈值（NMS）", "imgsz": "输入尺寸", "classes": "只显示这些类别（留空 = 全部）",
        "zone": "危险区报警", "zone_pts": "危险区多边形（0–1 归一化坐标 x,y; …）", "faster": "转为 ONNX 提速（CPU 推荐）",
        "tab_cam": "浏览器摄像头", "tab_local": "本机摄像头 / RTSP", "tab_video": "视频文件", "tab_image": "图片",
        "source": "摄像头编号或流地址", "source_hint": "0 = 第一个 USB 摄像头；也可以填 rtsp://… 或 http://… 视频流",
        "start": "开始", "stop": "停止", "scan": "检测本机摄像头", "record": "录制标注后的视频",
        "video": "上传视频", "image": "上传图片", "result": "结果", "stats": "统计", "download": "下载录制",
        "no_model": "请先选择模型", "alert": "⚠️ 危险区内有障碍物", "clear": "✅ 危险区畅通", "found": "检测到",
        "latency": "延迟", "fps": "帧率", "objects": "目标数", "info": "模型信息", "converted": "已转换并切换到",
        "cams_found": "找到摄像头", "cams_none": "没有找到本机摄像头", "cannot_open": "无法打开视频源",
        "task": "任务", "nclass": "类别数", "format": "格式", "device": "设备",
    },
    "en": {
        "title": "TrainKit Playground", "subtitle": "Check a model live on camera, video or images. Works with YOLO models from anywhere.",
        "model": "Model A", "model_b": "Model B (optional, side-by-side)", "upload": "Upload a model file (.pt / .onnx)",
        "conf": "Confidence threshold", "iou": "IoU threshold (NMS)", "imgsz": "Input size", "classes": "Only show these classes (empty = all)",
        "zone": "Danger-zone alert", "zone_pts": "Danger-zone polygon (normalized x,y; …)", "faster": "Convert to ONNX for speed (CPU)",
        "tab_cam": "Browser camera", "tab_local": "Local camera / RTSP", "tab_video": "Video file", "tab_image": "Image",
        "source": "Camera index or stream URL", "source_hint": "0 = first USB camera; or rtsp://… / http://… streams",
        "start": "Start", "stop": "Stop", "scan": "Find local cameras", "record": "Record the annotated video",
        "video": "Upload a video", "image": "Upload an image", "result": "Result", "stats": "Stats", "download": "Download recording",
        "no_model": "Pick a model first", "alert": "⚠️ Obstacle in the danger zone", "clear": "✅ Danger zone clear", "found": "Found",
        "latency": "Latency", "fps": "FPS", "objects": "Objects", "info": "Model info", "converted": "Converted and switched to",
        "cams_found": "Cameras found", "cams_none": "No local cameras found", "cannot_open": "Cannot open video source",
        "task": "Task", "nclass": "Classes", "format": "Format", "device": "Device",
    },
}
L = TEXT["en"]


# ---------------------------------------------------------------------------------------------- models -------
def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DEVICE = "cpu"


def discover_models() -> list[tuple[str, str]]:
    """(label, path) for trained runs, downloaded weights and unpacked bundles."""
    found: list[tuple[str, str]] = []
    if RUNS.is_dir():
        for rd in sorted(RUNS.iterdir(), reverse=True):
            best = rd / "train" / "weights" / "best.pt"
            if not best.exists():
                continue
            name = rd.name
            try:
                name = json.loads((rd / "run.json").read_text(encoding="utf-8")).get("name", rd.name)
            except (OSError, ValueError):
                pass
            found.append((f"🏋 {name}", str(best)))
    for root in (MODELS, WEIGHTS):
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if f.suffix in (".pt", ".onnx", ".engine") or (f.is_dir() and f.name.endswith("_openvino_model")):
                if "smoke" in str(f):
                    continue
                found.append((("📦 " if root == MODELS else "⬇ ") + f.name, str(f)))
    seen, out = set(), []
    for label, path in found:
        if path not in seen:
            seen.add(path)
            out.append((label, path))
    return out


@lru_cache(maxsize=4)
def load_model(path: str):
    from ultralytics import RTDETR, YOLO

    cls = RTDETR if "rtdetr" in Path(path).name.lower() else YOLO
    task = None
    if path.endswith(".onnx") or path.endswith("_openvino_model"):
        task = None  # read from export metadata
    return cls(path, task=task) if task else cls(path)


def model_names(path: str) -> list[str]:
    if not path or path == NONE:
        return []
    try:
        names = load_model(path).names
        return list(names.values()) if isinstance(names, dict) else list(names)
    except Exception:
        return []


def model_info(path: str) -> str:
    if not path or path == NONE:
        return L["no_model"]
    try:
        m = load_model(path)
        names = model_names(path)
        fmt = "OpenVINO" if path.endswith("_openvino_model") else Path(path).suffix.lstrip(".").upper()
        preview = ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
        return (f"**{Path(path).name}**  \n{L['task']}: `{m.task}` · {L['nclass']}: {len(names)} · {L['format']}: {fmt} · "
                f"{L['device']}: `{DEVICE}`  \n{preview}")
    except Exception as e:
        return f"❌ {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------------------------- inference ----
def parse_zone(text: str) -> np.ndarray | None:
    try:
        pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";") if p.strip()]
        if len(pts) >= 3 and all(len(p) == 2 for p in pts):
            return np.array(pts, dtype=np.float32)
    except ValueError:
        pass
    return None


def draw_zone(img: np.ndarray, zone: np.ndarray, boxes: list, names: dict) -> tuple[np.ndarray, list[str]]:
    """Boxes whose bottom-center (contact point with the floor) is inside the zone raise the alert."""
    h, w = img.shape[:2]
    poly = (zone * [w, h]).astype(np.int32)
    hits = []
    for x1, _y1, x2, y2, cls in boxes:
        foot = (float((x1 + x2) / 2), float(y2))
        if cv2.pointPolygonTest(poly, foot, False) >= 0:
            hits.append(names.get(int(cls), str(cls)))
    color = (220, 50, 50) if hits else (40, 170, 70)  # RGB
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color)
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
    return img, hits


def run_model(path: str, rgb: np.ndarray, conf: float, iou: float, imgsz: int, classes: list[str]):
    model = load_model(path)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    idx = [k for k, v in names.items() if v in (classes or [])] or None
    t0 = time.perf_counter()
    r = model.predict(rgb[..., ::-1], conf=conf, iou=iou, imgsz=int(imgsz), classes=idx, device=DEVICE, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    out = r.plot()[..., ::-1].copy()  # BGR → RGB
    boxes, counts = [], Counter()
    if getattr(r, "boxes", None) is not None and r.boxes is not None and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        boxes = [(*b, c) for b, c in zip(xyxy, cls)]
        counts = Counter(names.get(int(c), str(c)) for c in cls)
    elif getattr(r, "probs", None) is not None:
        top = int(r.probs.top1)
        counts = Counter({f"{names.get(top, top)} ({float(r.probs.top1conf):.2f})": 1})
    return out, boxes, counts, ms, names


def label_strip(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 12 + 11 * len(text)), 30), (20, 20, 20), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class Meter:
    def __init__(self) -> None:
        self.times: deque[float] = deque(maxlen=30)

    def tick(self) -> float:
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


def process(rgb, model_a, model_b, conf, iou, imgsz, classes, zone_on, zone_text, meter: Meter | None = None):
    if rgb is None:
        return None, ""
    if not model_a or model_a == NONE:
        return rgb, L["no_model"]
    zone = parse_zone(zone_text) if zone_on else None
    panels, lines, all_hits = [], [], []
    for tag, path in (("A", model_a), ("B", model_b)):
        if tag == "B" and (not path or path == NONE):
            continue
        try:
            out, boxes, counts, ms, names = run_model(path, rgb, conf, iou, imgsz, classes)
        except Exception as e:
            return rgb, f"❌ {Path(path).name}: {type(e).__name__}: {e}"
        if zone is not None:
            out, hits = draw_zone(out, zone, boxes, names)
            all_hits += hits
        if model_b and model_b != NONE:
            out = label_strip(out, f"{tag}: {Path(path).name}")
        panels.append(out)
        objs = ", ".join(f"{k} ×{v}" for k, v in counts.most_common(6)) or "—"
        lines.append(f"**{tag}** `{Path(path).name}` · {L['latency']} **{ms:.0f} ms** · {L['objects']} {sum(counts.values())}: {objs}")
    img = panels[0] if len(panels) == 1 else np.hstack([p if p.shape == panels[0].shape else
                                                        cv2.resize(p, panels[0].shape[1::-1]) for p in panels])
    if meter:
        lines.insert(0, f"{L['fps']}: **{meter.tick():.1f}**")
    if zone is not None:
        lines.insert(0, f"### {L['alert']}: {', '.join(sorted(set(all_hits)))}" if all_hits else f"### {L['clear']}")
    return img, "  \n".join(lines)


# ---------------------------------------------------------------------------------------------- sources ------
def open_source(src: str):
    src = (src or "0").strip()
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    return cap if cap.isOpened() else None


def scan_cameras(max_index: int = 6) -> str:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                found.append(f"{i} ({frame.shape[1]}×{frame.shape[0]})")
        cap.release()
    return f"{L['cams_found']}: {', '.join(found)}" if found else L["cams_none"]


def stream_source(src, record, model_a, model_b, conf, iou, imgsz, classes, zone_on, zone_text):
    cap = open_source(src)
    if cap is None:
        yield None, f"❌ {L['cannot_open']}: {src}", None
        return
    meter = Meter()
    writer, out_path = None, None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    is_file = not str(src).strip().isdigit() and Path(str(src)).exists()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img, stats = process(frame[..., ::-1], model_a, model_b, conf, iou, imgsz, classes, zone_on, zone_text, meter)
            if record and img is not None:
                if writer is None:
                    RECORDINGS.mkdir(parents=True, exist_ok=True)
                    out_path = RECORDINGS / f"trainkit-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, img.shape[1::-1])
                writer.write(img[..., ::-1])
            yield img, stats, None
            if not is_file:
                time.sleep(0.001)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if out_path:
        yield gr.skip(), gr.skip(), str(out_path)


def run_video(video, record, *settings):
    if not video:
        yield None, "", None
        return
    yield from stream_source(video, record, *settings)


def convert_onnx(path: str, imgsz: int):
    if not path or path == NONE:
        return gr.skip(), L["no_model"]
    if not path.endswith(".pt"):
        return gr.skip(), model_info(path)
    out = load_model(path).export(format="onnx", imgsz=int(imgsz))
    choices = discover_models()
    if not any(p == out for _, p in choices):
        choices.append((f"⚡ {Path(out).name}", out))
    return gr.update(choices=choices + [(NONE, NONE)], value=out), f"{L['converted']} `{Path(out).name}`\n\n" + model_info(out)


# ---------------------------------------------------------------------------------------------- UI -----------
CSS = """
.gradio-container { max-width: 1400px !important; }
#stats { min-height: 72px; }
footer { display: none !important; }
"""


def build(default_model: str | None, default_source: str | None) -> gr.Blocks:
    choices = discover_models()
    if default_model and not any(p == default_model for _, p in choices):
        choices.insert(0, (f"★ {Path(default_model).name}", default_model))
    value_a = default_model or (choices[0][1] if choices else None)
    with gr.Blocks(title=L["title"]) as demo:
        gr.Markdown(f"## {L['title']}\n{L['subtitle']}")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                model_a = gr.Dropdown(choices=choices, value=value_a, label=L["model"], allow_custom_value=True)
                model_b = gr.Dropdown(choices=choices + [(NONE, NONE)], value=NONE, label=L["model_b"], allow_custom_value=True)
                upload = gr.File(label=L["upload"], file_types=[".pt", ".onnx"], type="filepath")
                info = gr.Markdown(model_info(value_a) if value_a else L["no_model"])
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label=L["conf"])
                iou = gr.Slider(0.1, 0.9, value=0.7, step=0.05, label=L["iou"])
                imgsz = gr.Dropdown([320, 416, 480, 640, 960, 1280], value=640, label=L["imgsz"])
                classes = gr.Dropdown(choices=model_names(value_a) if value_a else [], multiselect=True, label=L["classes"])
                zone_on = gr.Checkbox(value=False, label=L["zone"])
                zone_text = gr.Textbox(value=DEFAULT_ZONE, label=L["zone_pts"])
                faster = gr.Button(L["faster"], size="sm")
            with gr.Column(scale=3):
                settings = [model_a, model_b, conf, iou, imgsz, classes, zone_on, zone_text]
                with gr.Tabs(selected="local" if default_source else "cam"):
                    with gr.Tab(L["tab_cam"], id="cam"):
                        with gr.Row():
                            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label=L["tab_cam"],
                                           webcam_options=gr.WebcamOptions(mirror=False))
                            cam_out = gr.Image(type="numpy", label=L["result"])
                        cam_stats = gr.Markdown(elem_id="stats")
                        cam_meter = Meter()
                        cam.stream(lambda f, *s: process(f, *s, meter=cam_meter), inputs=[cam, *settings],
                                   outputs=[cam_out, cam_stats], stream_every=0.1, concurrency_limit=4)
                    with gr.Tab(L["tab_local"], id="local"):
                        with gr.Row():
                            src = gr.Textbox(value=default_source or "0", label=L["source"], info=L["source_hint"], scale=3)
                            rec = gr.Checkbox(value=False, label=L["record"], scale=1)
                        with gr.Row():
                            start = gr.Button(L["start"], variant="primary")
                            stop = gr.Button(L["stop"])
                            scan = gr.Button(L["scan"])
                        local_out = gr.Image(type="numpy", label=L["result"])
                        local_stats = gr.Markdown(elem_id="stats")
                        local_file = gr.File(label=L["download"])
                        ev = start.click(stream_source, inputs=[src, rec, *settings], outputs=[local_out, local_stats, local_file])
                        stop.click(None, cancels=[ev])
                        scan.click(scan_cameras, outputs=local_stats)
                    with gr.Tab(L["tab_video"], id="video"):
                        with gr.Row():
                            video = gr.Video(label=L["video"], sources=["upload"])
                            video_out = gr.Image(type="numpy", label=L["result"])
                        with gr.Row():
                            vrec = gr.Checkbox(value=True, label=L["record"])
                            vstart = gr.Button(L["start"], variant="primary")
                            vstop = gr.Button(L["stop"])
                        video_stats = gr.Markdown(elem_id="stats")
                        video_file = gr.File(label=L["download"])
                        vev = vstart.click(run_video, inputs=[video, vrec, *settings], outputs=[video_out, video_stats, video_file])
                        vstop.click(None, cancels=[vev])
                    with gr.Tab(L["tab_image"], id="image"):
                        with gr.Row():
                            image = gr.Image(type="numpy", label=L["image"], sources=["upload", "clipboard"])
                            image_out = gr.Image(type="numpy", label=L["result"])
                        image_stats = gr.Markdown(elem_id="stats")
                        image.change(process, inputs=[image, *settings], outputs=[image_out, image_stats])
                        for c in settings:
                            c.change(process, inputs=[image, *settings], outputs=[image_out, image_stats])

        def on_model(path):
            return model_info(path), gr.update(choices=model_names(path), value=[])

        model_a.change(on_model, inputs=model_a, outputs=[info, classes])

        def on_upload(f):
            if not f:
                return gr.skip(), gr.skip()
            dest = MODELS / "uploads"
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / Path(f).name
            target.write_bytes(Path(f).read_bytes())
            ch = discover_models()
            return gr.update(choices=ch, value=str(target)), gr.update(choices=ch + [(NONE, NONE)])

        upload.upload(on_upload, inputs=upload, outputs=[model_a, model_b])
        faster.click(convert_onnx, inputs=[model_a, imgsz], outputs=[model_a, info])
    return demo


def main() -> None:
    global L, DEVICE
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--port", type=int, default=7721)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--lang", default="zh")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()
    L = TEXT.get(args.lang, TEXT["en"])
    DEVICE = best_device()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner" / "adapters"))
        from tk_common import configure_ultralytics

        configure_ultralytics()
    except Exception:
        pass
    demo = build(args.model, args.source)
    print(f"Playground: http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}", flush=True)
    demo.queue(default_concurrency_limit=4).launch(server_name=args.host, server_port=args.port, inbrowser=args.open,
                                                   css=CSS, theme=gr.themes.Soft(), show_error=True, quiet=True,
                                                   footer_links=[])


if __name__ == "__main__":
    main()
