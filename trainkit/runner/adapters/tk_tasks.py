"""Setup-time tasks that run inside a recipe environment.

Usage: python tk_tasks.py <task> [options]. The last stdout line is `TRAINKIT_JSON={...}`.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tk_common import best_device, configure_ultralytics, emit_result, load_model  # noqa: E402


def task_selftest(args) -> dict:
    r: dict = {"python": platform.python_version()}
    try:
        import torch

        r["torch"] = torch.__version__
        r["cuda_build"] = torch.version.cuda
        r["cuda_available"] = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None)
        r["mps_available"] = bool(mps is not None and mps.is_available())
        dev = "cuda" if r["cuda_available"] else ("mps" if r["mps_available"] else "cpu")
        r["device"] = dev
        if dev == "cuda":
            props = torch.cuda.get_device_properties(0)
            r["device_name"] = torch.cuda.get_device_name(0)
            r["vram_gb"] = round(props.total_memory / 1024**3, 1)
            r["compute_cap"] = f"{props.major}.{props.minor}"
            r["arch_list"] = torch.cuda.get_arch_list()
        t0 = time.time()
        x = torch.randn(512, 512, device=dev)
        float((x @ x).sum().item())
        r["matmul_ms"] = round((time.time() - t0) * 1000, 1)
        r["tensor_ok"] = True
    except Exception as e:
        r["tensor_ok"] = False
        r["tensor_error"] = f"{type(e).__name__}: {e}"
    for mod in ("ultralytics", "gradio", "onnxruntime", "cv2"):
        try:
            m = __import__(mod)
            r[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            r[f"{mod}_error"] = f"{type(e).__name__}: {e}"
    return r


def task_weights(args) -> dict:
    configure_ultralytics()
    model = load_model(args.name, args.model_class)
    path = Path(getattr(model, "ckpt_path", "") or args.name).resolve()
    return {"path": str(path), "task": model.task, "names": model.names}


def task_dataset(args) -> dict:
    configure_ultralytics()
    if args.task == "classify":
        from ultralytics.data.utils import check_cls_dataset

        data = check_cls_dataset(args.name)
        root = Path(str(data["train"])).parent
        return {"data": str(root) if Path(args.name).exists() else args.name, "root": str(root),
                "nc": data["nc"], "names": data["names"]}
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(args.name)
    return {"data": str(data.get("yaml_file", args.name)), "root": str(data.get("path", "")),
            "nc": data.get("nc"), "names": data.get("names")}


def task_smoke(args) -> dict:
    """Train one tiny epoch to prove the whole chain (data → model → device → checkpoint) works."""
    configure_ultralytics()
    device = args.device or best_device()
    model = load_model(args.weights, args.model_class)
    out = Path(tempfile.mkdtemp(prefix="trainkit-smoke-"))
    t0 = time.time()
    try:
        imgsz = 64 if args.task == "classify" else 320
        model.train(data=args.data, epochs=1, imgsz=imgsz, batch=4, workers=0, device=device, project=str(out),
                    name="smoke", exist_ok=True, plots=False, verbose=False, amp=device not in ("cpu", "mps"))
        ok = (out / "smoke" / "weights" / "last.pt").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return {"ok": ok, "device": device, "seconds": round(time.time() - t0, 1)}


def task_predict_test(args) -> dict:
    """Run one inference on Ultralytics' bundled sample image (used by the CPU preview tier)."""
    configure_ultralytics()
    import ultralytics

    img = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    model = load_model(args.weights, args.model_class)
    model.predict(str(img), verbose=False, device="cpu")  # warm-up
    t0 = time.time()
    res = model.predict(str(img), verbose=False, device="cpu")
    ms = (time.time() - t0) * 1000
    n = 0
    r = res[0]
    if getattr(r, "boxes", None) is not None:
        n = len(r.boxes)
    elif getattr(r, "probs", None) is not None:
        n = 1
    return {"ok": True, "detections": n, "ms": round(ms, 1)}


def task_info(args) -> dict:
    configure_ultralytics()
    model = load_model(args.model, args.model_class)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    imgsz = None
    try:
        imgsz = model.overrides.get("imgsz") or (model.ckpt or {}).get("train_args", {}).get("imgsz")
    except Exception:
        pass
    return {"task": model.task, "names": {int(k): v for k, v in names.items()}, "imgsz": imgsz,
            "model_class": type(model).__name__}


def task_export(args) -> dict:
    configure_ultralytics()
    model = load_model(args.model, args.model_class)
    kwargs = {"format": args.format}
    if args.imgsz:
        kwargs["imgsz"] = args.imgsz
    if args.half:
        kwargs["half"] = True
    if args.int8:
        kwargs["int8"] = True
    if args.device:
        kwargs["device"] = args.device
    if args.data:
        kwargs["data"] = args.data
    path = model.export(**kwargs)
    return {"path": str(Path(path).resolve())}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("weights")
    s.add_argument("name")
    s.add_argument("--model-class", default="YOLO")
    s = sub.add_parser("dataset")
    s.add_argument("name")
    s.add_argument("--task", default="detect")
    s = sub.add_parser("smoke")
    s.add_argument("--weights", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--task", default="detect")
    s.add_argument("--device", default="")
    s.add_argument("--model-class", default="YOLO")
    s = sub.add_parser("predict-test")
    s.add_argument("--weights", required=True)
    s.add_argument("--model-class", default="YOLO")
    s = sub.add_parser("info")
    s.add_argument("model")
    s.add_argument("--model-class", default=None)
    s = sub.add_parser("export")
    s.add_argument("model")
    s.add_argument("--format", default="onnx")
    s.add_argument("--imgsz", type=int, default=0)
    s.add_argument("--half", action="store_true")
    s.add_argument("--int8", action="store_true")
    s.add_argument("--device", default="")
    s.add_argument("--data", default="")
    s.add_argument("--model-class", default=None)
    args = p.parse_args()
    fn = globals()["task_" + args.cmd.replace("-", "_")]
    try:
        emit_result(fn(args))
    except Exception as e:
        traceback.print_exc()
        emit_result({"error": f"{type(e).__name__}: {e}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
