"""Web dashboard backend: REST + WebSocket over the same run files the CLI uses."""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from pathlib import Path
from typing import Any

import psutil
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, models
from .. import datasets as ds
from ..doctor import analyze, collect, recommended_profile, verdicts
from ..envs import manager
from ..recipes import all_recipes, get_recipe
from ..runner import job, runs

STATIC = Path(__file__).resolve().parent / "static"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class GpuSampler:
    """Live GPU (NVIDIA via NVML) and CPU/RAM utilization."""

    def __init__(self) -> None:
        self.nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
        except Exception:
            self.nvml = None
        psutil.cpu_percent(None)

    def sample(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        out: dict[str, Any] = {"t": time.time(), "cpu": psutil.cpu_percent(None), "ram": vm.percent,
                               "ram_used_gb": round(vm.used / 1024**3, 1), "ram_total_gb": round(vm.total / 1024**3, 1),
                               "gpus": []}
        if self.nvml:
            n = self.nvml
            try:
                for i in range(n.nvmlDeviceGetCount()):
                    h = n.nvmlDeviceGetHandleByIndex(i)
                    util = n.nvmlDeviceGetUtilizationRates(h)
                    mem = n.nvmlDeviceGetMemoryInfo(h)
                    g = {"name": n.nvmlDeviceGetName(h), "util": util.gpu, "mem_used_gb": round(mem.used / 1024**3, 2),
                         "mem_total_gb": round(mem.total / 1024**3, 2), "mem": round(100 * mem.used / mem.total, 1)}
                    try:
                        g["temp"] = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
                        g["power_w"] = round(n.nvmlDeviceGetPowerUsage(h) / 1000, 1)
                    except Exception:
                        pass
                    if isinstance(g["name"], bytes):
                        g["name"] = g["name"].decode()
                    out["gpus"].append(g)
            except Exception:
                pass
        return out


def _run_payload(run_id: str) -> dict[str, Any]:
    rd = runs.run_dir(run_id)
    if not (rd / "run.json").exists():
        raise HTTPException(404, "run not found")
    meta = runs.read_meta(rd)
    s, _ = runs.summarize(rd)
    best = runs.best_weights(rd)
    return {"meta": meta, "summary": s.to_dict(), "resumable": runs.resumable(rd, s),
            "has_weights": best is not None, "plots": [p.name for p in sorted((rd / "train").glob("*.png"))]
            if (rd / "train").is_dir() else []}


def create_app(token: str | None = None) -> FastAPI:
    app = FastAPI(title="TrainKit", version=__version__, docs_url=None, redoc_url=None)
    sampler = GpuSampler()
    system_cache: dict[str, Any] = {}

    def authorized(request_token: str | None, cookie: str | None, header: str | None) -> bool:
        if not token:
            return True
        return any(t and secrets.compare_digest(t, token) for t in (request_token, cookie, header))

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if token:
            q = request.query_params.get("token")
            if not authorized(q, request.cookies.get("tk_token"), request.headers.get("x-trainkit-token")):
                if request.url.path == "/api/ping":
                    return JSONResponse({"app": "trainkit", "auth": True}, status_code=401)
                return PlainTextResponse("TrainKit: missing or wrong token", status_code=401)
            response = await call_next(request)
            if q:
                response.set_cookie("tk_token", token, httponly=True, samesite="strict")
            return response
        return await call_next(request)

    @app.get("/api/ping")
    def ping():
        return {"app": "trainkit", "version": __version__}

    @app.get("/api/system")
    def system(refresh: bool = False):
        if refresh or not system_cache:
            info = collect(check_network=False)
            system_cache.update({
                "system": info.to_dict(),
                "profile": recommended_profile(info),
                "findings": [f.__dict__ for f in analyze(info)],
                "verdicts": {v.recipe_id: v.__dict__ for v in verdicts(info, all_recipes())},
            })
        envs = {name: manager.load_manifest(name) for name in ("ultralytics",)}
        return {**system_cache, "envs": envs, "version": __version__}

    @app.get("/api/gpu")
    def gpu():
        return sampler.sample()

    @app.get("/api/recipes")
    def recipes():
        return [r.to_dict() for r in all_recipes()]

    @app.get("/api/datasets")
    def datasets():
        return {"registered": ds.registry(),
                "builtin": {r.id: r.to_dict()["builtin_datasets"] for r in all_recipes()}}

    @app.get("/api/runs")
    def list_runs():
        job.reap()
        return runs.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return _run_payload(run_id)

    @app.get("/api/runs/{run_id}/log")
    def run_log(run_id: str, tail: int = 80):
        f = runs.run_dir(run_id) / "train.log"
        if not f.exists():
            return {"lines": []}
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64_000))
            text = fh.read().decode("utf-8", "replace")
        text = ANSI.sub("", text)
        lines = [ln.rstrip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
        return {"lines": lines[-tail:]}

    @app.get("/api/runs/{run_id}/files/{path:path}")
    def run_file(run_id: str, path: str):
        rd = runs.run_dir(run_id)
        f = (rd / path).resolve()
        if rd not in f.parents or not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".csv", ".yaml"):
            raise HTTPException(404)
        return FileResponse(f)

    @app.post("/api/runs")
    def start_run(body: dict = Body(...)):
        try:
            recipe = get_recipe(body.get("recipe", "yolo-detect"))
        except KeyError as e:
            raise HTTPException(400, str(e))
        data = (body.get("data") or "").strip()
        if not data:
            raise HTTPException(400, "data is required")
        try:
            resolved = ds.resolve(data, recipe.task)
            report = ds.check(resolved, recipe.task)
        except ds.DatasetError as e:
            raise HTTPException(400, str(e))
        if report and not report.ok:
            raise HTTPException(400, "; ".join(report.errors[:5]))
        ov: dict[str, Any] = {}
        for k in ("epochs", "imgsz", "batch", "patience", "device", "workers"):
            v = body.get(k)
            if v not in (None, ""):
                ov[k] = v
        try:
            rid = job.start(recipe, resolved.data, resolved.label, body.get("weights") or None, ov,
                            body.get("name") or None, int(body.get("samples_every", 1)))
        except manager.EnvError as e:
            raise HTTPException(400, str(e))
        return {"id": rid, "warnings": report.warnings if report else []}

    @app.post("/api/runs/{run_id}/control")
    def control(run_id: str, body: dict = Body(...)):
        cmd = body.get("cmd", "")
        if cmd not in ("pause", "continue", "save_stop", "stop"):
            raise HTTPException(400, "bad command")
        return {"result": job.control(run_id, cmd)}

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str, body: dict = Body(default={})):
        try:
            pid = job.resume(run_id, body or None)
        except (RuntimeError, manager.EnvError) as e:
            raise HTTPException(400, str(e))
        return {"pid": pid}

    @app.get("/api/runs/{run_id}/bundle")
    def bundle(run_id: str):
        rd = runs.run_dir(run_id)
        out = rd / f"{runs.read_meta(rd).get('name', run_id)}.zip"
        try:
            models.pack(run_id, out)
        except models.ModelError as e:
            raise HTTPException(400, str(e))
        return FileResponse(out, filename=out.name, media_type="application/zip")

    @app.websocket("/api/runs/{run_id}/ws")
    async def run_ws(ws: WebSocket, run_id: str):
        if not authorized(ws.query_params.get("token"), ws.cookies.get("tk_token"), None):
            await ws.close(code=4401)
            return
        await ws.accept()
        try:
            rd = runs.run_dir(run_id)
        except ValueError:
            await ws.close(code=4404)
            return
        s = runs.Summary()
        offset = 0
        last_push = 0.0
        last_gpu = 0.0
        try:
            while True:
                events, offset = runs.read_events(rd, offset)
                for ev in events:
                    s.feed(ev)
                prev_state = s.state
                runs.reconcile(s, rd)
                now = time.time()
                if events or prev_state != s.state or now - last_push > 5:
                    job.reap()
                    await ws.send_json({"type": "summary", "summary": s.to_dict(),
                                        "resumable": runs.resumable(rd, s),
                                        "has_weights": runs.best_weights(rd) is not None})
                    last_push = now
                if now - last_gpu >= 1.0:
                    await ws.send_json({"type": "gpu", "gpu": sampler.sample()})
                    last_gpu = now
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
