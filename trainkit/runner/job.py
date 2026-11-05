"""Start, resume and control training processes.

Training runs as a detached subprocess inside the recipe environment, so closing the terminal or the dashboard
never kills it. Control goes through <run_dir>/control.json; a hard stop kills the whole process tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from .. import paths
from ..envs import manager
from ..recipes import Recipe, get_recipe
from . import runs

ADAPTER = manager.ADAPTERS_DIR / "ultralytics_run.py"
_children: dict[str, subprocess.Popen] = {}  # keep handles so finished children get reaped


def _spawn(rd: Path, env_name: str, args: list[str]) -> int:
    log = open(rd / "train.log", "ab")
    kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                              "cwd": str(paths.weights_dir()), "env": manager.subprocess_env(env_name)}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(manager.python_cmd(env_name, ADAPTER, *args), **kwargs)
    log.close()
    _children[rd.name] = proc
    meta = runs.read_meta(rd)
    meta["pid"] = proc.pid
    meta["last_spawn"] = time.time()
    meta.setdefault("attempts", []).append({"pid": proc.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "resume": "--resume" in args})
    runs.write_meta(rd, meta)
    return proc.pid


def reap() -> None:
    for rid, proc in list(_children.items()):
        if proc.poll() is not None:
            _children.pop(rid, None)


def start(recipe: Recipe, data: str, data_label: str, weights: str | None = None,
          overrides: dict[str, Any] | None = None, name: str | None = None, samples_every: int = 1) -> str:
    if not manager.is_ready(recipe.env):
        raise manager.EnvError(f"Environment '{recipe.env}' is not ready. Run: trainkit setup {recipe.id}")
    paths.ensure_dirs()
    overrides = {**recipe.defaults, **(overrides or {})}
    overrides.pop("save", None)  # checkpoints are required for pause/resume
    rid = runs.new_run_id(recipe.id, name)
    rd = runs.run_dir(rid)
    rd.mkdir(parents=True)
    weights = weights or recipe.default_weights
    runs.write_meta(rd, {
        "id": rid, "name": name or rid, "recipe": recipe.id, "task": recipe.task, "model_class": recipe.model_class,
        "env": recipe.env, "weights": weights, "data": data, "data_label": data_label, "overrides": overrides,
        "samples_every": samples_every, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    runs.append_event(rd, "status", state="starting")
    _spawn(rd, recipe.env, ["--run-dir", str(rd), "--weights", weights, "--data", data,
                            "--model-class", recipe.model_class, "--overrides", json.dumps(overrides),
                            "--samples-every", str(samples_every)])
    return rid


def resume(run_id: str, overrides: dict[str, Any] | None = None) -> int:
    rd = runs.run_dir(run_id)
    meta = runs.read_meta(rd)
    s, _ = runs.summarize(rd)
    if s.state in runs.ACTIVE_STATES:
        raise RuntimeError("Run is still active")
    ckpt = rd / "resume.pt"
    if not ckpt.exists():
        raise RuntimeError("No checkpoint to resume from (the first epoch never finished)")
    recipe = get_recipe(meta["recipe"])
    if not manager.is_ready(recipe.env):
        raise manager.EnvError(f"Environment '{recipe.env}' is not ready. Run: trainkit setup {recipe.id}")
    runs.append_event(rd, "status", state="starting", msg="resume")
    ov = {k: v for k, v in {**meta.get("overrides", {}), **(overrides or {})}.items()
          if k in ("device", "batch", "workers", "imgsz", "patience")}
    return _spawn(rd, recipe.env, ["--run-dir", str(rd), "--resume", str(ckpt), "--model-class",
                                   meta.get("model_class", "YOLO"), "--overrides", json.dumps(ov),
                                   "--samples-every", str(meta.get("samples_every", 1))])


def send(run_id: str, cmd: str) -> None:
    """Write a control command (pause | continue | save_stop | stop) for the training process."""
    if cmd not in ("pause", "continue", "save_stop", "stop"):
        raise ValueError(cmd)
    rd = runs.run_dir(run_id)
    f = rd / "control.json"
    seq = 0
    try:
        seq = int(json.loads(f.read_text(encoding="utf-8")).get("seq", 0))
    except (OSError, ValueError):
        pass
    tmp = rd / "control.json.tmp"
    tmp.write_text(json.dumps({"cmd": cmd, "seq": seq + 1, "t": time.time()}), encoding="utf-8")
    os.replace(tmp, f)


def kill_tree(pid: int, timeout: float = 5.0) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def stop(run_id: str, grace: float = 10.0) -> str:
    """Ask the process to stop after the current batch; kill the process tree if it does not exit in time."""
    rd = runs.run_dir(run_id)
    s, _ = runs.summarize(rd)
    pid = s.pid or runs.read_meta(rd).get("pid")
    if not runs.pid_alive(pid):
        return "not-running"
    send(run_id, "stop")
    deadline = time.time() + grace
    while time.time() < deadline:
        reap()
        if not runs.pid_alive(pid):
            return "stopped"
        time.sleep(0.25)
    kill_tree(pid)
    reap()
    runs.append_event(rd, "status", state="stopped", msg="killed")
    return "killed"


def control(run_id: str, cmd: str) -> str:
    if cmd == "stop":
        return stop(run_id)
    send(run_id, cmd)
    return "sent"
