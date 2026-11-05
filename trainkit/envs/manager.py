"""Isolated per-recipe Python environments, created with uv.

The TrainKit CLI itself never imports torch. Everything heavy runs as a subprocess inside these environments.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .. import paths
from ..config import load_config
from ..doctor.probe import uv_binary
from ..recipes import env_spec

MANIFEST = "trainkit-env.json"
JSON_MARK = "TRAINKIT_JSON="
ADAPTERS_DIR = paths.package_dir() / "runner" / "adapters"


class EnvError(RuntimeError):
    pass


def env_path(name: str) -> Path:
    return paths.envs_dir() / name


def env_python(name: str) -> Path:
    base = env_path(name)
    return base / ("Scripts/python.exe" if paths.IS_WINDOWS else "bin/python")


def load_manifest(name: str) -> dict[str, Any] | None:
    p = env_path(name) / MANIFEST
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_ready(name: str) -> bool:
    m = load_manifest(name)
    return bool(m and m.get("ok") and env_python(name).exists())


def uv() -> str:
    b = uv_binary()
    if not b:
        raise EnvError("uv not found. Re-run the TrainKit installer or `pip install uv`.")
    return b


def subprocess_env(name: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment variables for running code inside a recipe env."""
    env = os.environ.copy()
    for k in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        env.pop(k, None)
    venv = env_path(name)
    bin_dir = venv / ("Scripts" if paths.IS_WINDOWS else "bin")
    uv_dir = Path(uv_binary() or "").parent
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = os.pathsep.join([str(bin_dir), str(uv_dir), env.get("PATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["YOLO_CONFIG_DIR"] = str(paths.ultralytics_config_dir())
    env["TRAINKIT_HOME"] = str(paths.home())
    env["TRAINKIT_DATASETS_DIR"] = str(paths.datasets_dir())
    env["TRAINKIT_WEIGHTS_DIR"] = str(paths.weights_dir())
    env["TRAINKIT_RUNS_DIR"] = str(paths.runs_dir())
    env["GRADIO_ANALYTICS_ENABLED"] = "False"
    if paths.IS_MAC:
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    idx = load_config().get("pypi_index")
    if idx:
        env["UV_DEFAULT_INDEX"] = idx
        env["PIP_INDEX_URL"] = idx
    if extra:
        env.update(extra)
    return env


def _stream(cmd: list[str], on_line: Callable[[str], None] | None, env: dict[str, str] | None = None,
            cwd: Path | None = None) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            env=env, cwd=cwd, encoding="utf-8")
    assert proc.stdout is not None
    for line in proc.stdout:
        # Progress bars redraw with carriage returns: keep only the final state of the line
        segments = [seg for seg in line.rstrip("\n").split("\r") if seg.strip()]
        if on_line and segments:
            on_line(segments[-1])
    return proc.wait()


def create(name: str, profile: str, torch_backend: str, on_line: Callable[[str], None] | None = None,
           reinstall: bool = False) -> dict[str, Any]:
    """Create (or update) the environment and install its packages."""
    spec = env_spec(name)
    venv = env_path(name)
    paths.ensure_dirs()
    if profile == "preview":
        torch_backend = "cpu"  # preview tier never needs GPU wheels; CPU wheels are ~10x smaller

    if reinstall or not env_python(name).exists():
        cmd = [uv(), "venv", str(venv), "--python", spec["python"], "--quiet"]
        if venv.exists():
            cmd.append("--clear")
        code = _stream(cmd, on_line)
        if code != 0:
            raise EnvError(f"uv venv failed (exit {code})")

    install = [uv(), "pip", "install", "--python", str(env_python(name)), *spec["packages"],
               "-c", str(spec["constraints_path"]), "--torch-backend", torch_backend]
    idx = load_config().get("pypi_index")
    if idx:
        install += ["--default-index", idx]
    env = os.environ.copy()
    env["UV_PREVIEW"] = "1"  # --torch-backend is a preview feature in some uv versions
    t0 = time.time()
    code = _stream(install, on_line, env=env)
    if code != 0:
        raise EnvError(f"Package installation failed (exit {code})")

    manifest = {
        "name": name,
        "profile": profile,
        "torch_backend": torch_backend,
        "python": spec["python"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "install_seconds": round(time.time() - t0, 1),
        "ok": False,
    }
    (venv / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def mark_ready(name: str, self_test_result: dict[str, Any]) -> dict[str, Any]:
    manifest = load_manifest(name) or {"name": name}
    manifest["ok"] = True
    manifest["self_test"] = self_test_result
    (env_path(name) / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def freeze(name: str) -> list[str]:
    out = subprocess.run([uv(), "pip", "freeze", "--python", str(env_python(name))], capture_output=True,
                         text=True, errors="replace")
    return [line for line in out.stdout.splitlines() if line.strip()]


def run_task(name: str, task: str, *args: str, on_line: Callable[[str], None] | None = None,
             cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run `tk_tasks.py <task> ...` inside the env and return its JSON result."""
    if not env_python(name).exists():
        raise EnvError(f"Environment '{name}' is not installed. Run `trainkit setup` first.")
    script = ADAPTERS_DIR / "tk_tasks.py"
    cmd = [str(env_python(name)), str(script), task, *args]
    result: dict[str, Any] = {}
    tail: list[str] = []

    def handle(line: str) -> None:
        nonlocal result
        if line.startswith(JSON_MARK):
            result = json.loads(line[len(JSON_MARK):])
            return
        tail.append(line)
        del tail[:-40]
        if on_line:
            on_line(line)

    code = _stream(cmd, handle, env=subprocess_env(name, extra_env), cwd=cwd or paths.weights_dir())
    if code != 0 and not result:
        raise EnvError(f"Task '{task}' failed (exit {code}):\n" + "\n".join(tail[-15:]))
    if result.get("error"):
        raise EnvError(result["error"])
    return result


def python_cmd(name: str, script: Path, *args: str) -> list[str]:
    return [str(env_python(name)), str(script), *args]


def describe(name: str) -> str:
    m = load_manifest(name)
    if not m:
        return "not installed"
    st = m.get("self_test", {})
    return f"{m.get('profile')} · torch {st.get('torch', '?')} · {st.get('device', '?')} · ultralytics {st.get('ultralytics', '?')}"


if __name__ == "__main__":  # pragma: no cover
    print(describe(sys.argv[1] if len(sys.argv) > 1 else "ultralytics"))
