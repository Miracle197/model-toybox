"""Collect facts about this machine: OS, memory, disk, GPU, driver, cameras, network.

Everything here must work without torch installed and must never raise: missing tools just mean "unknown".
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

from .. import paths


@dataclass
class GPU:
    vendor: str  # nvidia | apple | amd | intel
    name: str
    vram_gb: float = 0.0
    driver: str = ""
    cuda_driver: str = ""  # highest CUDA version the driver supports, e.g. "12.8"
    compute_cap: str = ""  # e.g. "8.6"


@dataclass
class SystemInfo:
    os: str
    os_version: str
    arch: str
    cpu: str
    cpu_count: int
    ram_gb: float
    disk_free_gb: float
    user_home: str
    trainkit_home: str
    python: str
    uv: str
    gpus: list[GPU] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    windows_long_paths: bool | None = None
    network: dict[str, bool] = field(default_factory=dict)

    @property
    def nvidia(self) -> GPU | None:
        return next((g for g in self.gpus if g.vendor == "nvidia"), None)

    @property
    def apple(self) -> GPU | None:
        return next((g for g in self.gpus if g.vendor == "apple"), None)

    @property
    def tier(self) -> str:
        """nvidia | mps | cpu — what TrainKit will use to train."""
        if self.nvidia:
            return "nvidia"
        if self.apple:
            return "mps"
        return "cpu"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier
        return d


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def parse_nvidia_query(csv_text: str) -> list[GPU]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,driver_version[,compute_cap] --format=csv,noheader,nounits`."""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mem, driver = parts[0], parts[1], parts[2]
        cc = parts[3] if len(parts) > 3 and re.fullmatch(r"\d+\.\d+", parts[3]) else ""
        try:
            vram = round(float(mem) / 1024, 1)
        except ValueError:
            vram = 0.0
        gpus.append(GPU(vendor="nvidia", name=name, vram_gb=vram, driver=driver, compute_cap=cc))
    return gpus


def parse_cuda_driver_version(smi_text: str) -> str:
    m = re.search(r"CUDA Version:\s*([\d.]+)", smi_text or "")
    return m.group(1) if m else ""


def probe_nvidia() -> list[GPU]:
    smi = shutil.which("nvidia-smi")
    if not smi and sys.platform == "win32":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(candidate) if candidate.exists() else None
    if not smi:
        return []
    out = _run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"])
    if out is None:  # old drivers don't know compute_cap
        out = _run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = parse_nvidia_query(out)
    cuda = parse_cuda_driver_version(_run([smi]) or "")
    for g in gpus:
        g.cuda_driver = cuda
    return gpus


def probe_apple() -> list[GPU]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return []
    chip = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon").strip()
    ram = round(psutil.virtual_memory().total / 1024**3, 1)
    return [GPU(vendor="apple", name=chip, vram_gb=ram)]  # unified memory


def probe_cameras() -> list[str]:
    if sys.platform.startswith("linux"):
        return sorted(str(p) for p in Path("/dev").glob("video*"))
    return []  # Windows/macOS cameras are listed by the playground (needs OpenCV)


def probe_windows_long_paths() -> bool | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return bool(value)
    except OSError:
        return False


NETWORK_TARGETS = {
    "pypi.org": "https://pypi.org/simple/pip/",
    "download.pytorch.org": "https://download.pytorch.org/whl/",
    "github.com": "https://github.com/ultralytics/assets/releases",
}


def probe_network(timeout: float = 6.0) -> dict[str, bool]:
    import httpx

    def check(url: str) -> bool:
        try:
            r = httpx.head(url, timeout=timeout, follow_redirects=True)
            return r.status_code < 500
        except Exception:
            return False

    with cf.ThreadPoolExecutor(max_workers=len(NETWORK_TARGETS)) as pool:
        futures = {name: pool.submit(check, url) for name, url in NETWORK_TARGETS.items()}
        return {name: f.result() for name, f in futures.items()}


def uv_binary() -> str | None:
    try:
        from uv import find_uv_bin

        return find_uv_bin()
    except Exception:
        return shutil.which("uv")


def uv_version() -> str:
    uv = uv_binary()
    if not uv:
        return ""
    out = _run([uv, "--version"])
    return (out or "").strip().removeprefix("uv ").split(" ")[0]


def cpu_name() -> str:
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def collect(check_network: bool = True) -> SystemInfo:
    home = paths.home()
    probe_dir = home if home.exists() else Path.home()
    try:
        free = shutil.disk_usage(probe_dir).free / 1024**3
    except OSError:
        free = 0.0
    if sys.platform == "win32":
        os_version = platform.version()
    elif sys.platform == "darwin":
        os_version = platform.mac_ver()[0]
    else:
        os_version = platform.release()
    return SystemInfo(
        os={"win32": "Windows", "darwin": "macOS"}.get(sys.platform, platform.system()),
        os_version=os_version,
        arch=platform.machine(),
        cpu=cpu_name(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        disk_free_gb=round(free, 1),
        user_home=str(Path.home()),
        trainkit_home=str(home),
        python=platform.python_version(),
        uv=uv_version(),
        gpus=probe_nvidia() + probe_apple(),
        cameras=probe_cameras(),
        windows_long_paths=probe_windows_long_paths(),
        network=probe_network() if check_network else {},
    )
