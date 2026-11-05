"""Turn raw machine facts into findings, fixes and per-recipe verdicts."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass

from ..i18n import tr
from ..recipes import Recipe
from .probe import SystemInfo

NVIDIA_DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"

# Minimum NVIDIA driver for CUDA 12.x wheels (minor-version compatibility) and for CUDA 11.8.
_CUDA12_MIN = {"win32": 528.33, "linux": 525.60}
_CUDA118_MIN = {"win32": 452.39, "linux": 450.80}


@dataclass
class Finding:
    level: str  # ok | info | warn | error
    title: str
    detail: str = ""
    fix: str = ""


@dataclass
class Verdict:
    recipe_id: str
    title: str
    train: str  # ok | warn | no
    preview: str  # ok | warn | no
    reasons: list[str]


def _driver_number(driver: str) -> float:
    try:
        major, _, rest = driver.partition(".")
        return float(f"{int(major)}.{rest.split('.')[0] or 0}")
    except ValueError:
        return 0.0


def _platform_key() -> str:
    return "win32" if sys.platform == "win32" else "linux"


def _compute_cap(info: SystemInfo) -> float | None:
    g = info.nvidia
    if not g or not g.compute_cap:
        return None
    try:
        return float(g.compute_cap)
    except ValueError:
        return None


def _is_blackwell(info: SystemInfo) -> bool:
    g = info.nvidia
    cc = _compute_cap(info)
    return bool(g) and ((cc is not None and cc >= 12.0) or " RTX 50" in f" {g.name}")


def nvidia_driver_blocked(info: SystemInfo) -> str:
    """Return a reason string if the NVIDIA driver cannot run current PyTorch at all."""
    g = info.nvidia
    if not g:
        return ""
    drv = _driver_number(g.driver)
    if drv and drv < _CUDA118_MIN[_platform_key()]:
        return tr(f"显卡驱动 {g.driver} 太旧，PyTorch 无法使用 GPU", f"Driver {g.driver} is too old for PyTorch GPU support")
    if _is_blackwell(info) and drv and drv < 570:
        return tr(f"RTX 50 系列需要驱动 ≥ 570（当前 {g.driver}）", f"RTX 50 series needs driver >= 570 (have {g.driver})")
    cc = _compute_cap(info)
    if cc is not None and cc < 5.0:
        return tr(f"显卡计算能力 {cc} 太低，当前 PyTorch 不再支持", f"GPU compute capability {cc} is no longer supported by PyTorch")
    return ""


def torch_backend_for(info: SystemInfo) -> str:
    """Value for `uv pip install --torch-backend`.

    uv's `auto` picks the newest CUDA build the driver supports. CUDA 13 builds dropped GPUs older than
    Turing (compute capability < 7.5, e.g. GTX 10-series), so those must stay on CUDA 12.6 builds.
    """
    if info.nvidia:
        if nvidia_driver_blocked(info):
            return "cpu"
        cc = _compute_cap(info)
        if cc is not None and cc < 7.5:
            return "cu126"
    return "auto"


def analyze(info: SystemInfo) -> list[Finding]:
    f: list[Finding] = []
    g = info.nvidia

    # Device tier
    if info.tier == "nvidia" and g:
        blocked = nvidia_driver_blocked(info)
        f.append(Finding("ok" if not blocked else "error",
                         tr(f"NVIDIA 显卡：{g.name}（{g.vram_gb} GB 显存）", f"NVIDIA GPU: {g.name} ({g.vram_gb} GB VRAM)"),
                         tr(f"驱动 {g.driver}，支持到 CUDA {g.cuda_driver or '?'}；计算能力 {g.compute_cap or '?'}",
                            f"Driver {g.driver}, supports up to CUDA {g.cuda_driver or '?'}; compute capability {g.compute_cap or '?'}"),
                         tr(f"更新驱动：{NVIDIA_DRIVER_URL}", f"Update the driver: {NVIDIA_DRIVER_URL}") if blocked else ""))
        if blocked:
            f.append(Finding("error", blocked, tr("在更新驱动前只能用 CPU 预览模型", "Until the driver is updated only CPU preview works"),
                             tr(f"下载最新驱动后重启：{NVIDIA_DRIVER_URL}", f"Install the latest driver and reboot: {NVIDIA_DRIVER_URL}")))
        else:
            drv = _driver_number(g.driver)
            if drv and drv < _CUDA12_MIN[_platform_key()]:
                f.append(Finding("warn", tr("显卡驱动较旧", "GPU driver is old"),
                                 tr("将使用较旧的 CUDA 11.8 版 PyTorch，建议更新驱动", "An older CUDA 11.8 PyTorch build will be used; updating is recommended"),
                                 NVIDIA_DRIVER_URL))
            if torch_backend_for(info) == "cu126":
                f.append(Finding("info", tr("老款显卡：将固定使用 CUDA 12.6 版 PyTorch", "Older GPU: pinning the CUDA 12.6 PyTorch build"),
                                 tr("CUDA 13 版本不再支持计算能力 7.5 以下的显卡（如 GTX 10 系列）", "CUDA 13 builds dropped GPUs below compute capability 7.5 (e.g. GTX 10-series)")))
        f.append(Finding("info", tr("无需安装 CUDA Toolkit", "No CUDA Toolkit needed"),
                         tr("PyTorch 安装包自带 CUDA 运行时，只需要显卡驱动", "PyTorch wheels ship their own CUDA runtime; only the driver is required")))
    elif info.tier == "mps":
        f.append(Finding("ok", tr(f"Apple 芯片：{info.apple.name}（统一内存 {info.ram_gb} GB）", f"Apple Silicon: {info.apple.name} ({info.ram_gb} GB unified memory)"),
                         tr("训练使用 MPS 加速，速度慢于同级 NVIDIA 显卡；已自动开启 MPS 回退", "Training uses MPS; slower than a comparable NVIDIA GPU. MPS fallback is enabled automatically")))
    else:
        detail = tr("没有找到可用的 NVIDIA 显卡或 Apple 芯片：可以试玩模型，训练请用有显卡的电脑或 Colab/Kaggle",
                    "No usable NVIDIA GPU or Apple Silicon: you can try models; train on a GPU machine or Colab/Kaggle")
        if sys.platform == "darwin":
            detail = tr("Intel 芯片的 Mac 没有 GPU 加速，进入 CPU 预览档", "Intel Macs have no GPU acceleration; using the CPU preview tier")
        f.append(Finding("info", tr("CPU 预览档", "CPU preview tier"), detail))

    # Memory and disk
    if info.ram_gb < 8:
        f.append(Finding("warn", tr(f"内存只有 {info.ram_gb} GB", f"Only {info.ram_gb} GB RAM"),
                         tr("训练建议 ≥ 8 GB；可减小 batch 和 imgsz", "Training wants >= 8 GB; lower batch and imgsz")))
    if info.disk_free_gb < 6:
        f.append(Finding("error", tr(f"磁盘剩余 {info.disk_free_gb} GB", f"Only {info.disk_free_gb} GB disk free"),
                         tr("安装环境约需 3–6 GB", "The environment needs about 3–6 GB"),
                         tr("清理磁盘，或设置环境变量 TRAINKIT_HOME 指向更大的盘", "Free up space or point TRAINKIT_HOME at a bigger drive")))
    elif info.disk_free_gb < 20:
        f.append(Finding("warn", tr(f"磁盘剩余 {info.disk_free_gb} GB", f"{info.disk_free_gb} GB disk free"),
                         tr("数据集和训练记录会占用较多空间", "Datasets and runs can use a lot of space"),
                         tr("可设置 TRAINKIT_HOME 指向更大的盘", "Set TRAINKIT_HOME to a bigger drive if needed")))

    # Windows pitfalls
    if sys.platform == "win32":
        if info.windows_long_paths is False:
            f.append(Finding("warn", tr("Windows 长路径未开启", "Windows long paths are disabled"),
                             tr("部分依赖的文件路径超过 260 字符会安装失败", "Some packages fail to install when paths exceed 260 characters"),
                             'PowerShell (Admin): New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force'))
        if info.trainkit_home != str(info.user_home) and not info.trainkit_home.startswith(info.user_home):
            f.append(Finding("info", tr(f"数据目录：{info.trainkit_home}", f"Data folder: {info.trainkit_home}"),
                             tr("你的用户目录含中文/特殊字符或在 OneDrive 中，已改用这个更安全的位置", "Your user folder has non-ASCII characters or is synced by OneDrive, so a safer location is used")))

    # Tooling and network
    if not info.uv:
        f.append(Finding("error", tr("没有找到 uv", "uv not found"),
                         tr("TrainKit 用 uv 安装 Python 和依赖", "TrainKit uses uv to install Python and packages"),
                         tr("重新运行安装脚本，或执行 pip install uv", "Re-run the installer or `pip install uv`")))
    bad = [name for name, ok in info.network.items() if not ok]
    if bad:
        f.append(Finding("warn", tr("部分下载源无法访问：", "Some download sources are unreachable: ") + ", ".join(bad),
                         tr("可能是网络或防火墙问题", "This is usually a network or firewall issue"),
                         tr("可配置 PyPI 镜像：trainkit config pypi_index https://pypi.tuna.tsinghua.edu.cn/simple",
                            "Configure a PyPI mirror: trainkit config pypi_index <url>")))
    return f


def verdicts(info: SystemInfo, recipes: list[Recipe]) -> list[Verdict]:
    out = []
    blocked = nvidia_driver_blocked(info)
    for r in recipes:
        reasons: list[str] = []
        need_t, need_p = r.needs("train"), r.needs("preview")
        # Training
        if info.tier == "cpu":
            train = "no"
            reasons.append(tr("无 GPU：仅预览", "No GPU: preview only"))
        elif info.tier == "nvidia":
            vram = info.nvidia.vram_gb if info.nvidia else 0
            need_v = need_t.get("min_vram_gb", 0)
            if blocked:
                train = "no"
                reasons.append(blocked)
            elif vram >= need_v:
                train = "ok"
            elif vram >= need_v / 2:
                train = "warn"
                reasons.append(tr(f"显存 {vram} GB 偏小（建议 {need_v} GB），请用小模型/小 batch", f"{vram} GB VRAM is tight (want {need_v} GB); use a small model/batch"))
            else:
                train = "no"
                reasons.append(tr(f"显存不足（{vram} GB < {need_v} GB）", f"Not enough VRAM ({vram} GB < {need_v} GB)"))
        else:  # mps
            train = "ok" if info.ram_gb >= need_t.get("min_ram_gb", 0) else "warn"
            if train == "warn":
                reasons.append(tr(f"内存 {info.ram_gb} GB 偏小", f"{info.ram_gb} GB memory is tight"))
        if train != "no" and info.ram_gb < need_t.get("min_ram_gb", 0):
            train = "warn"
            reasons.append(tr(f"内存 {info.ram_gb} GB < {need_t['min_ram_gb']} GB", f"RAM {info.ram_gb} GB < {need_t['min_ram_gb']} GB"))
        if train != "no" and info.disk_free_gb < need_t.get("min_disk_gb", 0):
            train = "warn"
            reasons.append(tr("磁盘空间偏小", "Low disk space"))
        # Preview
        preview = "ok"
        if info.ram_gb < need_p.get("min_ram_gb", 0):
            preview = "warn"
            reasons.append(tr("内存偏小，预览可能卡顿", "Low RAM; preview may stutter"))
        if info.disk_free_gb < need_p.get("min_disk_gb", 0):
            preview = "no"
            reasons.append(tr("磁盘空间不足", "Not enough disk space"))
        out.append(Verdict(r.id, r.title, train, preview, reasons))
    return out


def recommended_profile(info: SystemInfo) -> str:
    return "preview" if info.tier == "cpu" or nvidia_driver_blocked(info) else "train"


def to_dict(findings: list[Finding], vs: list[Verdict]) -> dict:
    return {"findings": [asdict(x) for x in findings], "verdicts": [asdict(v) for v in vs]}
