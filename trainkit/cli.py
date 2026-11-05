"""TrainKit command line."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import __version__, paths
from . import datasets as ds
from . import project as proj
from .config import DEFAULTS, load_config, set_value
from .doctor import analyze, collect, recommended_profile, torch_backend_for, verdicts
from .envs import manager
from .i18n import tr
from .recipes import all_recipes, get_recipe

if sys.platform == "win32":  # make Chinese output work in old Windows consoles
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="TrainKit: set up, train and try YOLO models.")
datasets_app = typer.Typer(no_args_is_help=True, help="Manage datasets.")
app.add_typer(datasets_app, name="datasets")

LEVEL_ICON = {"ok": "[green]✓[/]", "info": "[blue]i[/]", "warn": "[yellow]![/]", "error": "[red]✗[/]"}
VERDICT_ICON = {"ok": "[green]✓[/]", "warn": "[yellow]![/]", "no": "[red]✗[/]"}


def fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]✗ {msg}[/]")
    raise typer.Exit(code)


def step(i: int, n: int, text: str) -> None:
    console.print(f"\n[bold cyan][{i}/{n}][/] [bold]{text}[/]")


def dim_line(line: str) -> None:
    if line.strip():
        console.print(f"    [dim]{line[:160]}[/]", highlight=False)


# ------------------------------------------------------------------------------------------------- doctor -----
def _print_system(info) -> None:
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row(tr("系统", "System"), f"{info.os} {info.os_version} ({info.arch})")
    t.add_row("CPU", f"{info.cpu} · {info.cpu_count} {tr('核', 'cores')}")
    t.add_row(tr("内存", "Memory"), f"{info.ram_gb} GB")
    t.add_row(tr("磁盘可用", "Disk free"), f"{info.disk_free_gb} GB")
    gpu = info.nvidia or info.apple
    t.add_row("GPU", f"{gpu.name}" + (f" · {gpu.vram_gb} GB" if info.nvidia else "") if gpu else tr("无", "none"))
    if info.nvidia:
        t.add_row(tr("驱动", "Driver"), f"{info.nvidia.driver} (CUDA ≤ {info.nvidia.cuda_driver or '?'})")
    tier = {"nvidia": tr("NVIDIA 训练档", "NVIDIA training tier"), "mps": tr("Apple MPS 训练档", "Apple MPS training tier"),
            "cpu": tr("CPU 预览档（只运行模型，不训练）", "CPU preview tier (run models, no training)")}[info.tier]
    t.add_row(tr("档位", "Tier"), f"[bold]{tier}[/]")
    t.add_row(tr("数据目录", "Data folder"), info.trainkit_home)
    if info.cameras:
        t.add_row(tr("摄像头", "Cameras"), ", ".join(info.cameras))
    console.print(t)


@app.command()
def doctor(json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
           no_network: bool = typer.Option(False, "--no-network", help="Skip download source checks.")):
    """Check this computer and show which models it can train or preview."""
    info = collect(check_network=not no_network)
    findings = analyze(info)
    vs = verdicts(info, all_recipes())
    if json_out:
        from .doctor.report import to_dict

        print(json.dumps({"system": info.to_dict(), **to_dict(findings, vs),
                          "torch_backend": torch_backend_for(info), "profile": recommended_profile(info)},
                         indent=2, ensure_ascii=False))
        return
    console.rule(f"TrainKit {__version__} · doctor")
    _print_system(info)
    console.print()
    for f in findings:
        console.print(f"{LEVEL_ICON[f.level]} {f.title}")
        if f.detail:
            console.print(f"   [dim]{f.detail}[/]")
        if f.fix:
            console.print(f"   [cyan]→ {f.fix}[/]")
    console.print()
    t = Table(title=tr("模型兼容性", "Model compatibility"), title_justify="left")
    t.add_column(tr("配方", "Recipe"))
    t.add_column(tr("名称", "Name"))
    t.add_column(tr("训练", "Train"), justify="center")
    t.add_column(tr("预览", "Preview"), justify="center")
    t.add_column(tr("说明", "Notes"), style="dim")
    for v in vs:
        t.add_row(v.recipe_id, v.title, VERDICT_ICON[v.train], VERDICT_ICON[v.preview], "; ".join(v.reasons))
    console.print(t)
    env_state = manager.describe("ultralytics")
    console.print(f"\n{tr('YOLO 环境', 'YOLO environment')}: [bold]{env_state}[/]")
    if not manager.is_ready("ultralytics"):
        prof = recommended_profile(info)
        console.print(tr(f"下一步：[bold]trainkit setup yolo-detect --profile {prof}[/]",
                         f"Next: [bold]trainkit setup yolo-detect --profile {prof}[/]"))


@app.command()
def recipes():
    """List the model recipes TrainKit can set up."""
    t = Table()
    t.add_column("id")
    t.add_column(tr("名称", "Name"))
    t.add_column(tr("默认权重", "Default weights"))
    t.add_column(tr("说明", "About"), style="dim")
    for r in all_recipes():
        t.add_row(r.id, r.title, r.default_weights, r.summary)
    console.print(t)


# ------------------------------------------------------------------------------------------------- setup ------
def _resolve_target(target: Optional[str]) -> tuple:
    """Returns (recipe, project_dict_or_None)."""
    if target and proj.is_project_file(target):
        p = proj.load(target)
        return get_recipe(p["recipe"]), p
    if not target:
        rs = all_recipes()
        t = Table(show_header=False, box=None)
        for i, r in enumerate(rs, 1):
            t.add_row(f"[bold]{i}[/]", r.title, f"[dim]{r.summary}[/]")
        console.print(t)
        choice = Prompt.ask(tr("选择要使用的模型", "Choose a model"), choices=[str(i) for i in range(1, len(rs) + 1)],
                            default="1")
        return rs[int(choice) - 1], None
    try:
        return get_recipe(target), None
    except KeyError as e:
        fail(str(e))


def _prepare_dataset(spec: str, recipe, env_ready: bool, name: str | None = None) -> ds.Resolved:
    try:
        resolved = ds.resolve(spec, recipe.task, log=dim_line, name=name)
    except ds.DatasetError as e:
        fail(str(e))
    report = ds.check(resolved, recipe.task)
    if report:
        _print_report(report)
        if not report.ok:
            fail(tr("数据集有问题，请先修正", "The dataset has problems; fix them first"))
    elif resolved.kind == "builtin" and env_ready:
        console.print(f"    {tr('下载/检查内置数据集', 'Downloading/checking built-in dataset')} [bold]{spec}[/] …")
        try:
            r = manager.run_task(recipe.env, "dataset", spec, "--task", recipe.task, on_line=dim_line)
            names = r.get("names") or {}
            console.print(f"    [green]✓[/] {len(names)} {tr('个类别', 'classes')}")
        except manager.EnvError as e:
            fail(str(e))
    return resolved


def _print_report(rep) -> None:
    st = rep.stats
    if rep.task == "classify":
        console.print(f"    {tr('训练', 'train')} {st.get('train_images', 0)} · {tr('验证', 'val')} {st.get('val_images', 0)} · "
                      f"{st.get('nc', 0)} {tr('类', 'classes')}")
    else:
        console.print(f"    {tr('训练', 'train')} {st.get('train_images', 0)} · {tr('验证', 'val')} {st.get('val_images', 0)} · "
                      f"{st.get('nc', 0)} {tr('类', 'classes')}")
        inst = st.get("instances") or {}
        if inst:
            top = ", ".join(f"{k}:{v}" for k, v in list(inst.items())[:8])
            console.print(f"    [dim]{tr('标注数', 'instances')}: {top}{' …' if len(inst) > 8 else ''}[/]")
    for w in rep.warnings:
        console.print(f"    [yellow]! {w}[/]")
    for e in rep.errors:
        console.print(f"    [red]✗ {e}[/]")
    if rep.ok:
        console.print(f"    [green]✓ {tr('数据集检查通过', 'Dataset looks good')}[/]")


@app.command()
def setup(
    target: Optional[str] = typer.Argument(None, help="Recipe id (see `trainkit recipes`) or a trainkit.yaml file."),
    profile: Optional[str] = typer.Option(None, help="train | preview (default: picked from your hardware)."),
    torch_backend: Optional[str] = typer.Option(None, help="Override PyTorch build: auto, cpu, cu126, cu128, ..."),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Also prepare this dataset."),
    skip_smoke: bool = typer.Option(False, help="Skip the 1-epoch smoke test."),
    reinstall: bool = typer.Option(False, help="Delete and rebuild the environment."),
    force: bool = typer.Option(False, help="Continue even if the doctor reports errors."),
):
    """Install everything a model needs: Python, PyTorch (matched to your GPU), packages, weights, dataset."""
    recipe, project = _resolve_target(target)
    if project and not dataset:
        dataset = project.get("dataset")
    weights = (project or {}).get("weights") or recipe.default_weights
    n = 6
    step(1, n, tr("检测设备", "Checking this computer"))
    info = collect(check_network=True)
    _print_system(info)
    errors = [f for f in analyze(info) if f.level == "error"]
    for f in errors:
        console.print(f"  {LEVEL_ICON['error']} {f.title}  [cyan]{f.fix}[/]")
    if errors and not force:
        fail(tr("请先解决上面的问题（或加 --force 继续）", "Fix the problems above first (or pass --force)"))
    rec_profile = recommended_profile(info)
    profile = profile or rec_profile
    if profile not in ("train", "preview"):
        fail("--profile must be train or preview")
    if profile == "train" and rec_profile == "preview":
        console.print(tr("  [yellow]! 这台电脑没有可用 GPU，CPU 训练会非常慢（仅适合测试）[/]",
                         "  [yellow]! No usable GPU here: CPU training is very slow (testing only)[/]"))
    backend = torch_backend or torch_backend_for(info)
    console.print(f"  {tr('模式', 'Profile')}: [bold]{profile}[/] · PyTorch backend: [bold]{backend if profile == 'train' else 'cpu'}[/]")

    step(2, n, tr("创建 Python 环境并安装依赖（首次约 2–10 分钟）", "Creating the Python environment and installing packages (2–10 min the first time)"))
    t0 = time.time()
    try:
        manager.create(recipe.env, profile, backend, on_line=dim_line, reinstall=reinstall)
    except manager.EnvError as e:
        fail(str(e) + "\n" + tr("提示：网络问题可配置镜像 trainkit config pypi_index <url>", "Tip: behind a firewall, set a mirror with trainkit config pypi_index <url>"))
    console.print(f"  [green]✓[/] {tr('完成', 'done')} ({time.time() - t0:.0f}s)")

    step(3, n, tr("自检：PyTorch 与设备", "Self-test: PyTorch and device"))
    try:
        st = manager.run_task(recipe.env, "selftest")
    except manager.EnvError as e:
        fail(str(e))
    console.print(f"  torch {st.get('torch')} · device [bold]{st.get('device')}[/]"
                  + (f" ({st.get('device_name')})" if st.get("device_name") else "")
                  + f" · ultralytics {st.get('ultralytics', '?')} · gradio {st.get('gradio', '?')}")
    if not st.get("tensor_ok"):
        hint = ""
        if "no kernel image" in st.get("tensor_error", ""):
            hint = tr("  → 显卡太老，试试：trainkit setup --reinstall --torch-backend cu126", "  → GPU too old for this build; try: trainkit setup --reinstall --torch-backend cu126")
        fail(tr("在设备上运行张量失败：", "Running a tensor on the device failed: ") + st.get("tensor_error", "") + hint)
    if profile == "train" and info.tier == "nvidia" and not st.get("cuda_available"):
        fail(tr("装好的 PyTorch 看不到 GPU。试试：trainkit setup --reinstall --torch-backend cu126，或更新显卡驱动",
                "PyTorch cannot see the GPU. Try: trainkit setup --reinstall --torch-backend cu126, or update the driver"))
    for mod in ("ultralytics", "gradio"):
        if st.get(f"{mod}_error"):
            fail(f"{mod}: {st[f'{mod}_error']}")
    console.print(f"  [green]✓[/] {tr('设备可用', 'Device works')}" + (f" ({st.get('matmul_ms')} ms)" if st.get("matmul_ms") else ""))

    step(4, n, tr(f"下载预训练权重 {weights}", f"Downloading pretrained weights {weights}"))
    try:
        w = manager.run_task(recipe.env, "weights", weights, "--model-class", recipe.model_class, on_line=dim_line)
        console.print(f"  [green]✓[/] {w.get('path')}")
    except manager.EnvError as e:
        fail(str(e))

    step(5, n, tr("准备数据集", "Preparing the dataset"))
    if dataset:
        resolved = _prepare_dataset(dataset, recipe, True, name=(project or {}).get("name"))
        if resolved.kind != "builtin":
            reg_name = (project or {}).get("name") or resolved.label
            ds.register(reg_name, resolved.data, recipe.task, dataset)
            console.print(f"  {tr('已登记为', 'Registered as')} [bold]{reg_name}[/]")
    smoke_data = recipe.smoke_dataset
    if profile == "train" or not dataset:
        _prepare_dataset(smoke_data, recipe, True)

    step(6, n, tr("冒烟测试" if profile == "train" else "推理测试", "Smoke test" if profile == "train" else "Inference test"))
    try:
        if profile == "train" and not skip_smoke:
            console.print(tr(f"  用 {smoke_data} 训练 1 轮，验证整条流程…", f"  Training 1 epoch on {smoke_data} to prove the whole chain…"))
            r = manager.run_task(recipe.env, "smoke", "--weights", weights, "--data", smoke_data, "--task", recipe.task,
                                 "--model-class", recipe.model_class, on_line=None)
            if not r.get("ok"):
                fail(tr("冒烟测试没有产生检查点", "Smoke test did not produce a checkpoint"))
            console.print(f"  [green]✓[/] {tr('训练流程正常', 'Training works')} · {r.get('device')} · {r.get('seconds')}s")
        else:
            r = manager.run_task(recipe.env, "predict-test", "--weights", weights, "--model-class", recipe.model_class)
            console.print(f"  [green]✓[/] {tr('推理正常', 'Inference works')} · {r.get('ms')} ms/{tr('张', 'image')}")
    except manager.EnvError as e:
        fail(str(e))
    manager.mark_ready(recipe.env, st)
    console.rule(tr("[green]准备完成[/]", "[green]Ready[/]"))
    if profile == "train":
        data_hint = dataset or smoke_data
        tgt = target if project else recipe.id
        console.print(tr(f"开始训练：[bold]trainkit train {tgt}{'' if project else ' --data ' + str(data_hint)}[/]",
                         f"Start training: [bold]trainkit train {tgt}{'' if project else ' --data ' + str(data_hint)}[/]"))
    console.print(tr("试玩模型：[bold]trainkit play[/]", "Try models: [bold]trainkit play[/]"))


# ------------------------------------------------------------------------------------------------- datasets ---
@datasets_app.command("list")
def datasets_list():
    """Show registered datasets and the built-in ones for each recipe."""
    reg = ds.registry()
    if reg:
        t = Table(title=tr("已登记的数据集", "Registered datasets"), title_justify="left")
        t.add_column(tr("名称", "Name"))
        t.add_column(tr("任务", "Task"))
        t.add_column(tr("来源", "Source"), style="dim")
        for name, e in reg.items():
            t.add_row(name, e["task"], e["source"])
        console.print(t)
    t = Table(title=tr("内置数据集（首次使用时自动下载）", "Built-in datasets (downloaded on first use)"), title_justify="left")
    t.add_column(tr("配方", "Recipe"))
    t.add_column(tr("名称", "Name"))
    t.add_column(tr("说明", "Notes"), style="dim")
    for r in all_recipes():
        for d in r.to_dict()["builtin_datasets"]:
            t.add_row(r.id, d["name"], d["note"])
    console.print(t)


@datasets_app.command("add")
def datasets_add(name: str, source: str = typer.Argument(..., help="Folder, .zip, data.yaml or roboflow:ws/project/version"),
                 recipe: str = typer.Option("yolo-detect", help="Recipe the dataset is for.")):
    """Prepare a dataset, check it, and register it under a short name."""
    r = get_recipe(recipe)
    resolved = _prepare_dataset(source, r, manager.is_ready(r.env), name=name)
    ds.register(name, resolved.data, r.task, source)
    console.print(f"[green]✓[/] {tr('已登记', 'Registered')} [bold]{name}[/] → {resolved.data}")


@datasets_app.command("check")
def datasets_check(source: str, recipe: str = typer.Option("yolo-detect")):
    """Validate a dataset without registering it."""
    r = get_recipe(recipe)
    _prepare_dataset(source, r, manager.is_ready(r.env))


@datasets_app.command("remove")
def datasets_remove(name: str):
    """Forget a registered dataset (files are not deleted)."""
    console.print("[green]✓[/]" if ds.unregister(name) else tr("没有这个名称", "No such name"))


# ------------------------------------------------------------------------------------------------- train ------
def _start_dashboard_thread(port: int, host: str = "127.0.0.1", token: str | None = None) -> str:
    import httpx

    base = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    try:
        r = httpx.get(base + "/api/ping", timeout=1)
        if r.status_code in (200, 401) and "trainkit" in r.text:
            return base  # already running (another terminal)
    except Exception:
        pass
    import uvicorn

    from .server.app import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(token=token), host=host, port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(40):
        if server.started:
            break
        time.sleep(0.1)
    return base + (f"/?token={token}" if token else "")


def _parse_overrides(epochs, imgsz, batch, device, workers, patience, extra: list[str]) -> dict:
    ov: dict = {}
    for k, v in (("epochs", epochs), ("imgsz", imgsz), ("batch", batch), ("device", device), ("workers", workers),
                 ("patience", patience)):
        if v is not None:
            ov[k] = v
    for item in extra or []:
        if "=" not in item:
            fail(f"--set expects key=value, got {item}")
        k, v = item.split("=", 1)
        try:
            ov[k] = json.loads(v)
        except ValueError:
            ov[k] = v
    return ov


@app.command()
def train(
    target: Optional[str] = typer.Argument(None, help="Recipe id or trainkit.yaml."),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="Dataset: registered name, built-in, folder, zip, yaml, roboflow:…"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Starting weights (default from recipe)."),
    epochs: Optional[int] = typer.Option(None),
    imgsz: Optional[int] = typer.Option(None),
    batch: Optional[float] = typer.Option(None, help="-1 = auto from GPU memory."),
    device: Optional[str] = typer.Option(None, help="0, mps or cpu (default: best available)."),
    workers: Optional[int] = typer.Option(None),
    patience: Optional[int] = typer.Option(None),
    set_: Optional[list[str]] = typer.Option(None, "--set", help="Any Ultralytics train argument, e.g. --set lr0=0.005"),
    name: Optional[str] = typer.Option(None, help="Run name."),
    samples_every: int = typer.Option(1, help="Save sample predictions every N epochs (0 = off)."),
    dash: bool = typer.Option(True, help="Start the web dashboard."),
    tui: bool = typer.Option(True, help="Show the live terminal view."),
    open_browser: bool = typer.Option(False, "--open", help="Open the dashboard in a browser."),
    port: Optional[int] = typer.Option(None),
):
    """Start training with a live terminal view and web dashboard (pause / continue / save & exit / stop)."""
    from .runner import job

    recipe, project = _resolve_target(target)
    p = project or {}
    data = data or p.get("dataset")
    if not data:
        fail(tr("请用 --data 指定数据集（trainkit datasets list 查看可选项）", "Pass a dataset with --data (see trainkit datasets list)"))
    if not manager.is_ready(recipe.env):
        fail(tr(f"环境还没装好，先运行：trainkit setup {recipe.id}", f"Environment not ready. Run: trainkit setup {recipe.id}"))
    m = manager.load_manifest(recipe.env) or {}
    if m.get("profile") == "preview":
        console.print(tr("[yellow]! 当前是 CPU 预览环境，训练会很慢。有 GPU 的话请运行 trainkit setup --profile train --reinstall[/]",
                         "[yellow]! This is a CPU preview environment; training will be slow. With a GPU run trainkit setup --profile train --reinstall[/]"))
    batch_v = int(batch) if batch is not None and float(batch).is_integer() else batch
    ov = {**p.get("train", {}), **_parse_overrides(epochs, imgsz, batch_v, device, workers, patience, set_ or [])}
    console.print(tr("检查数据集…", "Checking the dataset…"))
    resolved = _prepare_dataset(data, recipe, True)
    try:
        rid = job.start(recipe, resolved.data, resolved.label, weights or p.get("weights"), ov, name or p.get("name"),
                        samples_every)
    except manager.EnvError as e:
        fail(str(e))
    console.print(f"[green]✓[/] {tr('训练已启动', 'Training started')}: [bold]{rid}[/]")
    url = None
    if dash:
        url = _start_dashboard_thread(port or load_config()["dashboard_port"])
        url = f"{url.rstrip('/')}/#/run/{rid}" if "token=" not in url else url
        console.print(f"{tr('网页面板', 'Dashboard')}: [cyan]{url}[/]")
        if open_browser:
            webbrowser.open(url)
    _follow(rid, url, tui, dash)


def _follow(rid: str, url: str | None, tui: bool, dash: bool) -> None:
    from .runner import job, runs
    from .tui.live import state_label, watch

    if tui and sys.stdin.isatty():
        state = watch(rid, url, console)
    else:
        console.print(tr("训练在后台运行。Ctrl+C 只会退出这个窗口。", "Training runs in the background. Ctrl+C only closes this window."))
        try:
            while True:
                time.sleep(2)
                job.reap()
                s, _ = runs.summarize(runs.run_dir(rid))
                console.print(f"  {state_label(s.state)} · epoch {s.epoch}/{s.epochs or '?'} · batch {s.batch}/{s.nb or '?'}",
                              highlight=False)
                if s.state in runs.TERMINAL_STATES:
                    state = s.state
                    break
        except KeyboardInterrupt:
            state = "detached"
    if state == "detached":
        console.print(tr(f"已退出视图，训练仍在继续。查看：trainkit watch {rid} · 停止：trainkit ctl {rid} stop",
                         f"Left the view; training continues. Watch: trainkit watch {rid} · stop: trainkit ctl {rid} stop"))
        return
    console.print(f"{tr('结果', 'Result')}: [bold]{state_label(state)}[/]")
    rd = runs.run_dir(rid)
    if state in ("saved", "stopped", "interrupted"):
        console.print(tr(f"继续训练：trainkit resume {rid}", f"Continue training: trainkit resume {rid}"))
    if state == "finished":
        best = runs.best_weights(rd)
        console.print(tr(f"最佳模型：{best}\n试玩：trainkit play {rid}    打包分享：trainkit pack {rid}",
                         f"Best model: {best}\nTry it: trainkit play {rid}    Share: trainkit pack {rid}"))
    if state == "failed":
        s, _ = runs.summarize(rd)
        console.print(f"[red]{s.error}[/]\n{tr('日志', 'Log')}: {rd / 'train.log'}")
    if dash and url and sys.stdin.isatty():
        try:
            Prompt.ask(tr("按 Enter 关闭网页面板", "Press Enter to close the dashboard"), default="", show_default=False)
        except (KeyboardInterrupt, EOFError):
            pass
    if state == "failed":
        raise typer.Exit(1)


@app.command()
def resume(run_id: str, tui: bool = typer.Option(True), dash: bool = typer.Option(True),
           device: Optional[str] = typer.Option(None), batch: Optional[int] = typer.Option(None)):
    """Continue a run that was saved, stopped or interrupted, from its last completed epoch."""
    from .runner import job

    ov = {k: v for k, v in (("device", device), ("batch", batch)) if v is not None}
    try:
        job.resume(run_id, ov)
    except (RuntimeError, manager.EnvError, ValueError) as e:
        fail(str(e))
    console.print(f"[green]✓[/] {tr('继续训练', 'Resumed')}: [bold]{run_id}[/]")
    url = _start_dashboard_thread(load_config()["dashboard_port"]) if dash else None
    if url:
        url = f"{url}/#/run/{run_id}"
        console.print(f"{tr('网页面板', 'Dashboard')}: [cyan]{url}[/]")
    _follow(run_id, url, tui, dash)


@app.command()
def watch(run_id: str):
    """Re-open the live terminal view of a run."""
    from .tui.live import watch as _watch

    _watch(run_id, None, console)


@app.command()
def ctl(run_id: str, command: str = typer.Argument(..., help="pause | continue | save-stop | stop")):
    """Control a running training from another terminal."""
    from .runner import job

    cmd = command.replace("-", "_")
    try:
        result = job.control(run_id, cmd)
    except ValueError:
        fail("command must be pause, continue, save-stop or stop")
    console.print(f"[green]✓[/] {cmd}: {result}")


@app.command("runs")
def runs_cmd(limit: int = typer.Option(20)):
    """List training runs."""
    from .runner import runs as _runs
    from .tui.live import state_label

    items = _runs.list_runs()[:limit]
    if not items:
        console.print(tr("还没有训练记录", "No runs yet"))
        return
    t = Table()
    t.add_column("id")
    t.add_column(tr("配方", "Recipe"))
    t.add_column(tr("数据", "Data"))
    t.add_column(tr("状态", "State"))
    t.add_column(tr("轮次", "Epoch"), justify="right")
    t.add_column("mAP50-95 / top1", justify="right")
    for r in items:
        m = r.get("metrics") or {}
        score = m.get("metrics/mAP50-95(B)", m.get("metrics/accuracy_top1"))
        state = state_label(r["state"]) + (" ↻" if r.get("resumable") else "")
        t.add_row(r["id"], r.get("recipe", ""), r.get("data_label", ""), state, f"{r['epoch']}/{r['epochs'] or '?'}",
                  f"{score:.3f}" if score is not None else "—")
    console.print(t)
    console.print(tr("[dim]↻ = 可以用 trainkit resume <id> 继续[/]", "[dim]↻ = can be continued with trainkit resume <id>[/]"))


# ------------------------------------------------------------------------------------------------- dashboard --
@app.command()
def dash(port: Optional[int] = typer.Option(None), lan: bool = typer.Option(False, help="Let teammates on your network view it (adds an access token)."),
         open_browser: bool = typer.Option(True, "--open/--no-open")):
    """Open the web dashboard: all runs, live charts, controls, and a form to start new training."""
    import secrets

    import uvicorn

    from .server.app import create_app

    port = port or load_config()["dashboard_port"]
    host = "0.0.0.0" if lan else "127.0.0.1"
    token = secrets.token_urlsafe(12) if lan else None
    url = f"http://127.0.0.1:{port}" + (f"/?token={token}" if token else "")
    console.print(f"{tr('网页面板', 'Dashboard')}: [cyan]{url}[/]")
    if lan:
        import socket

        try:
            ip = socket.gethostbyname(socket.gethostname())
            console.print(f"{tr('局域网地址', 'LAN address')}: [cyan]http://{ip}:{port}/?token={token}[/]")
        except OSError:
            pass
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(token=token), host=host, port=port, log_level="warning")


# ------------------------------------------------------------------------------------------------- playground -
def _ensure_play_env(yes: bool) -> str:
    env = "ultralytics"
    if manager.is_ready(env):
        return env
    console.print(tr("还没有可用的模型环境。", "No model environment yet."))
    if yes or Confirm.ask(tr("现在安装一个轻量的 CPU 预览环境吗？（约 1GB）", "Install a lightweight CPU preview environment now? (~1 GB)"), default=True):
        ctx_args = dict(target="yolo-detect", profile="preview", torch_backend=None, dataset=None, skip_smoke=True,
                        reinstall=False, force=True)
        setup(**ctx_args)
        return env
    raise typer.Exit(1)


@app.command()
def play(model: Optional[str] = typer.Argument(None, help="Run id, .pt/.onnx file, OpenVINO folder or model bundle .zip."),
         source: Optional[str] = typer.Option(None, help="Start with this source: camera index, RTSP URL, or video file."),
         port: Optional[int] = typer.Option(None), lan: bool = typer.Option(False),
         open_browser: bool = typer.Option(True, "--open/--no-open"), yes: bool = typer.Option(False, "-y")):
    """Try models live: browser camera, USB camera, RTSP, video or images. Works without the training part."""
    from . import models

    env = _ensure_play_env(yes)
    args = ["--port", str(port or load_config()["playground_port"]), "--host", "0.0.0.0" if lan else "127.0.0.1",
            "--lang", "zh" if tr("zh", "en") == "zh" else "en"]
    if model:
        try:
            args += ["--model", str(models.resolve(model))]
        except models.ModelError as e:
            fail(str(e))
    if source:
        args += ["--source", source]
    if open_browser:
        args.append("--open")
    script = paths.package_dir() / "playground" / "app.py"
    console.print(tr("启动试玩区（Ctrl+C 退出）…", "Starting the playground (Ctrl+C to quit)…"))
    try:
        subprocess.run(manager.python_cmd(env, script, *args), env=manager.subprocess_env(env), cwd=str(paths.weights_dir()))
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------------------------------------- export -----
@app.command()
def export(model: str = typer.Argument(..., help="Run id or model path."),
           format: str = typer.Option("onnx", "--format", "-f", help="onnx | openvino | ncnn | engine (TensorRT) | tflite | coreml"),
           imgsz: Optional[int] = typer.Option(None), half: bool = typer.Option(False), int8: bool = typer.Option(False),
           device: Optional[str] = typer.Option(None)):
    """Convert a model for deployment (e.g. ONNX, OpenVINO for CPUs, NCNN for Raspberry Pi, TensorRT for Jetson)."""
    from . import models

    try:
        path = models.resolve(model)
    except models.ModelError as e:
        fail(str(e))
    if not manager.is_ready("ultralytics"):
        fail(tr("先运行 trainkit setup", "Run trainkit setup first"))
    args = [str(path), "--format", format, "--model-class", models.model_class_for(path)]
    if imgsz:
        args += ["--imgsz", str(imgsz)]
    if half:
        args.append("--half")
    if int8:
        args.append("--int8")
    if device:
        args += ["--device", device]
    console.print(tr(f"导出 {path.name} → {format}（缺少的转换工具会自动安装）…", f"Exporting {path.name} → {format} (missing converters install automatically)…"))
    try:
        r = manager.run_task("ultralytics", "export", *args, on_line=dim_line)
    except manager.EnvError as e:
        fail(str(e))
    console.print(f"[green]✓[/] {r['path']}")


@app.command()
def pack(run_id: str, output: Optional[Path] = typer.Option(None, "-o", help="Output .zip path."),
         with_onnx: bool = typer.Option(False, help="Also include an ONNX export (for CPU-only teammates).")):
    """Bundle a trained model (weights + class names + metrics) into one .zip teammates can `trainkit play`."""
    from . import models
    from .runner import runs as _runs

    include = []
    if with_onnx:
        w = _runs.best_weights(_runs.run_dir(run_id))
        r = manager.run_task("ultralytics", "export", str(w), "--format", "onnx", on_line=dim_line)
        include.append(Path(r["path"]))
    try:
        out = models.pack(run_id, output, include)
    except models.ModelError as e:
        fail(str(e))
    console.print(f"[green]✓[/] {out}")
    console.print(tr(f"组员可以运行：trainkit play {out.name}", f"Teammates can run: trainkit play {out.name}"))


# ------------------------------------------------------------------------------------------------- misc -------
@app.command()
def init(path: Path = typer.Argument(Path("trainkit.yaml")), recipe: str = typer.Option("yolo-detect"),
         dataset: str = typer.Option("coco128.yaml"), name: Optional[str] = typer.Option(None)):
    """Create a trainkit.yaml project file to share with teammates."""
    if path.exists() and not Confirm.ask(tr(f"{path} 已存在，覆盖吗？", f"{path} exists. Overwrite?"), default=False):
        raise typer.Exit()
    r = get_recipe(recipe)
    proj.write_template(path, name or Path.cwd().name, r.id, r.default_weights, dataset,
                        int(r.defaults.get("epochs", 100)), int(r.defaults.get("imgsz", 640)))
    console.print(f"[green]✓[/] {path}")
    console.print(tr(f"组员只需：trainkit setup {path} && trainkit train {path}", f"Teammates run: trainkit setup {path} && trainkit train {path}"))


@app.command()
def config(key: Optional[str] = typer.Argument(None), value: Optional[str] = typer.Argument(None)):
    """Show or change settings (lang, pypi_index, roboflow_api_key, dashboard_port, playground_port)."""
    if key is None:
        cfg = load_config()
        for k in DEFAULTS:
            v = cfg.get(k)
            if k.endswith("api_key") and v:
                v = v[:4] + "…"
            console.print(f"{k} = {v!r}")
        console.print(f"[dim]{paths.config_file()}[/]")
        return
    if value is None:
        fail("usage: trainkit config <key> <value>")
    try:
        set_value(key, value)
    except KeyError:
        fail(f"unknown key {key}; choose from {', '.join(DEFAULTS)}")
    console.print(f"[green]✓[/] {key}")


@app.command()
def version():
    """Print the TrainKit version."""
    console.print(f"TrainKit {__version__}")


if __name__ == "__main__":
    app()
