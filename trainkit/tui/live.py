"""Terminal view of a training run. Reads the same events.jsonl as the web dashboard."""

from __future__ import annotations

import re
import time
import webbrowser

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ..i18n import tr
from ..runner import job, runs
from .keys import KeyReader

STATE_STYLE = {"running": "green", "paused": "yellow", "saving": "cyan", "saved": "cyan", "finished": "bold green",
               "failed": "bold red", "stopped": "red", "interrupted": "red", "preparing": "blue", "starting": "blue"}


def state_label(state: str) -> str:
    return {
        "starting": tr("启动中", "starting"), "preparing": tr("准备中", "preparing"), "running": tr("训练中", "running"),
        "paused": tr("已暂停", "paused"), "saving": tr("本轮结束后保存并退出", "saving after this epoch"),
        "saved": tr("已保存并退出", "saved"), "finished": tr("已完成", "finished"), "failed": tr("失败", "failed"),
        "stopped": tr("已停止", "stopped"), "interrupted": tr("已中断", "interrupted"),
    }.get(state, state)


def fmt_secs(s: float | None) -> str:
    if s is None:
        return "—"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def key_metrics(metrics: dict[str, float]) -> list[tuple[str, float]]:
    order = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50(M)", "metrics/mAP50-95(M)",
             "metrics/mAP50(P)", "metrics/mAP50-95(P)", "metrics/precision(B)", "metrics/recall(B)",
             "metrics/accuracy_top1", "metrics/accuracy_top5"]
    out = [(k.replace("metrics/", ""), metrics[k]) for k in order if k in metrics]
    return out[:6]


def render(run_id: str, meta: dict, s: runs.Summary, dash_url: str | None, log_tail: list[str]) -> Panel:
    head = Table.grid(expand=True)
    head.add_column()
    head.add_column(justify="right")
    title = Text.assemble((meta.get("name", run_id), "bold"), "  ", (meta.get("recipe", ""), "dim"))
    head.add_row(title, Text(state_label(s.state), style=STATE_STYLE.get(s.state, "")))
    dev = s.device_name or s.device or "—"
    head.add_row(Text(tr(f"设备 {dev} · batch {s.batch_size or '—'} · 数据 {meta.get('data_label', '')}",
                         f"device {dev} · batch {s.batch_size or '—'} · data {meta.get('data_label', '')}"), style="dim"),
                 Text(f"ETA {fmt_secs(s.eta_seconds() if s.state in runs.ACTIVE_STATES else None)}", style="dim"))

    bars = Table.grid(padding=(0, 1))
    bars.add_column(width=10)
    bars.add_column(ratio=1)
    bars.add_column(width=14, justify="right")
    ep_total = s.epochs or 1
    bars.add_row(tr("轮次", "Epoch"), ProgressBar(total=ep_total, completed=s.epoch), f"{s.epoch}/{s.epochs or '?'}")
    bars.add_row(tr("本轮", "Batch"), ProgressBar(total=s.nb or 1, completed=s.batch if s.current_epoch > s.epoch else 0),
                 f"{s.batch}/{s.nb or '?'}")

    stats = Table(box=None, show_header=True, header_style="dim", expand=True)
    for name, _ in list(s.losses.items())[:4]:
        stats.add_column(name.replace("train/", ""), justify="right")
    for name, _ in key_metrics(s.metrics):
        stats.add_column(name, justify="right")
    if stats.columns:
        stats.add_row(*[f"{v:.4f}" for v in list(s.losses.values())[:4]],
                      *[f"{v:.4f}" for _, v in key_metrics(s.metrics)])

    hints = Text(tr("[p] 暂停  [c] 继续  [s] 本轮结束后保存并退出  [x] 立即停止  [d] 打开网页面板  [q] 退出视图（训练继续）",
                    "[p] pause  [c] continue  [s] save & exit after epoch  [x] stop now  [d] open dashboard  [q] leave view (training continues)"),
                 style="dim")
    parts = [head, Text(""), bars, Text(""), stats]
    if s.error:
        parts += [Text(""), Text(s.error, style="red")]
    if log_tail:
        parts += [Text(""), Text("\n".join(log_tail[-3:]), style="dim", overflow="ellipsis", no_wrap=True)]
    if dash_url:
        parts += [Text(""), Text(tr("网页面板：", "Dashboard: ") + dash_url, style="cyan")]
    parts += [Text(""), hints]
    return Panel(Group(*parts), title="TrainKit", border_style="blue")


def _log_tail(rd, n: int = 3) -> list[str]:
    f = rd / "train.log"
    try:
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4000))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    lines = [ln.strip() for ln in text.replace("\r", "\n").splitlines() if ln.strip()]
    return lines[-n:]


def watch(run_id: str, dash_url: str | None = None, console: Console | None = None) -> str:
    """Show live progress until the run ends or the user presses q. Returns the final state."""
    from rich.live import Live

    console = console or Console()
    rd = runs.run_dir(run_id)
    meta = runs.read_meta(rd)
    s = runs.Summary()
    offset = 0
    message = ""
    with KeyReader() as keys, Live(console=console, refresh_per_second=4, transient=False) as live:
        while True:
            events, offset = runs.read_events(rd, offset)
            for ev in events:
                s.feed(ev)
            runs.reconcile(s, rd)
            job.reap()
            panel = render(run_id, meta, s, dash_url, _log_tail(rd))
            live.update(Group(panel, Text(message, style="yellow")) if message else panel)
            if s.state in runs.TERMINAL_STATES and not events:
                break
            key = keys.read()
            if key == "p":
                job.send(run_id, "pause")
                message = tr("已发送暂停：当前 batch 结束后暂停", "Pause sent: pauses after the current batch")
            elif key == "c":
                job.send(run_id, "continue")
                message = tr("已发送继续", "Continue sent")
            elif key == "s":
                job.send(run_id, "save_stop")
                message = tr("本轮结束后会保存并退出，之后用 trainkit resume 继续", "Will save & exit after this epoch; continue later with trainkit resume")
            elif key == "x":
                message = tr("正在停止…", "Stopping…")
                live.update(Group(panel, Text(message, style="yellow")))
                job.stop(run_id)
            elif key == "d" and dash_url:
                webbrowser.open(dash_url)
            elif key == "q":
                return "detached"
            time.sleep(0.25)
    return s.state
