"""Training runs on disk: run.json (metadata) + events.jsonl (written by the training process)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import psutil

from .. import paths

ACTIVE_STATES = {"starting", "preparing", "running", "paused", "saving"}
TERMINAL_STATES = {"finished", "saved", "stopped", "failed", "interrupted"}


def run_dir(run_id: str) -> Path:
    d = (paths.runs_dir() / run_id).resolve()
    if paths.runs_dir().resolve() not in d.parents:
        raise ValueError("invalid run id")
    return d


def new_run_id(recipe_id: str, name: str | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "".join(c if c.isalnum() or c in "-_" else "-" for c in (name or recipe_id))[:40]
    rid = f"{stamp}-{suffix}"
    i = 2
    while (paths.runs_dir() / rid).exists():
        rid = f"{stamp}-{suffix}-{i}"
        i += 1
    return rid


def read_meta(rd: Path) -> dict[str, Any]:
    try:
        return json.loads((rd / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_meta(rd: Path, meta: dict[str, Any]) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    tmp = rd / "run.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(rd / "run.json")


def read_events(rd: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Read complete JSON lines starting at byte `offset`. Returns (events, new_offset)."""
    f = rd / "events.jsonl"
    if not f.exists():
        return [], offset
    with open(f, "rb") as fh:
        fh.seek(offset)
        chunk = fh.read()
    if not chunk:
        return [], offset
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset  # only a partial line so far
    events = []
    for line in chunk[: end + 1].splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events, offset + end + 1


def append_event(rd: Path, type_: str, **data: Any) -> None:
    rec = {"t": round(time.time(), 3), "type": type_, **data}
    with open(rd / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


class Summary:
    """Folds the event stream into the current picture of a run."""

    def __init__(self) -> None:
        self.state = "starting"
        self.msg = ""
        self.epochs = 0
        self.epoch = 0  # last completed epoch
        self.current_epoch = 0
        self.batch = 0
        self.nb = 0
        self.rate = 0.0
        self.device = ""
        self.device_name = ""
        self.batch_size = None
        self.save_dir = ""
        self.task = ""
        self.names = None
        self.pid = None
        self.losses: dict[str, float] = {}
        self.metrics: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []
        self.final: dict[str, float] = {}
        self.best = None
        self.checkpoint_epoch = 0
        self.error = ""
        self.epoch_seconds: list[float] = []
        self.started = None
        self.updated = None
        self.samples: list[str] = []

    def feed(self, ev: dict[str, Any]) -> None:
        t = ev.get("type")
        self.updated = ev.get("t", self.updated)
        if self.started is None:
            self.started = ev.get("t")
        if t == "process":
            self.pid = ev.get("pid")
            self.error = ""
        elif t == "status":
            self.state = ev.get("state", self.state)
            self.msg = ev.get("msg", "")
        elif t == "start":
            self.epochs = ev.get("epochs") or self.epochs
            self.nb = ev.get("nb") or self.nb
            self.device = ev.get("device", "")
            self.device_name = ev.get("device_name", "")
            self.batch_size = ev.get("batch_size")
            self.save_dir = ev.get("save_dir", "")
            self.task = ev.get("task", "")
            self.names = ev.get("names")
            self.epoch = max(self.epoch, (ev.get("start_epoch") or 0))
        elif t == "epoch_start":
            self.current_epoch = ev.get("epoch", 0)
            self.epochs = ev.get("epochs") or self.epochs
            self.batch = 0
        elif t == "batch":
            self.current_epoch = ev.get("epoch", self.current_epoch)
            self.batch = ev.get("batch", 0)
            self.nb = ev.get("nb") or self.nb
            self.rate = ev.get("rate") or self.rate
            self.losses = ev.get("losses") or self.losses
        elif t == "epoch":
            e = ev.get("epoch", 0)
            self.epoch = e
            self.metrics = ev.get("metrics") or {}
            self.history = [h for h in self.history if h["epoch"] < e]
            self.history.append({"epoch": e, "metrics": self.metrics, "losses": ev.get("losses") or {},
                                 "lr": ev.get("lr") or {}, "seconds": ev.get("seconds")})
            if ev.get("seconds"):
                self.epoch_seconds.append(float(ev["seconds"]))
            if ev.get("samples"):
                self.samples = ev["samples"]
        elif t == "checkpoint":
            self.checkpoint_epoch = ev.get("epoch", 0)
            self.best = ev.get("best") or self.best
        elif t == "final":
            self.final = ev.get("metrics") or {}
            self.best = ev.get("best") or self.best
        elif t == "error":
            self.error = ev.get("msg", "")

    def eta_seconds(self) -> float | None:
        if not self.epochs:
            return None
        per_epoch = None
        if self.epoch_seconds:
            recent = self.epoch_seconds[-5:]
            per_epoch = sum(recent) / len(recent)
        elif self.rate and self.nb:
            per_epoch = self.nb / self.rate
        if not per_epoch:
            return None
        done_in_epoch = (self.batch / self.nb) if self.nb else 0
        remaining_epochs = max(self.epochs - self.epoch - (done_in_epoch if self.current_epoch > self.epoch else 0), 0)
        return remaining_epochs * per_epoch

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "epoch_seconds"}
        d["eta_seconds"] = self.eta_seconds()
        return d


def summarize(rd: Path) -> tuple[Summary, int]:
    s = Summary()
    events, offset = read_events(rd)
    for ev in events:
        s.feed(ev)
    reconcile(s, rd)
    return s, offset


def reconcile(s: Summary, rd: Path) -> None:
    """A run whose process died without a final status was interrupted (crash, reboot, killed)."""
    if s.state in ACTIVE_STATES and not pid_alive(s.pid or read_meta(rd).get("pid")):
        meta = read_meta(rd)
        started = meta.get("last_spawn", 0)
        if time.time() - started > 15:  # allow time for the process to write its first event
            s.state = "interrupted"


def resumable(rd: Path, s: Summary) -> bool:
    return s.state in {"saved", "stopped", "interrupted", "failed"} and (rd / "resume.pt").exists() \
        and (not s.epochs or s.checkpoint_epoch < s.epochs)


def list_runs() -> list[dict[str, Any]]:
    out = []
    root = paths.runs_dir()
    if not root.exists():
        return out
    for rd in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        meta = read_meta(rd)
        if not meta:
            continue
        s, _ = summarize(rd)
        out.append({**meta, "state": s.state, "epoch": s.epoch, "epochs": s.epochs, "metrics": s.metrics,
                    "best": s.best, "resumable": resumable(rd, s), "updated": s.updated})
    return out


def best_weights(rd: Path) -> Path | None:
    for cand in (rd / "train" / "weights" / "best.pt", rd / "train" / "weights" / "last.pt", rd / "resume.pt"):
        if cand.exists():
            return cand
    return None
