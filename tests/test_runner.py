import importlib.util
import sys

from trainkit.envs.manager import ADAPTERS_DIR
from trainkit.runner import job, runs

sys.path.insert(0, str(ADAPTERS_DIR))
tk_common = importlib.import_module("tk_common")


def test_read_events_ignores_partial_line(tmp_path):
    f = tmp_path / "events.jsonl"
    f.write_bytes(b'{"type": "status", "state": "running"}\n{"type": "ba')  # bytes: no \r\n on Windows
    evs, off = runs.read_events(tmp_path)
    assert len(evs) == 1 and off == len('{"type": "status", "state": "running"}\n')
    with open(f, "ab") as fh:
        fh.write(b'tch", "batch": 3, "nb": 10}\n')
    evs, off2 = runs.read_events(tmp_path, off)
    assert evs[0]["batch"] == 3


def test_summary_fold_and_eta():
    s = runs.Summary()
    for ev in [
        {"t": 1, "type": "process", "pid": 1},
        {"t": 2, "type": "start", "epochs": 10, "nb": 20, "device": "cpu", "task": "detect", "start_epoch": 0},
        {"t": 3, "type": "epoch_start", "epoch": 1, "epochs": 10},
        {"t": 4, "type": "batch", "epoch": 1, "batch": 10, "nb": 20, "rate": 2.0, "losses": {"train/box_loss": 1.2}},
        {"t": 5, "type": "epoch", "epoch": 1, "metrics": {"metrics/mAP50(B)": 0.3}, "seconds": 10},
        {"t": 6, "type": "status", "state": "paused"},
    ]:
        s.feed(ev)
    assert s.state == "paused" and s.epoch == 1 and s.losses["train/box_loss"] == 1.2
    assert s.eta_seconds() == 90  # 9 epochs * 10 s
    s.feed({"t": 7, "type": "epoch", "epoch": 1, "metrics": {"metrics/mAP50(B)": 0.4}, "seconds": 10})
    assert len(s.history) == 1 and s.history[0]["metrics"]["metrics/mAP50(B)"] == 0.4  # resumed epoch replaces


def test_control_protocol(tk_home):
    rid = "20260101-000000-test"
    rd = runs.run_dir(rid)
    rd.mkdir(parents=True)
    reader = tk_common.ControlReader(rd / "control.json")
    assert reader.poll() is None
    job.send(rid, "pause")
    cmd = reader.poll()
    assert cmd["cmd"] == "pause" and cmd["seq"] == 1
    assert reader.poll() is None  # same command is not delivered twice
    job.send(rid, "continue")
    assert reader.poll()["cmd"] == "continue"
    late = tk_common.ControlReader(rd / "control.json")  # a resumed process ignores old commands
    assert late.poll() is None


def test_interrupted_detection(tk_home):
    rid = "20260101-000001-test"
    rd = runs.run_dir(rid)
    runs.write_meta(rd, {"id": rid, "pid": 999999, "last_spawn": 0})
    runs.append_event(rd, "process", pid=999999)
    runs.append_event(rd, "status", state="running")
    (rd / "resume.pt").write_bytes(b"x")
    s, _ = runs.summarize(rd)
    assert s.state == "interrupted"
    assert runs.resumable(rd, s)


def test_run_dir_rejects_traversal(tk_home):
    import pytest

    with pytest.raises(ValueError):
        runs.run_dir("../../etc")
