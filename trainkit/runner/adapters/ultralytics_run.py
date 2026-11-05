"""Training process for Ultralytics models. Runs inside the recipe environment.

It writes progress to <run_dir>/events.jsonl and obeys commands in <run_dir>/control.json:
  pause      wait in place after the current batch (GPU memory stays allocated)
  continue   leave pause
  save_stop  finish the current epoch, save the checkpoint, exit (resumable)
  stop       exit after the current batch; the last completed epoch stays resumable

Why exit via exceptions: Ultralytics' final_eval() strips the optimizer from last.pt, which makes it impossible
to resume. Raising from a callback skips final_eval (BaseTrainer.train only wraps _do_train in try/finally).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tk_common import ControlReader, EventWriter, best_device, configure_ultralytics, model_class  # noqa: E402

EXIT_FINISHED, EXIT_FAILED, EXIT_SAVED, EXIT_STOPPED = 0, 1, 10, 11
BATCH_EMIT_INTERVAL = 0.5  # seconds between batch events


class SaveStop(Exception):
    pass


class StopNow(Exception):
    pass


def _to_float_dict(d) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[str(k)] = round(float(v), 6)
        except (TypeError, ValueError):
            continue
    return out


class Controller:
    def __init__(self, run_dir: Path, events: EventWriter, samples_every: int, model_cls_name: str):
        self.run_dir = run_dir
        self.ev = events
        self.ctl = ControlReader(run_dir / "control.json")
        self.samples_every = samples_every
        self.model_cls_name = model_cls_name
        self.batch = 0
        self.nb = 0
        self.last_emit = 0.0
        self.epoch_t0 = time.time()
        self.paused_total = 0.0
        self.want_save_stop = False
        self.in_epoch = False
        self.sample_files: list[str] = []

    # ---- control -------------------------------------------------------------------------------------------
    def _apply(self, cmd: dict | None, trainer) -> None:
        if not cmd:
            return
        name = cmd.get("cmd")
        self.ev.emit("ack", cmd=name, seq=cmd.get("seq"))
        if name == "pause":
            self._pause_loop(trainer)
        elif name == "save_stop":
            self.want_save_stop = True
            self.ev.emit("status", state="saving", epoch=trainer.epoch + 1)
        elif name == "stop":
            raise StopNow()

    def _pause_loop(self, trainer) -> None:
        t0 = time.time()
        self.ev.emit("status", state="paused", epoch=trainer.epoch + 1, batch=self.batch)
        while True:
            time.sleep(0.2)
            cmd = self.ctl.poll()
            if not cmd:
                continue
            name = cmd.get("cmd")
            self.ev.emit("ack", cmd=name, seq=cmd.get("seq"))
            if name == "continue":
                break
            if name == "stop":
                raise StopNow()
            if name == "save_stop":  # finish this epoch so the checkpoint is complete
                self.want_save_stop = True
                break
        dt = time.time() - t0
        self.paused_total += dt
        state = "saving" if self.want_save_stop else "running"
        self.ev.emit("status", state=state, epoch=trainer.epoch + 1, paused_seconds=round(dt, 1))

    # ---- callbacks -----------------------------------------------------------------------------------------
    def on_pretrain_routine_start(self, trainer):
        self.ev.emit("status", state="preparing", msg="dataset")

    def on_pretrain_routine_end(self, trainer):
        self.nb = len(trainer.train_loader)
        try:
            n_train = len(trainer.train_loader.dataset)
            n_val = len(trainer.test_loader.dataset)
        except Exception:
            n_train = n_val = None
        self._pick_sample_images(trainer)
        dev = trainer.device
        name = str(dev)
        try:
            import torch

            if dev.type == "cuda":
                name = torch.cuda.get_device_name(dev)
        except Exception:
            pass
        self.ev.emit("start", epochs=trainer.epochs, start_epoch=trainer.start_epoch, nb=self.nb,
                     batch_size=trainer.batch_size, imgsz=trainer.args.imgsz, device=str(dev), device_name=name,
                     save_dir=str(trainer.save_dir), n_train=n_train, n_val=n_val,
                     names=getattr(trainer, "data", {}).get("names"), task=trainer.args.task,
                     loss_names=list(getattr(trainer, "loss_names", []) or []))
        self.ev.emit("status", state="running")

    def on_train_epoch_start(self, trainer):
        self.batch = 0
        self.in_epoch = True
        self.epoch_t0 = time.time()
        self.paused_total = 0.0
        self.ev.emit("epoch_start", epoch=trainer.epoch + 1, epochs=trainer.epochs, nb=self.nb)

    def on_train_batch_end(self, trainer):
        self.batch += 1
        now = time.time()
        if now - self.last_emit >= BATCH_EMIT_INTERVAL or self.batch == self.nb:
            self.last_emit = now
            losses = {}
            try:
                losses = _to_float_dict(trainer.label_loss_items(trainer.tloss, prefix="train"))
            except Exception:
                pass
            mem = 0.0
            try:
                mem = float(trainer._get_memory())
            except Exception:
                pass
            active = max(now - self.epoch_t0 - self.paused_total, 1e-6)
            self.ev.emit("batch", epoch=trainer.epoch + 1, batch=self.batch, nb=self.nb, losses=losses,
                         mem_gb=round(mem, 2), rate=round(self.batch / active, 3))
        self._apply(self.ctl.poll(), trainer)

    def on_fit_epoch_end(self, trainer):
        metrics = _to_float_dict(trainer.metrics)
        lr = _to_float_dict(getattr(trainer, "lr", {}))
        if not self.in_epoch:  # final validation of best.pt after the training loop
            self.ev.emit("final", metrics=metrics, best=str(trainer.best))
            return
        self.in_epoch = False
        losses = {}
        try:
            losses = _to_float_dict(trainer.label_loss_items(trainer.tloss, prefix="train"))
        except Exception:
            pass
        fitness = getattr(trainer, "fitness", None)
        self.ev.emit("epoch", epoch=trainer.epoch + 1, epochs=trainer.epochs, metrics=metrics, losses=losses,
                     lr=lr, fitness=None if fitness is None else round(float(fitness), 6),
                     seconds=round(time.time() - self.epoch_t0 - self.paused_total, 2),
                     samples=self.sample_files)
        self.sample_files = []

    def on_model_save(self, trainer):
        # Keep a resumable copy. Ultralytics writes last.pt non-atomically and strips it after training ends.
        src = Path(trainer.last)
        if src.exists():
            tmp = self.run_dir / "resume.pt.tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, self.run_dir / "resume.pt")
            self.ev.emit("checkpoint", epoch=trainer.epoch + 1, path=str(self.run_dir / "resume.pt"),
                         best=str(trainer.best) if Path(trainer.best).exists() else None)
        if self.samples_every and (trainer.epoch + 1) % self.samples_every == 0:
            self._make_samples(trainer)
        if self.want_save_stop and not trainer.stop:
            self.on_fit_epoch_end(trainer)  # on_fit_epoch_end would run after this hook; emit the epoch now
            self.ev.emit("status", state="saved", epoch=trainer.epoch + 1)
            raise SaveStop()

    # ---- sample predictions --------------------------------------------------------------------------------
    def _pick_sample_images(self, trainer, k: int = 4) -> None:
        self.sample_images = []
        try:
            ds = trainer.test_loader.dataset
            files = list(getattr(ds, "im_files", None) or [s[0] for s in getattr(ds, "samples", [])])
            if files:
                step = max(1, len(files) // k)
                self.sample_images = [str(f) for f in files[::step][:k]]
        except Exception as e:
            self.ev.emit("log", level="warn", msg=f"sample images unavailable: {e}")

    def _make_samples(self, trainer) -> None:
        if not self.sample_images:
            return
        try:
            import cv2

            out_dir = self.run_dir / "samples"
            out_dir.mkdir(exist_ok=True)
            model = model_class(self.model_cls_name)(str(self.run_dir / "resume.pt"))
            imgsz = trainer.args.imgsz if isinstance(trainer.args.imgsz, int) else 640
            results = model.predict(self.sample_images, imgsz=imgsz, device=trainer.device, verbose=False,
                                    conf=0.25)
            files = []
            for i, r in enumerate(results):
                im = r.plot()
                h, w = im.shape[:2]
                if w > 480:
                    im = cv2.resize(im, (480, int(h * 480 / w)))
                name = f"epoch{trainer.epoch + 1:03d}_{i}.jpg"
                cv2.imwrite(str(out_dir / name), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
                files.append(f"samples/{name}")
            self.sample_files = files
            del model
        except Exception as e:
            self.ev.emit("log", level="warn", msg=f"sample prediction failed: {e}")

    def register(self, model) -> None:
        for name in ("on_pretrain_routine_start", "on_pretrain_routine_end", "on_train_epoch_start",
                     "on_train_batch_end", "on_fit_epoch_end", "on_model_save"):
            model.add_callback(name, getattr(self, name))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--weights", default="")
    p.add_argument("--data", default="")
    p.add_argument("--model-class", default="YOLO")
    p.add_argument("--resume", default="", help="checkpoint to resume from")
    p.add_argument("--overrides", default="{}", help="JSON dict of extra train() arguments")
    p.add_argument("--samples-every", type=int, default=1)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ev = EventWriter(run_dir / "events.jsonl")
    ev.emit("process", pid=os.getpid(), resume=bool(args.resume))
    ev.emit("status", state="preparing", msg="model")
    try:
        configure_ultralytics()
        overrides = json.loads(args.overrides or "{}")
        device = overrides.pop("device", "") or best_device()
        cls = model_class(args.model_class, args.weights or args.resume)
        ctl = Controller(run_dir, ev, args.samples_every, args.model_class)
        if args.resume:
            model = cls(args.resume)
            ctl.register(model)
            keep = {k: v for k, v in overrides.items() if k in ("batch", "workers", "imgsz", "patience")}
            model.train(resume=args.resume, device=device, **keep)
        else:
            model = cls(args.weights)
            ctl.register(model)
            model.train(data=args.data, project=str(run_dir), name="train", exist_ok=True, device=device,
                        **overrides)
        ev.emit("status", state="finished")
        return EXIT_FINISHED
    except SaveStop:
        return EXIT_SAVED
    except (StopNow, KeyboardInterrupt):
        ev.emit("status", state="stopped")
        return EXIT_STOPPED
    except Exception as e:
        ev.emit("error", msg=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        ev.emit("status", state="failed", msg=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return EXIT_FAILED
    finally:
        ev.close()


if __name__ == "__main__":
    sys.exit(main())
