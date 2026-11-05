# TrainKit

[![CI](https://github.com/Miracle197/model-toybox/actions/workflows/ci.yml/badge.svg)](https://github.com/Miracle197/model-toybox/actions/workflows/ci.yml)

**Set up the environment with one command, train YOLO with one click, and try your model anytime.** Built for robotics teams, course projects and small groups: no more afternoons lost to CUDA, PyTorch and dependency conflicts.

**English** | [简体中文](README.zh-CN.md)

```
trainkit doctor   →   trainkit setup   →   trainkit train    →   trainkit play
 check this PC         install everything    train + live views     camera / video demo
```

## What it fixes

| Before | With TrainKit |
|---|---|
| Manually matching GPU driver, CUDA and PyTorch versions | `setup` detects the GPU and installs the matching PyTorch build (**no CUDA Toolkit needed**) |
| Dependency conflicts, a CPU-only torch by mistake, `no kernel image` on older GPUs | One isolated environment per model; an automatic self-test plus a 1-epoch training run proves the whole chain works |
| A wrong dataset format that only fails halfway through training | Labels are checked before training; YOLO / VOC / COCO / Roboflow exports are recognized and split automatically |
| Staring at a terminal; Ctrl+C is the only way to stop | Terminal view + web dashboard: pause, continue, save & exit, resume, stop |
| Teammates must set up their own environment just to see results | `trainkit play`: a playground that also runs on CPU, with camera, video, images and side-by-side model comparison |

## Install

**Windows**: double-click `bootstrap/install-windows.bat`, or run in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/Miracle197/model-toybox/main/bootstrap/install.ps1 | iex"
```

**macOS / Linux**:

```bash
curl -LsSf https://raw.githubusercontent.com/Miracle197/model-toybox/main/bootstrap/install.sh | sh
```

The installer first installs [uv](https://github.com/astral-sh/uv), which also provides Python, then installs TrainKit. Python does not need to be installed beforehand.

From source (for development):

```bash
git clone https://github.com/Miracle197/model-toybox && cd model-toybox
uv tool install --force .
```

## Quick start

```bash
trainkit doctor                       # what can this computer train?
trainkit setup                        # pick a model; installs PyTorch, packages, weights and a test dataset
trainkit train yolo-detect --data coco128.yaml --epochs 50
trainkit play                         # open the playground
```

`train` opens a live terminal view and the web dashboard (default <http://127.0.0.1:7720>) at the same time.

### Terminal hotkeys

| Key | Action |
|---|---|
| `p` | Pause: waits after the current batch. **GPU memory stays allocated**, so resuming is instant |
| `c` | Continue |
| `s` | **Save & exit after this epoch**: finishes the epoch, saves a checkpoint and frees the GPU. Continue later with `trainkit resume <id>` |
| `x` | Stop now: keeps the checkpoint from the last completed epoch, which can also be resumed |
| `d` | Open the web dashboard in a browser |
| `q` | Close this view only; **training keeps running in the background** |

You can also control a run from another terminal: `trainkit ctl <id> pause | continue | save-stop | stop`.

### Why there are three ways to pause

YOLO training only saves a checkpoint at the end of each epoch, so:

- **Pause** is the fastest, but the GPU stays occupied. Good for briefly freeing the CPU or looking at sample predictions.
- **Save & exit** waits for the epoch to finish, but really frees the GPU. You can even shut the computer down.
- **Stop now** ends immediately; progress within the current epoch is lost.

Every epoch, TrainKit atomically copies the checkpoint to `resume.pt`. When Ultralytics training finishes normally it strips the optimizer state from its checkpoint, which makes resuming impossible; this copy avoids that.

## Web dashboard

Opens with `trainkit dash` (and automatically during training).

- Every run and its state; **start new training from the browser**, no command line needed
- Epoch and batch progress, time remaining, speed, device
- Training loss / validation loss / mAP · precision · recall / learning-rate charts (hover for values; switch to a table)
- GPU utilization, VRAM and temperature (NVIDIA); CPU and RAM when there is no discrete GPU
- Each epoch, the current model predicts on 4 fixed validation images, so you can see it improve
- Pause / continue / save & exit / stop / continue training / download model bundle
- `trainkit dash --lan` prints a link with an access token so teammates on the same network can watch

## Playground (independent of training)

```bash
trainkit play                         # lists your trained models and downloaded weights
trainkit play best.pt                 # any YOLO .pt / .onnx / OpenVINO model, from anywhere
trainkit play robot-model.zip         # a model bundle from a teammate
trainkit play --source 0              # open local USB camera 0 directly
trainkit play --source rtsp://192.168.1.20:8554/cam   # the robot's camera stream
```

- Inputs: browser camera, local USB camera (lower latency), RTSP/HTTP streams, video files, images
- Adjust confidence, IoU, input size, and filter classes; shows FPS and per-model latency
- **A/B comparison**: pick two models and see their outputs side by side
- **Danger-zone alert**: draw a polygon at the bottom of the frame (by default a trapezoid right in front of the robot). When an obstacle's "foot point" enters it, the zone is highlighted, which is handy for checking obstacle-avoidance logic quickly
- Record the annotated video; one-click conversion to ONNX for speed when there is no GPU

The playground does not depend on the training part. Teammates who only want to see results can run `trainkit play` on a computer without a GPU; it offers to install a ~1 GB CPU preview environment.

## Datasets

`--data` accepts:

| Form | Examples |
|---|---|
| Built-in datasets (downloaded on first use) | `coco8.yaml`, `coco128.yaml`, `HomeObjects-3K.yaml` (indoor objects, good for indoor robots), `kitti.yaml` |
| A registered name | after `trainkit datasets add obstacles ./my_data`, use `--data obstacles` |
| A folder | recognized automatically, see below |
| A `.zip` | extracted, then treated as a folder |
| A `data.yaml` | relative paths are fixed automatically |
| Roboflow | `roboflow:workspace/project/3` or a Universe URL (first run `trainkit config roboflow_api_key <key>`) |

Folders are recognized as one of these layouts:

- An existing `data.yaml` (including Roboflow exports)
- `images/` + `labels/`, or images and `.txt` labels in the same folder (LabelImg / X-AnyLabeling exports). Without a split, data is **split 80/20 automatically**; class names come from `classes.txt`
- `images/train` + `images/val`, or `train/images` + `valid/images`
- PASCAL VOC (`Annotations/*.xml` + `JPEGImages/`), converted automatically
- COCO JSON, converted automatically (detection and segmentation)
- Classification: one folder per class, split automatically

TrainKit never modifies your original files. Converted and split data goes to `~/.trainkit/datasets/_prepared/`, using hard links for images where possible so no extra space is used.

Checks before training: class ids out of range, coordinates not normalized, wrong number of columns, too many unlabeled images, classes with no labels at all, too few images, and more.

## Sharing with teammates

**Share the training setup**:

```bash
trainkit init                          # writes trainkit.yaml; commit it to your repo
# teammates:
trainkit setup trainkit.yaml
trainkit train trainkit.yaml
```

**Share a trained model**:

```bash
trainkit pack <run-id> -o obstacle-v3.zip   # weights + class names + metrics + training curves
# teammates:
trainkit play obstacle-v3.zip
```

**Deploy to the robot**:

```bash
trainkit export <run-id> --format onnx      # general purpose
trainkit export <run-id> --format openvino  # Intel CPUs
trainkit export <run-id> --format ncnn      # Raspberry Pi and other ARM boards
trainkit export <run-id> --format engine    # Jetson (TensorRT; run it on the Jetson)
```

## Supported models (v1)

| Recipe | Task | Default weights |
|---|---|---|
| `yolo-detect` | Object detection | yolo26n.pt (YOLO11 / YOLOv8 sizes also available) |
| `yolo-seg` | Instance segmentation | yolo26n-seg.pt |
| `yolo-pose` | Pose / keypoints | yolo26n-pose.pt |
| `yolo-cls` | Image classification | yolo26n-cls.pt |
| `yolo-obb` | Oriented boxes | yolo26n-obb.pt |
| `rtdetr` | RT-DETR detection (≥ 8 GB VRAM) | rtdetr-l.pt |

Planned: LLM LoRA fine-tuning (Unsloth), Whisper speech-recognition fine-tuning and diffusion LoRA. Adding a model family only takes a recipe YAML and a training adapter.

## Hardware tiers

| Tier | Requirement | What you can do |
|---|---|---|
| NVIDIA | An NVIDIA GPU with a recent enough driver | Train + play |
| Apple MPS | A Mac with an M-series chip | Train (slower than a comparable NVIDIA GPU) + play |
| CPU preview | Any other computer | Play with models only; train on a GPU machine or Colab/Kaggle |

What `doctor` handles:

- Driver too old: gives you the download link. It does not install drivers itself, because that needs admin rights and a reboot
- RTX 50 series needs driver ≥ 570
- Older GPUs such as the GTX 10 series (compute capability < 7.5) are not supported by PyTorch's CUDA 13 builds, so TrainKit pins the CUDA 12.6 build automatically
- Windows: when long paths are disabled, or the user name has non-ASCII characters or the profile lives in OneDrive, the data folder moves to `C:\trainkit`

## FAQ

**Installation is slow or downloads fail**: configure a PyPI mirror, e.g. `trainkit config pypi_index https://pypi.tuna.tsinghua.edu.cn/simple`, then run `trainkit setup --reinstall`.

**There is a GPU but PyTorch does not use it**: `trainkit setup --reinstall --torch-backend cu126`.

**Use a bigger drive**: set the environment variable `TRAINKIT_HOME=D:\trainkit`.

**Training was interrupted (crash, power loss)**: `trainkit runs` marks it "interrupted ↻"; `trainkit resume <id>` continues from the last epoch's checkpoint.

**Interface language**: `trainkit config lang en` or `zh`; the web dashboard also has a switch in the top-right corner.

## How it works

```
trainkit (CLI, lightweight, no torch dependency)
 ├─ doctor    probes GPU/driver/memory/disk/network → each recipe: "can train / preview only / unavailable"
 ├─ setup     uv creates an isolated environment ~/.trainkit/envs/ultralytics
 │            uv pip install --torch-backend <chosen for your GPU>  → self-test → smoke training
 ├─ train     starts the training process in the background (closing the terminal does not stop it)
 │              ultralytics_run.py ── writes → runs/<id>/events.jsonl ── read by → terminal view / web dashboard
 │                                 ←─ reads ── runs/<id>/control.json ←─ written by ── pause/continue/save & exit
 ├─ dash      FastAPI + WebSocket; static page, no build step
 └─ play      Gradio playground (runs inside the model environment, usable on its own)
```

All state lives in files, so if the dashboard or terminal crashes, training is unaffected and reopening picks up where it left off.

## Development

```bash
uv sync --group dev
uv run pytest
uv run trainkit doctor
```

## Credits and license

Built on [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), [uv](https://github.com/astral-sh/uv), [Gradio](https://github.com/gradio-app/gradio), [FastAPI](https://github.com/fastapi/fastapi), [Rich](https://github.com/Textualize/rich) and [uPlot](https://github.com/leeoniya/uPlot) (MIT).

Because it builds on AGPL-3.0 Ultralytics, TrainKit is also released under **AGPL-3.0**. The license of models you train depends on the model used (Ultralytics YOLO models are AGPL-3.0).
