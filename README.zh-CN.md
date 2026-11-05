# TrainKit

[![CI](https://github.com/Miracle197/model-toybox/actions/workflows/ci.yml/badge.svg)](https://github.com/Miracle197/model-toybox/actions/workflows/ci.yml)

**一条命令装好环境，一键训练 YOLO，随时试玩模型。** 适合机器人、课程项目和小团队：组员不用再花一下午排查 CUDA、PyTorch 和依赖冲突。

[English](README.md) | **简体中文**

```
trainkit doctor   →   trainkit setup   →   trainkit train   →   trainkit play
 检测这台电脑          装好一切              训练 + 实时面板         摄像头/视频试玩
```

## 它解决什么问题

| 以前 | 用 TrainKit |
|---|---|
| 手动查显卡驱动、CUDA、PyTorch 版本是否匹配 | `setup` 自动检测显卡并装对应的 PyTorch（**不需要装 CUDA Toolkit**） |
| 依赖冲突、装错 CPU 版 torch、老显卡报 `no kernel image` | 每个模型独立环境；装完自动自检，并用 1 轮小训练验证整条链路 |
| 数据集格式不对，训练到一半才报错 | 训练前检查标注；自动识别 YOLO / VOC / COCO / Roboflow 导出，自动划分训练/验证集 |
| 训练只能盯着终端，中途想停只能 Ctrl+C | 终端视图 + 网页面板：暂停、继续、保存并退出、续训、立即停止 |
| 组员想看效果还得自己配环境 | `trainkit play`：CPU 也能跑的试玩区，支持摄像头、视频、图片、两个模型对比 |

## 安装

**Windows**：双击 `bootstrap/install-windows.bat`，或在 PowerShell 里运行：

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/Miracle197/model-toybox/main/bootstrap/install.ps1 | iex"
```

**macOS / Linux**：

```bash
curl -LsSf https://raw.githubusercontent.com/Miracle197/model-toybox/main/bootstrap/install.sh | sh
```

安装脚本会先装 [uv](https://github.com/astral-sh/uv)（它会顺便提供 Python），再安装 TrainKit。电脑上不需要事先装 Python。

从源码安装（开发用）：

```bash
git clone https://github.com/Miracle197/model-toybox && cd model-toybox
uv tool install --force .
```

## 快速开始

```bash
trainkit doctor                       # 看看这台电脑能训练什么
trainkit setup                        # 选模型，自动装好 PyTorch、依赖、权重和测试数据集
trainkit train yolo-detect --data coco128.yaml --epochs 50
trainkit play                         # 打开试玩区
```

`train` 会同时打开终端实时视图和网页面板（默认 <http://127.0.0.1:7720>）。

### 终端里的快捷键

| 键 | 作用 |
|---|---|
| `p` | 暂停（当前 batch 结束后原地等待，**显存不释放**，可以秒恢复） |
| `c` | 继续 |
| `s` | **本轮结束后保存并退出**：跑完当前这一轮，存档，释放显卡。之后用 `trainkit resume <id>` 接着练 |
| `x` | 立即停止：保留上一轮结束时的存档，同样可以续训 |
| `d` | 在浏览器里打开网页面板 |
| `q` | 只关闭这个视图，**训练继续在后台跑** |

另一个终端也能控制：`trainkit ctl <id> pause | continue | save-stop | stop`。

### 为什么"暂停"分成三种

YOLO 训练只在每一轮（epoch）结束时存档，所以：

- **暂停**：最快，但显卡一直被占着。适合临时让一下 CPU，或者想先看看样例图。
- **保存并退出**：要等这一轮跑完，但会真正释放显卡，关机都没问题。
- **立即停止**：马上结束，这一轮做了一半的进度会丢掉。

TrainKit 每轮都会把存档原子地复制一份为 `resume.pt`。Ultralytics 训练正常结束时会删掉存档里的优化器状态，导致无法续训，这份副本就是为此准备的。

## 网页面板

`trainkit dash` 单独打开（训练时会自动启动）。

- 所有训练记录及其状态；可以在网页上**新建训练**，不用碰命令行
- 轮次和本轮进度、预计剩余时间、速度、设备
- 训练损失 / 验证损失 / mAP·精确率·召回率 / 学习率曲线（悬停看数值；可切换为表格）
- GPU 利用率、显存、温度（NVIDIA）；没有独显时显示 CPU 和内存
- 每一轮用当前模型对 4 张固定验证图做预测的样例图，直观看到模型在进步
- 暂停 / 继续 / 保存并退出 / 立即停止 / 继续训练 / 下载模型包
- `trainkit dash --lan`：生成带访问令牌的链接，组员在同一局域网可以围观

## 试玩区（独立于训练）

```bash
trainkit play                         # 自动列出训练出的模型和下载的权重
trainkit play best.pt                 # 任何来源的 YOLO .pt / .onnx / OpenVINO 模型
trainkit play robot-model.zip         # 组员发来的模型包
trainkit play --source 0              # 直接打开本机第 0 个 USB 摄像头
trainkit play --source rtsp://192.168.1.20:8554/cam   # 机器人相机的视频流
```

- 输入：浏览器摄像头、本机 USB 摄像头（延迟更低）、RTSP/HTTP 视频流、视频文件、图片
- 调节置信度、IoU、输入尺寸，只看指定类别；显示帧率和每个模型的延迟
- **A/B 对比**：选两个模型，画面并排显示
- **危险区报警**：在画面下方画一个多边形（默认是机器人正前方的梯形），障碍物的"脚点"进入时高亮报警，适合快速验证避障逻辑
- 录制标注后的视频；没有显卡时一键转成 ONNX 提速

试玩区不依赖训练部分。只想看效果的组员在没有显卡的电脑上运行 `trainkit play`，它会提示安装一个约 1GB 的 CPU 预览环境。

## 数据集

`--data` 可以接受：

| 写法 | 例子 |
|---|---|
| 内置数据集（首次使用自动下载） | `coco8.yaml`、`coco128.yaml`、`HomeObjects-3K.yaml`（室内物体，适合室内避障）、`kitti.yaml` |
| 已登记的名称 | `trainkit datasets add obstacles ./my_data` 之后用 `--data obstacles` |
| 文件夹 | 见下表，自动识别 |
| `.zip` | 自动解压后按文件夹处理 |
| `data.yaml` | 自动修正相对路径 |
| Roboflow | `roboflow:workspace/project/3` 或 Universe 链接（先 `trainkit config roboflow_api_key <key>`） |

文件夹会被自动识别成以下几种结构之一：

- 已有 `data.yaml`（包括 Roboflow 导出）
- `images/` + `labels/`，或图片和 `.txt` 标注放在同一个文件夹（LabelImg / X-AnyLabeling 导出）。没有划分时**自动按 8:2 划分**，类别名从 `classes.txt` 读取
- `images/train` + `images/val`，或 `train/images` + `valid/images`
- PASCAL VOC（`Annotations/*.xml` + `JPEGImages/`），自动转换
- COCO JSON，自动转换（检测和分割）
- 分类任务：每个类别一个文件夹，自动划分

TrainKit 不会修改你的原始文件。转换和划分的结果写在 `~/.trainkit/datasets/_prepared/`，图片尽量用硬链接，不额外占空间。

训练前的检查：类别编号越界、坐标没有归一化、列数不对、没有标注的图片太多、某些类别没有任何标注、图片太少等。

## 和组员共享

**共享训练配置**：

```bash
trainkit init                          # 生成 trainkit.yaml，提交到你们的仓库
# 组员：
trainkit setup trainkit.yaml
trainkit train trainkit.yaml
```

**共享训练好的模型**：

```bash
trainkit pack <run-id> -o obstacle-v3.zip   # 权重 + 类别名 + 指标 + 训练曲线
# 组员：
trainkit play obstacle-v3.zip
```

**部署到机器人**：

```bash
trainkit export <run-id> --format onnx      # 通用
trainkit export <run-id> --format openvino  # Intel CPU
trainkit export <run-id> --format ncnn      # 树莓派等 ARM 板子
trainkit export <run-id> --format engine    # Jetson（TensorRT，需在 Jetson 上运行）
```

## 支持的模型（v1）

| 配方 | 任务 | 默认权重 |
|---|---|---|
| `yolo-detect` | 目标检测 | yolo26n.pt（可选 YOLO11 / YOLOv8 各尺寸） |
| `yolo-seg` | 实例分割 | yolo26n-seg.pt |
| `yolo-pose` | 姿态 / 关键点 | yolo26n-pose.pt |
| `yolo-cls` | 图像分类 | yolo26n-cls.pt |
| `yolo-obb` | 旋转框 | yolo26n-obb.pt |
| `rtdetr` | RT-DETR 检测（≥ 8GB 显存） | rtdetr-l.pt |

后续计划加入大模型 LoRA 微调（Unsloth）、Whisper 语音识别微调和扩散模型 LoRA。新增模型只需要写一个配方 YAML 和一个训练适配器。

## 硬件档位

| 档位 | 条件 | 能做什么 |
|---|---|---|
| NVIDIA | 有 NVIDIA 显卡且驱动够新 | 训练 + 试玩 |
| Apple MPS | M 系列芯片的 Mac | 训练（比同级 NVIDIA 慢）+ 试玩 |
| CPU 预览 | 其他电脑 | 只试玩模型；训练请用有显卡的电脑或 Colab/Kaggle |

`doctor` 会处理的情况：

- 驱动太旧：给出下载链接。不会自动装驱动，因为装驱动需要管理员权限还要重启
- RTX 50 系列需要驱动 ≥ 570
- GTX 10 系列等老显卡（计算能力 < 7.5）：PyTorch 的 CUDA 13 版本已经不支持，自动固定使用 CUDA 12.6 版本
- Windows：长路径未开启、用户名含中文或在 OneDrive 里时，数据目录自动改到 `C:\trainkit`

## 常见问题

**装得很慢 / 下载失败**：配置 PyPI 镜像，例如 `trainkit config pypi_index https://pypi.tuna.tsinghua.edu.cn/simple`，然后 `trainkit setup --reinstall`。

**有显卡但 PyTorch 用不上**：`trainkit setup --reinstall --torch-backend cu126`。

**换到更大的硬盘**：设置环境变量 `TRAINKIT_HOME=D:\trainkit`。

**训练意外中断（死机、断电）**：`trainkit runs` 里会标为"已中断 ↻"，`trainkit resume <id>` 从最后一轮的存档继续。

**界面语言**：`trainkit config lang zh` 或 `en`；网页右上角也可以切换。

## 工作原理

```
trainkit (CLI，很轻，不依赖 torch)
 ├─ doctor    探测 GPU/驱动/内存/磁盘/网络 → 每个配方「可训练 / 仅预览 / 不可用」
 ├─ setup     uv 创建独立环境 ~/.trainkit/envs/ultralytics
 │            uv pip install --torch-backend <按显卡选择>  → 自检 → 冒烟训练
 ├─ train     在后台启动训练子进程（关掉终端也不影响）
 │              ultralytics_run.py ── 写 → runs/<id>/events.jsonl ── 读 → 终端视图 / 网页面板
 │                                 ←─ 读 ── runs/<id>/control.json ←─ 写 ── 暂停/继续/保存退出
 ├─ dash      FastAPI + WebSocket，静态页面无需构建
 └─ play      Gradio 试玩区（在模型环境里运行，可单独使用）
```

所有状态都在文件里，面板或终端崩了不影响训练，重新打开会接着显示。

## 开发

```bash
uv sync --group dev
uv run pytest
uv run trainkit doctor
```

## 致谢与许可证

基于 [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）、[uv](https://github.com/astral-sh/uv)、[Gradio](https://github.com/gradio-app/gradio)、[FastAPI](https://github.com/fastapi/fastapi)、[Rich](https://github.com/Textualize/rich) 和 [uPlot](https://github.com/leeoniya/uPlot)（MIT）。

因为调用了 AGPL-3.0 的 Ultralytics，TrainKit 同样以 **AGPL-3.0** 发布。用它训练出的模型的许可证取决于所用的模型（Ultralytics YOLO 模型为 AGPL-3.0）。
