# Pilot Project — SlowFast + ST-TripletAttention

**简体中文** | [English](README.en.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![uv](https://img.shields.io/badge/packaged%20with-uv-DE5FE9)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)](tests/)

在驾驶舱飞行员动作识别数据集上对比 SlowFast baseline 与 SlowFast+STTA（时空三重注意力）的效果，并支持消融实验。

项目地址：<https://github.com/ray854564531/STTA-SlowFast>

## 核心方法

项目包含两类模型：

- **SlowFast + STTA**（主方法）：在 SlowFast backbone 上插入时空三重注意力，支持消融实验
- **Baseline 对比模型**：TSN、TSM、C3D、I3D、R(2+1)D，通过同一 `train.py` 入口运行

**ST-TripletAttention (STTA)** 在 SlowFast 的 fast pathway 每个 stage 之后插入三路注意力分支：

| 分支 | 建模维度 | 压缩维度 |
|------|----------|----------|
| T-C-W | 时间-通道-宽度 | H |
| T-C-H | 时间-通道-高度 | W |
| T-H-W | 时间-高度-宽度 | C |

所有分支均包含时间维度，使用 7×7×7 3D 卷积核，输出三路平均融合后反馈给 slow pathway 的 lateral connection。

## 项目结构

```
pilot_project/
├── train.py                  # 训练入口（SlowFast + STTA / baseline 统一入口）
├── configs/
│   ├── base.yaml             # 公共超参数
│   ├── stta_fast_only.yaml      # 主方法：fast pathway 全部 STTA（最佳模型）
│   ├── stta_slow_only.yaml      # 对照：slow pathway 全部 STTA
│   ├── slowfast_baseline.yaml   # 无注意力基线
│   ├── ablation_tcw_only.yaml    # 消融：仅 T-C-W
│   ├── ablation_tch_only.yaml    # 消融：仅 T-C-H
│   ├── ablation_thw_only.yaml    # 消融：仅 T-H-W
│   ├── ablation_tcw_tch.yaml     # 消融：T-C-W + T-C-H
│   ├── ablation_tcw_thw.yaml     # 消融：T-C-W + T-H-W
│   ├── ablation_tch_thw.yaml     # 消融：T-C-H + T-H-W
│   ├── tsn.yaml              # TSN baseline（8 段，segment 采样）
│   ├── tsm.yaml              # TSM baseline（8 段，segment 采样）
│   ├── c3d.yaml              # C3D baseline（16 帧，112×112）
│   ├── i3d.yaml              # I3D baseline（32 帧，224×224）
│   └── r2plus1d.yaml         # R(2+1)D baseline（8 帧，224×224）
├── models/
│   ├── attention.py          # STTripletAttention 模块
│   ├── resnet3d.py           # ResNet3d 基础块
│   ├── slowfast.py           # ResNet3dSlowFast backbone
│   ├── slowfast_stta.py      # SlowFastWithSTTA（主模型）
│   ├── head.py               # SlowFastHead 分类头
│   ├── conv_utils.py         # ConvModule
│   ├── tsn.py                # TSN（ResNet-50 + 时序 consensus）
│   └── tsm.py                # TSM（ResNet-50 + temporal shift）
├── engine/
│   ├── module.py             # SlowFastSTTALightningModule
│   └── baseline_module.py    # BaselineLightningModule（TSN/TSM/C3D/I3D/R(2+1)D）
├── data/
│   ├── dataset.py            # KeyframeClipDataset（支持 uniform / segment 采样）
│   ├── datamodule.py         # LightningDataModule
│   └── transforms.py         # 多帧一致性数据增强
├── utils/
│   └── metrics.py            # top-k accuracy
└── tests/                    # pytest 测试（49 个）
```

## 环境要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（包管理）
- PyTorch 2.x + CUDA（推荐）或 CPU

```bash
cd pilot_project
uv sync          # 推荐方式（使用 uv.lock 锁定版本）
```

不使用 uv 的用户可退回 pip：

```bash
# 先安装匹配本机 CUDA 的 torch（示例为 CUDA 11.8）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 数据准备

数据目录需放置在 `../data/keyframes_pilot_cockpit/`（相对于本目录），结构如下：

```
data/keyframes_pilot_cockpit/
├── frames/
│   ├── 10/
│   │   ├── img_00001.jpg
│   │   ├── img_00002.jpg
│   │   └── ...
│   └── {video_id}/...
└── annotations/
    ├── train_annotations.csv
    └── val_annotations.csv
```

CSV 格式（无表头）：

```
video_id,keyframe_id,action_id
10,5,3
...
```

## 训练

```bash
cd pilot_project

# 主方法（最佳模型）：fast pathway 全部插入 STTA
uv run train.py --config configs/stta_fast_only.yaml

# 对照：slow pathway 全部插入 STTA
uv run train.py --config configs/stta_slow_only.yaml

# 无注意力基线
uv run train.py --config configs/slowfast_baseline.yaml

# 消融实验（示例）
uv run train.py --config configs/ablation_tcw_only.yaml
uv run train.py --config configs/ablation_tch_thw.yaml

# Baseline 对比模型
uv run train.py --config configs/tsn.yaml
uv run train.py --config configs/tsm.yaml
uv run train.py --config configs/c3d.yaml
uv run train.py --config configs/i3d.yaml
uv run train.py --config configs/r2plus1d.yaml
```

## Kinetics-400 训练（开箱即用）

本项目支持直接在 Kinetics-400 上训练 SlowFast + STTA，并自动执行标准 30-view 测试协议。

### 1. 下载数据

使用 [`cvdfoundation/kinetics-dataset`](https://github.com/cvdfoundation/kinetics-dataset) 等官方工具下载 K400，解压到项目同级目录 `../data/kinetics400/`：

```
data/kinetics400/
├── train/
│   ├── abseiling/
│   │   └── *.mp4
│   └── ... (400 个类别文件夹)
├── val/
│   ├── abseiling/
│   └── ...
└── annotations/        # 可选
```

### 2. 安装依赖

`decord` 会随 `uv sync` 自动安装：

```bash
uv sync
```

### 3. 训练 + 自动测试

```bash
uv run train.py --config configs/kinetics400_stta.yaml
```

训练结束后，Trainer 会用 `best` checkpoint 自动执行 **10 temporal clips × 3 spatial crops = 30 views** 的测试协议，对应 SlowFast 论文标准，指标日志为 `test/acc1` 与 `test/acc5`。

**关键配置项**（在 `configs/kinetics400_stta.yaml`）：

| 字段 | 含义 |
|------|------|
| `data.type: kinetics` | 切换到 Kinetics 数据管线 |
| `data.root` | K400 数据根目录 |
| `data.clip_len` / `data.frame_interval` | 时间采样：32 帧 × stride 2 |
| `test.num_clips` / `test.num_crops` | 30-view 默认 10 × 3 |
| `test.batch_size: 1` | 实际每条视频送入 30 个 clip |

多 GPU 训练通过 YAML 中的 `trainer.devices` 和 `trainer.strategy` 字段控制，无需修改代码：

```yaml
trainer:
  devices: 2       # GPU 数量
  strategy: ddp    # 分布式策略
```

**wandb 日志模式**通过 YAML 的 `logging.wandb_mode` 字段控制：

```yaml
logging:
  wandb_mode: online    # 默认，联网上传
  # wandb_mode: offline # 本地保存，之后可用 wandb sync 手动上传
  # wandb_mode: disabled # 完全禁用 wandb，仅保留 TensorBoard
```

也可通过环境变量临时覆盖：

```bash
WANDB_MODE=offline uv run train.py --config configs/stta_fast_only.yaml
```

**TensorBoard 查看训练日志**：每次训练会在 `lightning_logs/` 目录下生成 TensorBoard 事件文件，可用以下命令启动本地服务：

```bash
# 查看所有实验（默认端口 6006）
uv run -- tensorboard --logdir logs/tb/

# 指定端口（多实验对比时可开多个）
uv run -- tensorboard --logdir logs/tb/ --port 6007

# 只查看某次具体实验
uv run -- tensorboard --logdir logs/tb/stta_slowfast_fast_only/version_0/
```

启动后访问 `http://localhost:6006` 即可查看 loss、accuracy 等曲线。

## 关键超参数（base.yaml）

| 参数 | 值 |
|------|----|
| clip_len | 32 |
| batch_size | 4（单 GPU）/ 8 per GPU（多 GPU 复现） |
| optimizer | SGD |
| lr | 0.1 |
| momentum | 0.9 |
| weight_decay | 5e-4 |
| grad_clip | 40 |
| LR schedule | LinearLR warmup (5 epoch) + CosineAnnealingLR (150 epoch) |
| 归一化均值/标准差 | ImageNet ([0.485,0.456,0.406] / [0.229,0.224,0.225]) |
| SlowFast resample_rate | 4 |
| SlowFast speed_ratio | 4 |
| SlowFast channel_ratio | 8 |
| STTA kernel_size | 7 |

## 模型架构细节

### SlowFast + STTA

- Slow pathway: `base_channels=64`, `conv1_kernel=(1,7,7)`, `inflate=(0,0,1,1)`
- Fast pathway: `base_channels=8`, `conv1_kernel=(5,7,7)`, `inflate=(1,1,1,1)`
- STTA 插入位置：fast pathway 每个 stage 之后（共 4 个位置，均可独立开关）
- Lateral connection 使用 STTA 增强后的 fast 特征
- 分类头输入维度：`2304`（slow 2048 + fast 256）

### Baseline 对比模型

| 模型 | Backbone | 预训练权重 | 输入尺寸 | 采样方式 |
|------|----------|-----------|---------|---------|
| TSN | ResNet-50 | ImageNet | 8 段 × 224×224 | segment（每段随机/中间帧） |
| TSM | ResNet-50 + temporal shift | ImageNet | 8 段 × 224×224 | segment |
| C3D | VGG-style 3D CNN | Sports-1M（backbone only） | 16×112×112 | uniform |
| I3D | ResNet-50 3D | Kinetics-400 | 32×224×224 | uniform |
| R(2+1)D | ResNet-50 (2+1)D | Kinetics-400 | 8×224×224 | uniform |

所有 baseline 使用 `lr=0.01`，`max_epochs=100`（相比 SlowFast 的 `lr=0.1`、`150 epoch`）。

## 测试

```bash
cd pilot_project
uv run -- python -m pytest tests/ -v
```

共 49 个测试，覆盖：SlowFast+STTA 模型前向传播、训练步骤、各消融组合、端到端数据流、TSN/TSM/C3D/I3D/R(2+1)D baseline 模型前向传播与训练逻辑。

## License

本项目采用 [MIT License](LICENSE) 开源。你可以自由使用、修改、分发本项目的代码，但需保留原版权声明。
