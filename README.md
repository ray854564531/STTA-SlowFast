# Pilot Project — SlowFast + ST-TripletAttention

在驾驶舱飞行员动作识别数据集上对比 SlowFast baseline 与 SlowFast+STTA（时空三重注意力）的效果，并支持消融实验。

## 核心方法

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
├── train.py                  # 训练入口
├── configs/
│   ├── base.yaml             # 公共超参数
│   ├── slowfast_stta_reproduce.yaml  # 复现 94.84%（2 GPU × bs8）
│   ├── slowfast_stta_full.yaml   # 主实验（全部 STTA，单 GPU）
│   ├── slowfast_baseline.yaml    # 无注意力基线
│   ├── ablation_tcw_only.yaml    # 消融：仅 T-C-W
│   ├── ablation_tch_only.yaml    # 消融：仅 T-C-H
│   ├── ablation_thw_only.yaml    # 消融：仅 T-H-W
│   ├── ablation_tcw_tch.yaml     # 消融：T-C-W + T-C-H
│   ├── ablation_tcw_thw.yaml     # 消融：T-C-W + T-H-W
│   └── ablation_tch_thw.yaml     # 消融：T-C-H + T-H-W
├── models/
│   ├── attention.py          # STTripletAttention 模块
│   ├── resnet3d.py           # ResNet3d 基础块
│   ├── slowfast.py           # ResNet3dSlowFast backbone
│   ├── slowfast_stta.py      # SlowFastWithSTTA（主模型）
│   ├── head.py               # SlowFastHead 分类头
│   └── conv_utils.py         # ConvModule
├── engine/
│   └── module.py             # LightningModule（训练/验证逻辑）
├── data/
│   ├── dataset.py            # KeyframeClipDataset
│   ├── datamodule.py         # LightningDataModule
│   └── transforms.py         # 多帧一致性数据增强
├── utils/
│   └── metrics.py            # top-k accuracy
└── tests/                    # pytest 测试（28 个）
```

## 环境要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（包管理）
- PyTorch 2.x + CUDA（推荐）或 CPU

```bash
cd pilot_project
uv sync          # 安装依赖
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

# 复现 94.84%（2 GPU DDP，等效 batch=16）
uv run train.py --config configs/slowfast_stta_reproduce.yaml

# 主实验（SlowFast + 全 STTA，单 GPU）
uv run train.py --config configs/slowfast_stta_full.yaml

# 无注意力基线
uv run train.py --config configs/slowfast_baseline.yaml

# 消融实验（示例）
uv run train.py --config configs/ablation_tcw_only.yaml
uv run train.py --config configs/ablation_tch_thw.yaml
```

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
WANDB_MODE=offline uv run train.py --config configs/slowfast_stta_full.yaml
```

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

- Slow pathway: `base_channels=64`, `conv1_kernel=(1,7,7)`, `inflate=(0,0,1,1)`
- Fast pathway: `base_channels=8`, `conv1_kernel=(5,7,7)`, `inflate=(1,1,1,1)`
- STTA 插入位置：fast pathway 每个 stage 之后（共 4 个位置，均可独立开关）
- Lateral connection 使用 STTA 增强后的 fast 特征
- 分类头输入维度：`2304`（slow 2048 + fast 256）

## 测试

```bash
cd pilot_project
uv run -- python -m pytest tests/ -v
```

共 28 个测试，覆盖：模型前向传播、训练步骤、各消融组合、端到端数据流。
