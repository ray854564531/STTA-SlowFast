# Pilot Project — SlowFast + ST-TripletAttention

[简体中文](README.md) | **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![uv](https://img.shields.io/badge/packaged%20with-uv-DE5FE9)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)](tests/)

Comparison of a SlowFast baseline with SlowFast + STTA (Spatio-Temporal Triplet Attention) on a cockpit pilot action-recognition dataset, with support for ablation studies.

Repository: <https://github.com/ray854564531/STTA-SlowFast>

## Core Method

The project includes two families of models:

- **SlowFast + STTA** (main method): inserts Spatio-Temporal Triplet Attention into the SlowFast backbone, with ablation support
- **Baseline comparison models**: TSN, TSM, C3D, I3D, R(2+1)D, all runnable through the same `train.py` entry point

**ST-TripletAttention (STTA)** inserts three attention branches after each stage of the SlowFast fast pathway:

| Branch | Modeled dims | Squeezed dim |
|--------|--------------|--------------|
| T-C-W  | Time–Channel–Width  | H |
| T-C-H  | Time–Channel–Height | W |
| T-H-W  | Time–Height–Width   | C |

All branches include the temporal dimension and use 7×7×7 3D convolution kernels. The three branches are averaged and fed into the lateral connection of the slow pathway.

## Project Structure

```
pilot_project/
├── train.py                  # Training entry (unified for SlowFast+STTA / baselines)
├── configs/
│   ├── base.yaml             # Shared hyper-parameters
│   ├── stta_fast_only.yaml      # Main method: STTA on all fast-pathway stages (best model)
│   ├── stta_slow_only.yaml      # Contrast: STTA on all slow-pathway stages
│   ├── slowfast_baseline.yaml   # Baseline without attention
│   ├── ablation_tcw_only.yaml    # Ablation: T-C-W only
│   ├── ablation_tch_only.yaml    # Ablation: T-C-H only
│   ├── ablation_thw_only.yaml    # Ablation: T-H-W only
│   ├── ablation_tcw_tch.yaml     # Ablation: T-C-W + T-C-H
│   ├── ablation_tcw_thw.yaml     # Ablation: T-C-W + T-H-W
│   ├── ablation_tch_thw.yaml     # Ablation: T-C-H + T-H-W
│   ├── tsn.yaml              # TSN baseline (8 segments, segment sampling)
│   ├── tsm.yaml              # TSM baseline (8 segments, segment sampling)
│   ├── c3d.yaml              # C3D baseline (16 frames, 112×112)
│   ├── i3d.yaml              # I3D baseline (32 frames, 224×224)
│   └── r2plus1d.yaml         # R(2+1)D baseline (8 frames, 224×224)
├── models/
│   ├── attention.py          # STTripletAttention module
│   ├── resnet3d.py           # ResNet3d building blocks
│   ├── slowfast.py           # ResNet3dSlowFast backbone
│   ├── slowfast_stta.py      # SlowFastWithSTTA (main model)
│   ├── head.py               # SlowFastHead classification head
│   ├── conv_utils.py         # ConvModule
│   ├── tsn.py                # TSN (ResNet-50 + temporal consensus)
│   └── tsm.py                # TSM (ResNet-50 + temporal shift)
├── engine/
│   ├── module.py             # SlowFastSTTALightningModule
│   └── baseline_module.py    # BaselineLightningModule (TSN/TSM/C3D/I3D/R(2+1)D)
├── data/
│   ├── dataset.py            # KeyframeClipDataset (uniform / segment sampling)
│   ├── datamodule.py         # LightningDataModule
│   └── transforms.py         # Multi-frame consistent data augmentation
├── utils/
│   └── metrics.py            # top-k accuracy
└── tests/                    # pytest tests (49 in total)
```

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (package management)
- PyTorch 2.x + CUDA (recommended) or CPU

```bash
cd pilot_project
uv sync          # Recommended (uses uv.lock for reproducible versions)
```

Users who don't use uv can fall back to pip:

```bash
# First install a torch build matching your local CUDA (example: CUDA 11.8)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data Preparation

Place the dataset at `../data/keyframes_pilot_cockpit/` (relative to this directory) with the following layout:

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

CSV format (no header):

```
video_id,keyframe_id,action_id
10,5,3
...
```

## Training

```bash
cd pilot_project

# Main method (best model): STTA on all fast-pathway stages
uv run train.py --config configs/stta_fast_only.yaml

# Contrast: STTA on all slow-pathway stages
uv run train.py --config configs/stta_slow_only.yaml

# Baseline without attention
uv run train.py --config configs/slowfast_baseline.yaml

# Ablation experiments (examples)
uv run train.py --config configs/ablation_tcw_only.yaml
uv run train.py --config configs/ablation_tch_thw.yaml

# Baseline comparison models
uv run train.py --config configs/tsn.yaml
uv run train.py --config configs/tsm.yaml
uv run train.py --config configs/c3d.yaml
uv run train.py --config configs/i3d.yaml
uv run train.py --config configs/r2plus1d.yaml
```

Multi-GPU training is controlled via `trainer.devices` and `trainer.strategy` in YAML, with no code changes required:

```yaml
trainer:
  devices: 2       # Number of GPUs
  strategy: ddp    # Distributed strategy
```

**wandb logging mode** is controlled via the `logging.wandb_mode` field in YAML:

```yaml
logging:
  wandb_mode: online    # Default, upload online
  # wandb_mode: offline # Save locally, upload later with `wandb sync`
  # wandb_mode: disabled # Disable wandb entirely, keep TensorBoard only
```

You can also override it temporarily via environment variable:

```bash
WANDB_MODE=offline uv run train.py --config configs/stta_fast_only.yaml
```

**View training logs with TensorBoard**: each run writes TensorBoard event files under `lightning_logs/`. Start a local server with:

```bash
# View all experiments (default port 6006)
uv run -- tensorboard --logdir logs/tb/

# Specify a port (useful when comparing multiple runs)
uv run -- tensorboard --logdir logs/tb/ --port 6007

# View a specific experiment only
uv run -- tensorboard --logdir logs/tb/stta_slowfast_fast_only/version_0/
```

Then open `http://localhost:6006` to view loss, accuracy, and other curves.

## Key Hyper-parameters (base.yaml)

| Parameter | Value |
|-----------|-------|
| clip_len | 32 |
| batch_size | 4 (single GPU) / 8 per GPU (multi-GPU reproduction) |
| optimizer | SGD |
| lr | 0.1 |
| momentum | 0.9 |
| weight_decay | 5e-4 |
| grad_clip | 40 |
| LR schedule | LinearLR warmup (5 epochs) + CosineAnnealingLR (150 epochs) |
| Normalization mean/std | ImageNet ([0.485,0.456,0.406] / [0.229,0.224,0.225]) |
| SlowFast resample_rate | 4 |
| SlowFast speed_ratio | 4 |
| SlowFast channel_ratio | 8 |
| STTA kernel_size | 7 |

## Architecture Details

### SlowFast + STTA

- Slow pathway: `base_channels=64`, `conv1_kernel=(1,7,7)`, `inflate=(0,0,1,1)`
- Fast pathway: `base_channels=8`, `conv1_kernel=(5,7,7)`, `inflate=(1,1,1,1)`
- STTA insertion points: after each stage of the fast pathway (4 positions, each independently toggleable)
- Lateral connection uses STTA-enhanced fast features
- Classifier head input dim: `2304` (slow 2048 + fast 256)

### Baseline Models

| Model | Backbone | Pretrained weights | Input shape | Sampling |
|-------|----------|--------------------|-------------|----------|
| TSN | ResNet-50 | ImageNet | 8 segments × 224×224 | segment (random/center frame per segment) |
| TSM | ResNet-50 + temporal shift | ImageNet | 8 segments × 224×224 | segment |
| C3D | VGG-style 3D CNN | Sports-1M (backbone only) | 16×112×112 | uniform |
| I3D | ResNet-50 3D | Kinetics-400 | 32×224×224 | uniform |
| R(2+1)D | ResNet-50 (2+1)D | Kinetics-400 | 8×224×224 | uniform |

All baselines use `lr=0.01`, `max_epochs=100` (vs. `lr=0.1`, `150 epochs` for SlowFast).

## Testing

```bash
cd pilot_project
uv run -- python -m pytest tests/ -v
```

49 tests in total, covering: SlowFast+STTA forward pass, training step, all ablation combinations, end-to-end data flow, and forward/training logic for TSN/TSM/C3D/I3D/R(2+1)D baselines.

## License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute the code, provided the original copyright notice is retained.
