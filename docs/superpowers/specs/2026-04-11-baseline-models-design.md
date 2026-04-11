# Baseline Models Design: TSN, TSM, C3D, I3D, R(2+1)D

**Date**: 2026-04-11  
**Status**: Approved

## Overview

Add five baseline video recognition models (TSN, TSM, C3D, I3D, R(2+1)D) as comparison points against the SlowFast+STTA model. All models use publicly available pretrained checkpoints (no training from scratch on ImageNet/Sports-1M/Kinetics). Integration shares the existing `train.py` entry point via YAML-driven configuration.

## Architecture

### Entry Point

`train.py` reads `model.type` from the YAML config to select the appropriate LightningModule:

```python
model_type = cfg.model.get('type', 'slowfast_stta')
if model_type == 'slowfast_stta':
    module = SlowFastSTTALightningModule(cfg)
else:
    module = BaselineLightningModule(cfg)
```

### Module Design

`engine/baseline_module.py` — a new `BaselineLightningModule` that:
- Calls `model_factory(model_type, cfg)` to build the model
- Reuses the same SGD optimizer, linear-warmup + cosine-annealing scheduler, gradient clipping, and train/val logging as the existing STTA module
- Reads `feat_dim` and `num_classes` from config to attach the classification head

### Model Factory

`model_factory(model_type, cfg)` dispatches to:

| `model.type` | Source | Pretrained |
|---|---|---|
| `tsn` | `models/tsn.py` (custom) | ImageNet via torchvision ResNet-50 |
| `tsm` | `models/tsm.py` (custom) | ImageNet via torchvision ResNet-50 |
| `c3d` | `pytorchvideo.models.c3d` | Sports-1M pretrained checkpoint (exact URL confirmed during implementation) |
| `i3d` | `torch.hub` pytorchvideo `i3d_r50` | Kinetics-400 |
| `r2plus1d` | `torch.hub` pytorchvideo `r2plus1d_r50` | Kinetics-400 |

## Model-Specific Configurations

| Model | clip_len | img_size | feat_dim | Pretrained source |
|---|---|---|---|---|
| TSN | 8 segments | 224 | 2048 | torchvision ImageNet ResNet-50 |
| TSM | 8 frames | 224 | 2048 | torchvision ImageNet ResNet-50 |
| C3D | 16 frames | 112 | 4096 | Sports-1M (pytorchvideo) |
| I3D | 32 frames | 224 | 2048 | Kinetics-400 (torch.hub) |
| R(2+1)D | 16 frames | 112 | 512 | Kinetics-400 (torch.hub) |

## TSN Implementation (`models/tsn.py`)

- Backbone: `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`
- Remove final FC; attach new `nn.Linear(2048, num_classes)`
- Forward:
  1. Input `(B, N_seg, C, H, W)` → reshape `(B*N, C, H, W)`
  2. ResNet-50 up to `avgpool` → `(B*N, 2048)`
  3. Reshape `(B, N, 2048)` → mean over segment dim → `(B, 2048)`
  4. New FC → `(B, num_classes)`

## TSM Implementation (`models/tsm.py`)

- Backbone: same ImageNet ResNet-50
- Insert `TemporalShift` before the first Conv in each `Bottleneck` block:
  - Input `(B*T, C, H, W)` → reshape `(B, T, C, H, W)`
  - Shift `C//8` channels forward by 1 frame (pad with zeros)
  - Shift `C//8` channels backward by 1 frame (pad with zeros)
  - Remaining `C*3//4` channels unchanged
  - Reshape back to `(B*T, C, H, W)`
- Weights are identical to ImageNet pretrained ResNet-50 — only computation changes
- Forward: same reshape-and-mean consensus as TSN

## Data Pipeline Changes (`data/dataset.py`)

### New YAML fields

```yaml
data:
  sampling: segment     # "uniform" (default) or "segment" (TSN only)
  segment_window: 64    # total frame window for segment sampling
  img_size: 224         # controls transform resize/crop size
```

### Segment Sampling Logic

For TSN (`sampling: segment`):

```
Given: keyframe_id=100, clip_len=8, segment_window=64

Window: [keyframe - window//2 ... keyframe + window//2]
        = frames [68 ... 131]

Divide into clip_len=8 equal segments of 8 frames each:
  Seg 1: [68..75], Seg 2: [76..83], ..., Seg 8: [124..131]

Training:   randomly pick 1 frame per segment (stochastic)
Validation: pick middle frame per segment (deterministic)

Output: 8 frame indices → loaded as (8, C, H, W) → returned as (N_seg, C, H, W)
```

Implementation: add `_sample_segment_indices()` method (~30 lines) to `KeyframeClipDataset`; branch in `__getitem__` on `self.sampling`.

## New Files

```
models/tsn.py
models/tsm.py
engine/baseline_module.py
configs/tsn.yaml
configs/tsm.yaml
configs/c3d.yaml
configs/i3d.yaml
configs/r2plus1d.yaml
```

## Modified Files

```
train.py          +5 lines: model.type dispatch
data/dataset.py   +30 lines: segment sampling branch
```

## YAML Structure

All baseline configs inherit from `base.yaml` and override only what differs:

```yaml
# configs/tsn.yaml
_base_: base.yaml
model:
  type: tsn
  backbone: resnet50
  pretrained: true
  feat_dim: 2048
data:
  clip_len: 8
  sampling: segment
  segment_window: 64
  img_size: 224
train:
  lr: 0.01

# configs/c3d.yaml
_base_: base.yaml
model:
  type: c3d
  pretrained: true
  feat_dim: 4096
data:
  clip_len: 16
  img_size: 112
train:
  lr: 0.01
```

## Dependencies

Add to `pyproject.toml`:
```
pytorchvideo>=0.1.5
```

## Non-Goals

- No STTA applied to any baseline model
- No ablation configs for baseline models
- No changes to existing SlowFast/STTA code paths
