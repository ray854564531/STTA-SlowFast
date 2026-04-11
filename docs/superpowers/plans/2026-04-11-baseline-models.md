# Baseline Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TSN, TSM, C3D, I3D, R(2+1)D as comparison baselines that share the existing `train.py` entry point via YAML configs, with publicly available pretrained weights.

**Architecture:** New `engine/baseline_module.py` (`BaselineLightningModule`) reuses the existing optimizer/scheduler/logging logic; a `model_factory()` function dispatches to TSN/TSM (custom, torchvision ResNet-50 ImageNet pretrained) and C3D/I3D/R(2+1)D (pytorchvideo, Sports-1M/Kinetics-400 pretrained). `train.py` picks the LightningModule based on `model.type` in YAML.

**Tech Stack:** PyTorch, torchvision (ResNet-50), pytorchvideo (C3D/I3D/R(2+1)D), PyTorch Lightning, existing YAML config system.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `pyproject.toml` | Add `pytorchvideo>=0.1.5` |
| Modify | `data/dataset.py` | Add `_sample_segment_indices()` + `sampling` param |
| Modify | `data/datamodule.py` | Pass `img_size`, `sampling`, `segment_window` from config |
| Create | `models/tsn.py` | TSN: ResNet-50 backbone + segment consensus head |
| Create | `models/tsm.py` | TSM: ResNet-50 backbone + temporal shift + consensus head |
| Create | `engine/baseline_module.py` | `BaselineLightningModule` + `model_factory()` |
| Modify | `train.py` | Dispatch by `model.type` (+5 lines) |
| Create | `configs/tsn.yaml` | TSN config |
| Create | `configs/tsm.yaml` | TSM config |
| Create | `configs/c3d.yaml` | C3D config |
| Create | `configs/i3d.yaml` | I3D config |
| Create | `configs/r2plus1d.yaml` | R(2+1)D config |
| Create | `tests/test_tsn.py` | TSN unit tests |
| Create | `tests/test_tsm.py` | TSM unit tests |
| Create | `tests/test_baseline_module.py` | BaselineLightningModule unit tests |
| Modify | `tests/test_dataset.py` | Add segment sampling tests |

---

## Task 1: Add pytorchvideo dependency + fix datamodule img_size

**Files:**
- Modify: `pyproject.toml`
- Modify: `data/datamodule.py`

- [ ] **Step 1: Add pytorchvideo to pyproject.toml**

In `pyproject.toml`, add `pytorchvideo` to the `dependencies` list:

```toml
dependencies = [
    "torch==2.7.1",
    "torchvision==0.22.1",
    "pytorch-lightning>=2.0.0",
    "wandb",
    "tensorboard",
    "pyyaml",
    "pandas",
    "pillow",
    "einops",
    "rich",
    "pytorchvideo>=0.1.5",
]
```

- [ ] **Step 2: Install pytorchvideo**

```bash
uv sync
```

Expected: resolves and installs pytorchvideo without conflicts.

- [ ] **Step 3: Update datamodule to read img_size, sampling, segment_window from config**

Replace `data/datamodule.py` content with:

```python
"""LightningDataModule for pilot cockpit dataset."""
import torch
import pytorch_lightning as pl

_PIN_MEMORY = False
from torch.utils.data import DataLoader
from .dataset import KeyframeClipDataset
from .transforms import build_train_transforms, build_val_transforms


class PilotDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data

    def setup(self, stage=None):
        dc = self.cfg
        img_size = dc.get('img_size', 224)
        sampling = dc.get('sampling', 'uniform')
        segment_window = dc.get('segment_window', 64)
        self.train_dataset = KeyframeClipDataset(
            ann_file=dc.train_ann, data_root=dc.root,
            clip_len=dc.clip_len, frame_interval=dc.frame_interval,
            jitter_range=dc.jitter_range,
            sampling=sampling, segment_window=segment_window,
            transform=build_train_transforms(img_size=img_size))
        self.val_dataset = KeyframeClipDataset(
            ann_file=dc.val_ann, data_root=dc.root,
            clip_len=dc.clip_len, frame_interval=dc.frame_interval,
            jitter_range=0,
            sampling=sampling, segment_window=segment_window,
            transform=build_val_transforms(img_size=img_size))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,
                          shuffle=True, num_workers=self.cfg.num_workers,
                          pin_memory=_PIN_MEMORY,
                          persistent_workers=(self.cfg.num_workers > 0))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          pin_memory=_PIN_MEMORY,
                          persistent_workers=(self.cfg.num_workers > 0))
```

- [ ] **Step 4: Verify existing tests still pass**

```bash
uv run pytest tests/test_dataset.py tests/test_module.py -v
```

Expected: all tests PASS (no behaviour change yet since new params are optional with defaults).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml data/datamodule.py
git commit -m "feat: add pytorchvideo dep, pass img_size/sampling to datamodule"
```

---

## Task 2: Add segment sampling to KeyframeClipDataset (TDD)

**Files:**
- Modify: `data/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests for segment sampling**

Append to `tests/test_dataset.py`:

```python
def test_segment_sampling_returns_correct_count(fake_dataset):
    frames_dir, ann_file = fake_dataset
    ds = KeyframeClipDataset(
        ann_file=ann_file, data_root=frames_dir,
        clip_len=8, frame_interval=1,
        sampling='segment', segment_window=32)
    frames_list, _ = ds[0]
    # without transform, __getitem__ returns a list of 8 PIL images
    assert len(frames_list) == 8


def test_segment_sampling_indices_span_window(fake_dataset):
    """Each of the 8 segments should contribute exactly one index."""
    frames_dir, ann_file = fake_dataset
    ds = KeyframeClipDataset(
        ann_file=ann_file, data_root=frames_dir,
        clip_len=8, frame_interval=1,
        sampling='segment', segment_window=32, jitter_range=0)
    # keyframe=50, window=32 → [34..65], 8 segs of 4 frames each
    indices = ds._sample_segment_indices(
        keyframe_id=50, total_frames=100, is_train=False)
    assert len(indices) == 8
    # val: deterministic middle of each segment, must be sorted
    assert indices == sorted(indices)


def test_segment_shape_with_transform(fake_dataset):
    from data.transforms import build_val_transforms
    frames_dir, ann_file = fake_dataset
    ds = KeyframeClipDataset(
        ann_file=ann_file, data_root=frames_dir,
        clip_len=8, frame_interval=1,
        sampling='segment', segment_window=32,
        transform=build_val_transforms(img_size=224))
    frames, label = ds[0]
    # transform stacks: (C, T, H, W) = (3, 8, 224, 224)
    assert frames.shape == (3, 8, 224, 224)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_dataset.py::test_segment_sampling_returns_correct_count -v
```

Expected: FAIL with `TypeError` (unexpected keyword arg `sampling`).

- [ ] **Step 3: Implement segment sampling in dataset.py**

Replace `data/dataset.py` with:

```python
"""KeyframeClipDataset: pure PyTorch, no mm dependencies."""
import os
import random
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class KeyframeClipDataset(Dataset):
    """Dataset for keyframe-annotated action recognition.

    Args:
        ann_file: CSV with columns [video_id, keyframe_id, action_id].
        data_root: Root dir containing per-video frame folders.
        clip_len: Frames per clip (or segments for TSN). Default 32.
        frame_interval: Sampling interval (uniform mode only). Default 1.
        jitter_range: Max temporal jitter (0=no jitter). Default 0.
        sampling: 'uniform' (default) or 'segment' (TSN-style).
        segment_window: Total frame window for segment mode. Default 64.
        transform: Optional transform applied to list of PIL Images.
        filename_tmpl: Frame filename template. Default 'img_{:05d}.jpg'.
    """

    def __init__(self, ann_file, data_root, clip_len=32, frame_interval=1,
                 jitter_range=0, sampling='uniform', segment_window=64,
                 transform=None, filename_tmpl='img_{:05d}.jpg'):
        self.data_root = data_root
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.jitter_range = jitter_range
        self.sampling = sampling
        self.segment_window = segment_window
        self.transform = transform
        self.filename_tmpl = filename_tmpl
        self.samples = self._load_annotations(ann_file)
        self._total_frames_cache = self._build_frame_count_cache()
        # True during training (stochastic segment pick), False for val
        self._is_train = True

    def _load_annotations(self, ann_file):
        df = pd.read_csv(ann_file)
        samples = []
        for _, row in df.iterrows():
            samples.append({
                'video_id': str(int(row['video_id'])),
                'keyframe_id': int(row['keyframe_id']),
                'label': int(row['action_id']) - 1,
            })
        return samples

    def _build_frame_count_cache(self):
        cache = {}
        for s in self.samples:
            vid = s['video_id']
            if vid not in cache:
                frame_dir = os.path.join(self.data_root, vid)
                cache[vid] = len([f for f in os.listdir(frame_dir)
                                   if f.endswith('.jpg') or f.endswith('.png')])
        return cache

    def _get_total_frames(self, video_id):
        return self._total_frames_cache[video_id]

    def _sample_frame_indices(self, keyframe_id, total_frames):
        """Uniform (existing) sampling around keyframe."""
        center = keyframe_id
        if self.jitter_range > 0:
            center += random.randint(-self.jitter_range, self.jitter_range)
        half = (self.clip_len * self.frame_interval) // 2
        start = center - half
        indices = [start + i * self.frame_interval for i in range(self.clip_len)]
        return [max(1, min(total_frames, idx)) for idx in indices]

    def _sample_segment_indices(self, keyframe_id, total_frames, is_train=True):
        """TSN segment sampling: divide window into clip_len equal segments,
        pick one frame per segment (random during training, middle for val).

        Args:
            keyframe_id: Center frame of the window.
            total_frames: Total frames in the video (for boundary clamping).
            is_train: If True, pick random frame in each segment; else middle.

        Returns:
            List of clip_len frame indices (1-based, clamped to [1, total_frames]).
        """
        center = keyframe_id
        if self.jitter_range > 0 and is_train:
            center += random.randint(-self.jitter_range, self.jitter_range)

        half = self.segment_window // 2
        win_start = center - half          # inclusive, may be < 1
        win_end = win_start + self.segment_window  # exclusive

        seg_len = self.segment_window // self.clip_len
        indices = []
        for i in range(self.clip_len):
            seg_start = win_start + i * seg_len
            seg_end = seg_start + seg_len  # exclusive
            if is_train:
                frame_idx = random.randint(seg_start, seg_end - 1)
            else:
                frame_idx = (seg_start + seg_end - 1) // 2  # middle
            indices.append(max(1, min(total_frames, frame_idx)))
        return indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        total_frames = self._get_total_frames(sample['video_id'])
        if self.sampling == 'segment':
            frame_indices = self._sample_segment_indices(
                sample['keyframe_id'], total_frames, is_train=self._is_train)
        else:
            frame_indices = self._sample_frame_indices(
                sample['keyframe_id'], total_frames)
        frame_dir = os.path.join(self.data_root, sample['video_id'])
        frames = [Image.open(os.path.join(frame_dir, self.filename_tmpl.format(fi))).convert('RGB')
                  for fi in frame_indices]
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, sample['label']
```

- [ ] **Step 4: Update datamodule to set _is_train flag**

In `data/datamodule.py`, after creating datasets in `setup()`, add:

```python
        self.train_dataset._is_train = True
        self.val_dataset._is_train = False
```

Add these two lines right after the two `KeyframeClipDataset(...)` calls.

- [ ] **Step 5: Run the new tests**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: all PASS including the 3 new segment tests.

- [ ] **Step 6: Commit**

```bash
git add data/dataset.py data/datamodule.py tests/test_dataset.py
git commit -m "feat: add TSN segment sampling mode to KeyframeClipDataset"
```

---

## Task 3: Implement TSN model (TDD)

**Files:**
- Create: `models/tsn.py`
- Create: `tests/test_tsn.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tsn.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from models.tsn import TSN


def test_tsn_output_shape():
    model = TSN(num_classes=8, num_segments=8, pretrained=False)
    model.eval()
    x = torch.zeros(2, 3, 8, 224, 224)  # (B, C, T, H, W) from DataLoader
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8), f"Expected (2,8), got {out.shape}"


def test_tsn_different_batch_sizes():
    model = TSN(num_classes=8, num_segments=8, pretrained=False)
    model.eval()
    for B in [1, 4]:
        x = torch.zeros(B, 3, 8, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 8)


def test_tsn_num_classes():
    model = TSN(num_classes=10, num_segments=8, pretrained=False)
    model.eval()
    x = torch.zeros(1, 3, 8, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_tsn_consensus_reduces_time():
    """All-zero input should not raise; output finite."""
    model = TSN(num_classes=8, num_segments=4, pretrained=False)
    model.eval()
    x = torch.zeros(1, 3, 4, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_tsn.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models.tsn'`.

- [ ] **Step 3: Implement TSN**

Create `models/tsn.py`:

```python
"""Temporal Segment Networks (TSN) with ImageNet-pretrained ResNet-50 backbone.

Reference: Wang et al., "Temporal Segment Networks", ECCV 2016.
Input shape: (B, C, T, H, W) — matches the project's DataLoader output.
Forward:
  1. Permute to (B, T, C, H, W), reshape to (B*T, C, H, W)
  2. ResNet-50 feature extraction → (B*T, 2048)
  3. Temporal consensus (mean) → (B, 2048)
  4. Linear classifier → (B, num_classes)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class TSN(nn.Module):
    """TSN with ResNet-50 backbone and average-pooling consensus function.

    Args:
        num_classes: Number of output classes.
        num_segments: Number of temporal segments (= clip_len in config).
        pretrained: Load ImageNet-pretrained ResNet-50 weights.
    """

    def __init__(self, num_classes: int, num_segments: int = 8,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.num_segments = num_segments
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        # Remove avgpool and fc; keep everything up to layer4
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # → (B*T,2048,h,w)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, T, H, W) — DataLoader output, T == num_segments.
        Returns:
            logits: (B, num_classes)
        """
        B, C, T, H, W = x.shape
        # Fold time into batch
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)
        x = self.features(x)                          # (B*T, 2048, h, w)
        x = self.pool(x).flatten(1)                   # (B*T, 2048)
        # Temporal consensus: mean over segments
        x = x.view(B, T, -1).mean(dim=1)             # (B, 2048)
        return self.fc(x)                              # (B, num_classes)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_tsn.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/tsn.py tests/test_tsn.py
git commit -m "feat: add TSN model with ImageNet-pretrained ResNet-50 backbone"
```

---

## Task 4: Implement TSM model (TDD)

**Files:**
- Create: `models/tsm.py`
- Create: `tests/test_tsm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tsm.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from models.tsm import TSM, TemporalShift


def test_temporal_shift_shape_preserved():
    shift = TemporalShift(n_segment=8, n_div=8)
    x = torch.randn(16, 64, 7, 7)  # (B*T=2*8, C, H, W)
    out = shift(x)
    assert out.shape == x.shape


def test_temporal_shift_zero_padding():
    """First frame's backward-shifted channels must be zero (no past frame)."""
    shift = TemporalShift(n_segment=4, n_div=8)
    x = torch.ones(4, 8, 1, 1)   # B=1, T=4, C=8, H=W=1
    out = shift(x)
    # n_div=8 → fold = 8//8 = 1 channel shifted per direction
    # fold=1: out[batch=0, t=0, :1] should be 0 (no previous frame)
    assert out.view(1, 4, 8, 1, 1)[0, 0, 0, 0, 0].item() == 0.0


def test_tsm_output_shape():
    model = TSM(num_classes=8, num_segments=8, pretrained=False)
    model.eval()
    x = torch.zeros(2, 3, 8, 224, 224)  # (B, C, T, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8)


def test_tsm_num_classes():
    model = TSM(num_classes=10, num_segments=8, pretrained=False)
    model.eval()
    x = torch.zeros(1, 3, 8, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_tsm_has_temporal_shift_modules():
    """TemporalShift submodules must be registered on every Bottleneck block."""
    from torchvision.models.resnet import Bottleneck
    model = TSM(num_classes=8, num_segments=4, pretrained=False)
    shift_count = sum(1 for m in model.backbone.modules()
                      if isinstance(m, TemporalShift))
    # ResNet-50 has 3+4+6+3 = 16 Bottleneck blocks
    assert shift_count == 16, f"Expected 16 TemporalShift modules, got {shift_count}"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_tsm.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models.tsm'`.

- [ ] **Step 3: Implement TSM**

Create `models/tsm.py`:

```python
"""Temporal Shift Module (TSM) with ImageNet-pretrained ResNet-50 backbone.

Reference: Lin et al., "TSM: Temporal Shift Module for Efficient Video
Understanding", ICCV 2019.

The shift is inserted as a forward pre-hook into every Bottleneck block.
Since TemporalShift has no learnable parameters, this does not affect
the backbone's state_dict; pretrained weights load without key conflicts.

Input shape: (B, C, T, H, W) — matches the project's DataLoader output.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import Bottleneck


class TemporalShift(nn.Module):
    """Shift a fraction of channels along the temporal dimension.

    Given (B*T, C, H, W), reshapes to (B, T, C, H, W), shifts:
      - channels [:fold]      ← shift from t-1 (past context)
      - channels [fold:2*fold] ← shift from t+1 (future context)
      - channels [2*fold:]     unchanged
    Boundary frames are zero-padded.

    Args:
        n_segment: Number of temporal frames T.
        n_div: Denominator for fold size; fold = C // n_div.
    """

    def __init__(self, n_segment: int, n_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        n = nt // self.n_segment
        x = x.view(n, self.n_segment, c, h, w)
        fold = c // self.n_div
        out = torch.zeros_like(x)
        # Past context: out[t] ← x[t-1] for first `fold` channels
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # Future context: out[t] ← x[t+1] for next `fold` channels
        out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
        # Unchanged channels
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


def _insert_temporal_shift(backbone: nn.Module, n_segment: int,
                            n_div: int = 8) -> None:
    """Register TemporalShift pre-hooks on every Bottleneck block in-place."""
    block_idx = 0
    for module in backbone.modules():
        if isinstance(module, Bottleneck):
            shift = TemporalShift(n_segment=n_segment, n_div=n_div)
            # Register as a named submodule so it shows in named_modules()
            module.add_module(f'temporal_shift', shift)

            def _make_hook(s: TemporalShift):
                def hook(m, inputs):
                    return (s(inputs[0]),)
                return hook

            module.register_forward_pre_hook(_make_hook(shift))
            block_idx += 1


class TSM(nn.Module):
    """TSM with ResNet-50 backbone and average-pooling consensus function.

    Args:
        num_classes: Number of output classes.
        num_segments: Number of temporal segments (= clip_len in config).
        pretrained: Load ImageNet-pretrained ResNet-50 weights.
        n_div: Shift fraction denominator (default 8 → shift C/8 each dir).
    """

    def __init__(self, num_classes: int, num_segments: int = 8,
                 pretrained: bool = True, n_div: int = 8) -> None:
        super().__init__()
        self.num_segments = num_segments
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        # Remove avgpool and fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        _insert_temporal_shift(self.backbone, n_segment=num_segments, n_div=n_div)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, T, H, W) — DataLoader output.
        Returns:
            logits: (B, num_classes)
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)                          # (B*T, 2048, h, w)
        x = self.pool(x).flatten(1)                   # (B*T, 2048)
        x = x.view(B, T, -1).mean(dim=1)             # (B, 2048)
        return self.fc(x)                              # (B, num_classes)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_tsm.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/tsm.py tests/test_tsm.py
git commit -m "feat: add TSM model with temporal shift on ResNet-50 backbone"
```

---

## Task 5: Implement BaselineLightningModule (TDD)

**Files:**
- Create: `engine/baseline_module.py`
- Create: `tests/test_baseline_module.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_baseline_module.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from utils.config import Config
from engine.baseline_module import BaselineLightningModule


def _make_cfg(model_type, clip_len=8):
    return Config({
        'model': {
            'type': model_type,
            'pretrained': False,
            'feat_dim': 2048,
        },
        'data': {'num_classes': 8, 'clip_len': clip_len},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 50,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })


def test_tsn_module_forward():
    cfg = _make_cfg('tsn', clip_len=8)
    module = BaselineLightningModule(cfg)
    x = torch.zeros(2, 3, 8, 224, 224)
    logits = module(x)
    assert logits.shape == (2, 8)


def test_tsm_module_forward():
    cfg = _make_cfg('tsm', clip_len=8)
    module = BaselineLightningModule(cfg)
    x = torch.zeros(2, 3, 8, 224, 224)
    logits = module(x)
    assert logits.shape == (2, 8)


def test_training_step_returns_loss():
    cfg = _make_cfg('tsn')
    module = BaselineLightningModule(cfg)
    x = torch.randn(2, 3, 8, 224, 224)
    y = torch.tensor([0, 3])
    loss = module.training_step((x, y), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_configure_optimizers_returns_dict():
    cfg = _make_cfg('tsn')
    module = BaselineLightningModule(cfg)
    result = module.configure_optimizers()
    assert 'optimizer' in result
    assert 'lr_scheduler' in result
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_baseline_module.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'engine.baseline_module'`.

- [ ] **Step 3: Implement BaselineLightningModule**

Create `engine/baseline_module.py`:

```python
"""Generic LightningModule for baseline video recognition models.

Supports: tsn, tsm, c3d, i3d, r2plus1d.
Model is selected by cfg.model.type and built via model_factory().
Reuses the same optimizer, scheduler, and logging as SlowFastSTTALightningModule.
"""
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from utils.config import Config
from utils.metrics import top_k_accuracy


def model_factory(model_type: str, cfg: Config) -> nn.Module:
    """Build and return a model with pretrained weights loaded.

    Args:
        model_type: One of 'tsn', 'tsm', 'c3d', 'i3d', 'r2plus1d'.
        cfg: Full config object.

    Returns:
        nn.Module with classification head for cfg.data.num_classes.
    """
    mc = cfg.model
    dc = cfg.data
    num_classes = dc.get('num_classes', 8)
    pretrained = mc.get('pretrained', True)
    clip_len = dc.get('clip_len', 8)

    if model_type == 'tsn':
        from models.tsn import TSN
        return TSN(num_classes=num_classes, num_segments=clip_len,
                   pretrained=pretrained)

    if model_type == 'tsm':
        from models.tsm import TSM
        return TSM(num_classes=num_classes, num_segments=clip_len,
                   pretrained=pretrained)

    if model_type == 'c3d':
        return _build_c3d(num_classes=num_classes, pretrained=pretrained)

    if model_type == 'i3d':
        return _build_i3d(num_classes=num_classes, pretrained=pretrained)

    if model_type == 'r2plus1d':
        return _build_r2plus1d(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model_type: {model_type!r}. "
                     f"Choose from: tsn, tsm, c3d, i3d, r2plus1d")


def _build_c3d(num_classes: int, pretrained: bool) -> nn.Module:
    """C3D with Sports-1M pretrained weights.

    Downloads Sports-1M checkpoint on first call if pretrained=True.
    Architecture: pytorchvideo create_c3d (VGG-style 5 conv blocks + FC head).
    Head: FC(4096 → num_classes) replaces the pretrained proj layer.
    """
    from pytorchvideo.models.c3d import create_c3d
    # Build with Sports-1M num_classes (487) so backbone weights load cleanly;
    # head is replaced afterward regardless.
    model = create_c3d(model_num_class=487)
    if pretrained:
        import torch
        url = ('https://download.openmmlab.com/mmaction/recognition/c3d/'
               'c3d_sports1m_pretrain_20201016-dcc47ddc.pth')
        ckpt = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        # MMAction2 checkpoint wraps weights under 'state_dict'
        state = ckpt.get('state_dict', ckpt)
        # Remap MMAction2 keys → pytorchvideo C3D keys
        state = _remap_c3d_keys(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[C3D] Missing keys ({len(missing)}): {missing[:5]} ...")
        # Replace head projection for target num_classes
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    return model


def _remap_c3d_keys(state: dict) -> dict:
    """Remap MMAction2 C3D key names to pytorchvideo C3D key names.

    MMAction2 format:  backbone.conv1a.weight → pytorchvideo: blocks.0.conv.weight
    This mapping is approximate; strict=False loading handles residual mismatches.
    Verify and extend this mapping during integration testing if needed.
    """
    mapping = {
        'backbone.conv1a': 'blocks.0.conv',
        'backbone.conv2a': 'blocks.1.conv',
        'backbone.conv3a': 'blocks.2.conv',
        'backbone.conv3b': 'blocks.2.conv',
        'backbone.conv4a': 'blocks.3.conv',
        'backbone.conv4b': 'blocks.3.conv',
        'backbone.conv5a': 'blocks.4.conv',
        'backbone.conv5b': 'blocks.4.conv',
        'cls_head.fc1': 'blocks.5.fc1',
        'cls_head.fc2': 'blocks.5.fc2',
        'cls_head.fc_cls': 'blocks.5.proj',
    }
    new_state = {}
    for k, v in state.items():
        new_key = k
        for old_prefix, new_prefix in mapping.items():
            if k.startswith(old_prefix):
                new_key = k.replace(old_prefix, new_prefix, 1)
                break
        new_state[new_key] = v
    return new_state


def _build_i3d(num_classes: int, pretrained: bool) -> nn.Module:
    """I3D ResNet-50 with Kinetics-400 pretrained weights via pytorchvideo hub.

    Head: last block's proj (2048→400) replaced with Linear(2048, num_classes).
    """
    import torch
    model = torch.hub.load(
        'facebookresearch/pytorchvideo:main', 'i3d_r50',
        pretrained=pretrained)
    in_features = model.blocks[-1].proj.in_features   # 2048
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    return model


def _build_r2plus1d(num_classes: int, pretrained: bool) -> nn.Module:
    """R(2+1)D ResNet-50 with Kinetics-400 pretrained weights via pytorchvideo hub.

    Head: last block's proj (512→400) replaced with Linear(512, num_classes).
    """
    import torch
    model = torch.hub.load(
        'facebookresearch/pytorchvideo:main', 'r2plus1d_r50',
        pretrained=pretrained)
    in_features = model.blocks[-1].proj.in_features   # 512
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    return model


class BaselineLightningModule(pl.LightningModule):
    """Training/validation logic for baseline video recognition models.

    Args:
        cfg: Config object with model/train/data/logging sections.
            cfg.model.type must be one of: tsn, tsm, c3d, i3d, r2plus1d.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
        self.cfg = cfg
        tc = cfg.train
        model_type = cfg.model.get('type', 'tsn')
        self.model = model_factory(model_type, cfg)
        self.criterion = nn.CrossEntropyLoss()
        self._warmup_epochs = tc.warmup_epochs
        self._warmup_start_factor = tc.warmup_start_factor
        self._max_epochs = tc.max_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc1 = top_k_accuracy(logits, y, k=1)
        acc5 = top_k_accuracy(logits, y, k=min(5, logits.size(1)))
        if self._trainer is not None:
            self.log('train/loss', loss, on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self.log('train/acc1', acc1, on_step=False, on_epoch=True,
                     sync_dist=True)
            self.log('train/acc5', acc5, on_step=False, on_epoch=True,
                     sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc1 = top_k_accuracy(logits, y, k=1)
        acc5 = top_k_accuracy(logits, y, k=min(5, logits.size(1)))
        if self._trainer is not None:
            self.log('val/loss', loss, on_epoch=True, prog_bar=True,
                     sync_dist=True)
            self.log('val/acc1', acc1, on_epoch=True, prog_bar=True,
                     sync_dist=True)
            self.log('val/acc5', acc5, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        tc = self.cfg.train
        optimizer = SGD(
            self.parameters(),
            lr=tc.lr,
            momentum=tc.momentum,
            weight_decay=tc.weight_decay,
        )
        warmup_epochs = self._warmup_epochs
        start_factor = self._warmup_start_factor
        total_epochs = self._max_epochs

        def lr_lambda(epoch: int) -> float:
            cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / 160))
            if epoch < warmup_epochs:
                linear = start_factor + (1.0 - start_factor) * epoch / max(warmup_epochs, 1)
                return linear * cosine
            return cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        nn.utils.clip_grad_norm_(self.parameters(),
                                  max_norm=self.cfg.train.grad_clip)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_baseline_module.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
uv run pytest tests/ -v --ignore=tests/test_baseline_module.py -x
```

Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add engine/baseline_module.py tests/test_baseline_module.py
git commit -m "feat: add BaselineLightningModule and model_factory (tsn, tsm)"
```

---

## Task 6: Add C3D, I3D, R(2+1)D to model_factory (integration tests)

**Files:**
- Modify: `tests/test_baseline_module.py`

Note: These tests use `pretrained=False` to avoid downloading checkpoints during CI.
The pytorchvideo models are built but weights are not downloaded.

- [ ] **Step 1: Write tests for 3D models**

Append to `tests/test_baseline_module.py`:

```python
def test_c3d_module_forward():
    """C3D: input (B, C, 16, 112, 112), output (B, 8)."""
    cfg = Config({
        'model': {'type': 'c3d', 'pretrained': False, 'feat_dim': 4096},
        'data': {'num_classes': 8, 'clip_len': 16},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 50,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })
    module = BaselineLightningModule(cfg)
    module.eval()
    x = torch.zeros(1, 3, 16, 112, 112)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 8)


def test_i3d_module_forward():
    """I3D: input (B, C, 32, 224, 224), output (B, 8)."""
    cfg = Config({
        'model': {'type': 'i3d', 'pretrained': False, 'feat_dim': 2048},
        'data': {'num_classes': 8, 'clip_len': 32},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 50,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })
    module = BaselineLightningModule(cfg)
    module.eval()
    x = torch.zeros(1, 3, 32, 224, 224)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 8)


def test_r2plus1d_module_forward():
    """R(2+1)D: input (B, C, 16, 112, 112), output (B, 8)."""
    cfg = Config({
        'model': {'type': 'r2plus1d', 'pretrained': False, 'feat_dim': 512},
        'data': {'num_classes': 8, 'clip_len': 16},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 50,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })
    module = BaselineLightningModule(cfg)
    module.eval()
    x = torch.zeros(1, 3, 16, 112, 112)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 8)


def test_unknown_model_type_raises():
    cfg = _make_cfg('nonexistent')
    with pytest.raises(ValueError, match="Unknown model_type"):
        BaselineLightningModule(cfg)
```

- [ ] **Step 2: Run new tests**

```bash
uv run pytest tests/test_baseline_module.py -v
```

Expected: all 8 tests PASS (pytorchvideo builds models without downloading when pretrained=False).

> **Note on C3D key mapping:** When running with `pretrained=True` for the first time,
> `strict=False` loading will print missing/unexpected keys. If many backbone keys are
> missing, extend `_remap_c3d_keys()` in `engine/baseline_module.py` by inspecting
> the actual checkpoint: `print(list(state.keys())[:20])` after loading the ckpt.
> The pytorchvideo C3D key structure can be inspected with:
> `print([n for n, _ in model.named_parameters()])`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_baseline_module.py
git commit -m "test: add C3D/I3D/R(2+1)D forward shape tests for BaselineLightningModule"
```

---

## Task 7: Update train.py + create YAML configs

**Files:**
- Modify: `train.py`
- Create: `configs/tsn.yaml`, `configs/tsm.yaml`, `configs/c3d.yaml`, `configs/i3d.yaml`, `configs/r2plus1d.yaml`

- [ ] **Step 1: Update train.py to dispatch by model.type**

In `train.py`, replace:

```python
from engine.module import SlowFastSTTALightningModule
```

with:

```python
from engine.module import SlowFastSTTALightningModule
from engine.baseline_module import BaselineLightningModule
```

Then replace:

```python
    module = SlowFastSTTALightningModule(cfg)
```

with:

```python
    model_type = cfg.model.get('type', 'slowfast_stta')
    if model_type == 'slowfast_stta':
        module = SlowFastSTTALightningModule(cfg)
    else:
        module = BaselineLightningModule(cfg)
```

- [ ] **Step 2: Create configs/tsn.yaml**

```yaml
_base_: base.yaml

model:
  type: tsn
  pretrained: true
  feat_dim: 2048

data:
  clip_len: 8
  sampling: segment
  segment_window: 64
  img_size: 224

train:
  lr: 0.01
  warmup_epochs: 5
  warmup_start_factor: 0.1
  max_epochs: 100

logging:
  wandb_name: tsn_baseline
```

- [ ] **Step 3: Create configs/tsm.yaml**

```yaml
_base_: base.yaml

model:
  type: tsm
  pretrained: true
  feat_dim: 2048

data:
  clip_len: 8
  img_size: 224

train:
  lr: 0.01
  warmup_epochs: 5
  warmup_start_factor: 0.1
  max_epochs: 100

logging:
  wandb_name: tsm_baseline
```

- [ ] **Step 4: Create configs/c3d.yaml**

```yaml
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
  warmup_epochs: 5
  warmup_start_factor: 0.1
  max_epochs: 100

logging:
  wandb_name: c3d_baseline
```

- [ ] **Step 5: Create configs/i3d.yaml**

```yaml
_base_: base.yaml

model:
  type: i3d
  pretrained: true
  feat_dim: 2048

data:
  clip_len: 32
  img_size: 224

train:
  lr: 0.01
  warmup_epochs: 5
  warmup_start_factor: 0.1
  max_epochs: 100

logging:
  wandb_name: i3d_baseline
```

- [ ] **Step 6: Create configs/r2plus1d.yaml**

```yaml
_base_: base.yaml

model:
  type: r2plus1d
  pretrained: true
  feat_dim: 512

data:
  clip_len: 16
  img_size: 112

train:
  lr: 0.01
  warmup_epochs: 5
  warmup_start_factor: 0.1
  max_epochs: 100

logging:
  wandb_name: r2plus1d_baseline
```

- [ ] **Step 7: Verify config loading works for all new configs**

```bash
uv run python -c "
from utils.config import load_config
for name in ['tsn', 'tsm', 'c3d', 'i3d', 'r2plus1d']:
    cfg = load_config(f'configs/{name}.yaml')
    print(f'{name}: type={cfg.model.type}, clip_len={cfg.data.clip_len}, img_size={cfg.data.get(\"img_size\",224)}')
"
```

Expected output:
```
tsn: type=tsn, clip_len=8, img_size=224
tsm: type=tsm, clip_len=8, img_size=224
c3d: type=c3d, clip_len=16, img_size=112
i3d: type=i3d, clip_len=32, img_size=224
r2plus1d: type=r2plus1d, clip_len=16, img_size=112
```

- [ ] **Step 8: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add train.py configs/tsn.yaml configs/tsm.yaml configs/c3d.yaml configs/i3d.yaml configs/r2plus1d.yaml
git commit -m "feat: add train.py dispatch and YAML configs for all 5 baseline models"
```

---

## Appendix: Running a Baseline Experiment

To train any baseline model:

```bash
# TSN
uv run train.py --config configs/tsn.yaml

# C3D (downloads Sports-1M checkpoint on first run, ~400MB)
uv run train.py --config configs/c3d.yaml

# I3D (downloads Kinetics-400 checkpoint on first run, ~90MB)
uv run train.py --config configs/i3d.yaml

# R(2+1)D (downloads Kinetics-400 checkpoint on first run, ~130MB)
uv run train.py --config configs/r2plus1d.yaml
```

Checkpoints and logs are saved to `work_dirs/<config_name>/`.
