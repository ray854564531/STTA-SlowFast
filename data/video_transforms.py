"""Tensor-level video transforms for the Kinetics pipeline.

Input clips are (T, H, W, C) uint8 tensors (decord native layout).
Normalize is the terminal step and outputs (C, T, H, W) float32.
"""
from __future__ import annotations

import random
from typing import Sequence, Union

import torch
import torch.nn.functional as F


Size = Union[int, Sequence[int]]


class ShortSideScale:
    def __init__(self, size: Size):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if isinstance(self.size, int):
            target = self.size
        else:
            lo, hi = int(self.size[0]), int(self.size[1])
            target = random.randint(lo, hi)
        t, h, w, c = clip.shape
        short = min(h, w)
        scale = target / short
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        x = clip.permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear',
                          align_corners=False)
        return x.permute(0, 2, 3, 1).to(clip.dtype)


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        x = clip.permute(3, 0, 1, 2).float().div_(255.0)
        return (x - self.mean) / self.std


class RandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        t, h, w, c = clip.shape
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        return clip[:, top:top + self.size, left:left + self.size, :]


class CenterCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        t, h, w, c = clip.shape
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        return clip[:, top:top + self.size, left:left + self.size, :]


class ThreeCrop:
    """Three spatial crops along the long axis."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor):
        t, h, w, c = clip.shape
        s = self.size
        if w >= h:
            offsets = [(0, 0), (0, (w - s) // 2), (0, w - s)]
        else:
            offsets = [(0, 0), ((h - s) // 2, 0), (h - s, 0)]
        return [clip[:, top:top + s, left:left + s, :] for top, left in offsets]


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(clip, dims=[2])
        return clip


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def build_train_video_transform(short_side_range, crop_size, mean, std):
    return Compose([
        ShortSideScale(tuple(short_side_range)),
        RandomCrop(crop_size),
        RandomHorizontalFlip(0.5),
        Normalize(mean, std),
    ])


def build_val_video_transform(short_side, crop_size, mean, std):
    return Compose([
        ShortSideScale(short_side),
        CenterCrop(crop_size),
        Normalize(mean, std),
    ])


class _ThreeCropNormalize:
    def __init__(self, crop_size, mean, std):
        self.three_crop = ThreeCrop(crop_size)
        self.normalize = Normalize(mean, std)

    def __call__(self, clip):
        return [self.normalize(c) for c in self.three_crop(clip)]


def build_test_three_crop_transform(short_side, crop_size, mean, std):
    return Compose([
        ShortSideScale(short_side),
        _ThreeCropNormalize(crop_size, mean, std),
    ])
