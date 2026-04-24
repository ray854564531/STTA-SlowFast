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
