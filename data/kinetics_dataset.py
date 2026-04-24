"""KineticsVideoDataset: raw-mp4 online-decoding dataset for Kinetics-400."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

_VIDEO_EXTS = ('.mp4', '.mkv', '.webm', '.avi')


class KineticsVideoDataset(Dataset):
    """Kinetics dataset that decodes raw videos on the fly via decord.

    Args:
        root: Dataset root containing ``<split>/<class_name>/*.mp4``.
        split: ``train`` or ``val``.
        clip_len: Frames per clip.
        frame_interval: Sampling stride (in frames).
        num_clips: Number of temporal clips (>1 enables multi-view test mode).
        num_crops: 1 (train/val) or 3 (three-crop test mode).
        mode: ``train`` / ``val`` / ``test``.
        transform: Callable. For ``mode='test'`` with ``num_crops=3``, must
            return a list of 3 tensors.
        decode_backend: Currently only ``decord``.
    """

    def __init__(
        self,
        root: str,
        split: str,
        clip_len: int = 32,
        frame_interval: int = 2,
        num_clips: int = 1,
        num_crops: int = 1,
        mode: str = 'train',
        transform=None,
        decode_backend: str = 'decord',
    ):
        self.root = root
        self.split = split
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.num_crops = num_crops
        self.mode = mode
        self.transform = transform
        self.decode_backend = decode_backend

        split_dir = Path(root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f'Split dir not found: {split_dir}')

        self.classes = sorted(
            d.name for d in split_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            cls_path = split_dir / cls
            for name in sorted(os.listdir(cls_path)):
                if name.lower().endswith(_VIDEO_EXTS):
                    self.samples.append((str(cls_path / name), cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError  # filled in Task 5
