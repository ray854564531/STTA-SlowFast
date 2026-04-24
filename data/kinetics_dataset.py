"""KineticsVideoDataset: raw-mp4 online-decoding dataset for Kinetics-400."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
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

    # ---- helpers ----------------------------------------------------------

    def _decord_reader(self, path: str):
        import decord
        decord.bridge.set_bridge('native')
        return decord.VideoReader(path, num_threads=1)

    def _sample_train_indices(self, total_frames: int) -> np.ndarray:
        span = self.clip_len * self.frame_interval
        if total_frames >= span:
            start = random.randint(0, total_frames - span)
            idx = np.arange(start, start + span, self.frame_interval)
        else:
            base = np.arange(self.clip_len) * self.frame_interval
            idx = base % total_frames
        return idx.astype(np.int64)

    def _center_start(self, total_frames: int) -> int:
        span = self.clip_len * self.frame_interval
        return max(0, (total_frames - span) // 2)

    def _sample_test_starts(self, total_frames: int) -> List[int]:
        span = self.clip_len * self.frame_interval
        if self.num_clips == 1:
            return [self._center_start(total_frames)]
        if total_frames <= span:
            return [0] * self.num_clips
        max_start = total_frames - span
        return [
            int(round(i * max_start / (self.num_clips - 1)))
            for i in range(self.num_clips)
        ]

    def _sample_val_indices(self, total_frames: int) -> np.ndarray:
        span = self.clip_len * self.frame_interval
        if total_frames >= span:
            start = self._center_start(total_frames)
            idx = np.arange(start, start + span, self.frame_interval)
        else:
            base = np.arange(self.clip_len) * self.frame_interval
            idx = base % total_frames
        return idx.astype(np.int64)

    # ---- main -------------------------------------------------------------

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.mode == 'train':
            return self._get_train(path, label)
        if self.mode == 'val':
            return self._get_val(path, label)
        if self.mode == 'test':
            return self._get_test(path, label)
        raise ValueError(f'Unknown mode: {self.mode!r}')

    def _get_train(self, path: str, label: int):
        vr = self._decord_reader(path)
        total = len(vr)
        indices = self._sample_train_indices(total)
        indices = np.clip(indices, 0, total - 1)
        frames = vr.get_batch(indices).asnumpy()
        clip = torch.from_numpy(frames)
        if self.transform is not None:
            clip = self.transform(clip)
        return clip, label

    def _get_val(self, path: str, label: int):
        vr = self._decord_reader(path)
        total = len(vr)
        indices = np.clip(self._sample_val_indices(total), 0, total - 1)
        frames = vr.get_batch(indices).asnumpy()
        clip = torch.from_numpy(frames)
        if self.transform is not None:
            clip = self.transform(clip)
        return clip, label

    def _get_test(self, path: str, label: int):
        vr = self._decord_reader(path)
        total = len(vr)
        starts = self._sample_test_starts(total)
        span = self.clip_len * self.frame_interval
        all_views = []
        for start in starts:
            if total >= span:
                idx = np.arange(start, start + span, self.frame_interval)
            else:
                idx = (np.arange(self.clip_len) * self.frame_interval) % total
            idx = np.clip(idx.astype(np.int64), 0, total - 1)
            frames = torch.from_numpy(vr.get_batch(idx).asnumpy())
            out = self.transform(frames) if self.transform is not None else frames
            if isinstance(out, (list, tuple)):
                all_views.extend(out)
            else:
                all_views.append(out)
        return torch.stack(all_views, dim=0), label
