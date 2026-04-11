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
