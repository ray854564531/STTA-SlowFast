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
        clip_len: Frames per clip. Default 32.
        frame_interval: Sampling interval. Default 1.
        jitter_range: Max temporal jitter (0=no jitter). Default 0.
        transform: Optional transform applied to list of PIL Images.
        filename_tmpl: Frame filename template. Default 'img_{:05d}.jpg'.
    """

    def __init__(self, ann_file, data_root, clip_len=32, frame_interval=1,
                 jitter_range=0, transform=None, filename_tmpl='img_{:05d}.jpg'):
        self.data_root = data_root
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.jitter_range = jitter_range
        self.transform = transform
        self.filename_tmpl = filename_tmpl
        self.samples = self._load_annotations(ann_file)

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

    def _get_total_frames(self, video_id):
        frame_dir = os.path.join(self.data_root, video_id)
        return len([f for f in os.listdir(frame_dir)
                    if f.endswith('.jpg') or f.endswith('.png')])

    def _sample_frame_indices(self, keyframe_id, total_frames):
        center = keyframe_id
        if self.jitter_range > 0:
            center += random.randint(-self.jitter_range, self.jitter_range)
        half = (self.clip_len * self.frame_interval) // 2
        start = center - half
        indices = [start + i * self.frame_interval for i in range(self.clip_len)]
        return [max(1, min(total_frames, idx)) for idx in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        total_frames = self._get_total_frames(sample['video_id'])
        frame_indices = self._sample_frame_indices(sample['keyframe_id'], total_frames)
        frame_dir = os.path.join(self.data_root, sample['video_id'])
        frames = [Image.open(os.path.join(frame_dir, self.filename_tmpl.format(fi))).convert('RGB')
                  for fi in frame_indices]
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, sample['label']
