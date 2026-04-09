import pytest
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PIL.Image
from data.dataset import KeyframeClipDataset


@pytest.fixture
def fake_dataset(tmp_path):
    frames_dir = tmp_path / "frames"
    for vid_id in [1, 2]:
        vid_dir = frames_dir / str(vid_id)
        vid_dir.mkdir(parents=True)
        for i in range(1, 101):
            img = PIL.Image.new('RGB', (256, 256), color=(i % 255, i % 255, i % 255))
            img.save(vid_dir / f"img_{i:05d}.jpg")
    ann_file = tmp_path / "train.csv"
    ann_file.write_text("video_id,keyframe_id,action_id\n1,50,1\n2,60,3\n")
    return str(frames_dir), str(ann_file)


def test_dataset_length(fake_dataset):
    frames_dir, ann_file = fake_dataset
    ds = KeyframeClipDataset(ann_file=ann_file, data_root=frames_dir,
                              clip_len=16, frame_interval=1)
    assert len(ds) == 2


def test_dataset_label_zero_based(fake_dataset):
    frames_dir, ann_file = fake_dataset
    ds = KeyframeClipDataset(ann_file=ann_file, data_root=frames_dir,
                              clip_len=16, frame_interval=1)
    _, label = ds[0]
    assert label == 0   # action_id=1 → label=0
    _, label = ds[1]
    assert label == 2   # action_id=3 → label=2


def test_dataset_clip_shape(fake_dataset):
    from data.transforms import build_val_transforms
    frames_dir, ann_file = fake_dataset
    transform = build_val_transforms(img_size=224)
    ds = KeyframeClipDataset(ann_file=ann_file, data_root=frames_dir,
                              clip_len=16, frame_interval=1, transform=transform)
    frames, label = ds[0]
    assert frames.shape == (3, 16, 224, 224), f"Got {frames.shape}"
    assert isinstance(label, int)


def test_dataset_no_jitter(fake_dataset):
    frames_dir, ann_file = fake_dataset
    ds = KeyframeClipDataset(ann_file=ann_file, data_root=frames_dir,
                              clip_len=16, frame_interval=1, jitter_range=0)
    frames, _ = ds[0]
    assert len(frames) == 16
