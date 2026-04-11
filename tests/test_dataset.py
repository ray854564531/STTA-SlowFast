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
