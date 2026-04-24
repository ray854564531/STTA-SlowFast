import pytest

from tests._video_fixtures import make_kinetics_tree
from data.kinetics_dataset import KineticsVideoDataset


@pytest.fixture
def k400_tree(tmp_path):
    make_kinetics_tree(
        tmp_path, classes=['abseiling', 'air_drumming', 'archery'],
        videos_per_class=2, split='train', num_frames=60,
    )
    make_kinetics_tree(
        tmp_path, classes=['abseiling', 'air_drumming', 'archery'],
        videos_per_class=1, split='val', num_frames=60,
    )
    return str(tmp_path)


def test_train_index_len_and_class_map(k400_tree):
    ds = KineticsVideoDataset(root=k400_tree, split='train', mode='train',
                               clip_len=8, frame_interval=2)
    assert len(ds) == 6
    assert ds.class_to_idx == {
        'abseiling': 0, 'air_drumming': 1, 'archery': 2,
    }


def test_val_index_len(k400_tree):
    ds = KineticsVideoDataset(root=k400_tree, split='val', mode='val',
                               clip_len=8, frame_interval=2)
    assert len(ds) == 3


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        KineticsVideoDataset(root=str(tmp_path), split='train', mode='train',
                              clip_len=8, frame_interval=2)


import torch
from data.video_transforms import build_train_video_transform


def test_train_getitem_shape(k400_tree):
    tfm = build_train_video_transform((256, 320), 224,
                                      [0.45]*3, [0.225]*3)
    ds = KineticsVideoDataset(root=k400_tree, split='train', mode='train',
                               clip_len=8, frame_interval=2, transform=tfm)
    clip, label = ds[0]
    assert clip.shape == (3, 8, 224, 224)
    assert clip.dtype == torch.float32
    assert 0 <= label < 3


def test_train_random_across_calls(k400_tree):
    tfm = build_train_video_transform((256, 320), 224,
                                      [0.45]*3, [0.225]*3)
    ds = KineticsVideoDataset(root=k400_tree, split='train', mode='train',
                               clip_len=8, frame_interval=2, transform=tfm)
    out1, _ = ds[0]
    out2, _ = ds[0]
    assert out1.shape == out2.shape


def test_short_video_is_loop_padded(tmp_path):
    from tests._video_fixtures import make_mp4
    make_mp4(tmp_path / 'train' / 'x' / '0.mp4', num_frames=5)
    tfm = build_train_video_transform((64, 64), 32,
                                      [0.45]*3, [0.225]*3)
    ds = KineticsVideoDataset(root=str(tmp_path), split='train', mode='train',
                               clip_len=8, frame_interval=2, transform=tfm)
    clip, _ = ds[0]
    assert clip.shape[1] == 8


from data.video_transforms import build_val_video_transform


def test_val_getitem_shape_and_determinism(k400_tree):
    tfm = build_val_video_transform(256, 224, [0.45]*3, [0.225]*3)
    ds = KineticsVideoDataset(root=k400_tree, split='val', mode='val',
                               clip_len=8, frame_interval=2, transform=tfm)
    clip1, label1 = ds[0]
    clip2, label2 = ds[0]
    assert clip1.shape == (3, 8, 224, 224)
    assert label1 == label2
    assert torch.equal(clip1, clip2)


from data.video_transforms import build_test_three_crop_transform


def test_test_mode_30_view_shape(k400_tree):
    tfm = build_test_three_crop_transform(256, 224, [0.45]*3, [0.225]*3)
    ds = KineticsVideoDataset(root=k400_tree, split='val', mode='test',
                               clip_len=8, frame_interval=2,
                               num_clips=10, num_crops=3, transform=tfm)
    clips, label = ds[0]
    assert clips.shape == (30, 3, 8, 224, 224)
    assert isinstance(label, int)


def test_test_mode_single_crop(k400_tree):
    tfm = build_val_video_transform(256, 224, [0.45]*3, [0.225]*3)
    ds = KineticsVideoDataset(root=k400_tree, split='val', mode='test',
                               clip_len=8, frame_interval=2,
                               num_clips=5, num_crops=1, transform=tfm)
    clips, _ = ds[0]
    assert clips.shape == (5, 3, 8, 224, 224)
