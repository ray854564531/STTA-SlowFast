from pathlib import Path

import torch

from tests._video_fixtures import make_mp4
from data.video_transforms import (
    ShortSideScale, Normalize,
    RandomCrop, CenterCrop, ThreeCrop, RandomHorizontalFlip, Compose,
    build_train_video_transform, build_val_video_transform,
    build_test_three_crop_transform,
)


def test_fixture_makes_mp4(tmp_path):
    p = make_mp4(tmp_path / 'a.mp4', num_frames=10)
    assert Path(p).exists()
    assert Path(p).stat().st_size > 0


def _fake_clip(t=4, h=100, w=200, c=3):
    return torch.randint(0, 255, (t, h, w, c), dtype=torch.uint8)


def test_short_side_scale_fixed():
    clip = _fake_clip(h=100, w=200)
    out = ShortSideScale(256)(clip)
    assert out.shape[1] == 256
    assert out.shape[2] == 512
    assert out.dtype == torch.uint8


def test_short_side_scale_range_respects_range():
    clip = _fake_clip(h=100, w=200)
    out = ShortSideScale((200, 300))(clip)
    assert 200 <= out.shape[1] <= 300


def test_normalize_produces_c_t_h_w_float():
    clip = _fake_clip(t=4, h=224, w=224)
    out = Normalize([0.45]*3, [0.225]*3)(clip)
    assert out.shape == (3, 4, 224, 224)
    assert out.dtype == torch.float32


def test_center_crop_shape():
    clip = _fake_clip(h=256, w=320)
    out = CenterCrop(224)(clip)
    assert out.shape == (4, 224, 224, 3)


def test_random_crop_shape_and_range():
    clip = _fake_clip(h=256, w=320)
    out = RandomCrop(224)(clip)
    assert out.shape == (4, 224, 224, 3)


def test_three_crop_returns_three_tensors():
    clip = _fake_clip(h=256, w=320)
    crops = ThreeCrop(224)(clip)
    assert len(crops) == 3
    for c in crops:
        assert c.shape == (4, 224, 224, 3)


def test_horizontal_flip_deterministic_with_seed(monkeypatch):
    clip = _fake_clip(h=8, w=8)
    monkeypatch.setattr('random.random', lambda: 0.0)
    out = RandomHorizontalFlip(p=0.5)(clip)
    assert torch.equal(out, torch.flip(clip, dims=[2]))


def test_build_train_pipeline_shape():
    clip = _fake_clip(h=240, w=320)
    tfm = build_train_video_transform(
        short_side_range=(256, 320), crop_size=224,
        mean=[0.45]*3, std=[0.225]*3,
    )
    out = tfm(clip)
    assert out.shape == (3, 4, 224, 224)


def test_build_val_pipeline_shape():
    clip = _fake_clip(h=240, w=320)
    tfm = build_val_video_transform(
        short_side=256, crop_size=224,
        mean=[0.45]*3, std=[0.225]*3,
    )
    out = tfm(clip)
    assert out.shape == (3, 4, 224, 224)


def test_build_three_crop_pipeline_returns_three():
    clip = _fake_clip(h=240, w=320)
    tfm = build_test_three_crop_transform(
        short_side=256, crop_size=224,
        mean=[0.45]*3, std=[0.225]*3,
    )
    crops = tfm(clip)
    assert len(crops) == 3
    for c in crops:
        assert c.shape == (3, 4, 224, 224)
