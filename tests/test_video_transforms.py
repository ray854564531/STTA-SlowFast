from pathlib import Path

import torch

from tests._video_fixtures import make_mp4
from data.video_transforms import ShortSideScale, Normalize


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
