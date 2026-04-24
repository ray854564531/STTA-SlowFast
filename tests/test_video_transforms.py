from pathlib import Path

from tests._video_fixtures import make_mp4


def test_fixture_makes_mp4(tmp_path):
    p = make_mp4(tmp_path / 'a.mp4', num_frames=10)
    assert Path(p).exists()
    assert Path(p).stat().st_size > 0
