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
