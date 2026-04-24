import pytest

from tests._video_fixtures import make_kinetics_tree
from data.kinetics_datamodule import KineticsDataModule


class _Ns(dict):
    def __getattr__(self, k):
        v = self[k]
        return _Ns(v) if isinstance(v, dict) else v

    def get(self, k, default=None):
        v = dict.get(self, k, default)
        return _Ns(v) if isinstance(v, dict) else v


def _cfg(root):
    return _Ns({
        'data': {
            'type': 'kinetics',
            'root': root,
            'num_classes': 3,
            'clip_len': 8,
            'frame_interval': 2,
            'img_size': 224,
            'short_side_scale_train': [256, 320],
            'short_side_scale_eval': 256,
            'mean': [0.45, 0.45, 0.45],
            'std': [0.225, 0.225, 0.225],
            'batch_size': 2,
            'num_workers': 0,
            'decode_backend': 'decord',
        },
        'test': {
            'enabled': True,
            'num_clips': 10,
            'num_crops': 3,
            'batch_size': 1,
        },
    })


@pytest.fixture
def k400_tree(tmp_path):
    make_kinetics_tree(tmp_path, ['a', 'b', 'c'], 2, 'train', 60)
    make_kinetics_tree(tmp_path, ['a', 'b', 'c'], 1, 'val', 60)
    return str(tmp_path)


def test_train_batch_shape(k400_tree):
    dm = KineticsDataModule(_cfg(k400_tree))
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert x.shape == (2, 3, 8, 224, 224)
    assert y.shape == (2,)


def test_val_batch_shape(k400_tree):
    dm = KineticsDataModule(_cfg(k400_tree))
    dm.setup()
    batch = next(iter(dm.val_dataloader()))
    x, y = batch
    assert x.shape[0] >= 1
    assert x.shape[1:] == (3, 8, 224, 224)


def test_test_batch_shape(k400_tree):
    dm = KineticsDataModule(_cfg(k400_tree))
    dm.setup()
    batch = next(iter(dm.test_dataloader()))
    x, y = batch
    assert x.shape[1:] == (30, 3, 8, 224, 224)
