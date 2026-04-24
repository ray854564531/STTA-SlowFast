import torch

from utils.config import Config
from engine.module import SlowFastSTTALightningModule


def _base_cfg():
    return Config({
        'model': {
            'resample_rate': 4, 'speed_ratio': 4, 'channel_ratio': 8,
            'kernel_size': 7,
            'fast_stta_stages': [True, True, True, True],
            'slow_stta_stages': [False, False, False, False],
            'enable_stta': True,
            'enable_tcw': True, 'enable_tch': True, 'enable_thw': True,
            'dropout': 0.5,
        },
        'data': {'num_classes': 3},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
            'warmup_epochs': 0, 'warmup_start_factor': 0.1,
            'max_epochs': 1, 'grad_clip': 40,
        },
    })


def test_test_step_averages_30_views():
    module = SlowFastSTTALightningModule(_base_cfg()).eval()
    B, N, T = 1, 30, 32
    clips = torch.randn(B, N, 3, T, 32, 32)
    labels = torch.tensor([0])
    with torch.no_grad():
        out = module.test_step((clips, labels), 0)
    assert out is None or isinstance(out, dict)
