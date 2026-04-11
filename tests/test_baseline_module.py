import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from utils.config import Config
from engine.baseline_module import BaselineLightningModule


def _make_cfg(model_type, clip_len=8):
    return Config({
        'model': {
            'type': model_type,
            'pretrained': False,
            'feat_dim': 2048,
        },
        'data': {'num_classes': 8, 'clip_len': clip_len},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 50,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })


def test_tsn_module_forward():
    cfg = _make_cfg('tsn', clip_len=8)
    module = BaselineLightningModule(cfg)
    x = torch.zeros(2, 3, 8, 224, 224)
    logits = module(x)
    assert logits.shape == (2, 8)


def test_tsm_module_forward():
    cfg = _make_cfg('tsm', clip_len=8)
    module = BaselineLightningModule(cfg)
    x = torch.zeros(2, 3, 8, 224, 224)
    logits = module(x)
    assert logits.shape == (2, 8)


def test_training_step_returns_loss():
    cfg = _make_cfg('tsn')
    module = BaselineLightningModule(cfg)
    x = torch.randn(2, 3, 8, 224, 224)
    y = torch.tensor([0, 3])
    loss = module.training_step((x, y), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_configure_optimizers_returns_dict():
    cfg = _make_cfg('tsn')
    module = BaselineLightningModule(cfg)
    result = module.configure_optimizers()
    assert 'optimizer' in result
    assert 'lr_scheduler' in result


def test_invalid_model_type_raises():
    cfg = _make_cfg('nonexistent')
    with pytest.raises(ValueError, match="Unknown model_type"):
        BaselineLightningModule(cfg)


def test_validation_step_runs():
    cfg = _make_cfg('tsn')
    module = BaselineLightningModule(cfg)
    x = torch.randn(2, 3, 8, 224, 224)
    y = torch.tensor([0, 3])
    module.validation_step((x, y), batch_idx=0)  # must not raise


def test_c3d_module_forward():
    """C3D: input (B, C, 16, 112, 112), output (B, 8)."""
    cfg = Config({
        'model': {'type': 'c3d', 'pretrained': False, 'feat_dim': 4096},
        'data': {'num_classes': 8, 'clip_len': 16},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 50,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })
    module = BaselineLightningModule(cfg)
    module.eval()
    x = torch.zeros(1, 3, 16, 112, 112)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 8)


def test_i3d_module_forward():
    """I3D: input (B, C, 32, 224, 224), output (B, 8)."""
    cfg = Config({
        'model': {'type': 'i3d', 'pretrained': False, 'feat_dim': 2048},
        'data': {'num_classes': 8, 'clip_len': 32},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 50,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })
    module = BaselineLightningModule(cfg)
    module.eval()
    x = torch.zeros(1, 3, 32, 224, 224)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 8)


def test_r2plus1d_module_forward():
    """R(2+1)D: input (B, C, 16, 224, 224), output (B, 8).

    The pytorchvideo r2plus1d_r50 head uses AvgPool3d(kernel=(4,7,7));
    224×224 spatial input is required so the spatial dims are ≥7 after
    the backbone's stride-32 downsampling.
    """
    cfg = Config({
        'model': {'type': 'r2plus1d', 'pretrained': False, 'feat_dim': 512},
        'data': {'num_classes': 8, 'clip_len': 16},
        'train': {
            'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 50,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })
    module = BaselineLightningModule(cfg)
    module.eval()
    x = torch.zeros(1, 3, 16, 224, 224)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 8)
