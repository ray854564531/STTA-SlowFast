import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from models.tsm import TSM, TemporalShift


def test_temporal_shift_shape_preserved():
    shift = TemporalShift(n_segment=8, n_div=8)
    x = torch.randn(16, 64, 7, 7)  # (B*T=2*8, C, H, W)
    out = shift(x)
    assert out.shape == x.shape


def test_temporal_shift_zero_padding():
    """First frame's backward-shifted channels must be zero (no past frame)."""
    shift = TemporalShift(n_segment=4, n_div=8)
    x = torch.ones(4, 8, 1, 1)   # B=1, T=4, C=8, H=W=1
    out = shift(x)
    # n_div=8 → fold = 8//8 = 1 channel shifted per direction
    # fold=1: out[batch=0, t=0, :1] should be 0 (no previous frame)
    assert out.view(1, 4, 8, 1, 1)[0, 0, 0, 0, 0].item() == 0.0


def test_tsm_output_shape():
    model = TSM(num_classes=8, num_segments=8, pretrained=False)
    model.eval()
    x = torch.zeros(2, 3, 8, 224, 224)  # (B, C, T, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8)


def test_tsm_num_classes():
    model = TSM(num_classes=10, num_segments=8, pretrained=False)
    model.eval()
    x = torch.zeros(1, 3, 8, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_tsm_has_temporal_shift_modules():
    """TemporalShift submodules must be registered on every Bottleneck block."""
    from torchvision.models.resnet import Bottleneck
    model = TSM(num_classes=8, num_segments=4, pretrained=False)
    shift_count = sum(1 for m in model.backbone.modules()
                      if isinstance(m, TemporalShift))
    # ResNet-50 has 3+4+6+3 = 16 Bottleneck blocks
    assert shift_count == 16, f"Expected 16 TemporalShift modules, got {shift_count}"
