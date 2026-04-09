import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet3d import ResNet3d


def test_slow_pathway_shape():
    """Slow pathway: (1,3,8,224,224) -> (1,2048,8,14,14)."""
    model = ResNet3d(
        depth=50,
        in_channels=3,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        spatial_strides=(1, 2, 2, 1),
    )
    model.eval()
    x = torch.randn(1, 3, 8, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2048, 8, 14, 14), f"Got {out.shape}"


def test_fast_pathway_shape():
    """Fast pathway: base_channels=8, (1,3,32,224,224) -> (1,256,32,14,14)."""
    model = ResNet3d(
        depth=50,
        in_channels=3,
        base_channels=8,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 1),
    )
    model.eval()
    x = torch.randn(1, 3, 32, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 256, 32, 14, 14), f"Got {out.shape}"
