import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from models.tsn import TSN


def test_tsn_output_shape():
    model = TSN(num_classes=8, num_segments=8, pretrained=False)
    model.eval()
    x = torch.zeros(2, 3, 8, 224, 224)  # (B, C, T, H, W) from DataLoader
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8), f"Expected (2,8), got {out.shape}"


def test_tsn_different_batch_sizes():
    model = TSN(num_classes=8, num_segments=8, pretrained=False)
    model.eval()
    for B in [1, 4]:
        x = torch.zeros(B, 3, 8, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 8)


def test_tsn_num_classes():
    model = TSN(num_classes=10, num_segments=8, pretrained=False)
    model.eval()
    x = torch.zeros(1, 3, 8, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_tsn_consensus_reduces_time():
    """All-zero input should not raise; output finite."""
    model = TSN(num_classes=8, num_segments=4, pretrained=False)
    model.eval()
    x = torch.zeros(1, 3, 4, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()
