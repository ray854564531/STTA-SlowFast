import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.slowfast import ResNet3dSlowFast


def _make_slowfast():
    return ResNet3dSlowFast(
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            depth=50,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1),
        ),
        fast_pathway=dict(
            depth=50,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(1, 1, 1, 1),
            spatial_strides=(1, 2, 2, 1),
        ),
    )


def test_slowfast_output_shapes():
    """slow: (B,2048,8,14,14), fast: (B,256,32,14,14)."""
    model = _make_slowfast()
    model.eval()
    x = torch.randn(2, 3, 32, 224, 224)
    with torch.no_grad():
        slow_out, fast_out = model(x)
    assert slow_out.shape == (2, 2048, 8, 14, 14), f"slow: {slow_out.shape}"
    assert fast_out.shape == (2, 256, 32, 14, 14), f"fast: {fast_out.shape}"


def test_slowfast_combined_channels():
    """slow + fast channels = 2304."""
    model = _make_slowfast()
    model.eval()
    x = torch.randn(1, 3, 32, 224, 224)
    with torch.no_grad():
        slow_out, fast_out = model(x)
    assert slow_out.shape[1] + fast_out.shape[1] == 2304


from models.slowfast_stta import SlowFastWithSTTA


def _make_slowfast_stta(enable_tcw=True, enable_tch=True, enable_thw=True, enable_stta=True):
    return SlowFastWithSTTA(
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            depth=50,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1),
        ),
        fast_pathway=dict(
            depth=50,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(1, 1, 1, 1),
            spatial_strides=(1, 2, 2, 1),
        ),
        stta_kernel_size=7,
        stta_stages=[True, True, True, True],
        enable_stta=enable_stta,
        stta_enable_tcw=enable_tcw,
        stta_enable_tch=enable_tch,
        stta_enable_thw=enable_thw,
    )


def test_stta_output_shapes():
    """STTA does not change slow/fast output shapes."""
    model = _make_slowfast_stta()
    model.eval()
    x = torch.randn(2, 3, 32, 224, 224)
    with torch.no_grad():
        slow_out, fast_out = model(x)
    assert slow_out.shape == (2, 2048, 8, 14, 14)
    assert fast_out.shape == (2, 256, 32, 14, 14)


def test_stta_ablation_single_branch():
    """Single branch ablation: shapes unchanged."""
    for tcw, tch, thw in [(True,False,False),(False,True,False),(False,False,True)]:
        model = _make_slowfast_stta(enable_tcw=tcw, enable_tch=tch, enable_thw=thw)
        model.eval()
        x = torch.randn(1, 3, 32, 224, 224)
        with torch.no_grad():
            slow_out, fast_out = model(x)
        assert slow_out.shape == (1, 2048, 8, 14, 14)
        assert fast_out.shape == (1, 256, 32, 14, 14)


def test_stta_modules_attached():
    """STTA modules are attached to fast pathway."""
    model = _make_slowfast_stta()
    assert hasattr(model.fast_path, 'stta_modules')
    for i in range(1, 5):
        assert f'layer{i}_stta' in model.fast_path.stta_modules


def test_stta_baseline_no_modules():
    """enable_stta=False: no stta_modules created."""
    model = _make_slowfast_stta(enable_stta=False)
    assert not hasattr(model.fast_path, 'stta_modules')


from models.head import SlowFastHead


def test_head_output():
    head = SlowFastHead(in_channels=2304, num_classes=8, dropout=0.5)
    slow = torch.randn(2, 2048, 8, 14, 14)
    fast = torch.randn(2, 256, 32, 14, 14)
    logits = head((slow, fast))
    assert logits.shape == (2, 8), f"Got {logits.shape}"


def test_full_model_end_to_end():
    """Full forward: SlowFastWithSTTA + SlowFastHead."""
    from models.slowfast_stta import SlowFastWithSTTA
    backbone = _make_slowfast_stta()
    head = SlowFastHead(in_channels=2304, num_classes=8, dropout=0.5)
    backbone.eval(); head.eval()
    x = torch.randn(1, 3, 32, 224, 224)
    with torch.no_grad():
        logits = head(backbone(x))
    assert logits.shape == (1, 8)
