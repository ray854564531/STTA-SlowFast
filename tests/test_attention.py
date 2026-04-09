import pytest
import torch
import sys
sys.path.insert(0, '..')

from models.attention import STTripletAttention


def test_full_attention_shape():
    """三分支全开，输出形状与输入相同。"""
    attn = STTripletAttention(kernel_size=7, enable_tcw=True, enable_tch=True, enable_thw=True)
    x = torch.randn(2, 64, 8, 14, 14)
    out = attn(x)
    assert out.shape == x.shape


def test_single_branch_tcw():
    attn = STTripletAttention(kernel_size=7, enable_tcw=True, enable_tch=False, enable_thw=False)
    x = torch.randn(2, 64, 8, 14, 14)
    assert attn(x).shape == x.shape


def test_single_branch_tch():
    attn = STTripletAttention(kernel_size=7, enable_tcw=False, enable_tch=True, enable_thw=False)
    x = torch.randn(2, 64, 8, 14, 14)
    assert attn(x).shape == x.shape


def test_single_branch_thw():
    attn = STTripletAttention(kernel_size=7, enable_tcw=False, enable_tch=False, enable_thw=True)
    x = torch.randn(2, 64, 8, 14, 14)
    assert attn(x).shape == x.shape


def test_two_branches_tcw_tch():
    attn = STTripletAttention(kernel_size=7, enable_tcw=True, enable_tch=True, enable_thw=False)
    x = torch.randn(2, 64, 8, 14, 14)
    assert attn(x).shape == x.shape


def test_no_branch_raises():
    with pytest.raises(ValueError):
        STTripletAttention(enable_tcw=False, enable_tch=False, enable_thw=False)


def test_fast_pathway_shape():
    """模拟 fast pathway stage1 输出尺寸。"""
    attn = STTripletAttention(kernel_size=7)
    x = torch.randn(2, 32, 32, 56, 56)
    assert attn(x).shape == x.shape
