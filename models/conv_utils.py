"""Replacements for mmcv/mmengine utilities used in ResNet3d."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class ConvModule(nn.Module):
    """Conv3d + optional BN + optional ReLU.

    Drop-in replacement for mmcv.cnn.ConvModule used in ResNet3d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = False,
        with_bn: bool = True,
        with_relu: bool = True,
        # Accept and ignore mm-style cfg dicts
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        # Determine bn/relu from mm-style cfg if provided
        if norm_cfg is not None:
            with_bn = norm_cfg is not None
        if act_cfg is not None:
            with_relu = act_cfg is not None

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1) if with_bn else None
        self.relu = nn.ReLU(inplace=True) if with_relu else None

        # Expose norm as .bn for compatibility with inflate_weights
        self.norm = self.bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def kaiming_init(module: nn.Module, a: float = 0, mode: str = 'fan_out',
                 nonlinearity: str = 'relu') -> None:
    """Kaiming initialization for Conv layers."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    """Constant initialization."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
