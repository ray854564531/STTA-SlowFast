"""ResNet3d backbone (depth=50 only). Ported from mmaction2, mm-free."""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .conv_utils import ConvModule, constant_init, kaiming_init


class Bottleneck3d(nn.Module):
    """Bottleneck block for ResNet3d-50. expansion=4."""
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        inflate: bool = True,
        inflate_style: str = '3x1x1',
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        assert inflate_style in ('3x1x1', '3x3x3')
        self.with_cp = with_cp

        if inflate:
            if inflate_style == '3x1x1':
                conv1_kernel = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(inplanes, planes,
                                kernel_size=conv1_kernel,
                                stride=(temporal_stride, 1, 1),
                                padding=conv1_padding, bias=False)
        self.conv2 = ConvModule(planes, planes,
                                kernel_size=conv2_kernel,
                                stride=(1, spatial_stride, spatial_stride),
                                padding=conv2_padding,
                                dilation=(1, dilation, dilation), bias=False)
        self.conv3 = ConvModule(planes, planes * self.expansion,
                                kernel_size=1, bias=False, with_relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner(x):
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            return out + identity

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner, x)
        else:
            out = _inner(x)
        return self.relu(out)


class ResNet3d(nn.Module):
    """ResNet3d backbone (depth=50 only).

    Args:
        depth: 50 only.
        in_channels: Input channels. Default 3.
        base_channels: 64 for slow pathway, 8 for fast pathway.
        spatial_strides: Per-stage spatial stride tuple. Default (1,2,2,1).
        temporal_strides: Per-stage temporal stride tuple. Default (1,1,1,1).
        dilations: Per-stage dilation tuple. Default (1,1,1,1).
        conv1_kernel: (1,7,7) for slow, (5,7,7) for fast.
        conv1_stride_t: Temporal stride for conv1. Default 1.
        pool1_stride_t: Temporal stride for maxpool. Default 1.
        inflate: Per-stage inflate flag. (0,0,1,1) for slow, (1,1,1,1) for fast.
        inflate_style: '3x1x1' or '3x3x3'. Default '3x1x1'.
        lateral_inplanes: Extra input channels per stage from lateral connections.
    """
    arch_settings = {50: (Bottleneck3d, (3, 4, 6, 3))}

    def __init__(
        self,
        depth: int = 50,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
        spatial_strides: Tuple = (1, 2, 2, 1),
        temporal_strides: Tuple = (1, 1, 1, 1),
        dilations: Tuple = (1, 1, 1, 1),
        conv1_kernel: Tuple = (5, 7, 7),
        conv1_stride_t: int = 1,
        pool1_stride_t: int = 1,
        inflate: Union[Tuple, int] = (1, 1, 1, 1),
        inflate_style: str = '3x1x1',
        with_cp: bool = False,
        lateral_inplanes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        assert depth in self.arch_settings
        block, stage_blocks_all = self.arch_settings[depth]
        self.block = block
        self.stage_blocks = stage_blocks_all[:num_stages]
        self.base_channels = base_channels
        self.lateral_inplanes = lateral_inplanes or [0] * num_stages

        if isinstance(inflate, int):
            inflate = (inflate,) * num_stages
        self.inflate = inflate

        conv1_padding = tuple((k - 1) // 2 for k in conv1_kernel)
        self.conv1 = ConvModule(
            in_channels, base_channels,
            kernel_size=conv1_kernel,
            stride=(conv1_stride_t, 2, 2),
            padding=conv1_padding, bias=False,
        )
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(pool1_stride_t, 2, 2),
            padding=(0, 1, 1),
        )

        self.res_layers = []
        self.inplanes = base_channels
        for i in range(num_stages):
            planes = base_channels * (2 ** i)
            stage_inplanes = self.inplanes + self.lateral_inplanes[i]
            res_layer = self._make_layer(
                block=block,
                inplanes=stage_inplanes,
                planes=planes,
                blocks=self.stage_blocks[i],
                spatial_stride=spatial_strides[i],
                temporal_stride=temporal_strides[i],
                dilation=dilations[i],
                inflate=inflate[i],
                inflate_style=inflate_style,
                with_cp=with_cp,
            )
            self.inplanes = planes * block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._init_weights()

    def _make_layer(self, block, inplanes, planes, blocks,
                    spatial_stride=1, temporal_stride=1, dilation=1,
                    inflate=1, inflate_style='3x1x1', with_cp=False):
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * block.expansion, kernel_size=1,
                          stride=(temporal_stride, spatial_stride, spatial_stride),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(inplanes=inplanes, planes=planes,
                        spatial_stride=spatial_stride,
                        temporal_stride=temporal_stride,
                        dilation=dilation, downsample=downsample,
                        inflate=(inflate > 0), inflate_style=inflate_style,
                        with_cp=with_cp)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=inplanes, planes=planes,
                                dilation=dilation, inflate=(inflate > 0),
                                inflate_style=inflate_style, with_cp=with_cp))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer_name in self.res_layers:
            x = getattr(self, layer_name)(x)
        return x
