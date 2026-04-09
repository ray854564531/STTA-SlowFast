"""ResNet3dPathway and ResNet3dSlowFast. Ported from mmaction2, mm-free."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_utils import ConvModule, kaiming_init
from .resnet3d import Bottleneck3d, ResNet3d


class ResNet3dPathway(ResNet3d):
    """ResNet3d pathway with optional lateral connections (for slow pathway).

    When lateral=True, builds conv1_lateral and layer{1,2,3}_lateral modules
    that transform fast-pathway features to be concatenated into slow pathway.

    Args:
        lateral: Enable lateral connections. True for slow, False for fast.
        speed_ratio: Fast/slow temporal ratio (alpha). Default 4.
        channel_ratio: Slow/fast channel ratio (beta). Default 8.
        fusion_kernel: Temporal kernel size for lateral convs. Default 5.
        lateral_infl: Channel inflation factor for lateral convs. Default 2.
        lateral_activate: Which stages get lateral input.
            Tuple of 4 ints (conv1, l1, l2, l3). Default (1,1,1,1).
        **kwargs: Passed to ResNet3d.
    """

    def __init__(
        self,
        lateral: bool = False,
        speed_ratio: int = 4,
        channel_ratio: int = 8,
        fusion_kernel: int = 5,
        lateral_infl: int = 2,
        lateral_activate: Tuple = (1, 1, 1, 1),
        **kwargs,
    ) -> None:
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        self.lateral_infl = lateral_infl
        self.lateral_activate = list(lateral_activate)

        # Compute lateral_inplanes to pass into ResNet3d
        # lateral_inplanes[i] = extra channels added BEFORE stage i
        # (from the lateral conv applied after stage i-1)
        if lateral:
            base_channels = kwargs.get('base_channels', 64)
            lat_inplanes = self._calc_lateral_inplanes(base_channels)
        else:
            lat_inplanes = None

        super().__init__(lateral_inplanes=lat_inplanes, **kwargs)

        # Build lateral conv modules (slow pathway only)
        self.lateral_connections = []
        if self.lateral:
            self._build_lateral_convs()

    def _calc_lateral_inplanes(self, slow_base_channels: int) -> List[int]:
        """Compute extra channels injected before each of the 4 stages.

        Layout:
          before layer1 <- conv1_lateral output (from fast conv1)
          before layer2 <- layer1_lateral output (from fast layer1)
          before layer3 <- layer2_lateral output (from fast layer2)
          before layer4 <- layer3_lateral output (from fast layer3)

        Fast pathway channel at conv1 output: slow_base // channel_ratio
        Fast pathway channel at layerN output: slow_base*(2^(N-1))*expansion // channel_ratio
        """
        expansion = Bottleneck3d.expansion  # 4

        # conv1_lateral: fast conv1 out = slow_base // channel_ratio
        # lateral conv output = slow_base * lateral_infl // channel_ratio
        conv1_lat = (slow_base_channels * self.lateral_infl // self.channel_ratio
                     if self.lateral_activate[0] else 0)

        stage_lats = []
        for i in range(3):  # stages 0,1,2 -> feeds into stage 1,2,3
            # fast layer(i+1) output channels = fast_planes * expansion
            # fast_planes = slow_base * (2**i) // channel_ratio ... but actually:
            # fast pathway base_channels = slow_base // channel_ratio
            # fast layer(i+1) output = fast_base * (2**i) * expansion
            # = slow_base // channel_ratio * (2**i) * expansion
            # slow layer(i+1) output = slow_base * (2**i) * expansion
            # lateral conv output = slow_layer_out * lateral_infl // channel_ratio
            slow_stage_out = slow_base_channels * (2 ** i) * expansion
            lat = (slow_stage_out * self.lateral_infl // self.channel_ratio
                   if self.lateral_activate[i + 1] else 0)
            stage_lats.append(lat)

        # lateral_inplanes for ResNet3d layers 0..3:
        return [conv1_lat] + stage_lats

    def _build_lateral_convs(self) -> None:
        """Build lateral conv modules on the slow pathway."""
        base_channels = self.base_channels
        expansion = Bottleneck3d.expansion

        # conv1_lateral: converts fast conv1 output -> slow-compatible channels
        # fast conv1 output channels = slow_base // channel_ratio
        if self.lateral_activate[0]:
            fast_conv1_out = base_channels // self.channel_ratio
            lateral_out = base_channels * self.lateral_infl // self.channel_ratio
            self.conv1_lateral = ConvModule(
                fast_conv1_out,
                lateral_out,
                kernel_size=(self.fusion_kernel, 1, 1),
                stride=(self.speed_ratio, 1, 1),
                padding=((self.fusion_kernel - 1) // 2, 0, 0),
                bias=False,
                with_bn=True,
                with_relu=True,
            )

        # layer{1,2,3}_lateral (not layer4 - last stage has no lateral)
        for i in range(3):
            if not self.lateral_activate[i + 1]:
                continue
            # fast layer(i+1) output = slow_layer(i+1)_out // channel_ratio
            slow_stage_out = base_channels * (2 ** i) * expansion
            fast_stage_out = slow_stage_out // self.channel_ratio
            lateral_out = slow_stage_out * self.lateral_infl // self.channel_ratio
            lateral_name = f'layer{i + 1}_lateral'
            conv = ConvModule(
                fast_stage_out,
                lateral_out,
                kernel_size=(self.fusion_kernel, 1, 1),
                stride=(self.speed_ratio, 1, 1),
                padding=((self.fusion_kernel - 1) // 2, 0, 0),
                bias=False,
                with_bn=True,
                with_relu=True,
            )
            setattr(self, lateral_name, conv)
            self.lateral_connections.append(lateral_name)


class ResNet3dSlowFast(nn.Module):
    """SlowFast backbone with two ResNet3d pathways.

    Ported from mmaction2. STTripletAttention is NOT included here.

    Args:
        resample_rate: Temporal downsampling for slow pathway (tau). Default 4.
        speed_ratio: Fast/slow temporal ratio (alpha). Default 4.
        channel_ratio: Slow/fast channel ratio (beta). Default 8.
        slow_pathway: kwargs for slow ResNet3dPathway (lateral=True).
        fast_pathway: kwargs for fast ResNet3dPathway (lateral=False).
    """

    def __init__(
        self,
        resample_rate: int = 4,
        speed_ratio: int = 4,
        channel_ratio: int = 8,
        slow_pathway: Optional[Dict] = None,
        fast_pathway: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        slow_cfg = (slow_pathway or {}).copy()
        fast_cfg = (fast_pathway or {}).copy()

        # Inject speed/channel ratio into slow pathway for lateral conv sizing
        if slow_cfg.get('lateral', False):
            slow_cfg.setdefault('speed_ratio', speed_ratio)
            slow_cfg.setdefault('channel_ratio', channel_ratio)

        self.slow_path = ResNet3dPathway(**slow_cfg)
        self.fast_path = ResNet3dPathway(**fast_cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Slow pathway: temporal downsample by resample_rate
        # Input has T frames; slow sees T//resample_rate frames
        # Fast pathway: temporal downsample by resample_rate//speed_ratio
        # Fast sees T//(resample_rate//speed_ratio) = T*speed_ratio//resample_rate frames
        T = x.shape[2]
        slow_T = T // self.resample_rate
        fast_T = T // (self.resample_rate // self.speed_ratio)

        # Select frames via slicing (avoids interpolate float issues)
        slow_indices = torch.linspace(0, T - 1, slow_T).long()
        fast_indices = torch.linspace(0, T - 1, fast_T).long()
        x_slow = x[:, :, slow_indices, :, :]
        x_fast = x[:, :, fast_indices, :, :]

        # Initial conv + pool for both pathways
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        # Initial lateral: fast conv1 -> slow (before layer1)
        if self.slow_path.lateral and hasattr(self.slow_path, 'conv1_lateral'):
            lat = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat([x_slow, lat], dim=1)

        # Stage-by-stage forward with lateral connections after each stage
        for i, layer_name in enumerate(self.slow_path.res_layers):
            x_slow = getattr(self.slow_path, layer_name)(x_slow)
            x_fast = getattr(self.fast_path, layer_name)(x_fast)

            # Lateral after stage i, if not the last stage
            if (i < len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral
                    and i < len(self.slow_path.lateral_connections)):
                lateral_name = self.slow_path.lateral_connections[i]
                lat = getattr(self.slow_path, lateral_name)(x_fast)
                x_slow = torch.cat([x_slow, lat], dim=1)

        return x_slow, x_fast
