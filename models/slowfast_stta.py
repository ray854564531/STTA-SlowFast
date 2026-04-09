"""SlowFast + STTripletAttention. Ported from mmaction2, mm-free.

STTA is applied to fast pathway AFTER each ResNet stage,
BEFORE the lateral connection feeds enhanced features into slow pathway.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import STTripletAttention
from .slowfast import ResNet3dPathway, ResNet3dSlowFast


class ResNet3dPathwayWithSTTA(ResNet3dPathway):
    """Fast pathway with ST-TripletAttention after each stage.

    STTA modules are stored in self.stta_modules (nn.ModuleDict).
    Only created when enable_stta=True.
    """

    def __init__(
        self,
        enable_stta: bool = True,
        stta_kernel_size: int = 7,
        stta_stages: List[bool] = (True, True, True, True),
        stta_enable_tcw: bool = True,
        stta_enable_tch: bool = True,
        stta_enable_thw: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.enable_stta = enable_stta

        if self.enable_stta:
            self.stta_modules = nn.ModuleDict()
            for i, enabled in enumerate(stta_stages):
                if enabled:
                    self.stta_modules[f'layer{i+1}_stta'] = STTripletAttention(
                        kernel_size=stta_kernel_size,
                        enable_tcw=stta_enable_tcw,
                        enable_tch=stta_enable_tch,
                        enable_thw=stta_enable_thw,
                    )


class SlowFastWithSTTA(nn.Module):
    """SlowFast backbone with stage-wise STTripletAttention in fast pathway.

    STTA is applied to fast pathway after each stage, BEFORE lateral
    connections transfer features to slow pathway.

    Args:
        resample_rate: Temporal downsampling for slow (tau). Default 4.
        speed_ratio: Fast/slow temporal ratio (alpha). Default 4.
        channel_ratio: Slow/fast channel ratio (beta). Default 8.
        slow_pathway: kwargs for slow ResNet3dPathway (lateral=True).
        fast_pathway: kwargs for fast ResNet3dPathwayWithSTTA (lateral=False).
        enable_stta: Master switch for STTA. False = baseline (no STTA).
        stta_kernel_size: 3D conv kernel for STTA. Default 7.
        stta_stages: Which stages get STTA. Default all 4.
        stta_enable_tcw: Enable T-C-W branch. Default True.
        stta_enable_tch: Enable T-C-H branch. Default True.
        stta_enable_thw: Enable T-H-W branch. Default True.
    """

    def __init__(
        self,
        resample_rate: int = 4,
        speed_ratio: int = 4,
        channel_ratio: int = 8,
        slow_pathway: Optional[Dict] = None,
        fast_pathway: Optional[Dict] = None,
        enable_stta: bool = True,
        stta_kernel_size: int = 7,
        stta_stages: List[bool] = (True, True, True, True),
        stta_enable_tcw: bool = True,
        stta_enable_tch: bool = True,
        stta_enable_thw: bool = True,
    ) -> None:
        super().__init__()
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        slow_cfg = (slow_pathway or {}).copy()
        fast_cfg = (fast_pathway or {}).copy()

        if slow_cfg.get('lateral', False):
            slow_cfg.setdefault('speed_ratio', speed_ratio)
            slow_cfg.setdefault('channel_ratio', channel_ratio)

        self.slow_path = ResNet3dPathway(**slow_cfg)
        self.fast_path = ResNet3dPathwayWithSTTA(
            enable_stta=enable_stta,
            stta_kernel_size=stta_kernel_size,
            stta_stages=stta_stages,
            stta_enable_tcw=stta_enable_tcw,
            stta_enable_tch=stta_enable_tch,
            stta_enable_thw=stta_enable_thw,
            **fast_cfg,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.shape[2]
        slow_T = T // self.resample_rate
        fast_T = T // (self.resample_rate // self.speed_ratio)

        # Select frames via slicing (consistent with ResNet3dSlowFast)
        slow_indices = torch.linspace(0, T - 1, slow_T).long()
        fast_indices = torch.linspace(0, T - 1, fast_T).long()
        x_slow = x[:, :, slow_indices, :, :]
        x_fast = x[:, :, fast_indices, :, :]

        # Initial conv + pool
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        # Initial lateral (before layer1)
        if self.slow_path.lateral and hasattr(self.slow_path, 'conv1_lateral'):
            x_slow = torch.cat(
                [x_slow, self.slow_path.conv1_lateral(x_fast)], dim=1)

        # Stage-by-stage: fast stage → STTA → lateral → slow
        for i, layer_name in enumerate(self.slow_path.res_layers):
            x_slow = getattr(self.slow_path, layer_name)(x_slow)
            x_fast = getattr(self.fast_path, layer_name)(x_fast)

            # Apply STTA to fast pathway (BEFORE lateral)
            stta_name = f'layer{i+1}_stta'
            if (hasattr(self.fast_path, 'stta_modules') and
                    stta_name in self.fast_path.stta_modules):
                x_fast = self.fast_path.stta_modules[stta_name](x_fast)

            # Lateral: STTA-enhanced fast → slow (not after last stage)
            if (i < len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral
                    and i < len(self.slow_path.lateral_connections)):
                lateral_name = self.slow_path.lateral_connections[i]
                x_slow = torch.cat(
                    [x_slow, getattr(self.slow_path, lateral_name)(x_fast)],
                    dim=1)

        return x_slow, x_fast
