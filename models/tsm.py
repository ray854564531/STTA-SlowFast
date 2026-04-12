"""Temporal Shift Module (TSM) with ImageNet-pretrained ResNet-50 backbone.

Reference: Lin et al., "TSM: Temporal Shift Module for Efficient Video
Understanding", ICCV 2019.

The shift is inserted as a forward pre-hook into every Bottleneck block.
Since TemporalShift has no learnable parameters, this does not affect
the backbone's state_dict; pretrained weights load without key conflicts.

Input shape: (B, C, T, H, W) — matches the project's DataLoader output.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck


class TemporalShift(nn.Module):
    """Shift a fraction of channels along the temporal dimension.

    Given (B*T, C, H, W), reshapes to (B, T, C, H, W), shifts:
      - channels [:fold]      ← shift from t-1 (past context)
      - channels [fold:2*fold] ← shift from t+1 (future context)
      - channels [2*fold:]     unchanged
    Boundary frames are zero-padded.

    Args:
        n_segment: Number of temporal frames T.
        n_div: Denominator for fold size; fold = C // n_div.
    """

    def __init__(self, n_segment: int, n_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        n = nt // self.n_segment
        x = x.view(n, self.n_segment, c, h, w)
        fold = c // self.n_div
        out = torch.zeros_like(x)
        # Past context: out[t] ← x[t-1] for first `fold` channels
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # Future context: out[t] ← x[t+1] for next `fold` channels
        out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
        # Unchanged channels
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class _TemporalShiftHook:
    """Picklable pre-hook wrapper. Module-level class makes it pickle-safe for DDP."""

    def __init__(self, shift: TemporalShift) -> None:
        self.shift = shift

    def __call__(self, module: nn.Module, inputs: tuple) -> tuple:
        return (self.shift(inputs[0]),)


def _insert_temporal_shift(backbone: nn.Module, n_segment: int,
                            n_div: int = 8) -> None:
    """Register TemporalShift pre-hooks on every Bottleneck block in-place."""
    for module in backbone.modules():
        if isinstance(module, Bottleneck):
            shift = TemporalShift(n_segment=n_segment, n_div=n_div)
            # Register as a named submodule so it shows in named_modules()
            module.add_module('temporal_shift', shift)

            module.register_forward_pre_hook(_TemporalShiftHook(shift))


class TSM(nn.Module):
    """TSM with ResNet-50 backbone and average-pooling consensus function.

    Args:
        num_classes: Number of output classes.
        num_segments: Number of temporal segments (= clip_len in config).
        pretrained: Load ImageNet-pretrained ResNet-50 weights.
        n_div: Shift fraction denominator (default 8 → shift C/8 each dir).
    """

    def __init__(self, num_classes: int, num_segments: int = 8,
                 pretrained: bool = True, n_div: int = 8) -> None:
        super().__init__()
        self.num_segments = num_segments
        backbone = resnet50(weights=None)
        if pretrained:
            state = torch.load('./checkpoints/resnet50-0676ba61.pth', map_location='cpu')
            backbone.load_state_dict(state)
        # Remove avgpool and fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        _insert_temporal_shift(self.backbone, n_segment=num_segments, n_div=n_div)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, T, H, W) — DataLoader output.
        Returns:
            logits: (B, num_classes)
        """
        B, C, T, H, W = x.shape
        assert T == self.num_segments, (
            f"Input T={T} does not match num_segments={self.num_segments}. "
            "Ensure clip_len in config matches num_segments."
        )
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)                          # (B*T, 2048, h, w)
        x = self.pool(x).flatten(1)                   # (B*T, 2048)
        x = x.view(B, T, -1).mean(dim=1)             # (B, 2048)
        return self.fc(x)                              # (B, num_classes)
