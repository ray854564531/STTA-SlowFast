"""Temporal Segment Networks (TSN) with ImageNet-pretrained ResNet-50 backbone.

Reference: Wang et al., "Temporal Segment Networks", ECCV 2016.
Input shape: (B, C, T, H, W) — matches the project's DataLoader output.
Forward:
  1. Permute to (B, T, C, H, W), reshape to (B*T, C, H, W)
  2. ResNet-50 feature extraction → (B*T, 2048)
  3. Temporal consensus (mean) → (B, 2048)
  4. Linear classifier → (B, num_classes)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50


class TSN(nn.Module):
    """TSN with ResNet-50 backbone and average-pooling consensus function.

    Args:
        num_classes: Number of output classes.
        num_segments: Number of temporal segments (= clip_len in config).
        pretrained: Load ImageNet-pretrained ResNet-50 weights.
    """

    def __init__(self, num_classes: int, num_segments: int = 8,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.num_segments = num_segments
        backbone = resnet50(weights=None)
        if pretrained:
            state = torch.load('./checkpoints/resnet50-0676ba61.pth', map_location='cpu')
            backbone.load_state_dict(state)
        # Remove avgpool and fc; keep everything up to layer4
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # → (B*T,2048,h,w)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, T, H, W) — DataLoader output, T == num_segments.
        Returns:
            logits: (B, num_classes)
        """
        B, C, T, H, W = x.shape
        assert T == self.num_segments, (
            f"Input T={T} does not match num_segments={self.num_segments}. "
            "Ensure clip_len in config matches num_segments."
        )
        # Fold time into batch
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)
        x = self.features(x)                          # (B*T, 2048, h, w)
        x = self.pool(x).flatten(1)                   # (B*T, 2048)
        # Temporal consensus: mean over segments
        x = x.view(B, T, -1).mean(dim=1)             # (B, 2048)
        return self.fc(x)                              # (B, num_classes)
