"""SlowFast classification head."""
import torch
import torch.nn as nn
from typing import Tuple


class SlowFastHead(nn.Module):
    """Classification head for SlowFast.

    Global avg pool both pathways, concat, dropout, linear.

    Args:
        in_channels: Total channels after concat (slow+fast). Default 2304.
        num_classes: Number of action classes. Default 8.
        dropout: Dropout probability. Default 0.5.
    """

    def __init__(self, in_channels: int = 2304, num_classes: int = 8,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        slow_x, fast_x = inputs
        slow_x = self.avg_pool(slow_x).flatten(1)   # (B, 2048)
        fast_x = self.avg_pool(fast_x).flatten(1)   # (B, 256)
        x = torch.cat([slow_x, fast_x], dim=1)       # (B, 2304)
        x = self.dropout(x)
        return self.fc(x)
