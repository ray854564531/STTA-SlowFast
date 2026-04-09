# Copyright (c) OpenMMLab. All rights reserved.
"""ST-TripletAttention: Spatio-Temporal Triplet Attention Module for 3D Video Understanding.

This module implements an improved triplet attention mechanism that captures
three-dimensional interactions in 3D feature maps through three branches:
- T-C-W interaction: Temporal-Channel-Width relationship
- T-C-H interaction: Temporal-Channel-Height relationship
- T-H-W interaction: Temporal-Height-Width spatial relationship

Key improvements over traditional approaches:
1. Each branch preserves 3 dimensions and only compresses 1 dimension
2. Uses consistent 7x7x7 3D convolutions for all branches
3. Better temporal modeling as all branches include time dimension
4. More efficient with 2 input channels per branch instead of 4

Author: Improved design based on user feedback
"""
import torch
import torch.nn as nn
from typing import Union, Tuple


class BasicConv3D(nn.Module):
    """Basic 3D Convolution block with optional batch normalization and ReLU.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
        stride (int | tuple[int]): Stride of the convolution. Defaults to 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Defaults to 0.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Defaults to 1.
        groups (int): Number of blocked connections from input channels to
            output channels. Defaults to 1.
        relu (bool): Whether to use ReLU activation. Defaults to True.
        bn (bool): Whether to use batch normalization. Defaults to True.
        bias (bool): Whether to use bias. Defaults to False.
    """

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1, groups: int = 1,
                 relu: bool = True, bn: bool = True, bias: bool = False):
        super(BasicConv3D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool1D(nn.Module):
    """1D Z-Pooling layer for compressing a single dimension.

    Computes both max and average pooling along specified dimension,
    then concatenates the results.

    Args:
        dim (int): The dimension to perform pooling on.
    """

    def __init__(self, dim: int):
        super(ZPool1D, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Max and average pooling along specified dimension, then concatenate
        max_pool = torch.max(x, dim=self.dim, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=self.dim, keepdim=True)
        output = torch.cat((max_pool, avg_pool), dim=self.dim)
        return output


class STAttentionBranch(nn.Module):
    """Single branch of ST-TripletAttention.

    Performs dimension permutation, compression, and 3D convolution
    to model interactions between three specific dimensions.

    Args:
        compress_dim (int): The dimension to compress (1-indexed relative to input).
        target_permutation (tuple): Permutation to bring target dims to last 3 positions.
        recovery_permutation (tuple): Permutation to recover original dimension order.
        kernel_size (int): 3D convolution kernel size. Defaults to 7.
    """

    def __init__(self, compress_dim: int, target_permutation: tuple,
                 recovery_permutation: tuple, kernel_size: int = 7):
        super(STAttentionBranch, self).__init__()
        self.compress_dim = compress_dim
        self.target_permutation = target_permutation
        self.recovery_permutation = recovery_permutation

        # Z-pooling to compress the specified dimension
        self.compress = ZPool1D(compress_dim)

        # 3D convolution with 7x7x7 kernel
        padding = (kernel_size - 1) // 2
        self.conv3d = BasicConv3D(
            in_planes=2,  # 2 channels from max+avg pooling
            out_planes=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            relu=False  # No ReLU before sigmoid
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (B, C, T, H, W)

        Returns:
            Attention-enhanced tensor with shape (B, C, T, H, W)
        """
        # Step 1: Permute dimensions to bring target dims to last 3 positions
        x_perm = x.permute(*self.target_permutation)

        # Step 2: Compress the specified dimension
        x_compressed = self.compress(x_perm)  # One dim: original_size -> 2

        # Step 3: Apply 3D convolution
        attention_map = self.conv3d(x_compressed)  # (B, 2, ...) -> (B, 1, ...)

        # Step 4: Apply sigmoid activation
        attention_weights = self.sigmoid(attention_map)

        # Step 5: Apply attention to compressed features
        x_attended = x_perm * attention_weights

        # Step 6: Recover original dimension order
        x_recovered = x_attended.permute(*self.recovery_permutation)

        return x_recovered


class STTripletAttention(nn.Module):
    """ST-TripletAttention: Spatio-Temporal Triplet Attention for 3D video understanding.

    This module captures three-dimensional interactions in 3D feature maps through
    three specialized branches:
    - T-C-W branch: Models Temporal-Channel-Width interactions (compresses H)
    - T-C-H branch: Models Temporal-Channel-Height interactions (compresses W)
    - T-H-W branch: Models Temporal-Height-Width spatial interactions (compresses C)

    Each branch:
    1. Rearranges dimensions to put target 3D interaction at the last 3 positions
    2. Compresses 1 irrelevant dimension using max+avg pooling
    3. Applies 7x7x7 3D convolution to model 3D interactions
    4. Recovers original dimension arrangement
    5. Applies attention weights to enhance features

    Args:
        kernel_size (int): Size of 3D convolution kernel. Defaults to 7.
        enable_tcw (bool): Whether to enable T-C-W branch. Defaults to True.
        enable_tch (bool): Whether to enable T-C-H branch. Defaults to True.
        enable_thw (bool): Whether to enable T-H-W branch. Defaults to True.

    Example:
        >>> # Input: (B, C, T, H, W) = (2, 64, 8, 14, 14)
        >>> attention = STTripletAttention(kernel_size=7)
        >>> enhanced_features = attention(input_tensor)
        >>> # Output: (2, 64, 8, 14, 14) - same shape as input
        >>>
        >>> # Ablation: Only use T-C-W branch
        >>> attention_tcw = STTripletAttention(enable_tcw=True, enable_tch=False, enable_thw=False)
    """

    def __init__(self, kernel_size: int = 7,
                 enable_tcw: bool = True,
                 enable_tch: bool = True,
                 enable_thw: bool = True):
        super(STTripletAttention, self).__init__()

        self.enable_tcw = enable_tcw
        self.enable_tch = enable_tch
        self.enable_thw = enable_thw

        # At least one branch should be enabled
        if not (enable_tcw or enable_tch or enable_thw):
            raise ValueError("At least one branch must be enabled")

        # T-C-W branch: compress H dimension, model T-C-W interactions
        # Input: (B, C, T, H, W) -> Target: (..., T, C, W)
        # Permutation: (0, 3, 1, 2, 4) -> (B, H, C, T, W), compress dim=1 (H)
        # Recovery: (0, 2, 3, 1, 4) -> (B, C, T, H, W)
        if enable_tcw:
            self.tcw_branch = STAttentionBranch(
                compress_dim=1,
                target_permutation=(0, 3, 1, 2, 4),  # (B, H, C, T, W)
                recovery_permutation=(0, 2, 3, 1, 4),  # (B, C, T, H, W)
                kernel_size=kernel_size
            )

        # T-C-H branch: compress W dimension, model T-C-H interactions
        # Input: (B, C, T, H, W) -> Target: (..., T, C, H)
        # Permutation: (0, 4, 1, 2, 3) -> (B, W, C, T, H), compress dim=1 (W)
        # Recovery: (0, 2, 3, 4, 1) -> (B, C, T, H, W)
        if enable_tch:
            self.tch_branch = STAttentionBranch(
                compress_dim=1,
                target_permutation=(0, 4, 1, 2, 3),  # (B, W, C, T, H)
                recovery_permutation=(0, 2, 3, 4, 1),  # (B, C, T, H, W)
                kernel_size=kernel_size
            )

        # T-H-W branch: compress C dimension, model T-H-W spatial interactions
        # Input: (B, C, T, H, W) -> Target: (..., T, H, W)
        # Permutation: (0, 1, 2, 3, 4) -> (B, C, T, H, W), compress dim=1 (C)
        # Recovery: (0, 1, 2, 3, 4) -> (B, C, T, H, W)
        if enable_thw:
            self.thw_branch = STAttentionBranch(
                compress_dim=1,
                target_permutation=(0, 1, 2, 3, 4),  # (B, C, T, H, W)
                recovery_permutation=(0, 1, 2, 3, 4),  # (B, C, T, H, W)
                kernel_size=kernel_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, T, H, W).

        Returns:
            torch.Tensor: Enhanced tensor with same shape as input.
        """
        if len(x.shape) != 5:
            raise ValueError(f"Input must be 5D tensor (B, C, T, H, W), got {x.shape}")

        # Collect outputs from enabled branches
        branch_outputs = []

        # Apply enabled branches
        if self.enable_tcw:
            tcw_out = self.tcw_branch(x)  # T-C-W interactions
            branch_outputs.append(tcw_out)

        if self.enable_tch:
            tch_out = self.tch_branch(x)  # T-C-H interactions
            branch_outputs.append(tch_out)

        if self.enable_thw:
            thw_out = self.thw_branch(x)  # T-H-W spatial interactions
            branch_outputs.append(thw_out)

        # Average fusion of enabled branches
        enhanced_features = sum(branch_outputs) / len(branch_outputs)

        return enhanced_features

    def __repr__(self):
        enabled_branches = []
        if self.enable_tcw:
            enabled_branches.append('T-C-W')
        if self.enable_tch:
            enabled_branches.append('T-C-H')
        if self.enable_thw:
            enabled_branches.append('T-H-W')
        return f"{self.__class__.__name__}(branches={enabled_branches}, kernel_size=7x7x7)"
