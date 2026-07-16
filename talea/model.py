"""Small frozen neural head used by Talea at inference time."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class ConvBlock(nn.Module):
    """Two convolution, normalization, and GELU stages."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = max(1, min(4, output_channels))
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class AttentionStateTeacherUNet(nn.Module):
    """Frozen attention decoder with residue-state and architecture outputs."""

    def __init__(
        self,
        input_channels: int,
        teacher_classes: int,
        base_channels: int = 8,
    ):
        super().__init__()
        self.teacher_classes = teacher_classes
        self.encoder1 = ConvBlock(input_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.decoder2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, 1 + teacher_classes, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(values)
        second = self.encoder2(functional.max_pool2d(first, 2))
        bottleneck = self.bottleneck(functional.max_pool2d(second, 2))
        up_second = functional.interpolate(
            bottleneck,
            size=second.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_second = self.decoder2(torch.cat([up_second, second], dim=1))
        up_first = functional.interpolate(
            decoded_second,
            size=first.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_first = self.decoder1(torch.cat([up_first, first], dim=1))
        return self.output(decoded_first)
