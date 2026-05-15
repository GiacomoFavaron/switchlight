"""Small residual U-Net for SwitchLight refinement."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _num_groups(channels: int) -> int:
    """Pick a GroupNorm group count that divides the channel count."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        conv_groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=conv_groups,
                bias=False,
            ),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ConvNormAct(out_channels + skip_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class RefinementUNet(nn.Module):
    """Predict a bounded residual for a rendered portrait.

    Input channels are rendered RGB, albedo RGB, and normal XYZ.
    Output is a residual to add to rendered RGB, bounded to [-0.5, 0.5].
    """

    def __init__(self, in_channels: int = 9, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.down1 = DownBlock(in_channels, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)
        self.down4 = DownBlock(c3, c4)

        # Grouped bottleneck keeps the model near the 1M parameter target.
        self.bottleneck = ConvNormAct(c4, c4, conv_groups=2)

        self.up4 = UpBlock(c4, c4, c3)
        self.up3 = UpBlock(c3, c3, c2)
        self.up2 = UpBlock(c2, c2, c1)
        self.up1 = UpBlock(c1, c1, c1)

        self.head = nn.Conv2d(c1, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.bottleneck(x)

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        return torch.tanh(self.head(x)) * 0.5


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    net = RefinementUNet()
    params = count_parameters(net)
    print(f"Total parameters: {params:,}")

    dummy = torch.randn(1, 9, 384, 384)
    with torch.no_grad():
        residual = net(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(residual.shape)}")
    print(f"Output range: [{residual.min().item():.3f}, {residual.max().item():.3f}]")
