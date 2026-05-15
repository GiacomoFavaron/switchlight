"""Losses for SwitchLight refinement training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def _as_batched_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return mask.to(dtype=reference.dtype, device=reference.device).clamp(0.0, 1.0)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over foreground pixels only."""
    mask = _as_batched_mask(mask, pred)
    loss = (pred - target).abs() * mask
    denom = (mask.sum() * pred.shape[1]).clamp_min(1.0)
    return loss.sum() / denom


class VGGPerceptualLoss(nn.Module):
    """Foreground-masked VGG16 perceptual loss on relu2_2 and relu3_3."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            features = models.vgg16(weights=weights).features
        except AttributeError:
            features = models.vgg16(pretrained=pretrained).features

        self.features = features[:16].eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.feature_layers = {8, 15}

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image.clamp(0.0, 1.0) - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = self._normalize(pred)
        target = self._normalize(target)

        total = pred.new_tensor(0.0)
        x_pred = pred
        x_target = target
        for idx, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            with torch.no_grad():
                x_target = layer(x_target)
            if idx in self.feature_layers:
                feat_mask = _as_batched_mask(mask, x_pred)
                diff = (x_pred - x_target).abs() * feat_mask
                denom = (feat_mask.sum() * x_pred.shape[1]).clamp_min(1.0)
                total = total + diff.sum() / denom
        return total / len(self.feature_layers)


@dataclass
class LossOutput:
    total: torch.Tensor
    l1: torch.Tensor
    perceptual: torch.Tensor


class RefinementLoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        vgg_weight: float = 0.5,
        vgg_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.vgg_weight = vgg_weight
        self.perceptual = (
            VGGPerceptualLoss(pretrained=vgg_pretrained) if vgg_weight > 0.0 else None
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> LossOutput:
        l1 = masked_l1_loss(pred, target, mask)
        if self.perceptual is None:
            perceptual = pred.new_tensor(0.0)
        else:
            perceptual = self.perceptual(pred, target, mask)
        total = self.l1_weight * l1 + self.vgg_weight * perceptual
        return LossOutput(total=total, l1=l1.detach(), perceptual=perceptual.detach())
