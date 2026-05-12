"""Shared utilities for image I/O and tensor conventions.

The whole pipeline operates in linear RGB internally. sRGB <-> linear
conversion happens only at file I/O boundaries.

Tensor conventions:
    image:     [3, H, W], float32, range [0, 1], linear RGB
    normal:    [3, H, W], float32, range [-1, 1], camera space
    albedo:    [3, H, W], float32, range [0, 1], linear RGB
    roughness: [1, H, W], float32, range [0.05, 1.0]
    specular:  [1, H, W], float32, range [0, 1]
    mask:      [1, H, W], float32, range [0, 1], foreground = 1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert sRGB-encoded [0,1] tensor to linear RGB [0,1].

    Uses the standard IEC 61966-2-1 piecewise transform.
    """
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB [0,1] tensor to sRGB-encoded [0,1]."""
    x = x.clamp(0.0, 1.0)  # safe clamp before the transform
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * (x ** (1.0 / 2.4)) - 0.055,
    )


def load_image_linear(path: str | Path) -> torch.Tensor:
    """Load an sRGB image file and return as linear RGB tensor [3, H, W] in [0,1]."""
    path = Path(path)
    pil = Image.open(path).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0  # [H, W, 3], sRGB
    t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    return srgb_to_linear(t)


def save_image_linear(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a linear RGB tensor [3, H, W] (or [H, W, 3]) as sRGB PNG.

    Clamps to [0, 1] before encoding.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tensor.dim() == 3 and tensor.shape[0] == 3:
        t = tensor
    elif tensor.dim() == 3 and tensor.shape[-1] == 3:
        t = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected image tensor with 3 channels, got shape {tuple(tensor.shape)}")

    srgb = linear_to_srgb(t.detach().cpu().float()).clamp(0.0, 1.0)
    arr = (srgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def resize_image(tensor: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    """Resize [3, H, W] or [1, H, W] tensor to target size using bilinear interp."""
    if isinstance(size, int):
        size = (size, size)
    # add batch dim for F.interpolate
    t = tensor.unsqueeze(0)
    out = torch.nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return out.squeeze(0)


def visualize_normal(normal: torch.Tensor) -> torch.Tensor:
    """Convert a normal map in [-1, 1] to a viewable RGB image in [0, 1]."""
    return (normal + 1.0) / 2.0


def visualize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a [1, H, W] mask to [3, H, W] for saving as RGB."""
    if mask.dim() == 3 and mask.shape[0] == 1:
        return mask.repeat(3, 1, 1)
    return mask
