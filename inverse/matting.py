"""Foreground matting using rembg.

This is the simplest piece of the inverse rendering frontend.
We use rembg's u2net model by default — good quality, fast enough on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


# rembg lazily creates a session. We cache it module-level to avoid reload on every call.
_REMBG_SESSION = None


def _get_session():
    """Lazily create and cache the rembg session.

    We use 'u2net_human_seg' which is specifically tuned for human subjects
    and produces cleaner masks for portraits than the generic 'u2net'.
    """
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION = new_session("u2net_human_seg")
    return _REMBG_SESSION


def extract_mask(image_path: str | Path | Image.Image) -> torch.Tensor:
    """Extract a foreground mask for a portrait image.

    Args:
        image_path: Path to image file, or a PIL Image.

    Returns:
        Mask tensor [1, H, W], float32, range [0, 1]. Foreground = 1.
    """
    from rembg import remove

    if isinstance(image_path, (str, Path)):
        pil = Image.open(image_path).convert("RGB")
    else:
        pil = image_path.convert("RGB")

    session = _get_session()

    # rembg returns RGBA with alpha as the mask
    result = remove(pil, session=session)
    if result.mode != "RGBA":
        result = result.convert("RGBA")

    alpha = np.asarray(result.split()[-1], dtype=np.float32) / 255.0  # [H, W]
    return torch.from_numpy(alpha).unsqueeze(0)  # [1, H, W]


def extract_mask_from_tensor(image: torch.Tensor) -> torch.Tensor:
    """Extract a foreground mask from a [3, H, W] linear RGB tensor in [0, 1].

    Converts to sRGB PIL Image first since rembg works on standard images.
    """
    from .io_utils import linear_to_srgb

    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] tensor, got {tuple(image.shape)}")

    srgb = linear_to_srgb(image.detach().cpu()).clamp(0.0, 1.0)
    arr = (srgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(arr)
    return extract_mask(pil)
