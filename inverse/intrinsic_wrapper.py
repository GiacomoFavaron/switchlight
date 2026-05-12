"""Albedo estimation using Ordinal Shading / Intrinsic Image Decomposition
from Careaga & Aksoy (compphoto/Intrinsic).

Like DSINE, this is not pip-installable. Clone the repo to third_party/Intrinsic/:

    git clone https://github.com/compphoto/Intrinsic.git third_party/Intrinsic

Pretrained weights are downloaded automatically on first model creation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .io_utils import get_device, linear_to_srgb


_INTRINSIC_PIPELINE = None
_INTRINSIC_DEVICE = None


def _find_intrinsic_root() -> Path:
    """Locate the Intrinsic repo."""
    candidates = []
    if env := os.environ.get("INTRINSIC_ROOT"):
        candidates.append(Path(env))

    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / "third_party" / "Intrinsic")
    candidates.append(project_root.parent / "Intrinsic")

    for c in candidates:
        if c.exists() and (c / "intrinsic").exists():
            return c

    raise RuntimeError(
        "Could not find Intrinsic repo. Clone it with:\n"
        "  git clone https://github.com/compphoto/Intrinsic.git third_party/Intrinsic\n"
        "Or set the INTRINSIC_ROOT environment variable."
    )


def _load_intrinsic_pipeline(device: torch.device):
    """Load the Intrinsic pipeline. Cached at module level."""
    global _INTRINSIC_PIPELINE, _INTRINSIC_DEVICE

    if _INTRINSIC_PIPELINE is not None and _INTRINSIC_DEVICE == device:
        return _INTRINSIC_PIPELINE

    intrinsic_root = _find_intrinsic_root()
    sys.path.insert(0, str(intrinsic_root))

    # The Intrinsic repo exposes a load_models() helper and a run_pipeline() function
    from intrinsic.pipeline import load_models, run_pipeline  # type: ignore

    pipeline = load_models("paper_weights", device=str(device))

    _INTRINSIC_PIPELINE = (pipeline, run_pipeline)
    _INTRINSIC_DEVICE = device
    return _INTRINSIC_PIPELINE


def estimate_albedo(
    image: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Estimate diffuse albedo from a linear RGB image.

    Args:
        image: [3, H, W] tensor, linear RGB, range [0, 1].
        device: target device.

    Returns:
        albedo: [3, H, W] tensor, linear RGB, range [0, 1].
    """
    if device is None:
        device = get_device()

    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] tensor, got {tuple(image.shape)}")

    pipeline, run_pipeline = _load_intrinsic_pipeline(device)

    # The Intrinsic pipeline expects an sRGB image as a NumPy array in [0, 1]
    srgb = linear_to_srgb(image).clamp(0.0, 1.0)
    np_img = srgb.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    with torch.no_grad():
        result = run_pipeline(pipeline, np_img, device=str(device))

    # result is a dict — keys vary by version. Common: 'albedo', 'hr_alb', 'inv_shading'.
    # Prefer high-res albedo if available.
    albedo_np = None
    for key in ("hr_alb", "albedo", "alb"):
        if key in result:
            albedo_np = result[key]
            break

    if albedo_np is None:
        raise RuntimeError(
            f"Intrinsic pipeline returned unexpected keys: {list(result.keys())}. "
            "Update the wrapper to handle the new output format."
        )

    # albedo_np is typically HWC float32 in [0, 1], in linear color space
    if albedo_np.ndim == 3 and albedo_np.shape[-1] == 3:
        albedo = torch.from_numpy(albedo_np).permute(2, 0, 1).float()
    else:
        albedo = torch.from_numpy(albedo_np).float()
        if albedo.dim() == 3 and albedo.shape[0] != 3:
            albedo = albedo.permute(2, 0, 1)

    albedo = albedo.clamp(0.0, 1.0)

    # Sanity: if the pipeline returned a different resolution than input, resize back.
    _, h, w = image.shape
    if albedo.shape[-2:] != (h, w):
        from .io_utils import resize_image
        albedo = resize_image(albedo, (h, w))

    return albedo
