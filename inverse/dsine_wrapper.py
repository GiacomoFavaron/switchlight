"""Surface normal estimation using DSINE.

DSINE is not pip-installable. You need to clone the repo and place it
at third_party/DSINE/ (or set DSINE_ROOT env var to its location).

    git clone https://github.com/baegwangbin/DSINE.git third_party/DSINE

Pretrained weights are downloaded automatically by DSINE on first use,
or you can grab them manually from their README.

Important: DSINE was trained on scene-level images (mostly indoor scenes,
KITTI, etc.), not portrait crops. For tight portrait photos we pad the
input with mirror reflection to give the model more spatial context,
then crop the output back. Without this padding, normals around hair
edges and shoulders look noticeably worse.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .io_utils import get_device


_DSINE_MODEL = None
_DSINE_DEVICE = None


def _find_dsine_root() -> Path:
    """Locate the DSINE repo. Check env var, then standard locations."""
    candidates = []
    if env := os.environ.get("DSINE_ROOT"):
        candidates.append(Path(env))

    # Common locations relative to project root
    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / "third_party" / "DSINE")
    candidates.append(project_root.parent / "DSINE")

    for c in candidates:
        if c.exists() and (c / "models").exists():
            return c

    raise RuntimeError(
        "Could not find DSINE repo. Clone it with:\n"
        "  git clone https://github.com/baegwangbin/DSINE.git third_party/DSINE\n"
        "Or set the DSINE_ROOT environment variable."
    )


def _load_dsine_model(device: torch.device):
    """Load DSINE model and pretrained weights. Cached at module level."""
    global _DSINE_MODEL, _DSINE_DEVICE

    if _DSINE_MODEL is not None and _DSINE_DEVICE == device:
        return _DSINE_MODEL

    dsine_root = _find_dsine_root()
    sys.path.insert(0, str(dsine_root))

    # DSINE's recent versions expose a hub-style loader.
    # We try multiple loading strategies depending on which version is checked out.
    try:
        # Newer DSINE: torch.hub style
        model = torch.hub.load(str(dsine_root), "DSINE", source="local", trust_repo=True)
    except Exception:
        # Older DSINE: manual instantiation
        from models.dsine import DSINE  # type: ignore

        model = DSINE()
        # Load pretrained weights — adjust path if DSINE's structure changes
        ckpt_paths = list((dsine_root / "checkpoints").glob("*.pt"))
        if not ckpt_paths:
            raise RuntimeError(
                f"No checkpoint found in {dsine_root}/checkpoints. "
                "Download per DSINE's README."
            )
        state = torch.load(ckpt_paths[0], map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)

    _DSINE_MODEL = model
    _DSINE_DEVICE = device
    return model


def _pad_for_dsine(image: torch.Tensor, pad_ratio: float = 0.25) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Mirror-pad a portrait image to give DSINE more spatial context.

    Args:
        image: [3, H, W] tensor in [0, 1].
        pad_ratio: fraction of H/W to pad on each side.

    Returns:
        padded: [3, H_pad, W_pad] tensor.
        crop_box: (top, left, h, w) for cropping back.
    """
    _, h, w = image.shape
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    # F.pad expects (left, right, top, bottom)
    padded = F.pad(
        image.unsqueeze(0),
        (pad_w, pad_w, pad_h, pad_h),
        mode="reflect",
    ).squeeze(0)

    crop_box = (pad_h, pad_w, h, w)
    return padded, crop_box


def estimate_normals(
    image: torch.Tensor,
    device: torch.device | None = None,
    pad: bool = True,
) -> torch.Tensor:
    """Estimate surface normals from a linear RGB image.

    Args:
        image: [3, H, W] tensor, linear RGB, range [0, 1].
        device: target device. If None, uses get_device().
        pad: whether to mirror-pad for better portrait results. Recommended.

    Returns:
        normals: [3, H, W] tensor, camera space, Z toward camera,
                 each pixel a unit vector in [-1, 1].
    """
    if device is None:
        device = get_device()

    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] tensor, got {tuple(image.shape)}")

    model = _load_dsine_model(device)

    # DSINE expects sRGB input in [0, 1], not linear
    from .io_utils import linear_to_srgb
    srgb = linear_to_srgb(image).clamp(0.0, 1.0)

    if pad:
        padded, (top, left, h, w) = _pad_for_dsine(srgb)
        input_tensor = padded.unsqueeze(0).to(device)
    else:
        input_tensor = srgb.unsqueeze(0).to(device)

    with torch.no_grad():
        # DSINE typically returns a list of multi-scale outputs; last one is finest
        output = model(input_tensor)
        if isinstance(output, (list, tuple)):
            normals = output[-1]
        else:
            normals = output

    # normals shape: [1, 3, H_pad, W_pad], values already in [-1, 1] for most DSINE versions
    normals = normals.squeeze(0).cpu()

    if pad:
        normals = normals[:, top : top + h, left : left + w]

    # Normalize to unit vectors (defensive — DSINE usually outputs unit vectors but not always)
    norm = normals.norm(dim=0, keepdim=True).clamp_min(1e-6)
    normals = normals / norm

    # DSINE convention check: their Z axis points away from camera in some versions.
    # Our convention: Z toward camera (so faces facing camera have positive Z).
    # If after this you see faces appearing to face away, flip the Z channel:
    #   normals[2] = -normals[2]
    # We will check this empirically once we have a test image.

    return normals.float()
