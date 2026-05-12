"""Surface normal estimation using DSINE (Bae & Davison, CVPR 2024).

DSINE is not pip-installable. You need to clone the repo:

    git clone https://github.com/baegwangbin/DSINE.git third_party/DSINE

The pretrained weights are downloaded automatically from HuggingFace
(camenduru/DSINE) by torch.hub on first call. Cached under
~/.cache/torch/hub/checkpoints/.

DSINE's official hubconf hardcodes CUDA. We bypass that by directly using
its model code and weight-loading logic, which lets us target MPS or CPU.

For portrait inputs, we mirror-pad the image before inference. DSINE was
trained on scene-level imagery and produces noticeably better normals
around hair and shoulders when given more spatial context.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from .io_utils import get_device, linear_to_srgb


_DSINE_MODEL = None
_DSINE_DEVICE = None

# Standard ImageNet normalization, used by DSINE's official Predictor
_IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def _find_dsine_root() -> Path:
    """Locate the DSINE repo. Check env var, then standard locations."""
    candidates = []
    if env := os.environ.get("DSINE_ROOT"):
        candidates.append(Path(env))

    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / "third_party" / "DSINE")
    candidates.append(project_root.parent / "DSINE")

    for c in candidates:
        if c.exists() and (c / "hubconf.py").exists():
            return c

    raise RuntimeError(
        "Could not find DSINE repo. Clone it with:\n"
        "  git clone https://github.com/baegwangbin/DSINE.git third_party/DSINE\n"
        "Or set the DSINE_ROOT environment variable."
    )


def _load_dsine_model(device: torch.device):
    """Load DSINE model and pretrained weights. Cached at module level.

    The DSINE_v02 constructor takes an argparse-style Namespace. We construct
    one with the exact values from projects/dsine/experiments/exp001_cvpr2024/dsine.txt
    (the CVPR 2024 config that matches the released weights).
    """
    global _DSINE_MODEL, _DSINE_DEVICE

    if _DSINE_MODEL is not None and _DSINE_DEVICE == device:
        return _DSINE_MODEL

    from argparse import Namespace

    dsine_root = _find_dsine_root()
    if str(dsine_root) not in sys.path:
        sys.path.insert(0, str(dsine_root))

    from models.dsine.v02 import DSINE_v02  # type: ignore

    # Hyperparameters from projects/dsine/experiments/exp001_cvpr2024/dsine.txt
    # plus defaults from projects/dsine/config.py for args not in the config file.
    # These match the weights at huggingface.co/camenduru/DSINE.
    args = Namespace(
        NNET_architecture="v02",
        NNET_encoder_B=5,
        NNET_decoder_NF=2048,
        NNET_decoder_BN=False,
        NNET_decoder_down=8,
        NNET_learned_upsampling=True,
        NNET_output_dim=3,
        NNET_feature_dim=64,
        NNET_hidden_dim=64,
        NRN_prop_ps=5,
        NRN_num_iter_train=5,
        NRN_num_iter_test=5,
        NRN_ray_relu=True,
    )

    weight_url = "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
    state = torch.hub.load_state_dict_from_url(
        weight_url,
        file_name="dsine.pt",
        map_location="cpu",
    )
    if "model" in state:
        state = state["model"]

    model = DSINE_v02(args)
    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.to(device)
    if hasattr(model, "pixel_coords"):
        model.pixel_coords = model.pixel_coords.to(device)

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
        crop_box: (top, left, h, w) for cropping back to original size.
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

    Mirrors the canonical test pipeline in DSINE's projects/dsine/test.py:
      1. ImageNet-normalize the (sRGB) image
      2. Compute padding to multiples of 32 with utils.get_padding
      3. Apply utils.pad_input which pads the image and shifts intrinsics
      4. Forward pass; crop back to original size

    Args:
        image: [3, H, W] tensor, linear RGB, range [0, 1].
        device: target device. If None, uses get_device().
        pad: whether to mirror-pad for better portrait results (our addition,
             on top of DSINE's standard pipeline).

    Returns:
        normals: [3, H, W] tensor, camera space, each pixel a unit vector
                 in [-1, 1]. Z convention: positive Z points toward camera.
    """
    if device is None:
        device = get_device()

    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] tensor, got {tuple(image.shape)}")

    model = _load_dsine_model(device)

    # DSINE expects sRGB-encoded input in [0, 1]
    srgb = linear_to_srgb(image).clamp(0.0, 1.0)

    # Optional mirror-pad for portrait context (our addition, not DSINE-canonical)
    if pad:
        padded, (top, left, h, w) = _pad_for_dsine(srgb)
        x = padded
    else:
        x = srgb
        _, h, w = srgb.shape
        top, left = 0, 0

    from utils import utils as dsine_utils  # type: ignore  # from DSINE repo

    # Build the [1, 3, H, W] tensor that DSINE's pipeline expects
    img = x.unsqueeze(0).to(device)
    _, _, orig_H, orig_W = img.shape

    # Apply ImageNet normalization BEFORE pad_input (pad_input pads with the
    # normalized-zero values, which only makes sense after normalization)
    img = _IMAGENET_NORMALIZE(img.squeeze(0)).unsqueeze(0)

    # Build default intrinsics for 60° FoV (matches DSINE's Predictor default).
    # Their function lives in utils.projection, not utils.utils.
    from utils.projection import intrins_from_fov  # type: ignore  # from DSINE repo
    intrins = intrins_from_fov(
        new_fov=60.0, H=orig_H, W=orig_W, device=device
    ).unsqueeze(0)

    # Pad image and adjust intrinsics to multiples of 32
    lrtb = dsine_utils.get_padding(orig_H, orig_W)
    img, intrins = dsine_utils.pad_input(img, intrins, lrtb)

    with torch.no_grad():
        # Try with mode='test' first (their canonical test-time call), fall
        # back to no-mode call if the model doesn't accept it.
        try:
            out = model(img, intrins=intrins, mode="test")
        except TypeError:
            out = model(img, intrins=intrins)

        pred_norm = out[-1] if isinstance(out, (list, tuple)) else out
        # Crop the padded part: their code uses [t:t+orig_H, l:l+orig_W]
        l_p, _, t_p, _ = lrtb
        pred_norm = pred_norm[:, :3, t_p:t_p + orig_H, l_p:l_p + orig_W]

    pred_norm = pred_norm.squeeze(0).cpu()

    # Crop back from our reflect-padding for portrait context
    if pad:
        pred_norm = pred_norm[:, top : top + h, left : left + w]

    # Defensive renormalize
    pred_norm = pred_norm / pred_norm.norm(dim=0, keepdim=True).clamp_min(1e-6)

    # DSINE's Z convention already matches ours (positive Z = toward camera),
    # so no flip needed. (Verified empirically against a portrait visualization.)

    return pred_norm.float()