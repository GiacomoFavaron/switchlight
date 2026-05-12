"""Cook-Torrance forward rendering with image-based lighting.

Split-sum approximation:
    L_specular = L_prefiltered × (F0 · scale + bias)
where the prefiltered cubemap and the (scale, bias) LUT are precomputed
once per HDRI by render/hdri.py.

Diffuse uses a precomputed diffuse irradiance cubemap:
    L_diffuse = albedo × sample(diffuse_cubemap, N)

This module does the per-pixel shading. It assumes:
    - Orthographic camera looking down -Z (so V = +Z per pixel)
    - Buffer tensors follow the conventions in 00_OVERVIEW.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hdri import sample_cubemap, sample_brdf_lut, sample_prefiltered_specular


@dataclass
class PrefilteredHDRI:
    """Bundle of all prefiltered data for a single HDRI."""
    diffuse_cubemap: torch.Tensor      # [6, 3, F, F]
    specular_mips: list[torch.Tensor]  # list of [6, 3, F_i, F_i] tensors
    brdf_lut: torch.Tensor             # [2, S, S]


def cook_torrance_shade(
    normal: torch.Tensor,      # [3, H, W] unit camera-space, +Z toward camera
    albedo: torch.Tensor,      # [3, H, W] linear RGB
    roughness: torch.Tensor,   # [1, H, W] in [0, 1]
    specular: torch.Tensor,    # [1, H, W] F0 scalar in [0, 1]
    mask: torch.Tensor,        # [1, H, W] in [0, 1]
    prefiltered: PrefilteredHDRI,
) -> torch.Tensor:
    """Render an image using Cook-Torrance + split-sum IBL.

    Args:
        normal, albedo, roughness, specular, mask: see buffer bundle contract.
        prefiltered: precomputed IBL data for the target HDRI.

    Returns:
        [3, H, W] linear-RGB radiance, masked to foreground. Values are
        unclamped HDR — apply tone-mapping for display.
    """
    if normal.dim() != 3 or normal.shape[0] != 3:
        raise ValueError(f"Expected normal [3, H, W], got {tuple(normal.shape)}")

    _, H, W = normal.shape
    device, dtype = normal.device, normal.dtype

    # View direction: orthographic camera looking down -Z, so V is +Z per pixel.
    # Per-pixel vector for math clarity; broadcasts cheaply.
    V = torch.zeros(3, H, W, device=device, dtype=dtype)
    V[2] = 1.0

    # N·V (per pixel)
    NoV = (normal * V).sum(dim=0, keepdim=True).clamp_min(0.0)  # [1, H, W]

    # Reflection direction: R = 2 (N·V) N - V
    R = 2.0 * NoV * normal - V                                  # [3, H, W]
    R = R / R.norm(dim=0, keepdim=True).clamp_min(1e-8)

    # --- Diffuse term ---
    # Sample the diffuse irradiance cubemap along the normal direction.
    # sample_cubemap expects [..., 3] directions and returns [..., 3] colors.
    N_for_sample = normal.permute(1, 2, 0)                       # [H, W, 3]
    diffuse_irradiance = sample_cubemap(prefiltered.diffuse_cubemap, N_for_sample)
    diffuse_irradiance = diffuse_irradiance.permute(2, 0, 1)     # [3, H, W]
    L_diffuse = albedo * diffuse_irradiance                       # [3, H, W]

    # --- Specular term (split-sum) ---
    # 1. Sample prefiltered specular cubemap along R, at mip = roughness × (num_mips - 1)
    R_for_sample = R.permute(1, 2, 0)                            # [H, W, 3]
    rough_for_sample = roughness.permute(1, 2, 0)                # [H, W, 1]
    L_prefiltered = sample_prefiltered_specular(
        prefiltered.specular_mips,
        R_for_sample,
        rough_for_sample,
    )                                                             # [H, W, 3]
    L_prefiltered = L_prefiltered.permute(2, 0, 1)               # [3, H, W]

    # 2. BRDF LUT lookup: returns (scale, bias)
    NoV_flat = NoV.squeeze(0)                                    # [H, W]
    rough_flat = roughness.squeeze(0)                            # [H, W]
    scale_bias = sample_brdf_lut(prefiltered.brdf_lut, NoV_flat, rough_flat)
    # scale_bias: [H, W, 2]
    scale = scale_bias[..., 0:1].permute(2, 0, 1)                # [1, H, W]
    bias = scale_bias[..., 1:2].permute(2, 0, 1)                 # [1, H, W]

    # 3. Combine: L_spec = L_prefiltered × (F0 × scale + bias)
    F0 = specular                                                 # [1, H, W]
    L_specular = L_prefiltered * (F0 * scale + bias)              # [3, H, W]

    # --- Final radiance, masked ---
    output = (L_diffuse + L_specular) * mask                      # [3, H, W]
    return output


def normalize_exposure(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scale 'pred' so its mean luminance over the foreground matches 'gt'.

    Our renderer's absolute brightness scale is arbitrary (depends on HDRI
    intensity and tone-map exposure). For meaningful pixel-wise comparison,
    we match exposure first.
    """
    fg = mask > 0.5
    if fg.sum() < 1:
        return pred
    # Mean luminance over foreground
    def lum(x):
        return 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]
    pred_lum = lum(pred)[fg.squeeze(0)].mean()
    gt_lum = lum(gt)[fg.squeeze(0)].mean()
    scale = (gt_lum / pred_lum.clamp_min(1e-8)).clamp(0.01, 100.0)
    return pred * scale



def compute_mae(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Foreground-masked mean absolute error in linear RGB.

    Args:
        pred: [3, H, W] prediction.
        gt:   [3, H, W] ground truth.
        mask: [1, H, W] foreground mask in [0, 1].

    Returns:
        scalar MAE averaged over foreground pixels and color channels.
    """
    fg = (mask > 0.5).squeeze(0).float()
    abs_diff = (pred - gt).abs() * fg.unsqueeze(0)
    return (abs_diff.sum() / (fg.sum() * 3).clamp_min(1)).item()