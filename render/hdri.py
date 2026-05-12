"""HDRI loading and visualization.

HDRIs are environment maps stored in latlong (equirectangular) projection:
a [H, 2H] image where the horizontal axis maps to azimuth (0 to 2π) and
the vertical axis maps to polar angle (0 = top, π = bottom).

Files come in .hdr (Radiance RGBE) or .exr (OpenEXR) format. Both store
linear-RGB float32 values that can exceed 1.0 — that's the whole point of
HDR. A bright sun pixel might be 10,000+ nits while sky pixels are ~5.

This module handles loading, conversion to cubemaps, and (later)
prefiltering for image-based lighting.

Conventions:
  - Cubemap face order: +X, -X, +Y, -Y, +Z, -Z (standard OpenGL)
  - World axes: +X right, +Y up, +Z toward viewer
  - Latlong: azimuth 0 → +Z direction, increasing toward +X (right-handed)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_hdri(path: str | Path) -> torch.Tensor:
    """Load a .hdr or .exr environment map.

    Args:
        path: Path to .hdr (Radiance RGBE) or .exr (OpenEXR) file.

    Returns:
        Tensor [3, H, W], float32, linear RGB, values in [0, +inf).
        H and W follow the file (typically 2:1 aspect for latlong).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDRI not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".hdr":
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"Failed to load HDR: {path}")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif suffix == ".exr":
        import imageio.v3 as iio
        arr = iio.imread(str(path))
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported HDRI format: {suffix}. Use .hdr or .exr.")

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    tensor = tensor.clamp_min(0.0)
    return tensor


def tonemap_reinhard(hdri: torch.Tensor, exposure: float = 1.0) -> torch.Tensor:
    """Reinhard tone-map an HDRI to displayable [0, 1] linear range.

    The classic Reinhard operator: L' = L / (1 + L). Simple, robust,
    handles arbitrary dynamic range. Result is still linear — pass through
    linear_to_srgb() before saving as PNG.

    Args:
        hdri: [3, H, W] tensor, linear RGB, range [0, +inf).
        exposure: scalar multiplier applied before tone-mapping. Higher
                  values reveal more detail in dark areas but blow out
                  bright ones.

    Returns:
        [3, H, W] tensor in [0, 1].
    """
    x = hdri * exposure
    return x / (1.0 + x)


def hdri_stats(hdri: torch.Tensor) -> dict:
    """Compute summary statistics on an HDRI for debugging."""
    per_channel_max = hdri.amax(dim=(1, 2))
    luminance = 0.2126 * hdri[0] + 0.7152 * hdri[1] + 0.0722 * hdri[2]
    return {
        "shape": tuple(hdri.shape),
        "min": float(hdri.min()),
        "max": float(hdri.max()),
        "mean": float(hdri.mean()),
        "max_R": float(per_channel_max[0]),
        "max_G": float(per_channel_max[1]),
        "max_B": float(per_channel_max[2]),
        "max_luminance": float(luminance.max()),
        "mean_luminance": float(luminance.mean()),
    }

# -----------------------------------------------------------------------------
# Cubemap conversion
# -----------------------------------------------------------------------------
#
# A cubemap is a set of 6 square faces, each a perspective view from the cube
# center along one of the 6 axis directions (+X, -X, +Y, -Y, +Z, -Z).
#
# For each face, we need a function that takes a pixel coordinate (u, v) in
# the face and returns the world-space 3D direction it represents. Then we
# convert that direction to a latlong (lat, lon) coordinate to sample the
# source environment map.
#
# Convention chosen (standard OpenGL cubemap, right-handed world space):
#   +X face: looking down +X axis. u → -Z, v → -Y.
#   -X face: looking down -X axis. u → +Z, v → -Y.
#   +Y face: looking down +Y axis (up). u → +X, v → +Z.
#   -Y face: looking down -Y axis (down). u → +X, v → -Z.
#   +Z face: looking down +Z axis (toward viewer). u → +X, v → -Y.
#   -Z face: looking down -Z axis (away from viewer). u → -X, v → -Y.
#
# Latlong convention:
#   - longitude φ in [-π, π], with φ=0 → +Z direction, increasing toward +X
#   - latitude θ in [-π/2, π/2], with θ=π/2 → +Y (top of image, v=0)

FACE_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def _face_directions(face_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute the unit world-space direction for every pixel of every cube face.

    Returns:
        Tensor [6, face_size, face_size, 3], unit-length direction vectors.
    """
    coords = (torch.arange(face_size, device=device, dtype=dtype) + 0.5) / face_size
    coords = coords * 2.0 - 1.0  # [-1+1/F, 1-1/F]

    uu, vv = torch.meshgrid(coords, coords, indexing="xy")
    ones = torch.ones_like(uu)

    faces = torch.zeros(6, face_size, face_size, 3, device=device, dtype=dtype)

    # +X: ma=+X, sc=-z, tc=-y
    faces[0, ..., 0] =  ones
    faces[0, ..., 1] = -vv
    faces[0, ..., 2] = -uu

    # -X
    faces[1, ..., 0] = -ones
    faces[1, ..., 1] = -vv
    faces[1, ..., 2] =  uu

    # +Y (up)
    faces[2, ..., 0] =  uu
    faces[2, ..., 1] =  ones
    faces[2, ..., 2] =  vv

    # -Y (down)
    faces[3, ..., 0] =  uu
    faces[3, ..., 1] = -ones
    faces[3, ..., 2] = -vv

    # +Z (forward)
    faces[4, ..., 0] =  uu
    faces[4, ..., 1] = -vv
    faces[4, ..., 2] =  ones

    # -Z (back)
    faces[5, ..., 0] = -uu
    faces[5, ..., 1] = -vv
    faces[5, ..., 2] = -ones

    faces = faces / faces.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return faces


def direction_to_latlong_uv(directions: torch.Tensor) -> torch.Tensor:
    """Convert world-space directions to latlong sampling UVs in [-1, 1]."""
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    lon = torch.atan2(x, z)
    lat = torch.asin(y.clamp(-1.0, 1.0))

    u = lon / math.pi               # [-1, 1]
    v = -lat / (math.pi / 2.0)      # [-1, 1], -1 = top

    return torch.stack([u, v], dim=-1)


def latlong_to_cubemap(latlong: torch.Tensor, face_size: int = 256) -> torch.Tensor:
    """Convert a latlong environment map to a cubemap.

    Args:
        latlong: [3, H, W] tensor (typically W = 2H), linear RGB float32.
        face_size: output face size.

    Returns:
        [6, 3, face_size, face_size] tensor in OpenGL face order.
    """
    if latlong.dim() != 3 or latlong.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] latlong, got {tuple(latlong.shape)}")

    device, dtype = latlong.device, latlong.dtype
    dirs = _face_directions(face_size, device, dtype)   # [6, F, F, 3]
    uvs = direction_to_latlong_uv(dirs)                 # [6, F, F, 2]

    src = latlong.unsqueeze(0).expand(6, -1, -1, -1).contiguous()  # [6, 3, H, W]
    sampled = F.grid_sample(
        src, uvs,
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    return sampled


def sample_cubemap(cubemap: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Sample a cubemap along given world-space directions.

    Args:
        cubemap: [6, 3, F, F] tensor.
        directions: [..., 3] unit vectors.

    Returns:
        Sampled colors, shape [..., 3].
    """
    *batch_shape, three = directions.shape
    assert three == 3
    dirs_flat = directions.reshape(-1, 3)
    N = dirs_flat.shape[0]
    device = dirs_flat.device

    abs_d = dirs_flat.abs()
    max_axis = abs_d.argmax(dim=-1)

    face = torch.zeros(N, dtype=torch.long, device=device)
    is_x = max_axis == 0
    is_y = max_axis == 1
    is_z = max_axis == 2
    face = torch.where(is_x & (dirs_flat[:, 0] >= 0), torch.tensor(0, device=device), face)
    face = torch.where(is_x & (dirs_flat[:, 0] <  0), torch.tensor(1, device=device), face)
    face = torch.where(is_y & (dirs_flat[:, 1] >= 0), torch.tensor(2, device=device), face)
    face = torch.where(is_y & (dirs_flat[:, 1] <  0), torch.tensor(3, device=device), face)
    face = torch.where(is_z & (dirs_flat[:, 2] >= 0), torch.tensor(4, device=device), face)
    face = torch.where(is_z & (dirs_flat[:, 2] <  0), torch.tensor(5, device=device), face)

    x, y, z = dirs_flat[:, 0], dirs_flat[:, 1], dirs_flat[:, 2]
    sc = torch.zeros(N, device=device)
    tc = torch.zeros(N, device=device)
    ma = torch.zeros(N, device=device)

    m = face == 0;  sc = torch.where(m, -z, sc); tc = torch.where(m, -y, tc); ma = torch.where(m,  x, ma)
    m = face == 1;  sc = torch.where(m,  z, sc); tc = torch.where(m, -y, tc); ma = torch.where(m, -x, ma)
    m = face == 2;  sc = torch.where(m,  x, sc); tc = torch.where(m,  z, tc); ma = torch.where(m,  y, ma)
    m = face == 3;  sc = torch.where(m,  x, sc); tc = torch.where(m, -z, tc); ma = torch.where(m, -y, ma)
    m = face == 4;  sc = torch.where(m,  x, sc); tc = torch.where(m, -y, tc); ma = torch.where(m,  z, ma)
    m = face == 5;  sc = torch.where(m, -x, sc); tc = torch.where(m, -y, tc); ma = torch.where(m, -z, ma)

    ma = ma.clamp_min(1e-8)
    s = 0.5 * (sc / ma + 1.0)
    t = 0.5 * (tc / ma + 1.0)
    grid_uv = torch.stack([s * 2.0 - 1.0, t * 2.0 - 1.0], dim=-1)

    out = torch.zeros(N, 3, device=cubemap.device, dtype=cubemap.dtype)
    for f in range(6):
        sel = face == f
        if not sel.any():
            continue
        uv_f = grid_uv[sel].view(1, -1, 1, 2)
        face_tex = cubemap[f].unsqueeze(0)
        sampled = F.grid_sample(
            face_tex, uv_f,
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        out[sel] = sampled.squeeze(-1).squeeze(0).T

    return out.reshape(*batch_shape, 3)


def cubemap_to_latlong(cubemap: torch.Tensor, height: int = 512) -> torch.Tensor:
    """Reconstruct a latlong from a cubemap (validation helper).

    Args:
        cubemap: [6, 3, F, F] tensor.
        height: output latlong height. Width will be 2*height.

    Returns:
        [3, height, 2*height] tensor.
    """
    if cubemap.dim() != 4 or cubemap.shape[0] != 6:
        raise ValueError(f"Expected [6, 3, F, F] cubemap, got {tuple(cubemap.shape)}")

    width = 2 * height
    device, dtype = cubemap.device, cubemap.dtype

    v_coords = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    u_coords = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width

    lon = (u_coords * 2.0 - 1.0) * math.pi
    lat = (0.5 - v_coords) * math.pi

    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
    cos_lat = torch.cos(lat_grid)
    x = cos_lat * torch.sin(lon_grid)
    y = torch.sin(lat_grid)
    z = cos_lat * torch.cos(lon_grid)
    dirs = torch.stack([x, y, z], dim=-1)

    sampled = sample_cubemap(cubemap, dirs)   # [H, W, 3]
    return sampled.permute(2, 0, 1).contiguous()


def cubemap_cross_layout(cubemap: torch.Tensor) -> torch.Tensor:
    """Lay out the 6 cube faces in a horizontal cross.

    Layout:
              [+Y]
        [-X][+Z][+X][-Z]
              [-Y]
    """
    six, three, F_size, _ = cubemap.shape
    assert six == 6 and three == 3

    canvas = torch.zeros(3, 3 * F_size, 4 * F_size, device=cubemap.device, dtype=cubemap.dtype)

    def paste(face_idx: int, row: int, col: int):
        y0, x0 = row * F_size, col * F_size
        canvas[:, y0:y0 + F_size, x0:x0 + F_size] = cubemap[face_idx]

    paste(2, 0, 1)  # +Y top
    paste(1, 1, 0)  # -X
    paste(4, 1, 1)  # +Z forward
    paste(0, 1, 2)  # +X right
    paste(5, 1, 3)  # -Z back
    paste(3, 2, 1)  # -Y bottom

    return canvas


# -----------------------------------------------------------------------------
# Diffuse irradiance prefilter
# -----------------------------------------------------------------------------
#
# For Lambertian shading, the integral
#     L_diffuse(n) = ∫_hemisphere L_env(ω) × cos(θ) dω  /  π
# depends only on the normal direction n. Precompute this for every direction
# and store as a "diffuse irradiance cubemap". At shade time:
#     L_diffuse(n) = albedo × sample(irradiance_cubemap, n) / π
#
# We use Monte Carlo with cosine-weighted hemisphere sampling and Hammersley
# low-discrepancy sequences.


def _radical_inverse_vdc(bits: torch.Tensor) -> torch.Tensor:
    """Van der Corput radical inverse, base 2. Vectorized over a tensor of ints."""
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    bits = bits & 0xFFFFFFFF
    return bits.float() * (1.0 / float(0x100000000))


def hammersley_2d(num_samples: int, device: torch.device) -> torch.Tensor:
    """Generate a 2D Hammersley low-discrepancy sequence.

    Returns:
        Tensor [num_samples, 2] of values in [0, 1).
    """
    indices = torch.arange(num_samples, device=device, dtype=torch.long)
    x = indices.float() / num_samples
    y = _radical_inverse_vdc(indices)
    return torch.stack([x, y], dim=-1)


def _cosine_weighted_hemisphere_sample(xi: torch.Tensor) -> torch.Tensor:
    """Cosine-weighted hemisphere sampling in tangent space (+Z is the normal).

    Uses Malley's method: sample disk uniformly, project up.

    Args:
        xi: [..., 2] uniform [0, 1) values.

    Returns:
        [..., 3] unit vectors in the upper hemisphere, distributed ∝ cos(θ).
    """
    u, v = xi[..., 0], xi[..., 1]
    r = torch.sqrt(u.clamp_min(0.0))
    phi = 2.0 * math.pi * v
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt((1.0 - x * x - y * y).clamp_min(0.0))
    return torch.stack([x, y, z], dim=-1)


def _build_tangent_frame(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a continuous orthonormal tangent frame (t, b) for each normal n.

    Uses Frisvad's method (JCGT 2012): closed-form, branchless, no discontinuities.
    A continuous frame matters for Monte Carlo prefiltering because pixels with
    similar normals should sample similar hemisphere regions — otherwise variance
    explodes near singularities.

    Reference: Frisvad, "Building an Orthonormal Basis from a 3D Unit Vector
    Without Normalization", JCGT 1(2), 2012.

    Args:
        n: [..., 3] unit normals.

    Returns:
        (tangent, bitangent), each [..., 3].
    """
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    # The branchless variant: handle the n.z = -1 singularity by clamping
    # the divisor. Near the +Y axis our chosen formulation is well-defined.
    sign = torch.where(nz >= 0.0, torch.ones_like(nz), -torch.ones_like(nz))
    a = -1.0 / (sign + nz)
    b_term = nx * ny * a
    t = torch.stack(
        [
            1.0 + sign * nx * nx * a,
            sign * b_term,
            -sign * nx,
        ],
        dim=-1,
    )
    bt = torch.stack(
        [
            b_term,
            sign + ny * ny * a,
            -ny,
        ],
        dim=-1,
    )
    return t, bt


def _tangent_to_world(v_tangent: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Transform a tangent-space vector to world space, given world normal n."""
    t, b = _build_tangent_frame(n)
    return (
        v_tangent[..., 0:1] * t
        + v_tangent[..., 1:2] * b
        + v_tangent[..., 2:3] * n
    )


def prefilter_diffuse(
    cubemap: torch.Tensor,
    face_size: int = 32,
    num_samples: int = 2048,
    firefly_clamp: float = 500.0,
) -> torch.Tensor:
    """Precompute the diffuse irradiance cubemap from an environment cubemap.

    For each direction n (each pixel of the output cubemap), this approximates:
        E(n) = (1/π) ∫ L(ω) cos(θ) dω
    by Monte Carlo with cosine-weighted samples.

    For HDRIs with extreme dynamic range (e.g. a small bright sun), Monte
    Carlo variance can produce visible speckle at low sample counts. We clamp
    each radiance sample to `firefly_clamp` to trade a small amount of bias
    for a large reduction in variance. This is standard practice in real-time
    IBL prefilters and barely visible in the diffuse output (the energy from
    the sun is preserved as a soft glow rather than a noisy speckle).

    Args:
        cubemap: [6, 3, F_in, F_in] full-resolution environment cubemap.
        face_size: output face size. 32 is plenty for diffuse.
        num_samples: Monte Carlo sample count per pixel.
        firefly_clamp: max per-sample luminance. Set to a large number (e.g. 1e9)
                       to disable. ~50 works well for studio HDRIs.

    Returns:
        [6, 3, face_size, face_size] diffuse irradiance cubemap.
    """
    device, dtype = cubemap.device, cubemap.dtype

    out_dirs = _face_directions(face_size, device, dtype)  # [6, F, F, 3]
    xi = hammersley_2d(num_samples, device)                # [S, 2]
    h_tangent = _cosine_weighted_hemisphere_sample(xi)     # [S, 3]

    accum = torch.zeros(6, face_size, face_size, 3, device=device, dtype=dtype)
    for s in range(num_samples):
        sample_tangent = h_tangent[s].view(1, 1, 1, 3).expand_as(out_dirs)
        sample_world = _tangent_to_world(sample_tangent, out_dirs)  # [6, F, F, 3]
        radiance = sample_cubemap(cubemap, sample_world)            # [6, F, F, 3]

        # Firefly clamp: limit each sample's luminance contribution
        if firefly_clamp is not None and firefly_clamp > 0:
            lum = 0.2126 * radiance[..., 0] + 0.7152 * radiance[..., 1] + 0.0722 * radiance[..., 2]
            scale = torch.minimum(
                torch.ones_like(lum),
                firefly_clamp / lum.clamp_min(1e-8),
            ).unsqueeze(-1)  # [6, F, F, 1]
            radiance = radiance * scale

        accum = accum + radiance

    irradiance = accum / num_samples
    return irradiance.permute(0, 3, 1, 2).contiguous()


# -----------------------------------------------------------------------------
# Specular prefilter (split-sum approximation)
# -----------------------------------------------------------------------------
#
# For the specular term of Cook-Torrance, the integral
#     L_spec(n, v) = ∫ L(ω) × BRDF(ω, v, n) × cos(θ) dω
# depends on n, v, AND roughness — too much to precompute directly.
#
# Karis 2013 (UE4 PBR notes) introduced the "split-sum approximation":
#     L_spec(n, v) ≈ L_env_prefiltered(R, roughness) × BRDF_LUT(N·V, roughness)
#
# where R = reflect(-v, n). The prefiltered cubemap stores, for each
# direction and each roughness, the GGX-importance-sampled environment
# integral. At shade time, sample along R at mip = roughness × (mips-1).
#
# Karis additionally assumes n = v = R during prefiltering. This is wrong
# off-axis but lets us collapse (n, v) into a single direction R and store
# the result in a standard cubemap. The error is part of the inherent
# approximation of split-sum IBL.


# Number of mips and the per-mip resolution schedule
SPECULAR_NUM_MIPS = 6
SPECULAR_MIP_FACE_SIZES = [256, 128, 64, 32, 16, 8]


def _importance_sample_ggx(
    xi: torch.Tensor,
    roughness: float,
    n: torch.Tensor,
) -> torch.Tensor:
    """GGX importance-sample a half-vector around normal n.

    Standard formulation from Karis 2013 / Walter et al. 2007.

    Args:
        xi: [S, 2] tensor of uniform [0, 1) values.
        roughness: scalar in [0, 1]. The GGX α parameter is roughness².
        n: [..., 3] normals to sample around.

    Returns:
        Half-vector samples in world space, shape broadcast of (xi, n).
    """
    a = roughness * roughness  # GGX uses α = roughness²

    u, v = xi[..., 0], xi[..., 1]  # [S]
    phi = 2.0 * math.pi * u
    # Inverse CDF for GGX NDF in cos(θ):
    cos_theta = torch.sqrt((1.0 - v) / (1.0 + (a * a - 1.0) * v).clamp_min(1e-8))
    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp_min(0.0))

    # Tangent-space half-vector
    h_x = sin_theta * torch.cos(phi)
    h_y = sin_theta * torch.sin(phi)
    h_z = cos_theta
    h_tangent = torch.stack([h_x, h_y, h_z], dim=-1)  # [S, 3]

    # Broadcast and transform to world space
    # n is [..., 3], h_tangent is [S, 3] — we want [..., S, 3]
    n_b = n.unsqueeze(-2)                          # [..., 1, 3]
    h_b = h_tangent.expand(*n_b.shape[:-2], -1, -1)  # [..., S, 3]
    return _tangent_to_world(h_b, n_b.expand_as(h_b))


def prefilter_specular(
    cubemap: torch.Tensor,
    num_samples: int = 1024,
    firefly_clamp: float = 500.0,
) -> list[torch.Tensor]:
    """Precompute the GGX-prefiltered specular mip chain.

    Uses Karis 2013 split-sum approximation with **mip-biased importance
    sampling**: low-PDF samples (rare in solid angle) look up at a blurred
    mip level of the source cubemap. This dramatically reduces speckle from
    bright pixels at low sample counts.

    Args:
        cubemap: [6, 3, F_in, F_in] full-resolution environment cubemap.
        num_samples: GGX importance samples per output pixel.
        firefly_clamp: per-sample luminance clamp.

    Returns:
        List of cubemap tensors per mip level.
    """
    device, dtype = cubemap.device, cubemap.dtype

    # Build a mip pyramid of the SOURCE cubemap (each level is a 2x downsample).
    # We'll sample from these progressively-blurred levels at low PDF.
    source_mips: list[torch.Tensor] = [cubemap]
    cur = cubemap
    while cur.shape[-1] > 1:
        next_size = max(1, cur.shape[-1] // 2)
        cur = F.interpolate(cur, size=(next_size, next_size), mode="bilinear", align_corners=False)
        source_mips.append(cur)

    src_F = float(cubemap.shape[-1])
    # Per-pixel solid angle for the source cubemap (rough approximation)
    src_omega_per_pixel = 4.0 * math.pi / (6.0 * src_F * src_F)

    mips: list[torch.Tensor] = []

    for mip_level in range(SPECULAR_NUM_MIPS):
        face_size = SPECULAR_MIP_FACE_SIZES[mip_level]
        roughness = mip_level / (SPECULAR_NUM_MIPS - 1)

        if mip_level == 0:
            # Mirror: just resample source
            if cubemap.shape[-1] == face_size:
                mips.append(cubemap.clone())
            else:
                mips.append(F.interpolate(
                    cubemap, size=(face_size, face_size),
                    mode="bilinear", align_corners=False,
                ))
            continue

        a = roughness * roughness  # GGX α
        out_dirs = _face_directions(face_size, device, dtype)  # [6, F, F, 3]
        xi = hammersley_2d(num_samples, device)

        weighted_radiance = torch.zeros(6, face_size, face_size, 3, device=device, dtype=dtype)
        total_weight = torch.zeros(6, face_size, face_size, 1, device=device, dtype=dtype)

        for s in range(num_samples):
            xi_s = xi[s:s+1]
            # GGX-sample a half-vector around the reflection direction (Karis: n=v=R)
            h = _importance_sample_ggx(xi_s, roughness, out_dirs).squeeze(-2)  # [6, F, F, 3]
            n_dot_h = (out_dirs * h).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            l = 2.0 * n_dot_h * h - out_dirs
            l = l / l.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            n_dot_l = (out_dirs * l).sum(dim=-1, keepdim=True).clamp_min(0.0)

            # GGX PDF at this sample (for mip biasing).
            #   D(h) = α² / (π × ((n·h)² × (α² - 1) + 1)²)
            #   pdf  = D(h) × (n·h) / (4 × (v·h))   ; with v = n, v·h = n·h
            #        = D(h) / 4
            denom = (n_dot_h * n_dot_h * (a * a - 1.0) + 1.0)
            D = (a * a) / (math.pi * denom * denom + 1e-8)
            pdf = D / 4.0

            # Solid angle of one sample under this PDF
            sample_omega = 1.0 / (num_samples * pdf + 1e-8)
            # Pick mip level such that sample_omega ≈ pixel_omega at that mip
            # (each higher mip has 4× the per-pixel solid angle)
            mip_bias = 0.5 * torch.log2(sample_omega / src_omega_per_pixel + 1e-8)
            mip_bias = mip_bias.clamp(0.0, float(len(source_mips) - 1))

            # For simplicity we round to the nearest integer mip and use that
            # level's cubemap as the source. (Trilinear blending across mips
            # is doable but adds complexity for marginal gain in our use case.)
            mip_idx = mip_bias.round().long().squeeze(-1)  # [6, F, F]

            # Bucket samples by mip level (max len(source_mips) buckets, usually ≤ 6).
            radiance = torch.zeros_like(out_dirs)
            for src_level in range(len(source_mips)):
                sel = mip_idx == src_level
                if not sel.any():
                    continue
                l_sel = l[sel]
                rad_sel = sample_cubemap(source_mips[src_level], l_sel)
                radiance[sel] = rad_sel

            # Firefly clamp
            if firefly_clamp is not None and firefly_clamp > 0:
                lum = (
                    0.2126 * radiance[..., 0]
                    + 0.7152 * radiance[..., 1]
                    + 0.0722 * radiance[..., 2]
                )
                scale = torch.minimum(
                    torch.ones_like(lum),
                    firefly_clamp / lum.clamp_min(1e-8),
                ).unsqueeze(-1)
                radiance = radiance * scale

            weighted_radiance = weighted_radiance + radiance * n_dot_l
            total_weight = total_weight + n_dot_l

        prefiltered = weighted_radiance / total_weight.clamp_min(1e-8)
        mips.append(prefiltered.permute(0, 3, 1, 2).contiguous())

    return mips


def sample_prefiltered_specular(
    mips: list[torch.Tensor],
    directions: torch.Tensor,
    roughness: torch.Tensor,
) -> torch.Tensor:
    """Trilinear lookup of the prefiltered specular cubemap at shade time.

    Args:
        mips: list of N mip cubemaps as returned by prefilter_specular.
        directions: [..., 3] unit reflection directions to look up.
        roughness: [..., 1] or scalar broadcastable to directions' batch.

    Returns:
        Sampled specular radiance, shape [..., 3].
    """
    num_mips = len(mips)
    # Map roughness → continuous mip level in [0, num_mips-1]
    mip_f = roughness.clamp(0.0, 1.0) * (num_mips - 1)
    mip_lo = mip_f.floor().long().clamp(0, num_mips - 2)
    mip_hi = mip_lo + 1
    blend = (mip_f - mip_lo.float())

    # Sample each pair of mips and blend. We do this in a small loop over
    # the active mip pairs, which is at most num_mips - 1 ≤ 5.
    out = torch.zeros(*directions.shape[:-1], 3, device=directions.device, dtype=directions.dtype)

    # Flatten for index handling
    *batch_shape, three = directions.shape
    dirs_flat = directions.reshape(-1, 3)
    rough_flat = roughness.reshape(-1, 1) if roughness.dim() >= 1 else roughness.expand(dirs_flat.shape[0], 1)
    mip_lo_flat = mip_lo.reshape(-1)
    mip_hi_flat = mip_hi.reshape(-1)
    blend_flat = blend.reshape(-1, 1)

    out_flat = torch.zeros(dirs_flat.shape[0], 3, device=directions.device, dtype=directions.dtype)

    for pair_lo in range(num_mips - 1):
        sel = mip_lo_flat == pair_lo
        if not sel.any():
            continue
        d_sel = dirs_flat[sel]
        b_sel = blend_flat[sel]
        s_lo = sample_cubemap(mips[pair_lo], d_sel)
        s_hi = sample_cubemap(mips[pair_lo + 1], d_sel)
        out_flat[sel] = s_lo * (1.0 - b_sel) + s_hi * b_sel

    # Handle the edge case where roughness == 1.0 exactly
    sel = mip_lo_flat == (num_mips - 1)
    if sel.any():
        d_sel = dirs_flat[sel]
        out_flat[sel] = sample_cubemap(mips[-1], d_sel)

    return out_flat.reshape(*batch_shape, 3)


# -----------------------------------------------------------------------------
# BRDF integration LUT (split-sum approximation, second half)
# -----------------------------------------------------------------------------
#
# Karis split-sum:
#   L_spec(n, v) ≈ L_prefiltered(R, rough) × (F0 × scale + bias)
#
# where (scale, bias) is a 2D LUT indexed by (N·V, roughness). Built by Monte
# Carlo integration of the GGX BRDF against a uniform white environment.
# This LUT depends only on the BRDF, not on the HDRI — compute once, reuse.


def _geometry_smith_ggx_ibl(n_dot_v: torch.Tensor, n_dot_l: torch.Tensor, roughness: float) -> torch.Tensor:
    """Smith geometry term for GGX, IBL formulation.

    For IBL prefilters the convention uses k = α/2 (not (rough+1)²/8 which is for direct lighting).
    """
    a = roughness * roughness
    k = a / 2.0
    g_v = n_dot_v / (n_dot_v * (1.0 - k) + k + 1e-8)
    g_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 1e-8)
    return g_v * g_l


def integrate_brdf_lut(
    size: int = 128,
    num_samples: int = 1024,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the BRDF integration LUT for the split-sum approximation.

    Fully vectorized: builds the entire (roughness, NoV, sample) tensor and
    reduces over the sample dim. For 128x128 with 1024 samples, runs in well
    under a second on CPU and effectively instant on GPU/MPS.

    Args:
        size: LUT resolution (size × size). 128 is plenty; 256 is overkill.
        num_samples: Monte Carlo samples per LUT pixel.
        device: target device. CPU is fine — this is small and one-time.

    Returns:
        [2, size, size] tensor. Channel 0 = scale, channel 1 = bias.
        Indexed as LUT[:, roughness_idx, NoV_idx].
    """
    if device is None:
        device = torch.device("cpu")

    dtype = torch.float32

    # Axes — small epsilon to avoid singularities at boundaries.
    nov_axis = torch.linspace(1e-3, 1.0, size, device=device, dtype=dtype)        # [W]
    rough_axis = torch.linspace(1e-3, 1.0, size, device=device, dtype=dtype)      # [H]

    # Hammersley samples [S, 2]
    xi = hammersley_2d(num_samples, device).to(dtype)                              # [S, 2]
    u = xi[:, 0]
    w = xi[:, 1]

    # Reshape for broadcasting:
    #   rough_axis: [H, 1, 1]  -> broadcasts over W and S
    #   nov_axis:   [1, W, 1]  -> broadcasts over H and S
    #   u, w:       [1, 1, S]  -> broadcasts over H and W
    R = rough_axis.view(-1, 1, 1)
    NOV = nov_axis.view(1, -1, 1)
    U = u.view(1, 1, -1)
    W = w.view(1, 1, -1)

    a = R * R                                          # [H, 1, 1]

    # View direction (in tangent space, N=+Z): V = (sqrt(1 - NoV²), 0, NoV)
    v_x = torch.sqrt((1.0 - NOV * NOV).clamp_min(0.0))   # [1, W, 1]
    v_z = NOV                                            # [1, W, 1]
    # v_y is 0 by construction.

    # GGX importance-sampled half-vector in tangent space
    phi = 2.0 * math.pi * U                              # [1, 1, S]
    cos_theta = torch.sqrt(((1.0 - W) / (1.0 + (a * a - 1.0) * W)).clamp_min(1e-12))  # [H, 1, S]
    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp_min(0.0))
    h_x = sin_theta * torch.cos(phi)                     # [H, 1, S]
    h_y = sin_theta * torch.sin(phi)                     # [H, 1, S]
    h_z = cos_theta                                      # [H, 1, S]

    # V · H  =  v_x * h_x + v_y * h_y + v_z * h_z   (v_y = 0)
    v_dot_h = (v_x * h_x + v_z * h_z).clamp_min(0.0)     # [H, W, S]

    # L = 2 (V·H) H - V
    l_x = 2.0 * v_dot_h * h_x - v_x                       # [H, W, S]
    l_y = 2.0 * v_dot_h * h_y                             # [H, W, S]
    l_z = 2.0 * v_dot_h * h_z - v_z                       # [H, W, S]

    n_dot_l = l_z.clamp_min(0.0)                          # [H, W, S]
    n_dot_h = h_z.clamp_min(0.0)                          # [H, 1, S]
    n_dot_v = NOV.expand_as(n_dot_l)                      # [H, W, S]

    # Smith geometry term (IBL convention: k = α/2)
    k = a / 2.0                                            # [H, 1, 1]
    g_v = n_dot_v / (n_dot_v * (1.0 - k) + k + 1e-8)
    g_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 1e-8)
    g = g_v * g_l                                         # [H, W, S]

   # LearnOpenGL/Karis canonical formulation. Note: NO multiplication by n_dot_l —
    # the integrand is just G_Vis weighted by (1 - fc) and fc.
    #   G_Vis = (G * V·H) / (N·H * N·V)
    #   fc    = (1 - V·H)^5
    #   scale += G_Vis * (1 - fc)
    #   bias  += G_Vis * fc
    fc = (1.0 - v_dot_h) ** 5                              # [H, W, S]
    g_vis = (g * v_dot_h) / (n_dot_h * NOV + 1e-8)         # [H, W, S]

    # Mask: only count samples where N·L > 0
    valid = (n_dot_l > 0).to(dtype)                        # [H, W, S]

    scale_terms = g_vis * (1.0 - fc) * valid               # [H, W, S]
    bias_terms = g_vis * fc * valid                        # [H, W, S]

    # Average over the sample dimension
    scale = scale_terms.mean(dim=-1)                       # [H, W]
    bias = bias_terms.mean(dim=-1)                         # [H, W]

    lut = torch.stack([scale, bias], dim=0)                # [2, H, W]
    return lut


def sample_brdf_lut(lut: torch.Tensor, n_dot_v: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup into the BRDF LUT.

    Args:
        lut: [2, size, size] LUT from integrate_brdf_lut.
        n_dot_v: [...] N·V values in [0, 1].
        roughness: [...] roughness values in [0, 1].

    Returns:
        [..., 2] tensor with (scale, bias).
    """
    # Build a [..., 1, 1, 2] grid for F.grid_sample.
    nov = n_dot_v.clamp(0.0, 1.0)
    rough = roughness.clamp(0.0, 1.0)
    # grid_sample uses normalized coords in [-1, 1], x=cols=NoV, y=rows=roughness
    grid_x = nov * 2.0 - 1.0
    grid_y = rough * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 1, 1, 2)
    # Sample (LUT is [2, H, W]; we need batch dim, so add it once)
    lut_b = lut.unsqueeze(0)  # [1, 2, H, W]
    # Tile lut to match batch
    n_samples = grid.shape[0]
    lut_b = lut_b.expand(n_samples, -1, -1, -1).contiguous()
    sampled = F.grid_sample(lut_b, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # sampled: [N, 2, 1, 1] -> [N, 2]
    sampled = sampled.squeeze(-1).squeeze(-1)
    return sampled.view(*n_dot_v.shape, 2)