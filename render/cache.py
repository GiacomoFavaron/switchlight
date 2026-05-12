"""Disk cache for prefiltered HDRI data.

Each HDRI's diffuse cubemap, specular mip chain, and the (universal)
BRDF LUT take ~3 minutes to compute. We cache the per-HDRI results
keyed by (file path + mtime hash) so subsequent calls are instant.

The BRDF LUT is cached globally (one file in data/brdf_lut.pt).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .cook_torrance import PrefilteredHDRI
from .hdri import (
    integrate_brdf_lut,
    latlong_to_cubemap,
    load_hdri,
    prefilter_diffuse,
    prefilter_specular,
)


DEFAULT_CACHE_DIR = Path("cache/hdri")
DEFAULT_BRDF_LUT_PATH = Path("data/brdf_lut.pt")


def _hdri_cache_key(hdri_path: Path) -> str:
    """Build a stable cache key for an HDRI: SHA1 of (absolute path + size + mtime)."""
    p = hdri_path.resolve()
    st = p.stat()
    h = hashlib.sha1(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _get_or_build_brdf_lut(
    device: torch.device,
    lut_size: int = 128,
    num_samples: int = 1024,
    cache_path: Path = DEFAULT_BRDF_LUT_PATH,
) -> torch.Tensor:
    """Load the BRDF LUT from disk, or compute and save it if missing."""
    if cache_path.exists():
        return torch.load(cache_path, map_location=device)
    lut = integrate_brdf_lut(size=lut_size, num_samples=num_samples, device=device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lut, cache_path)
    return lut


def get_prefiltered_hdri(
    hdri_path: Path,
    device: torch.device,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    source_face_size: int = 256,
    diffuse_samples: int = 2048,
    specular_samples: int = 1024,
    force_recompute: bool = False,
) -> PrefilteredHDRI:
    """Return a PrefilteredHDRI bundle for the given HDRI, using disk cache."""
    hdri_path = Path(hdri_path)
    key = _hdri_cache_key(hdri_path)
    cache_file = cache_dir / f"{key}.pt"

    if cache_file.exists() and not force_recompute:
        print(f"  [cache hit] prefiltered HDRI for {hdri_path.name}")
        data = torch.load(cache_file, map_location=device)
        brdf_lut = _get_or_build_brdf_lut(device)
        return PrefilteredHDRI(
            diffuse_cubemap=data["diffuse_cubemap"].to(device),
            specular_mips=[m.to(device) for m in data["specular_mips"]],
            brdf_lut=brdf_lut.to(device),
        )

    print(f"  [cache miss] prefiltering HDRI for {hdri_path.name} (this may take ~3 min)...")
    hdri = load_hdri(hdri_path).to(device)
    cubemap = latlong_to_cubemap(hdri, face_size=source_face_size)
    diffuse_cube = prefilter_diffuse(cubemap, num_samples=diffuse_samples)
    specular_mips = prefilter_specular(cubemap, num_samples=specular_samples)
    brdf_lut = _get_or_build_brdf_lut(device)

    # Save the per-HDRI part (BRDF LUT is universal and cached separately)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "diffuse_cubemap": diffuse_cube.cpu(),
        "specular_mips": [m.cpu() for m in specular_mips],
        "hdri_path": str(hdri_path),
    }, cache_file)
    print(f"  [cached] {cache_file}")

    return PrefilteredHDRI(
        diffuse_cubemap=diffuse_cube,
        specular_mips=specular_mips,
        brdf_lut=brdf_lut,
    )