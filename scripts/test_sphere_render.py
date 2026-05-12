"""Step 6 validation: render a synthetic sphere using Cook-Torrance + IBL.

Generates the canonical "PBR material grid" — a row of spheres at
varying roughness, optionally with varying F0 too. Validates that:
  - shading looks physically correct (visible light direction, soft falloff)
  - roughness sweeps from mirror to matte continuously
  - F0 controls reflectance strength as expected

Usage:
    python scripts/test_sphere_render.py --hdri data/hdris/studio_small_03_2k.hdr
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from inverse.io_utils import get_device, save_image_linear
from render.cook_torrance import PrefilteredHDRI, cook_torrance_shade
from render.hdri import (
    integrate_brdf_lut,
    latlong_to_cubemap,
    load_hdri,
    prefilter_diffuse,
    prefilter_specular,
    tonemap_reinhard,
)


def make_sphere_buffers(
    size: int,
    roughness: float,
    specular_F0: float,
    albedo_rgb: tuple[float, float, float],
    device: torch.device,
) -> dict:
    """Build a buffer bundle for a sphere centered in a [size, size] image.

    The sphere occupies a circle of radius (size/2 - margin). Background
    pixels (outside the circle) are zeroed and masked out.
    """
    margin = size // 16
    radius = (size // 2) - margin

    # Pixel coordinates centered at image center, in [-radius, radius]
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32) - size / 2,
        torch.arange(size, device=device, dtype=torch.float32) - size / 2,
        indexing="ij",
    )
    # Normalize to sphere local coords in [-1, 1]
    nx = xx / radius
    ny = -yy / radius          # flip y: image y goes down, world y goes up
    nz_sq = 1.0 - nx * nx - ny * ny
    inside = nz_sq > 0.0
    nz = torch.where(inside, torch.sqrt(nz_sq.clamp_min(0.0)), torch.zeros_like(nz_sq))

    normal = torch.stack([nx, ny, nz], dim=0)                    # [3, H, W]
    # zero out background normals, but we'll mask them anyway
    normal = normal * inside.unsqueeze(0).float()

    mask = inside.unsqueeze(0).float()                            # [1, H, W]

    # Constant material
    H, W = size, size
    albedo = torch.tensor(albedo_rgb, device=device, dtype=torch.float32).view(3, 1, 1).expand(3, H, W).contiguous()
    rough = torch.full((1, H, W), roughness, device=device, dtype=torch.float32).clamp_min(0.05)
    spec = torch.full((1, H, W), specular_F0, device=device, dtype=torch.float32)

    # Dummy image (unused for shading, just for completeness)
    image = torch.zeros(3, H, W, device=device, dtype=torch.float32)

    return {
        "image": image,
        "normal": normal,
        "albedo": albedo,
        "roughness": rough,
        "specular": spec,
        "mask": mask,
    }


def main():
    parser = argparse.ArgumentParser(description="Render a sphere grid with Cook-Torrance + IBL.")
    parser.add_argument("--hdri", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("outputs/render_debug"))
    parser.add_argument("--sphere-size", type=int, default=256)
    parser.add_argument("--source-face-size", type=int, default=256)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--diffuse-samples", type=int, default=2048)
    parser.add_argument("--specular-samples", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prefilter the HDRI once
    print(f"Loading HDRI: {args.hdri}")
    hdri = load_hdri(args.hdri).to(device)
    print("  Building source cubemap...")
    cubemap = latlong_to_cubemap(hdri, face_size=args.source_face_size)
    print("  Prefiltering diffuse irradiance...")
    t0 = time.time()
    diffuse_cube = prefilter_diffuse(cubemap, num_samples=args.diffuse_samples)
    print(f"    {time.time() - t0:.1f}s")
    print("  Prefiltering specular mip chain...")
    t0 = time.time()
    specular_mips = prefilter_specular(cubemap, num_samples=args.specular_samples)
    print(f"    {time.time() - t0:.1f}s")
    print("  Computing BRDF LUT...")
    brdf_lut = integrate_brdf_lut(size=128, num_samples=1024, device=device)

    prefiltered = PrefilteredHDRI(
        diffuse_cubemap=diffuse_cube,
        specular_mips=specular_mips,
        brdf_lut=brdf_lut,
    )

    # 2. Render a grid of spheres at varying roughness, with two F0 rows
    roughness_values = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Row 1: dielectric (F0 = 0.04, white albedo) — like plastic / skin
    # Row 2: metal-ish (F0 = 0.9, no albedo — pure mirror) — bright reflections
    # We pick warm albedo for dielectric to see color through diffuse, white for metal.
    rows = [
        {"label": "dielectric F0=0.04", "F0": 0.04, "albedo": (0.8, 0.7, 0.6)},
        {"label": "metallic F0=0.9",   "F0": 0.9,  "albedo": (0.0, 0.0, 0.0)},  # no diffuse for "metal"
    ]

    S = args.sphere_size
    grid_rows = []

    print("\nRendering spheres...")
    for row in rows:
        print(f"  {row['label']}")
        row_tiles = []
        for r in roughness_values:
            buffers = make_sphere_buffers(
                size=S,
                roughness=r,
                specular_F0=row["F0"],
                albedo_rgb=row["albedo"],
                device=device,
            )
            t0 = time.time()
            rendered = cook_torrance_shade(
                normal=buffers["normal"],
                albedo=buffers["albedo"],
                roughness=buffers["roughness"],
                specular=buffers["specular"],
                mask=buffers["mask"],
                prefiltered=prefiltered,
            )
            print(f"    roughness={r:.2f}  max={rendered.max():.2f}  mean={rendered.mean():.3f}  ({time.time()-t0:.2f}s)")
            row_tiles.append(rendered)
        # Concatenate along width
        grid_rows.append(torch.cat(row_tiles, dim=2))

    # Concatenate rows along height
    grid = torch.cat(grid_rows, dim=1).cpu()

    # Save both HDR-tonemapped and clamped versions for comparison
    save_image_linear(tonemap_reinhard(grid, exposure=args.exposure),
                      args.output_dir / "sphere_grid_tonemapped.png")
    save_image_linear(grid.clamp(0, 1),
                      args.output_dir / "sphere_grid_clamped.png")

    print(f"\nSaved sphere grid to {args.output_dir}/")
    print("\nWhat to look for:")
    print("  - Top row (dielectric, warm albedo):")
    print("      Leftmost (roughness=0.05): sharp HDRI reflections visible")
    print("      Rightmost (roughness=1.0): no reflections, just diffuse warm color")
    print("      Smooth visible falloff across the row")
    print("  - Bottom row (metallic, F0=0.9, no albedo):")
    print("      Leftmost: near-perfect mirror, you should see the HDRI reflected")
    print("      Rightmost: rough metal, very blurry environment reflection")
    print("      Brighter overall than top row (high F0)")
    print("  - Both rows: the lit side of every sphere should face the same direction")
    print("    (toward the dominant light in the HDRI — for studio_small_03,")
    print("    that's the umbrella softbox)")


if __name__ == "__main__":
    main()