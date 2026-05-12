"""Relight a portrait under a target HDRI using Cook-Torrance + IBL.

Inputs:
  - A buffer bundle (.pt from extract_buffers.py) OR a raw portrait image
  - A target HDRI (.hdr or .exr)

Outputs:
  - The relit image (linear HDR + tone-mapped sRGB PNG)
  - A debug grid showing input, normal, albedo, mask, target HDRI, relit result

Usage:
    # From a bundle:
    python scripts/relight.py --bundle bundle.pt --hdri studio.hdr --output relit.png

    # From a raw image (extracts buffers on the fly):
    python scripts/relight.py --input portrait.jpg --hdri studio.hdr --output relit.png

    # Batch mode:
    python scripts/relight.py --bundle bundle.pt --hdri-dir data/hdris/ --output-dir outputs/relit/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from inverse.io_utils import (
    get_device,
    linear_to_srgb,
    load_image_linear,
    resize_image,
    save_image_linear,
    visualize_mask,
    visualize_normal,
)
from render.cache import get_prefiltered_hdri
from render.cook_torrance import cook_torrance_shade
from render.hdri import load_hdri, tonemap_reinhard


def load_bundle(bundle_path: Path, device: torch.device) -> dict:
    """Load a buffer bundle and move all tensors to the target device."""
    b = torch.load(bundle_path, map_location=device, weights_only=False)
    out = {}
    for k, v in b.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def make_bundle_from_image(image_path: Path, device: torch.device, size: int) -> dict:
    """Extract a buffer bundle from a raw image (slow path; reuses extract_buffers logic)."""
    # Local import keeps the fast path (loading a .pt) cheap
    from scripts.extract_buffers import extract_buffers
    return extract_buffers(image_path, target_size=size, device=device)


def make_debug_grid(
    bundle: dict,
    hdri_thumbnail: torch.Tensor,
    rendered: torch.Tensor,
    rendered_clamped: torch.Tensor,
) -> torch.Tensor:
    """Build a 2x3 grid: [input | albedo | normal] / [mask | HDRI | rendered]."""
    H, W = bundle["image"].shape[-2:]

    def fit(t: torch.Tensor) -> torch.Tensor:
        """Resize any [C, h, w] tensor to [3, H, W] (broadcasting grayscale if needed)."""
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        if t.shape[-2:] != (H, W):
            t = F.interpolate(t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        return t.clamp(0.0, 1.0)

    row1 = torch.cat([
        fit(bundle["image"]),
        fit(bundle["albedo"]),
        fit(visualize_normal(bundle["normal"])),
    ], dim=2)
    row2 = torch.cat([
        fit(visualize_mask(bundle["mask"])),
        fit(hdri_thumbnail),
        fit(rendered_clamped),
    ], dim=2)
    return torch.cat([row1, row2], dim=1)


def relight_one(
    bundle: dict,
    hdri_path: Path,
    device: torch.device,
    exposure: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the renderer on one (bundle, HDRI) pair.

    Returns (rendered_hdr_linear, rendered_tonemapped) — both [3, H, W].
    """
    prefiltered = get_prefiltered_hdri(hdri_path, device=device)
    t0 = time.time()
    rendered = cook_torrance_shade(
        normal=bundle["normal"],
        albedo=bundle["albedo"],
        roughness=bundle["roughness"],
        specular=bundle["specular"],
        mask=bundle["mask"],
        prefiltered=prefiltered,
    )
    print(f"  shade time: {time.time()-t0:.2f}s")
    print(f"  rendered range: [{rendered.min():.3f}, {rendered.max():.3f}]")
    rendered_tm = tonemap_reinhard(rendered, exposure=exposure)
    return rendered, rendered_tm


def main():
    parser = argparse.ArgumentParser(description="Relight a portrait under a target HDRI.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", type=Path, help="Path to a buffer bundle (.pt from extract_buffers.py)")
    src.add_argument("--input", type=Path, help="Path to a raw portrait image (will extract buffers inline)")

    tgt = parser.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--hdri", type=Path, help="Single HDRI file")
    tgt.add_argument("--hdri-dir", type=Path, help="Directory of HDRIs (batch mode)")

    parser.add_argument("--output", type=Path, default=None,
                        help="Single output PNG path (only with --hdri)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/relit"),
                        help="Output directory (used by --hdri-dir or as default)")
    parser.add_argument("--size", type=int, default=768,
                        help="Target resolution. Ignored when loading an existing bundle.")
    parser.add_argument("--exposure", type=float, default=1.0,
                        help="Tone-map exposure multiplier for the PNG output.")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu, mps, or cuda. Defaults to best available.")
    parser.add_argument("--save-debug-grid", action="store_true",
                        help="Also save the [input|albedo|normal / mask|HDRI|rendered] grid.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    # 1. Load or build the bundle
    if args.bundle:
        print(f"Loading bundle: {args.bundle}")
        bundle = load_bundle(args.bundle, device)
    else:
        print(f"Extracting buffers from: {args.input}")
        bundle = make_bundle_from_image(args.input, device, args.size)
        # extract_buffers returns CPU tensors; move to target device
        bundle = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in bundle.items()}
    src_name = (args.bundle or args.input).stem

    # 2. Determine which HDRIs to render against
    if args.hdri:
        hdri_paths = [args.hdri]
    else:
        hdri_paths = sorted([p for p in args.hdri_dir.iterdir()
                             if p.suffix.lower() in (".hdr", ".exr")])
        if not hdri_paths:
            print(f"No .hdr or .exr files in {args.hdri_dir}")
            sys.exit(1)
        print(f"Found {len(hdri_paths)} HDRI(s) in {args.hdri_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Render each (bundle, HDRI) combination
    for hdri_path in hdri_paths:
        print(f"\nRelighting under: {hdri_path.name}")
        rendered_hdr, rendered_tm = relight_one(
            bundle, hdri_path, device, exposure=args.exposure,
        )

        # Pick output path
        if args.output and args.hdri:
            out_path = args.output
        else:
            out_path = args.output_dir / f"{src_name}__{hdri_path.stem}.png"

        save_image_linear(rendered_tm, out_path)
        print(f"  saved: {out_path}")

        if args.save_debug_grid:
            # Tonemap a thumbnail of the HDRI for the grid
            hdri_raw = load_hdri(hdri_path)
            hdri_thumb = tonemap_reinhard(hdri_raw, exposure=args.exposure)
            hdri_thumb = resize_image(hdri_thumb, (bundle["image"].shape[-2], bundle["image"].shape[-1]))

            grid = make_debug_grid(
                bundle={k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in bundle.items()},
                hdri_thumbnail=hdri_thumb,
                rendered=rendered_hdr.cpu(),
                rendered_clamped=rendered_tm.cpu(),
            )
            grid_path = out_path.with_name(out_path.stem + "_grid.png")
            save_image_linear(grid, grid_path)
            print(f"  saved grid: {grid_path}")


if __name__ == "__main__":
    main()