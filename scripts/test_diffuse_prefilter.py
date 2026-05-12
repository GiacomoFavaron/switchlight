"""Step 3 validation: precompute and visualize the diffuse irradiance cubemap.

Usage:
    python scripts/test_diffuse_prefilter.py --input data/hdris/studio_small_03_2k.hdr
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inverse.io_utils import get_device, save_image_linear
from render.hdri import (
    cubemap_cross_layout,
    cubemap_to_latlong,
    hdri_stats,
    latlong_to_cubemap,
    load_hdri,
    prefilter_diffuse,
    tonemap_reinhard,
)


def main():
    parser = argparse.ArgumentParser(description="Test diffuse irradiance prefilter.")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("outputs/hdri_debug"))
    parser.add_argument("--source-face-size", type=int, default=256,
                        help="Cubemap face size for the source environment.")
    parser.add_argument("--diffuse-face-size", type=int, default=32,
                        help="Cubemap face size for the diffuse irradiance output. 32 is plenty.")
    parser.add_argument("--num-samples", type=int, default=2048,
                        help="Monte Carlo samples per output pixel.")
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None,
                        help="cpu, mps, or cuda. Defaults to best available.")
    args = parser.parse_args()

    import torch
    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    print(f"Loading: {args.input}")
    hdri = load_hdri(args.input).to(device)
    print("Stats:", hdri_stats(hdri))

    print(f"\nBuilding source cubemap (face_size={args.source_face_size})...")
    t0 = time.time()
    cubemap = latlong_to_cubemap(hdri, face_size=args.source_face_size)
    print(f"  {time.time() - t0:.1f}s")

    print(f"\nPrefiltering diffuse irradiance "
          f"(out face_size={args.diffuse_face_size}, samples={args.num_samples})...")
    t0 = time.time()
    diffuse_cube = prefilter_diffuse(
        cubemap,
        face_size=args.diffuse_face_size,
        num_samples=args.num_samples,
    )
    print(f"  {time.time() - t0:.1f}s")
    print(f"  Diffuse cubemap shape: {tuple(diffuse_cube.shape)}")
    print(f"  Diffuse cubemap range: [{diffuse_cube.min():.3f}, {diffuse_cube.max():.3f}]")

    print("\nPer-face diffuse irradiance (luminance):")
    for i, name in enumerate(["+X", "-X", "+Y", "-Y", "+Z", "-Z"]):
        face = diffuse_cube[i]
        lum = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
        print(f"  {name}  mean: {lum.mean():.3f}   max: {lum.max():.3f}   min: {lum.min():.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    # Save the diffuse cubemap in a few ways
    cross = cubemap_cross_layout(diffuse_cube)
    save_image_linear(tonemap_reinhard(cross, exposure=args.exposure),
                      args.output_dir / f"{stem}_diffuse_cross.png")
    print(f"\n  Saved: {stem}_diffuse_cross.png")

    # Also reproject to latlong so we can see it as a panorama
    diffuse_latlong = cubemap_to_latlong(diffuse_cube, height=256)
    save_image_linear(tonemap_reinhard(diffuse_latlong, exposure=args.exposure),
                      args.output_dir / f"{stem}_diffuse_latlong.png")
    print(f"  Saved: {stem}_diffuse_latlong.png")

    print("\nWhat to look for:")
    print("  - Diffuse irradiance is EXTREMELY blurry — no sharp features.")
    print("  - Each face should be a smooth color blob.")
    print("  - Faces facing the dominant light(s) should be brighter than the others.")
    print("  - Max value should be much SMALLER than the source cubemap's max")
    print("    (averaging a hemisphere flattens out the sharp highlights).")
    print("  - If you see anything sharp or speckled, the integration is wrong.")


if __name__ == "__main__":
    main()