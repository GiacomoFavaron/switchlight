"""Step 4 validation: precompute and visualize the specular mip chain.

For each mip level, save:
  - That mip's cubemap reprojected to latlong
  - All 6 mips composited into one vertical strip for direct comparison

Usage:
    python scripts/test_specular_prefilter.py --input data/hdris/studio_small_03_2k.hdr
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from inverse.io_utils import get_device, save_image_linear
from render.hdri import (
    SPECULAR_NUM_MIPS,
    cubemap_to_latlong,
    hdri_stats,
    latlong_to_cubemap,
    load_hdri,
    prefilter_specular,
    tonemap_reinhard,
)


def main():
    parser = argparse.ArgumentParser(description="Test the specular prefilter mip chain.")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("outputs/hdri_debug"))
    parser.add_argument("--source-face-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    print(f"Loading: {args.input}")
    hdri = load_hdri(args.input).to(device)
    print("Stats:", hdri_stats(hdri))

    print(f"\nBuilding source cubemap (face_size={args.source_face_size})...")
    t0 = time.time()
    cubemap = latlong_to_cubemap(hdri, face_size=args.source_face_size)
    print(f"  {time.time() - t0:.1f}s")

    print(f"\nPrefiltering specular mip chain ({SPECULAR_NUM_MIPS} mips, samples={args.num_samples})...")
    t0 = time.time()
    mips = prefilter_specular(cubemap, num_samples=args.num_samples)
    print(f"  Total: {time.time() - t0:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    # Convert each mip to latlong and stack vertically for direct comparison
    print("\nReprojecting each mip to latlong...")
    panoramas = []
    common_height = 256  # all mips visualized at same size for comparison
    for i, mip in enumerate(mips):
        roughness = i / (SPECULAR_NUM_MIPS - 1)
        ll = cubemap_to_latlong(mip, height=common_height)  # [3, h, 2h]
        # Tonemap each at the same exposure for fair comparison
        ll_tm = tonemap_reinhard(ll, exposure=args.exposure)
        panoramas.append(ll_tm)

        # Per-mip stats
        lum = 0.2126 * mip[..., 0, :, :] + 0.7152 * mip[..., 1, :, :] + 0.0722 * mip[..., 2, :, :]
        print(f"  Mip {i}  roughness={roughness:.2f}  shape={tuple(mip.shape[-2:])}  "
              f"max_lum={lum.max():.2f}  mean_lum={lum.mean():.2f}")

    # Stack the panoramas vertically — sharpest at top, blurriest at bottom
    strip = torch.cat(panoramas, dim=1)  # [3, num_mips * h, 2h]
    save_image_linear(strip, args.output_dir / f"{stem}_specular_mips_strip.png")
    print(f"\n  Saved: {stem}_specular_mips_strip.png")

    # Also save individual mips for closer inspection
    for i, ll_tm in enumerate(panoramas):
        save_image_linear(ll_tm, args.output_dir / f"{stem}_specular_mip{i}.png")

    print("\nWhat to look for in the strip (top = mip 0, sharp; bottom = mip 5, blurry):")
    print("  - Mip 0 should be NEARLY IDENTICAL to the original HDRI")
    print("  - Each successive mip should be visibly MORE BLURRED than the one above")
    print("  - Lamp/umbrella details should soften, then disappear")
    print("  - Last mip (5) should look similar to the diffuse irradiance — broad gradients only")
    print("  - No speckle/noise at any level. If a mid mip has speckle, it's a bug.")


if __name__ == "__main__":
    main()