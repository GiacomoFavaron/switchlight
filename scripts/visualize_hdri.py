"""Visualize an HDRI for sanity-check.

Loads a .hdr / .exr environment map, prints summary statistics, and
saves several tone-mapped PNGs at different exposures so you can see
both the bright (sun, highlights) and dark (shadow, ambient) regions.

Usage:
    python scripts/visualize_hdri.py --input data/hdris/studio_small_03_2k.hdr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inverse.io_utils import save_image_linear
from render.hdri import hdri_stats, load_hdri, tonemap_reinhard


def main():
    parser = argparse.ArgumentParser(description="Visualize an HDRI.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to .hdr / .exr")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("outputs/hdri_debug"),
        help="Directory to save tone-mapped previews."
    )
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    hdri = load_hdri(args.input)

    print("\nStatistics:")
    stats = hdri_stats(hdri)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:10.4f}")
        else:
            print(f"  {k:18s}: {v}")

    exposures = [0.1, 0.5, 1.0, 4.0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.input.stem
    print(f"\nSaving tone-mapped previews to {args.output_dir}/")
    for exp in exposures:
        tonemapped = tonemap_reinhard(hdri, exposure=exp)
        out_path = args.output_dir / f"{stem}_exposure_{exp:g}.png"
        save_image_linear(tonemapped, out_path)
        print(f"  {out_path.name}")

    print("\nWhat to look for:")
    print("  - The HDRI should look right-side-up (sky at top, ground at bottom)")
    print("  - Latlong wraps horizontally: left edge and right edge should match")
    print("  - At low exposure (0.1, 0.5) you'll see bright highlights / sun position")
    print("  - At high exposure (4.0) you'll see ambient / shadow detail")
    print("  - Max luminance much higher than mean luminance => true HDR")


if __name__ == "__main__":
    main()