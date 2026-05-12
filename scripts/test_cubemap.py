"""Step 2 validation: latlong → cubemap → latlong round-trip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inverse.io_utils import save_image_linear
from render.hdri import (
    cubemap_cross_layout,
    cubemap_to_latlong,
    hdri_stats,
    latlong_to_cubemap,
    load_hdri,
    tonemap_reinhard,
)


def main():
    parser = argparse.ArgumentParser(description="Test latlong↔cubemap conversion.")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("outputs/hdri_debug"))
    parser.add_argument("--face-size", type=int, default=256)
    parser.add_argument("--exposure", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    hdri = load_hdri(args.input)
    print("Stats:", hdri_stats(hdri))

    print(f"\nConverting to cubemap (face_size={args.face_size})...")
    cubemap = latlong_to_cubemap(hdri, face_size=args.face_size)
    print(f"  Cubemap shape: {tuple(cubemap.shape)}")
    print(f"  Cubemap range: [{cubemap.min():.3f}, {cubemap.max():.3f}]")

    print("\nReprojecting cubemap back to latlong (round-trip test)...")
    reconstructed = cubemap_to_latlong(cubemap, height=hdri.shape[1] // 2)
    print(f"  Reconstructed shape: {tuple(reconstructed.shape)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    cross = cubemap_cross_layout(cubemap)
    save_image_linear(tonemap_reinhard(cross, exposure=args.exposure),
                      args.output_dir / f"{stem}_cubemap_cross.png")
    print(f"  Saved: {stem}_cubemap_cross.png")

    save_image_linear(tonemap_reinhard(reconstructed, exposure=args.exposure),
                      args.output_dir / f"{stem}_cubemap_to_latlong.png")
    print(f"  Saved: {stem}_cubemap_to_latlong.png")

    save_image_linear(tonemap_reinhard(hdri, exposure=args.exposure),
                      args.output_dir / f"{stem}_original_latlong.png")
    print(f"  Saved: {stem}_original_latlong.png")

    print("\nPer-face brightness statistics:")
    face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    for i, name in enumerate(face_names):
        face = cubemap[i]
        lum = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
        print(f"  {name}  mean luminance: {lum.mean():.3f}   max: {lum.max():.3f}")

    print("\nWhat to look for:")
    print("  - cross layout: 6 faces in a + arrangement. Adjacent faces should ")
    print("    connect smoothly across boundaries")
    print("  - cubemap_to_latlong: should look NEARLY IDENTICAL to original_latlong")
    print("  - per-face stats: the face facing the dominant light source")
    print("    should have the highest max luminance")


if __name__ == "__main__":
    main()