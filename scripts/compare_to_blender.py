"""Compare our Cook-Torrance renderer against Blender's reference beauty render.

Given a Blender-produced bundle (which contains GT buffers AND the beauty
render in the 'image' field), run our renderer on the same buffers + HDRI,
then produce a side-by-side comparison + error map.

This is the load-bearing checkpoint of A2: if our renderer's output agrees
with Blender's under the same inputs, the physics path is validated.

Usage:
    python scripts/compare_to_blender.py \\
        --bundle data/blender/smoke_test/frame_0000.pt
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
from render.cache import get_prefiltered_hdri
from render.cook_torrance import cook_torrance_shade, normalize_exposure, compute_mae
from render.hdri import tonemap_reinhard





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--exposure", type=float, default=1.0,
                        help="Tone-map exposure for the final PNG.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    # Load Blender's bundle
    print(f"Loading bundle: {args.bundle}")
    bundle = torch.load(args.bundle, map_location=device, weights_only=False)
    for k in ("image", "normal", "albedo", "roughness", "specular", "mask"):
        if isinstance(bundle[k], torch.Tensor):
            bundle[k] = bundle[k].to(device)

    blender_image = bundle["image"]
    mask = bundle["mask"]
    hdri_path = Path(bundle["hdri_path"])
    # The bundle was packed on Hrithik's machine with an absolute path that won't
    # resolve here. Fall back to looking for the HDRI by basename in data/blender/hdri/.
    if not hdri_path.exists():
        local_repo_hdri = Path("data/blender/hdri") / hdri_path.name
        if local_repo_hdri.exists():
            print(f"[path remap] {hdri_path} -> {local_repo_hdri}")
            hdri_path = local_repo_hdri
        else:
            print(f"!! HDRI not found at {hdri_path} or {local_repo_hdri}")
            sys.exit(1)
    print(f"HDRI used by Blender: {hdri_path}")

    # Prefilter the HDRI (or load from cache)
    prefiltered = get_prefiltered_hdri(hdri_path, device=device)

    # Run our renderer on Blender's GT buffers
    print("\nRunning Cook-Torrance on Blender GT buffers...")
    t0 = time.time()
    ours = cook_torrance_shade(
        normal=bundle["normal"],
        albedo=bundle["albedo"],
        roughness=bundle["roughness"],
        specular=bundle["specular"],
        mask=mask,
        prefiltered=prefiltered,
    )
    print(f"  shade: {time.time()-t0:.2f}s")
    print(f"  ours range:    [{ours.min():.4f}, {ours.max():.4f}]  mean={ours.mean():.4f}")
    print(f"  blender range: [{blender_image.min():.4f}, {blender_image.max():.4f}]  mean={blender_image.mean():.4f}")

    # Exposure-match ours to Blender's
    ours_matched = normalize_exposure(ours, blender_image, mask)
    print(f"  ours (exposure-matched) range: [{ours_matched.min():.4f}, {ours_matched.max():.4f}]")

    # Foreground-masked error
    fg = (mask > 0.5).squeeze(0).float()
    abs_diff = (ours_matched - blender_image).abs() * fg.unsqueeze(0)
    mae = abs_diff.sum() / (fg.sum() * 3).clamp_min(1)
    print(f"  MAE over foreground (linear RGB): {mae:.4f}")

    # Side-by-side comparison: [Blender | Ours | Error×5]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blender_tm = tonemap_reinhard(blender_image, exposure=args.exposure).clamp(0, 1)
    ours_tm    = tonemap_reinhard(ours_matched, exposure=args.exposure).clamp(0, 1)
    error_amp  = (abs_diff * 5.0).clamp(0, 1)  # amplify x5 for visibility

    triptych = torch.cat([blender_tm, ours_tm, error_amp], dim=2).cpu()
    save_image_linear(triptych, args.output_dir / "compare_blender_vs_ours.png")

    # Also save individual images
    save_image_linear(blender_tm.cpu(), args.output_dir / "blender_reference.png")
    save_image_linear(ours_tm.cpu(),    args.output_dir / "ours_matched.png")
    save_image_linear(error_amp.cpu(),  args.output_dir / "error_x5.png")

    print(f"\nSaved comparison to {args.output_dir}/")
    print("\nWhat to look for in compare_blender_vs_ours.png (3 panels left-to-right):")
    print("  - Panel 1 (Blender):    reference beauty render")
    print("  - Panel 2 (Ours):       our Cook-Torrance + IBL output, exposure-matched")
    print("  - Panel 3 (Error × 5):  per-pixel absolute difference, amplified 5x")
    print("\nThe two should look qualitatively similar (same lighting direction,")
    print("similar shadow placement, similar overall tone). The error panel should")
    print("be mostly dark — bright spots indicate where we disagree with Blender.")


if __name__ == "__main__":
    main()