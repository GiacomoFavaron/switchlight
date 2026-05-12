"""Batch-render bundles under one or more HDRIs.

For each (bundle, HDRI) pair:
  1. Run our Cook-Torrance + IBL renderer on the bundle's buffers
  2. Exposure-match to the bundle's `image` (Blender's GT)
  3. Save an extended bundle with a `rendered_input` key holding the
     linear-HDR exposure-matched render

This produces the augmented dataset Alex's UNet trains against:
  - input:  bundle['rendered_input']  (our physics output)
  - target: bundle['image']            (Blender's reference)
  - loss:   L1 or L2 between UNet(rendered_input, ...) and image

Existing bundles aren't overwritten — augmented bundles are saved to
--output-dir under the same filename, with the `rendered_input` key added.

Usage:
    # Single HDRI: each bundle relit under one fixed environment
    python scripts/batch_render.py \\
        --bundle-dir data/blender/dataset \\
        --hdri data/hdris/studio.hdr \\
        --output-dir data/blender/dataset_augmented

    # Multi-HDRI: each bundle relit under every HDRI in a directory
    # (useful for augmentation: same scene under different lighting)
    python scripts/batch_render.py \\
        --bundle-dir data/blender/dataset \\
        --hdri-dir data/hdris \\
        --output-dir data/blender/dataset_augmented
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from inverse.io_utils import get_device, save_image_linear
from render.cache import get_prefiltered_hdri
from render.cook_torrance import cook_torrance_shade, normalize_exposure, compute_mae
from render.hdri import tonemap_reinhard


REQUIRED_KEYS = ("image", "normal", "albedo", "roughness", "specular", "mask", "hdri_path")


def validate_bundle(bundle: dict, path: Path) -> None:
    """Raise informatively if the bundle doesn't match the contract."""
    for k in REQUIRED_KEYS:
        if k not in bundle:
            raise ValueError(f"{path.name}: missing key '{k}'")
    expected_shapes = {
        "image":     (3, None, None),
        "normal":    (3, None, None),
        "albedo":    (3, None, None),
        "roughness": (1, None, None),
        "specular":  (1, None, None),
        "mask":      (1, None, None),
    }
    H, W = None, None
    for k, expected in expected_shapes.items():
        t = bundle[k]
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{path.name}: '{k}' is not a tensor (got {type(t).__name__})")
        if t.dtype != torch.float32:
            raise ValueError(f"{path.name}: '{k}' has dtype {t.dtype}, expected float32")
        if t.dim() != 3 or t.shape[0] != expected[0]:
            raise ValueError(f"{path.name}: '{k}' has shape {tuple(t.shape)}, expected [{expected[0]}, H, W]")
        if H is None:
            H, W = t.shape[1], t.shape[2]
        elif (t.shape[1], t.shape[2]) != (H, W):
            raise ValueError(f"{path.name}: '{k}' has shape {tuple(t.shape)}, expected [..., {H}, {W}]")


def resolve_hdri_path(hdri_field: str) -> Path:
    """Resolve the bundle's hdri_path field. Falls back to local repo copy."""
    p = Path(hdri_field)
    if p.exists():
        return p
    local = Path("data/blender/hdri") / p.name
    if local.exists():
        return local
    local2 = Path("data/hdris") / p.name
    if local2.exists():
        return local2
    raise FileNotFoundError(f"HDRI not found: {hdri_field}")


def render_bundle_under_hdri(
    bundle: dict,
    hdri_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Render one bundle under one HDRI. Returns (rendered_input, stats)."""
    prefiltered = get_prefiltered_hdri(hdri_path, device=device)

    ours = cook_torrance_shade(
        normal=bundle["normal"],
        albedo=bundle["albedo"],
        roughness=bundle["roughness"],
        specular=bundle["specular"],
        mask=bundle["mask"],
        prefiltered=prefiltered,
    )

    # Exposure-match to Blender's reference (the bundle's `image`).
    # This is what the UNet trains against — keeping the brightness scale
    # consistent across bundles makes the residual bounded.
    ours_matched = normalize_exposure(ours, bundle["image"], bundle["mask"])
    mae = compute_mae(ours_matched, bundle["image"], bundle["mask"])

    stats = {
        "mae": mae,
        "ours_mean": ours.mean().item(),
        "ours_max": ours.max().item(),
        "blender_mean": bundle["image"].mean().item(),
        "blender_max": bundle["image"].max().item(),
    }
    return ours_matched, stats


def main():
    parser = argparse.ArgumentParser(description="Batch-render bundles under one or more HDRIs.")
    parser.add_argument("--bundle-dir", type=Path, required=True,
                        help="Directory containing .pt bundles to process.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdri", type=Path, help="Render each bundle under this single HDRI.")
    src.add_argument("--hdri-dir", type=Path, help="Render each bundle under every HDRI in this dir.")
    src.add_argument("--use-bundle-hdri", action="store_true",
                     help="Render each bundle under the HDRI named in its own hdri_path field.")

    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for augmented bundles.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-preview", action="store_true",
                        help="Also save a tonemapped PNG preview per render.")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if output already exists.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    # Collect bundles
    bundle_paths = sorted([p for p in args.bundle_dir.iterdir() if p.suffix == ".pt"])
    if not bundle_paths:
        print(f"No .pt files in {args.bundle_dir}")
        sys.exit(1)
    print(f"Found {len(bundle_paths)} bundle(s) in {args.bundle_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Tally stats across all renders
    total_renders = 0
    total_skipped = 0
    total_failed = 0
    mae_values = []

    t_start = time.time()

    for bp in bundle_paths:
        print(f"\n[bundle] {bp.name}")
        try:
            bundle = torch.load(bp, map_location=device, weights_only=False)
            validate_bundle(bundle, bp)
            for k in ("image", "normal", "albedo", "roughness", "specular", "mask"):
                bundle[k] = bundle[k].to(device)
        except Exception as e:
            print(f"  [skip] failed to load: {e}")
            total_failed += 1
            continue

        # Determine which HDRI(s) to render under
        if args.hdri:
            hdri_paths = [args.hdri]
        elif args.use_bundle_hdri:
            try:
                hdri_paths = [resolve_hdri_path(bundle["hdri_path"])]
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                total_failed += 1
                continue
        else:  # --hdri-dir
            hdri_paths = sorted([p for p in args.hdri_dir.iterdir()
                                if p.suffix.lower() in (".hdr", ".exr")])

        for hdri_path in hdri_paths:
            # Output naming: <bundle_stem>__<hdri_stem>.pt (if multi-HDRI)
            # or just <bundle_stem>.pt (if single-HDRI)
            if len(hdri_paths) == 1 and (args.hdri or args.use_bundle_hdri):
                out_name = bp.name
            else:
                out_name = f"{bp.stem}__{hdri_path.stem}.pt"
            out_path = args.output_dir / out_name

            if out_path.exists() and not args.force:
                print(f"  [skip-exists] {out_name}")
                total_skipped += 1
                continue

            try:
                t0 = time.time()
                rendered, stats = render_bundle_under_hdri(bundle, hdri_path, device)
                dt = time.time() - t0

                # Augmented bundle: original + rendered_input
                augmented = dict(bundle)  # shallow copy
                augmented["rendered_input"] = rendered.cpu()
                augmented["render_hdri_path"] = str(hdri_path)
                augmented["render_mae"] = stats["mae"]
                # Move other tensors back to CPU for saving
                for k, v in augmented.items():
                    if isinstance(v, torch.Tensor):
                        augmented[k] = v.cpu()
                torch.save(augmented, out_path)

                print(f"  [ok] {out_name}  ({dt:.1f}s)  MAE={stats['mae']:.4f}")
                total_renders += 1
                mae_values.append(stats["mae"])

                if args.save_preview:
                    preview_dir = args.output_dir / "previews"
                    preview_dir.mkdir(exist_ok=True)
                    tm = tonemap_reinhard(rendered.cpu(), exposure=1.0).clamp(0, 1)
                    save_image_linear(tm, preview_dir / f"{out_path.stem}.png")

            except Exception as e:
                print(f"  [fail] {out_name}: {e}")
                traceback.print_exc()
                total_failed += 1

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done. {total_renders} rendered, {total_skipped} skipped, {total_failed} failed.")
    print(f"Total time: {elapsed:.1f}s ({elapsed/max(total_renders,1):.1f}s per render)")
    if mae_values:
        import statistics
        print(f"MAE over {len(mae_values)} renders:")
        print(f"  mean: {statistics.mean(mae_values):.4f}")
        print(f"  min:  {min(mae_values):.4f}")
        print(f"  max:  {max(mae_values):.4f}")
        if len(mae_values) > 1:
            print(f"  std:  {statistics.stdev(mae_values):.4f}")


if __name__ == "__main__":
    main()