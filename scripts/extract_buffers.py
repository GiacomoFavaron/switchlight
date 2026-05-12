"""Extract a buffer bundle from an input portrait image.

Runs the full inverse rendering frontend:
    image -> normal (DSINE) + albedo (Ordinal Shading) + mask (rembg)
    + constant roughness (0.5) and specular (0.04)

Outputs a single .pt file matching the bundle contract in 00_OVERVIEW.md.

Usage:
    python scripts/extract_buffers.py --input photo.jpg --output bundle.pt
    python scripts/extract_buffers.py --input photo.jpg --output bundle.pt --visualize debug/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

# Make the project importable without installing it as a package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inverse.dsine_wrapper import estimate_normals
from inverse.intrinsic_wrapper import estimate_albedo
from inverse.io_utils import (
    get_device,
    load_image_linear,
    resize_image,
    save_image_linear,
    visualize_normal,
    visualize_mask,
)
from inverse.matting import extract_mask_from_tensor


DEFAULT_ROUGHNESS = 0.5
DEFAULT_SPECULAR = 0.04  # F0 for dielectric materials (skin, plastic)


def extract_buffers(
    image_path: Path,
    target_size: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """Run the full inverse rendering frontend on an input image.

    Args:
        image_path: path to input image (any format Pillow can read).
        target_size: if given, resize image to (target_size, target_size) before processing.
        device: target device.

    Returns:
        bundle dict matching the contract in 00_OVERVIEW.md.
    """
    if device is None:
        device = get_device()

    # 1. Load image, optionally resize
    image = load_image_linear(image_path)
    if target_size is not None:
        image = resize_image(image, target_size)

    _, h, w = image.shape
    print(f"  Image: {image.shape}, range [{image.min():.3f}, {image.max():.3f}]")

    # 2. Foreground mask (rembg) — fastest, run first to confirm the pipeline works
    t0 = time.time()
    mask = extract_mask_from_tensor(image)
    # rembg may return a different resolution; ensure match
    if mask.shape[-2:] != (h, w):
        mask = resize_image(mask, (h, w))
    print(f"  Mask: {mask.shape}, range [{mask.min():.3f}, {mask.max():.3f}]  ({time.time()-t0:.1f}s)")

    # 3. Surface normals (DSINE)
    t0 = time.time()
    normal = estimate_normals(image, device=device)
    if normal.shape[-2:] != (h, w):
        normal = resize_image(normal, (h, w))
        # re-normalize after resize
        normal = normal / normal.norm(dim=0, keepdim=True).clamp_min(1e-6)
    print(f"  Normal: {normal.shape}, range [{normal.min():.3f}, {normal.max():.3f}]  ({time.time()-t0:.1f}s)")

    # 4. Albedo (Ordinal Shading / Intrinsic)
    t0 = time.time()
    albedo = estimate_albedo(image, device=device)
    print(f"  Albedo: {albedo.shape}, range [{albedo.min():.3f}, {albedo.max():.3f}]  ({time.time()-t0:.1f}s)")

    # 5. Constant material parameters (refine later if there's time)
    roughness = torch.full((1, h, w), DEFAULT_ROUGHNESS, dtype=torch.float32)
    specular = torch.full((1, h, w), DEFAULT_SPECULAR, dtype=torch.float32)

    bundle = {
        "image": image.float(),
        "normal": normal.float(),
        "albedo": albedo.float(),
        "roughness": roughness,
        "specular": specular,
        "mask": mask.float(),
        "hdri_path": None,  # not applicable for real-input bundles
        "meta": {
            "source": str(image_path),
            "h": h,
            "w": w,
            "extraction_device": str(device),
        },
    }
    return bundle


def save_visualization(bundle: dict, viz_dir: Path) -> None:
    """Save per-buffer PNGs to the given directory for visual sanity check."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    save_image_linear(bundle["image"], viz_dir / "01_image.png")
    save_image_linear(visualize_normal(bundle["normal"]), viz_dir / "02_normal.png")
    save_image_linear(bundle["albedo"], viz_dir / "03_albedo.png")
    save_image_linear(visualize_mask(bundle["mask"]), viz_dir / "04_mask.png")
    print(f"  Visualizations saved to {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract a buffer bundle from a portrait image.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input image path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .pt bundle path.")
    parser.add_argument(
        "--size", type=int, default=None,
        help="Optional resize to (size, size) before processing.",
    )
    parser.add_argument(
        "--visualize", type=Path, default=None,
        help="Optional directory to save per-buffer PNGs for inspection.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override: 'cpu', 'mps', or 'cuda'.",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")
    print(f"Processing: {args.input}")

    bundle = extract_buffers(args.input, target_size=args.size, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"Bundle saved: {args.output}")

    if args.visualize is not None:
        save_visualization(bundle, args.visualize)


if __name__ == "__main__":
    main()
