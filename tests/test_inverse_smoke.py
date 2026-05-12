"""Smoke test for the inverse rendering frontend.

This test runs only the pieces that don't require external repo clones
(matting + I/O utilities). It's a fast sanity check that the project
structure is correctly importable.

Run with: python tests/test_inverse_smoke.py path/to/portrait.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from inverse.io_utils import (
    get_device,
    linear_to_srgb,
    load_image_linear,
    save_image_linear,
    srgb_to_linear,
    visualize_mask,
)
from inverse.matting import extract_mask_from_tensor


def test_io_round_trip():
    """sRGB -> linear -> sRGB should be (approximately) identity."""
    x = torch.rand(3, 16, 16)
    y = linear_to_srgb(srgb_to_linear(x))
    err = (x - y).abs().max().item()
    assert err < 1e-4, f"Round-trip error too high: {err}"
    print(f"  sRGB round-trip max error: {err:.2e}  PASS")


def test_image_load(image_path: Path):
    """Confirm an image loads with correct shape and value range."""
    img = load_image_linear(image_path)
    assert img.dim() == 3, f"Expected 3D tensor, got {img.dim()}D"
    assert img.shape[0] == 3, f"Expected 3 channels, got {img.shape[0]}"
    assert img.dtype == torch.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    print(f"  Loaded {image_path.name}: shape {tuple(img.shape)}, range [{img.min():.3f}, {img.max():.3f}]  PASS")
    return img


def test_matting(image_path: Path, output_dir: Path):
    """Confirm matting produces a sensible mask."""
    img = load_image_linear(image_path)
    print("  Running rembg (first call downloads model — may take a minute)...")
    mask = extract_mask_from_tensor(img)
    assert mask.dim() == 3 and mask.shape[0] == 1
    assert mask.min() >= 0.0 and mask.max() <= 1.0

    fg_fraction = (mask > 0.5).float().mean().item()
    print(f"  Mask shape: {tuple(mask.shape)}, foreground fraction: {fg_fraction:.2%}")

    if fg_fraction < 0.05 or fg_fraction > 0.95:
        print(f"  WARN: foreground fraction {fg_fraction:.2%} is unusual. Check input image.")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_image_linear(img, output_dir / "smoke_01_image.png")
    save_image_linear(visualize_mask(mask), output_dir / "smoke_02_mask.png")
    print(f"  Saved to {output_dir}/  PASS")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_inverse_smoke.py <portrait_image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    print(f"Device: {get_device()}")
    print()

    print("Test 1: sRGB <-> linear round trip")
    test_io_round_trip()
    print()

    print(f"Test 2: image load")
    test_image_load(image_path)
    print()

    print(f"Test 3: foreground matting")
    test_matting(image_path, Path("outputs/smoke"))
    print()

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
