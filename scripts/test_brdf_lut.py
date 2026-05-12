"""Step 5 validation: compute and visualize the BRDF integration LUT.

The LUT is universal — it doesn't depend on any HDRI. We compute it once
and save it. Visualize with R = scale channel, G = bias channel.

Usage:
    python scripts/test_brdf_lut.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from inverse.io_utils import save_image_linear
from render.hdri import integrate_brdf_lut


def main():
    parser = argparse.ArgumentParser(description="Build and visualize the BRDF integration LUT.")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("outputs/hdri_debug"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--save-pt", type=Path, default=Path("data/brdf_lut.pt"))
    args = parser.parse_args()

    print(f"Computing BRDF LUT ({args.size}x{args.size}, samples={args.num_samples})...")
    t0 = time.time()
    lut = integrate_brdf_lut(size=args.size, num_samples=args.num_samples, device=torch.device("cpu"))
    print(f"  {time.time() - t0:.1f}s")

    print(f"\nLUT shape: {tuple(lut.shape)}")
    print(f"  scale (channel 0): min={lut[0].min():.4f}  max={lut[0].max():.4f}  mean={lut[0].mean():.4f}")
    print(f"  bias  (channel 1): min={lut[1].min():.4f}  max={lut[1].max():.4f}  mean={lut[1].mean():.4f}")

    # Sanity checks
    print("\nSanity checks:")
    print(f"  Top-left  (low N·V, low rough): scale={lut[0, 0, 0]:.4f}  bias={lut[1, 0, 0]:.4f}  (expect both near 0)")
    print(f"  Top-right (high N·V, low rough): scale={lut[0, 0, -1]:.4f}  bias={lut[1, 0, -1]:.4f}  (expect scale near 1, bias near 0)")
    print(f"  Bottom-left  (low N·V, high rough): scale={lut[0, -1, 0]:.4f}  bias={lut[1, -1, 0]:.4f}  (expect both small)")
    print(f"  Bottom-right (high N·V, high rough): scale={lut[0, -1, -1]:.4f}  bias={lut[1, -1, -1]:.4f}")

    # Visualize: R = scale, G = bias, B = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viz = torch.zeros(3, args.size, args.size)
    viz[0] = lut[0]  # R channel = scale
    viz[1] = lut[1]  # G channel = bias
    save_image_linear(viz.clamp(0, 1), args.output_dir / "brdf_lut.png")
    print(f"\n  Saved visualization: {args.output_dir}/brdf_lut.png")

    # Save the actual tensor for use by the renderer
    args.save_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lut, args.save_pt)
    print(f"  Saved tensor: {args.save_pt}")

    print("\nWhat to look for:")
    print("  - The visualization should show a clear smooth gradient (no noise)")
    print("  - Top-left corner: dark/black (rough surfaces seen edge-on reflect little)")
    print("  - Top-right area: bright red/yellow (smooth surfaces seen head-on reflect strongly)")
    print("  - This LUT is universal — look up 'BRDF LUT' images online to compare")
    print("  - It should look similar across UE4/Unity/Filament/any PBR renderer")

    
    lut = torch.load('data/brdf_lut.pt')
    # size=128, so index 64 ≈ middle (value 0.5)
    print(f"NoV=0.5, rough=0.5:  scale={lut[0, 64, 64]:.3f}  bias={lut[1, 64, 64]:.3f}")


if __name__ == "__main__":
    main()