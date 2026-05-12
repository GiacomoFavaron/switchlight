import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.visualize import save_grid


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * (x ** (1.0 / 2.4)) - 0.055,
    )


def tensor_to_rgb_image(tensor, linear=True):
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu()

    arr = tensor.numpy()

    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = np.clip(arr, 0.0, 1.0)

    if linear:
        arr = linear_to_srgb(arr)

    arr = (arr * 255.0).astype(np.uint8)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def normal_to_bgr(normal):
    if torch.is_tensor(normal):
        normal = normal.detach().cpu()

    arr = normal.numpy()

    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))

    arr = (arr + 1.0) * 0.5
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def scalar_to_bgr(tensor):
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu()

    arr = tensor.numpy()

    if arr.ndim == 3:
        arr = arr[0]

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)

    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

    if args.output is None:
        out_path = Path("outputs/inspect") / f"{bundle_path.stem}_grid.png"
    else:
        out_path = Path(args.output)

    items = []

    if "image" in bundle:
        items.append(("GT Image", tensor_to_rgb_image(bundle["image"], linear=True)))

    if "albedo" in bundle:
        items.append(("Albedo", tensor_to_rgb_image(bundle["albedo"], linear=True)))

    if "normal" in bundle:
        items.append(("Camera Normal", normal_to_bgr(bundle["normal"])))

    if "mask" in bundle:
        items.append(("Mask", scalar_to_bgr(bundle["mask"])))

    if "roughness" in bundle:
        items.append(("Roughness", scalar_to_bgr(bundle["roughness"])))

    if "specular" in bundle:
        items.append(("Specular", scalar_to_bgr(bundle["specular"])))

    save_grid(items, out_path, cols=3, cell_size=(320, 320))

    print(f"Saved inspection grid to: {out_path}")
    print("Bundle keys:", list(bundle.keys()))
    print("HDRI:", bundle.get("hdri_path"))


if __name__ == "__main__":
    main()