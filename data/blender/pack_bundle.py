import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path("/Users/hrithikg/SWRepos/switchlight")


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def read_rgb_png(path: Path, linearize: bool = True) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    if linearize:
        img = srgb_to_linear(img)

    return torch.from_numpy(img).permute(2, 0, 1).float()


def read_mask_png(path: Path) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = mask.astype(np.float32) / 255.0
    mask = np.clip(mask, 0.0, 1.0)

    return torch.from_numpy(mask).unsqueeze(0).float()


def read_camera_normal_png(path: Path) -> torch.Tensor:
    normal_rgb = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if normal_rgb is None:
        raise RuntimeError(f"Could not read normal: {path}")

    if normal_rgb.ndim == 3 and normal_rgb.shape[2] == 4:
        normal_rgb = cv2.cvtColor(normal_rgb, cv2.COLOR_BGRA2RGB)
    else:
        normal_rgb = cv2.cvtColor(normal_rgb, cv2.COLOR_BGR2RGB)

    normal_rgb = normal_rgb.astype(np.float32) / 255.0

    normal = normal_rgb * 2.0 - 1.0

    length = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.maximum(length, 1e-8)

    return torch.from_numpy(normal).permute(2, 0, 1).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/blender/smoke_test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--frame", default="0000")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir

    if args.output is None:
        out_path = input_dir / f"frame_{args.frame}.pt"
    else:
        out_path = ROOT / args.output

    meta_path = input_dir / "meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    image = read_rgb_png(input_dir / "beauty_0000.png", linearize=True)
    albedo = read_rgb_png(input_dir / "albedo_0000.png", linearize=True)
    normal = read_camera_normal_png(input_dir / "normal_camera_0000.png")
    mask = read_mask_png(input_dir / "mask_0000.png")

    h, w = mask.shape[1:]

    roughness = torch.full((1, h, w), 0.5, dtype=torch.float32)
    specular = torch.full((1, h, w), 0.04, dtype=torch.float32)

    hdri_path = meta.get("hdri_path")

    bundle = {
        "image": image,
        "normal": normal,
        "albedo": albedo,
        "roughness": roughness,
        "specular": specular,
        "mask": mask,
        "hdri_path": hdri_path,
        "meta": meta,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)

    print(f"Saved bundle to: {out_path}")

    for k, v in bundle.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype, float(v.min()), float(v.max()))
        else:
            print(k, type(v), v)


if __name__ == "__main__":
    main()