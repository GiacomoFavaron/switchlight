"""Dataset loader for SwitchLight synthetic buffer bundles."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


TENSOR_KEYS = ("image", "normal", "albedo", "roughness", "specular", "mask")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resize_chw(tensor: torch.Tensor, size: int, *, mode: str = "bilinear") -> torch.Tensor:
    if tensor.shape[-2:] == (size, size):
        return tensor.float()
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor.unsqueeze(0).float(), size=(size, size), **kwargs)
    return resized.squeeze(0)


class SwitchLightDataset(Dataset):
    """Loads Blender bundles and returns rendered input, GT image, and buffers."""

    def __init__(
        self,
        root: str | Path,
        *,
        image_size: int = 384,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.files = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt bundles found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        path = self.files[index]
        bundle = _torch_load(path)

        buffers: dict[str, Any] = {}
        for key in TENSOR_KEYS:
            if key not in bundle:
                raise KeyError(f"{path} is missing required key '{key}'")
            mode = "nearest" if key == "mask" else "bilinear"
            buffers[key] = _resize_chw(bundle[key], self.image_size, mode=mode)

        buffers["image"] = buffers["image"].clamp(0.0, 1.0)
        buffers["albedo"] = buffers["albedo"].clamp(0.0, 1.0)
        buffers["roughness"] = buffers["roughness"].clamp(0.05, 1.0)
        buffers["specular"] = buffers["specular"].clamp(0.0, 1.0)
        buffers["mask"] = buffers["mask"].clamp(0.0, 1.0)
        buffers["normal"] = F.normalize(buffers["normal"], dim=0, eps=1e-6)

        if "rendered_input" in bundle:
            rendered_input = _resize_chw(bundle["rendered_input"], self.image_size).clamp(0.0, 1.0)
        else:
            # B1 placeholder until Cook-Torrance pre-rendered inputs are generated.
            rendered_input = buffers["image"].clone()

        gt_image = buffers["image"]

        if self.augment and random.random() < 0.5:
            rendered_input = torch.flip(rendered_input, dims=(-1,))
            gt_image = torch.flip(gt_image, dims=(-1,))
            for key, value in list(buffers.items()):
                if torch.is_tensor(value):
                    buffers[key] = torch.flip(value, dims=(-1,))
            buffers["normal"][0] = -buffers["normal"][0]

        buffers["path"] = str(path)
        buffers["hdri_path"] = bundle.get("hdri_path")
        return rendered_input, gt_image, buffers
