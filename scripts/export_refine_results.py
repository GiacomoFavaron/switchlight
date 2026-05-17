"""Export refinement comparison grids from a trained UNet checkpoint.

Each output grid has three columns:
    physics input | UNet output | Blender GT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from inverse.io_utils import linear_to_srgb
from refine.dataset import SwitchLightDataset
from refine.unet import RefinementUNet


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def to_pil(image: torch.Tensor) -> Image.Image:
    image = linear_to_srgb(image.detach().cpu().float().clamp(0.0, 1.0))
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(array)


def label(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 32), fill=(0, 0, 0))
    draw.text((10, 9), text, fill=(255, 255, 255))
    return image


def make_grid(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    labeled = [label(image, title) for title, image in panels]
    width, height = labeled[0].size
    grid = Image.new("RGB", (width * len(labeled), height), (0, 0, 0))
    for index, panel in enumerate(labeled):
        grid.paste(panel, (index * width, 0))
    return grid


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UNet refinement result grids.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/blender/dataset_augmented"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--frame",
        type=str,
        default=None,
        help="Optional frame stem to export, for example frame_0004. Exports all frames by default.",
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    dataset = SwitchLightDataset(args.data, image_size=args.image_size, augment=False)

    model = RefinementUNet().to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} bundles={len(dataset)} checkpoint={args.checkpoint}")

    with torch.no_grad():
        for index in range(len(dataset)):
            rendered_input, gt_image, buffers = dataset[index]
            frame_name = Path(buffers["path"]).stem
            if args.frame and frame_name != args.frame:
                continue

            rendered_batch = rendered_input.unsqueeze(0).to(device)
            albedo_batch = buffers["albedo"].unsqueeze(0).to(device)
            normal_batch = buffers["normal"].unsqueeze(0).to(device)

            model_input = torch.cat([rendered_batch, albedo_batch, normal_batch], dim=1)
            residual = model(model_input).squeeze(0)
            unet_output = (rendered_input.to(device) + residual).clamp(0.0, 1.0)

            physics_image = to_pil(rendered_input)
            unet_image = to_pil(unet_output)
            gt = to_pil(gt_image)

            physics_image.save(args.output_dir / f"{frame_name}_physics.png")
            unet_image.save(args.output_dir / f"{frame_name}_unet.png")
            gt.save(args.output_dir / f"{frame_name}_gt.png")

            grid = make_grid(
                [
                    ("Physics input", physics_image),
                    ("UNet output", unet_image),
                    ("Blender GT", gt),
                ]
            )
            grid_path = args.output_dir / f"{frame_name}_grid.png"
            grid.save(grid_path)
            print(f"saved {grid_path}")


if __name__ == "__main__":
    main()
