import argparse
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.visualize import save_grid, mask_to_rgb, normal_to_rgb


def fake_mask(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, w // 4: 3 * w // 4] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return np.clip(mask, 0, 1)


def fake_normal(image):
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)

    normal = np.stack([
        -nx * 0.35,
        -ny * 0.25,
        np.ones_like(nx),
    ], axis=-1)

    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    return normal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output", default="outputs/figures/demo_grid.png")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    background = cv2.imread(args.background)

    if image is None:
        raise RuntimeError(f"Could not read input: {args.input}")

    if background is None:
        raise RuntimeError(f"Could not read background: {args.background}")

    h, w = image.shape[:2]
    background = cv2.resize(background, (w, h))

    mask = fake_mask(image)
    normal = fake_normal(image)

    mask_3 = np.stack([mask, mask, mask], axis=-1)

    raw_composite = (
        image.astype(np.float32) * mask_3 +
        background.astype(np.float32) * (1.0 - mask_3)
    ).astype(np.uint8)

    # Simple placeholder relight for visualization
    relit = image.astype(np.float32) * 1.12
    relit = np.clip(relit, 0, 255).astype(np.uint8)

    relit_composite = (
        relit.astype(np.float32) * mask_3 +
        background.astype(np.float32) * (1.0 - mask_3)
    ).astype(np.uint8)

    save_grid(
        [
            ("Input", image),
            ("Target Background", background),
            ("Mask", mask_to_rgb(mask)),
            ("Normal Map", normal_to_rgb(normal)),
            ("Raw Composite", raw_composite),
            ("Relit Composite", relit_composite),
        ],
        args.output,
        cols=3,
    )

    print(f"Saved demo grid to: {args.output}")


if __name__ == "__main__":
    main()