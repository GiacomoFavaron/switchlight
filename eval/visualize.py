from pathlib import Path
import cv2
import numpy as np


def _to_uint8(img):
    img = np.asarray(img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _resize(img, size=(320, 320)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def make_grid(items, cols=4, cell_size=(320, 320), bg_value=30):
    """
    items: list of (label, image)
    returns BGR uint8 grid image
    """
    panels = []

    for label, img in items:
        img = _to_uint8(img)
        img = _resize(img, cell_size)
        img = _label(img, label)
        panels.append(img)

    if not panels:
        raise ValueError("No images provided for grid.")

    rows = int(np.ceil(len(panels) / cols))
    h, w = cell_size[1], cell_size[0]

    blank = np.full((h, w, 3), bg_value, dtype=np.uint8)

    while len(panels) < rows * cols:
        panels.append(blank.copy())

    row_imgs = []

    for r in range(rows):
        row = panels[r * cols:(r + 1) * cols]
        row_imgs.append(np.hstack(row))

    return np.vstack(row_imgs)


def save_grid(items, output_path, cols=4, cell_size=(320, 320)):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_grid(items, cols=cols, cell_size=cell_size)
    cv2.imwrite(str(output_path), grid)

    return grid


def normal_to_rgb(normal):
    """
    Converts normal map from [-1,1] to [0,255].
    Accepts HWC or CHW.
    """
    normal = np.asarray(normal)

    if normal.ndim == 3 and normal.shape[0] == 3:
        normal = np.transpose(normal, (1, 2, 0))

    rgb = (normal + 1.0) * 0.5
    rgb = np.clip(rgb, 0.0, 1.0)

    return (rgb * 255).astype(np.uint8)


def mask_to_rgb(mask):
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    mask = np.clip(mask, 0.0, 1.0)
    mask = (mask * 255).astype(np.uint8)

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def error_map(pred, target):
    pred = _to_float01(pred)
    target = _to_float01(target)

    err = np.abs(pred - target)
    err = np.mean(err, axis=-1)

    err = err / (err.max() + 1e-8)
    err = (err * 255).astype(np.uint8)

    return cv2.applyColorMap(err, cv2.COLORMAP_JET)


def _to_float01(img):
    img = np.asarray(img).astype(np.float32)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.max() > 1.5:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)