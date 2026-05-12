import math
from typing import Dict, Optional

import cv2
import numpy as np


def _to_float01(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img).astype(np.float32)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.max() > 1.5:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def _prepare_mask(mask: Optional[np.ndarray], shape) -> np.ndarray:
    h, w = shape[:2]

    if mask is None:
        return np.ones((h, w), dtype=np.float32)

    mask = np.asarray(mask).astype(np.float32)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    if mask.max() > 1.5:
        mask = mask / 255.0

    return np.clip(mask, 0.0, 1.0)


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    pred = _to_float01(pred)
    target = _to_float01(target)
    mask = _prepare_mask(mask, pred.shape)

    diff = np.abs(pred - target)
    diff = diff * mask[..., None]

    denom = np.sum(mask) * 3.0 + 1e-8
    return float(np.sum(diff) / denom)


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    pred = _to_float01(pred)
    target = _to_float01(target)
    mask = _prepare_mask(mask, pred.shape)

    diff = (pred - target) ** 2
    diff = diff * mask[..., None]

    denom = np.sum(mask) * 3.0 + 1e-8
    return float(np.sum(diff) / denom)


def masked_psnr(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    mse = masked_mse(pred, target, mask)

    if mse <= 1e-12:
        return float("inf")

    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def masked_ssim_simple(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Lightweight SSIM approximation.
    """

    pred = _to_float01(pred)
    target = _to_float01(target)
    mask = _prepare_mask(mask, pred.shape)

    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = cv2.GaussianBlur(pred_gray, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(target_gray, (11, 11), 1.5)

    sigma_x = cv2.GaussianBlur(pred_gray * pred_gray, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(target_gray * target_gray, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(pred_gray * target_gray, (11, 11), 1.5) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) *
        (sigma_x + sigma_y + c2) + 1e-8
    )

    weighted = ssim_map * mask

    return float(np.sum(weighted) / (np.sum(mask) + 1e-8))


def evaluate_pair(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:

    return {
        "mae": masked_mae(pred, target, mask),
        "mse": masked_mse(pred, target, mask),
        "psnr": masked_psnr(pred, target, mask),
        "ssim": masked_ssim_simple(pred, target, mask),
    }