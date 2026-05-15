"""Train the SwitchLight refinement UNet."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from refine.dataset import SwitchLightDataset
from refine.losses import RefinementLoss
from refine.unet import RefinementUNet, count_parameters


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_buffers(buffers: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in buffers.items()}


def checkpoint_payload(
    *,
    model: RefinementUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    best_loss: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_loss": best_loss,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(rng: dict[str, Any]) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def save_checkpoint(
    checkpoint_dir: Path,
    payload: dict[str, Any],
    *,
    is_best: bool = False,
    keep_last: int = 3,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step = payload["step"]
    path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(payload, path)

    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)

    if is_best:
        torch.save(payload, checkpoint_dir / "best.pt")
    return path


def make_grad_scaler(device: torch.device):
    if device.type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML training config.")
    parser.add_argument("--data", default=None, help="Override dataset root.")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 5405)))

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})

    data_root = Path(args.data or data_cfg.get("root", "data/blender/dataset"))
    image_size = int(data_cfg.get("image_size", 384))
    batch_size = int(train_cfg.get("batch_size", 4))
    max_steps = int(args.max_steps or train_cfg.get("max_steps", 200))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 500))
    log_every = int(train_cfg.get("log_every", 10))
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints/b1"))
    log_path = Path(train_cfg.get("log_path", "outputs/logs/b1_train.jsonl"))

    device = choose_device()
    dataset = SwitchLightDataset(
        data_root,
        image_size=image_size,
        augment=bool(data_cfg.get("augment", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = RefinementUNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    criterion = RefinementLoss(
        l1_weight=float(loss_cfg.get("l1_weight", 1.0)),
        vgg_weight=float(loss_cfg.get("vgg_weight", 0.5)),
        vgg_pretrained=bool(loss_cfg.get("vgg_pretrained", True)),
    ).to(device)
    scaler = make_grad_scaler(device)

    start_step = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        if "rng" in checkpoint:
            restore_rng(checkpoint["rng"])

    print(f"device={device} bundles={len(dataset)} params={count_parameters(model):,}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    data_iter = cycle(loader)
    model.train()

    with open(log_path, "a", encoding="utf-8") as log_file:
        last_step_is_best = False
        for step in range(start_step + 1, max_steps + 1):
            rendered_input, gt_image, buffers = next(data_iter)
            rendered_input = rendered_input.to(device)
            gt_image = gt_image.to(device)
            buffers = move_buffers(buffers, device)

            model_input = torch.cat([rendered_input, buffers["albedo"], buffers["normal"]], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                residual = model(model_input)
                pred = (rendered_input + residual).clamp(0.0, 1.0)
                loss_output = criterion(pred, gt_image, buffers["mask"])

            if scaler is None:
                loss_output.total.backward()
                optimizer.step()
            else:
                scaler.scale(loss_output.total).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            loss_value = float(loss_output.total.detach().cpu())
            is_best = loss_value < best_loss
            if is_best:
                best_loss = loss_value
            last_step_is_best = is_best

            log_record = {
                "step": step,
                "loss": loss_value,
                "l1": float(loss_output.l1.cpu()),
                "perceptual": float(loss_output.perceptual.cpu()),
                "lr": scheduler.get_last_lr()[0],
            }
            log_file.write(json.dumps(log_record) + "\n")
            log_file.flush()

            if step % log_every == 0 or step == 1:
                print(
                    f"step={step} loss={loss_value:.6f} "
                    f"l1={log_record['l1']:.6f} perc={log_record['perceptual']:.6f}"
                )

            if step % checkpoint_every == 0:
                payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_loss=best_loss,
                    config=config,
                )
                path = save_checkpoint(checkpoint_dir, payload, is_best=is_best)
                print(f"saved checkpoint {path}")

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=max_steps,
            best_loss=best_loss,
            config=config,
        )
        path = save_checkpoint(checkpoint_dir, payload, is_best=last_step_is_best)
        print(f"saved final checkpoint {path}")


if __name__ == "__main__":
    main()
