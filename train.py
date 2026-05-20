#!/usr/bin/env python3
"""Train a per-class U-Net for pixel-level anomaly detection.

Usage examples:
    # Train all discovered classes (default dataset location):
    python train.py --data-root ./adl-2025-2026-anomaly-detection-normal

    # Train only two specific classes:
    python train.py --data-root ./data --classes class_01 class_03

    # Override hyperparameters:
    python train.py --data-root ./data --epochs 50 --lr 3e-4 --batch-size 32

    # Enable synthetic anomaly augmentation (auto-detects ../bg_cache):
    python train.py --data-root ./data --fg-mask-dir ../bg_cache

    # Adjust batch composition fractions:
    python train.py --data-root ./data --fg-mask-dir ../bg_cache \\
                    --real-anomaly-frac 0.3 --synthetic-anomaly-frac 0.2
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from config import (
    BATCH_SIZE,
    BCE_POS_WEIGHT,
    BCE_WEIGHT,
    DEVICE,
    DICE_WEIGHT,
    EPOCHS,
    IMAGE_SIZE,
    LR,
    MAX_NORMALS_PER_CLASS,
    REAL_ANOMALY_FRAC,
    SEED,
    SYNTHETIC_ANOMALY_FRAC,
    VAL_RATIO,
    WEIGHT_DECAY,
)
from dataset import build_class_splits, build_loaders, discover_classes
from model import SegLoss, build_model
from utils import safe_pixel_ap, seed_everything, show_class_predictions, show_train_samples


# ── Training & evaluation loops ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device=DEVICE):
    """Run one training epoch; return the mean loss."""
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss, parts = criterion(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss) * images.size(0)
        n += images.size(0)
        pbar.set_postfix(
            loss=f"{float(loss):.3f}",
            bce=f"{parts['bce']:.3f}",
            dice=f"{parts['dice']:.3f}",
        )
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    """Return validation pixel AP over all pixels of all val images."""
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    all_y, all_s = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].cpu().numpy()
        prob = torch.sigmoid(model(images)).cpu().numpy()
        all_y.append(masks.reshape(-1))
        all_s.append(prob.reshape(-1))
    y = np.concatenate(all_y)
    s = np.concatenate(all_s)
    return safe_pixel_ap(y, s)


# ── Per-class training ──────────────────────────────────────────────────────

def fit_class(class_name, *, data_root, checkpoint_dir, epochs=EPOCHS,
              lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
              image_size=IMAGE_SIZE, max_normals=MAX_NORMALS_PER_CLASS,
              val_ratio=VAL_RATIO, device=DEVICE, visualize=False,
              fg_mask_dir=None, real_anomaly_frac=REAL_ANOMALY_FRAC,
              synthetic_anomaly_frac=SYNTHETIC_ANOMALY_FRAC):
    """Train a U-Net for a single object class and save the best checkpoint.

    When *fg_mask_dir* is provided, a patch bank is extracted from the real
    anomalies and used to synthesise additional training examples on top of
    normal images.

    Returns (model, splits, history_list) or (None, None, None) when no
    training data is available.
    """
    seed_everything(SEED)
    splits = build_class_splits(class_name, root=data_root,
                                max_normals=max_normals, val_ratio=val_ratio)

    if len(splits["train_normals"]) + len(splits["train_anomalies"]) == 0:
        print(f"[{class_name}] no training data found — skipping.")
        return None, None, None

    # Build the synthetic patch bank when fg_mask_dir is available.
    patch_bank = None
    if fg_mask_dir is not None:
        from synthetic import extract_patch_bank
        class_root = Path(data_root) / class_name
        patch_bank = extract_patch_bank(class_root, image_size)
        print(f"[{class_name}] patch bank: {len(patch_bank)} patches extracted")

    print(
        f"[{class_name}] train: {len(splits['train_normals'])} normals + "
        f"{len(splits['train_anomalies'])} anomalies | "
        f"val: {len(splits['val_normals'])} normals + "
        f"{len(splits['val_anomalies'])} anomalies"
    )
    if patch_bank:
        print(
            f"[{class_name}] batch target: {real_anomaly_frac:.0%} real anomaly, "
            f"{synthetic_anomaly_frac:.0%} synthetic, "
            f"{1 - real_anomaly_frac - synthetic_anomaly_frac:.0%} normal"
        )

    if visualize:
        show_train_samples(splits, class_name, image_size=image_size, n_each=3)

    train_ds, val_ds, train_loader, val_loader = build_loaders(
        splits, batch_size=batch_size, image_size=image_size,
        patch_bank=patch_bank, fg_mask_dir=fg_mask_dir,
        data_root=data_root,
        real_anomaly_frac=real_anomaly_frac,
        synthetic_anomaly_frac=synthetic_anomaly_frac,
    )

    model = build_model(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SegLoss().to(device)

    history = []
    best_ap = -1.0
    ckpt_path = Path(checkpoint_dir) / f"{class_name}.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device=device)
        scheduler.step()
        val_ap = evaluate(model, val_loader, device=device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_pixel_ap": val_ap})
        print(f"  epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"| val_pixel_ap={val_ap:.4f}")
        if not math.isnan(val_ap) and val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), ckpt_path)

    if best_ap < 0:
        torch.save(model.state_dict(), ckpt_path)
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[{class_name}] best val pixel AP = {best_ap:.4f} -> {ckpt_path}")

    if visualize:
        show_class_predictions(class_name, model, splits,
                               image_size=image_size, device=device)

    return model, splits, history


# ── CLI entry point ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train per-class U-Net anomaly segmentation models.",
    )
    p.add_argument("--data-root", type=str, required=True,
                   help="Path to the dataset root (contains class_XX folders).")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Directory for checkpoints and logs (default: outputs).")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Class names to train (e.g. class_01 class_03). "
                        "If omitted, all discovered classes are trained.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--max-normals", type=int, default=MAX_NORMALS_PER_CLASS,
                   help="Max normal images per class for training.")
    p.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--visualize", action="store_true",
                   help="Show sample visualisations during training.")
    p.add_argument("--fg-mask-dir", type=str, default=None,
                   help="Path to foreground-mask folder (bg_cache).  Enables "
                        "synthetic anomaly generation.  Defaults to "
                        "<data-root>/../bg_cache when the folder exists.")
    p.add_argument("--real-anomaly-frac", type=float,
                   default=REAL_ANOMALY_FRAC,
                   help="Target fraction of real anomaly samples per batch "
                        f"(default: {REAL_ANOMALY_FRAC}).")
    p.add_argument("--synthetic-anomaly-frac", type=float,
                   default=SYNTHETIC_ANOMALY_FRAC,
                   help="Target fraction of synthetic anomaly samples per "
                        f"batch (default: {SYNTHETIC_ANOMALY_FRAC}).")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    assert data_root.exists(), f"Dataset folder not found: {data_root}"

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resolve foreground-mask directory (auto-detect sibling bg_cache).
    fg_mask_dir = None
    if args.fg_mask_dir is not None:
        fg_mask_dir = Path(args.fg_mask_dir)
    else:
        default_fg = data_root.parent / "bg_cache"
        if default_fg.exists():
            fg_mask_dir = default_fg
    if fg_mask_dir is not None:
        print(f"Foreground mask dir: {fg_mask_dir}")
    else:
        print("No foreground mask dir found — synthetic augmentation disabled.")

    if args.classes:
        classes = args.classes
    else:
        classes = discover_classes(data_root)
    print(f"Classes to train: {classes}")

    for cls in classes:
        print("=" * 80)
        print(f"Training class: {cls}")
        print("=" * 80)
        fit_class(
            cls,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            image_size=args.image_size,
            max_normals=args.max_normals,
            val_ratio=args.val_ratio,
            device=args.device,
            visualize=args.visualize,
            fg_mask_dir=fg_mask_dir,
            real_anomaly_frac=args.real_anomaly_frac,
            synthetic_anomaly_frac=args.synthetic_anomaly_frac,
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
