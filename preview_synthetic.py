#!/usr/bin/env python3
"""Preview synthetic anomaly generation before training.

For each requested class, this script:
1.  Builds the patch bank from real anomaly images.
2.  Generates N synthetic samples by pasting augmented patches onto random
    normal images.
3.  Saves each result as a 4-panel grid:
        original image | foreground mask | synthetic image | anomaly mask

All outputs go to ``<output-dir>/synthetic_preview/<class>/``.

Usage examples:
    # Preview 20 synthetic samples per class (all classes):
    python preview_synthetic.py --data-root ./data --fg-mask-dir ../bg_cache

    # Preview specific classes, 50 samples each:
    python preview_synthetic.py --data-root ./data --fg-mask-dir ../bg_cache \\
                                --classes class_01 class_03 --num-samples 50
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import IMAGE_SIZE, SEED
from dataset import build_class_splits, discover_classes
from synthetic import (
    augment_patch,
    extract_patch_bank,
    load_fg_mask,
    paste_synthetic_anomaly,
    resolve_fg_mask_path,
)
from utils import seed_everything


def _save_grid(original, fg_mask, synthetic, anomaly_mask, out_path):
    """Save a 4-panel grid to disk.

    Panels: original image | foreground mask | synthetic image | anomaly mask.
    Masks use a fixed [0, 1] colour range for global consistency.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(np.asarray(original))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(fg_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Foreground mask")
    axes[1].axis("off")

    axes[2].imshow(np.asarray(synthetic))
    axes[2].set_title("Synthetic")
    axes[2].axis("off")

    mask_np = np.asarray(anomaly_mask, dtype=np.float32) / 255.0
    axes[3].imshow(mask_np, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Anomaly mask")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def preview_class(class_name, *, data_root, fg_mask_dir, output_dir,
                  image_size=IMAGE_SIZE, num_samples=20):
    """Generate and save synthetic preview grids for one class."""
    class_root = Path(data_root) / class_name
    patch_bank = extract_patch_bank(class_root, image_size)
    if not patch_bank:
        print(f"[{class_name}] no patches extracted — skipping.")
        return

    splits = build_class_splits(class_name, root=data_root)
    normals = splits["train_normals"] + splits["val_normals"]
    if not normals:
        print(f"[{class_name}] no normal images available — skipping.")
        return

    out_dir = Path(output_dir) / "synthetic_preview" / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{class_name}] patch bank: {len(patch_bank)} patches, "
          f"normals pool: {len(normals)}, generating {num_samples} previews")

    for i in range(num_samples):
        rec = random.choice(normals)

        img = Image.open(rec["image_path"]).convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        original = img.copy()

        fg_path = resolve_fg_mask_path(rec["image_path"], data_root, fg_mask_dir)
        if fg_path is None:
            print(f"  [warn] no foreground mask for {rec['image_path']}, skipping")
            continue
        fg_mask_np = load_fg_mask(fg_path, image_size)

        patch_img_np, patch_mask_np = random.choice(patch_bank)
        patch_img, patch_mask = augment_patch(patch_img_np, patch_mask_np)
        synthetic_img, anomaly_mask = paste_synthetic_anomaly(
            img, fg_mask_np, patch_img, patch_mask, image_size,
        )

        out_path = out_dir / f"sample_{i:04d}.png"
        _save_grid(original, fg_mask_np, synthetic_img, anomaly_mask, out_path)

    print(f"[{class_name}] saved {num_samples} previews to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Preview synthetic anomaly generation (no training).",
    )
    p.add_argument("--data-root", type=str, required=True,
                   help="Path to the dataset root (contains class_XX folders).")
    p.add_argument("--fg-mask-dir", type=str, default=None,
                   help="Path to foreground-mask folder (bg_cache). "
                        "Defaults to <data-root>/../bg_cache.")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Base output directory (default: outputs).")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Class names to preview. If omitted, all classes.")
    p.add_argument("--num-samples", type=int, default=20,
                   help="Number of synthetic samples to generate per class "
                        "(default: 20).")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                   help=f"Working resolution (default: {IMAGE_SIZE}).")
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"Random seed (default: {SEED}).")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    data_root = Path(args.data_root)
    assert data_root.exists(), f"Dataset folder not found: {data_root}"

    fg_mask_dir = None
    if args.fg_mask_dir is not None:
        fg_mask_dir = Path(args.fg_mask_dir)
    else:
        default_fg = data_root.parent / "bg_cache"
        if default_fg.exists():
            fg_mask_dir = default_fg
    assert fg_mask_dir is not None and fg_mask_dir.exists(), (
        f"Foreground mask directory not found. Pass --fg-mask-dir explicitly."
    )
    print(f"Foreground mask dir: {fg_mask_dir}")

    classes = args.classes or discover_classes(data_root)
    print(f"Classes to preview: {classes}")

    for cls in classes:
        print("=" * 60)
        preview_class(
            cls,
            data_root=data_root,
            fg_mask_dir=fg_mask_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
            num_samples=args.num_samples,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
