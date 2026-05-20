#!/usr/bin/env python3
"""Generate a submission CSV by running inference with trained U-Net checkpoints.

Usage examples:
    # Predict all classes (checkpoints must exist for each):
    python test.py --data-root ./adl-2025-2026-anomaly-detection-normal \\
                   --checkpoint-dir outputs/checkpoints

    # Predict only two classes:
    python test.py --data-root ./data --checkpoint-dir outputs/checkpoints \\
                   --classes class_01 class_03

    # Disable TTA for faster inference:
    python test.py --data-root ./data --checkpoint-dir outputs/checkpoints --no-tta
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from config import DEVICE, IMAGE_SIZE, PRED_BLUR_SIGMA, USE_TTA
from dataset import discover_classes, discover_test_samples, list_image_files
from model import build_model
from utils import (
    float_matrix_to_q8rle,
    predict_test_image,
    q8rle_to_float_matrix,
    seed_everything,
)


# ── Submission generation ───────────────────────────────────────────────────

def make_submission(classes, *, data_root, checkpoint_dir, out_path,
                    image_size=IMAGE_SIZE, device=DEVICE, use_tta=USE_TTA,
                    blur_sigma=PRED_BLUR_SIGMA):
    """Generate a submission CSV for the given classes.

    For each class, loads the corresponding checkpoint from *checkpoint_dir*
    and predicts an anomaly map for every test image.  If no checkpoint exists
    the class is filled with all-zero predictions.
    """
    rows = []
    for cls in classes:
        class_root = Path(data_root) / cls
        test_samples = discover_test_samples(class_root)

        ckpt_path = Path(checkpoint_dir) / f"{cls}.pt"
        if ckpt_path.exists():
            print(f"[{cls}] loading checkpoint {ckpt_path}")
            model = build_model(device=device)
            import torch
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
        else:
            print(f"[{cls}] no checkpoint found at {ckpt_path}, "
                  f"writing all-zero predictions for {len(test_samples)} images.")
            model = None

        for s in tqdm(test_samples, desc=f"infer {cls}", leave=False):
            sample_id = s["image_path"].stem
            if model is None:
                with Image.open(s["image_path"]) as im:
                    w, h = im.size
                amap = np.zeros((h, w), dtype=np.float32)
            else:
                amap = predict_test_image(model, s["image_path"],
                                          image_size=image_size, device=device,
                                          use_tta=use_tta, blur_sigma=blur_sigma)
            rows.append({"ID": sample_id, "Label": float_matrix_to_q8rle(amap)})

    df = pd.DataFrame(rows, columns=["ID", "Label"])
    df.to_csv(out_path, index=False)
    print(f"Wrote submission: {out_path}  ({len(df)} rows)")
    return df


# ── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check_submission(df, data_root, classes, n_per_class=2):
    """Decode a few submission rows and verify dimensions match the originals."""
    rng = np.random.default_rng(7)

    id_to_path = {}
    for cls in classes:
        class_root = Path(data_root) / cls
        for p in list_image_files(class_root / "test"):
            id_to_path[p.stem] = p

    for cls in classes:
        class_root = Path(data_root) / cls
        test_ids = [p.stem for p in list_image_files(class_root / "test")]
        if not test_ids:
            continue
        for sid in rng.choice(test_ids, size=min(n_per_class, len(test_ids)),
                              replace=False):
            row = df[df["ID"] == sid]
            if len(row) == 0:
                print(f"  [warn] {sid} missing from submission")
                continue
            amap = q8rle_to_float_matrix(row.iloc[0]["Label"])
            with Image.open(id_to_path[sid]) as im:
                ow, oh = im.size
            ok = (amap.shape == (oh, ow)) and (0.0 <= amap.min() <= amap.max() <= 1.0)
            print(f"[{cls}] {sid}: shape={amap.shape} expected=({oh},{ow}) "
                  f"min={amap.min():.3f} max={amap.max():.3f} ok={ok}")


# ── CLI entry point ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with trained U-Net models and produce a "
                    "submission CSV.",
    )
    p.add_argument("--data-root", type=str, required=True,
                   help="Path to the dataset root (contains class_XX folders).")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Directory containing per-class .pt checkpoints.")
    p.add_argument("--output", type=str, default="submission.csv",
                   help="Path for the output submission CSV (default: submission.csv).")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Class names to predict (e.g. class_01 class_03). "
                        "If omitted, all discovered classes are predicted.")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--no-tta", action="store_true",
                   help="Disable test-time augmentation.")
    p.add_argument("--blur-sigma", type=float, default=PRED_BLUR_SIGMA,
                   help="Gaussian blur sigma for post-processing (0 to disable).")
    p.add_argument("--sanity-check", action="store_true",
                   help="Run a sanity check on the produced submission.")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything()

    data_root = Path(args.data_root)
    assert data_root.exists(), f"Dataset folder not found: {data_root}"

    if args.classes:
        classes = args.classes
    else:
        classes = discover_classes(data_root)
    print(f"Classes to predict: {classes}")

    use_tta = not args.no_tta
    out_path = Path(args.output)

    df = make_submission(
        classes,
        data_root=data_root,
        checkpoint_dir=args.checkpoint_dir,
        out_path=out_path,
        image_size=args.image_size,
        device=args.device,
        use_tta=use_tta,
        blur_sigma=args.blur_sigma,
    )

    if args.sanity_check:
        print("\n--- Sanity check ---")
        sanity_check_submission(df, data_root, classes)

    print("\nDone.")


if __name__ == "__main__":
    main()
