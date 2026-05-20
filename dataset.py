"""Dataset discovery, PyTorch Dataset, and DataLoader construction.

Handles the Spacepresso directory layout:
    class_XX/train/good/                    -> clean training images
    class_XX/train/anomaly_YY/              -> labelled anomalous examples
    class_XX/ground_truth_train/anomaly_YY/ -> binary masks for those examples
    class_XX/test/                           -> unlabeled leaderboard images
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import (
    ANOMALY_BATCH_FRACTION,
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_NORMALS_PER_CLASS,
    NUM_WORKERS,
    PIN_MEMORY,
    REAL_ANOMALY_FRAC,
    SEED,
    SYNTHETIC_ANOMALY_FRAC,
    VAL_RATIO,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ── Discovery helpers ────────────────────────────────────────────────────────

def list_image_files(folder):
    """Return a sorted list of image file paths inside *folder*."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def discover_classes(root):
    """Return sorted list of class directory names (e.g. ['class_01', ...])."""
    root = Path(root)
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("class_")
    )


def discover_train_samples(class_root):
    """Return list of dicts {image_path, mask_path|None, label, defect_type}.

    Normal samples are paired with a None mask (interpreted as all zeros).
    Anomaly samples are paired with their per-pixel ground-truth mask.
    """
    class_root = Path(class_root)
    samples = []

    good_dir = class_root / "train" / "good"
    for img_path in list_image_files(good_dir):
        samples.append({
            "image_path": img_path,
            "mask_path": None,
            "label": 0,
            "defect_type": "good",
        })

    train_dir = class_root / "train"
    gt_dir = class_root / "ground_truth_train"
    if train_dir.exists():
        for defect_dir in sorted(train_dir.iterdir()):
            if not defect_dir.is_dir() or defect_dir.name == "good":
                continue
            defect_type = defect_dir.name
            for img_path in list_image_files(defect_dir):
                mask_path = gt_dir / defect_type / img_path.name
                if not mask_path.exists():
                    alt = gt_dir / defect_type / f"{img_path.stem}_mask{img_path.suffix}"
                    if alt.exists():
                        mask_path = alt
                    else:
                        print(f"  [warn] missing mask for {img_path}")
                        continue
                samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": 1,
                    "defect_type": defect_type,
                })

    return samples


def discover_test_samples(class_root):
    """Return list of dicts for unlabeled leaderboard images."""
    class_root = Path(class_root)
    test_dir = class_root / "test"
    return [
        {"image_path": p, "mask_path": None, "label": -1, "defect_type": "unknown"}
        for p in list_image_files(test_dir)
    ]


def summarize_dataset(root):
    """Build a summary DataFrame of per-class sample counts."""
    import pandas as pd

    rows = []
    for cls in discover_classes(root):
        class_root = Path(root) / cls
        n_good = len(list_image_files(class_root / "train" / "good"))
        train_dir = class_root / "train"
        anomalies = []
        if train_dir.exists():
            for d in sorted(train_dir.iterdir()):
                if d.is_dir() and d.name != "good":
                    anomalies.append((d.name, len(list_image_files(d))))
        n_anom = sum(c for _, c in anomalies)
        n_test = len(list_image_files(class_root / "test"))
        rows.append({
            "class": cls,
            "train_good": n_good,
            "train_anomaly_imgs": n_anom,
            "anomaly_types": ", ".join(name for name, _ in anomalies) or "-",
            "test": n_test,
        })
    return pd.DataFrame(rows)


# ── Image pre-processing ────────────────────────────────────────────────────

def pil_to_chw_float(img):
    """Convert a PIL image to a (C, H, W) float32 tensor in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def normalize_imagenet(x):
    """Apply ImageNet channel-wise normalisation to a (C, H, W) tensor."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class SegAnomalyDataset(Dataset):
    """Image + binary anomaly mask for one object class.

    When *patch_bank*, *fg_mask_dir*, and *data_root* are provided and
    ``synthetic_prob > 0``, normal samples have a chance of receiving a
    synthetically pasted anomaly patch at each ``__getitem__`` call.
    """

    def __init__(self, samples, image_size=IMAGE_SIZE, train=False,
                 patch_bank=None, fg_mask_dir=None, data_root=None,
                 synthetic_prob=0.0):
        self.samples = list(samples)
        self.image_size = image_size
        self.train = train
        self.patch_bank = patch_bank or []
        self.fg_mask_dir = fg_mask_dir
        self.data_root = data_root
        self.synthetic_prob = synthetic_prob

    def __len__(self):
        return len(self.samples)

    def _load(self, record):
        img = Image.open(record["image_path"]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        if record["mask_path"] is None:
            mask = Image.new("L", (self.image_size, self.image_size), 0)
        else:
            mask = Image.open(record["mask_path"]).convert("L")
            mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        return img, mask

    def _augment(self, img, mask):
        if random.random() < 0.75:
            k = random.randint(0, 3)
            if k:
                img = img.rotate(90 * k, resample=Image.BILINEAR)
                mask = mask.rotate(90 * k, resample=Image.NEAREST)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            arr = np.asarray(img, dtype=np.float32) / 255.0
            brightness = 1.0 + random.uniform(-0.15, 0.15)
            contrast = 1.0 + random.uniform(-0.15, 0.15)
            arr = arr * brightness
            arr = (arr - 0.5) * contrast + 0.5
            arr = np.clip(arr, 0.0, 1.0)
            img = Image.fromarray((arr * 255).astype(np.uint8))
        return img, mask

    def _apply_synthetic(self, img, rec):
        """Try to paste a synthetic anomaly patch onto a normal image.

        Returns ``(img, mask, applied)`` where *applied* is True when a
        patch was successfully pasted.
        """
        from synthetic import (
            augment_patch,
            load_fg_mask,
            paste_synthetic_anomaly,
            resolve_fg_mask_path,
        )

        if not self.patch_bank or self.fg_mask_dir is None or self.data_root is None:
            return img, Image.new("L", (self.image_size, self.image_size), 0), False

        fg_path = resolve_fg_mask_path(
            rec["image_path"], self.data_root, self.fg_mask_dir,
        )
        if fg_path is None:
            return img, Image.new("L", (self.image_size, self.image_size), 0), False

        fg_mask_np = load_fg_mask(fg_path, self.image_size)
        patch_img_np, patch_mask_np = random.choice(self.patch_bank)
        patch_img, patch_mask = augment_patch(patch_img_np, patch_mask_np)
        img, mask = paste_synthetic_anomaly(
            img, fg_mask_np, patch_img, patch_mask, self.image_size,
        )
        return img, mask, True

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img, mask = self._load(rec)

        is_synthetic = False
        if self.train:
            # Synthetic anomaly injection for normal samples.
            if (rec["label"] == 0
                    and self.patch_bank
                    and random.random() < self.synthetic_prob):
                img, mask, is_synthetic = self._apply_synthetic(img, rec)

            img, mask = self._augment(img, mask)

        img_t = pil_to_chw_float(img)
        img_n = normalize_imagenet(img_t)
        mask_t = pil_to_chw_float(mask)
        mask_t = (mask_t > 0.5).float()

        label = 1 if is_synthetic else rec["label"]
        defect = "synthetic" if is_synthetic else rec["defect_type"]

        return {
            "image": img_n,
            "image_raw": img_t,
            "mask": mask_t,
            "label": torch.tensor(label, dtype=torch.long),
            "defect_type": defect,
            "path": str(rec["image_path"]),
        }


# ── Train / val split & DataLoader construction ─────────────────────────────

def build_class_splits(class_name, root, max_normals=MAX_NORMALS_PER_CLASS,
                       val_ratio=VAL_RATIO, seed=SEED):
    """Build train / val sample lists for a single class.

    Subsamples normals to *max_normals* and holds out *val_ratio* of normals
    plus ~20 % of anomaly samples for validation.
    """
    class_root = Path(root) / class_name
    samples = discover_train_samples(class_root)
    rng = np.random.default_rng(seed)

    normals = [s for s in samples if s["label"] == 0]
    anomalies = [s for s in samples if s["label"] == 1]

    rng.shuffle(normals)
    if max_normals is not None and len(normals) > max_normals:
        normals = normals[:max_normals]

    n_val_normals = max(1, int(round(len(normals) * val_ratio)))
    val_normals = normals[:n_val_normals]
    train_normals = normals[n_val_normals:]

    rng.shuffle(anomalies)
    if len(anomalies) >= 5:
        n_val_anom = max(1, len(anomalies) // 5)
    else:
        n_val_anom = 0
    val_anomalies = anomalies[:n_val_anom]
    train_anomalies = anomalies[n_val_anom:]

    return {
        "train_normals": train_normals,
        "train_anomalies": train_anomalies,
        "val_normals": val_normals,
        "val_anomalies": val_anomalies,
    }


def build_loaders(splits, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                  patch_bank=None, fg_mask_dir=None, data_root=None,
                  real_anomaly_frac=REAL_ANOMALY_FRAC,
                  synthetic_anomaly_frac=SYNTHETIC_ANOMALY_FRAC):
    """Build train and validation DataLoaders.

    Batch composition (when *patch_bank* is provided):
        - ``real_anomaly_frac`` of the batch are real anomaly samples.
        - ``synthetic_anomaly_frac`` are normal images with synthetic patches.
        - The remainder are pure normal images.

    When *patch_bank* is ``None`` or empty, falls back to the original
    two-class weighting using ``ANOMALY_BATCH_FRACTION``.
    """
    train_samples = splits["train_normals"] + splits["train_anomalies"]
    val_samples = splits["val_normals"] + splits["val_anomalies"]

    use_synthetic = bool(patch_bank) and fg_mask_dir is not None and synthetic_anomaly_frac > 0
    if use_synthetic:
        normal_frac = 1.0 - real_anomaly_frac
        synthetic_prob = synthetic_anomaly_frac / normal_frac if normal_frac > 0 else 0.0
    else:
        synthetic_prob = 0.0

    train_ds = SegAnomalyDataset(
        train_samples, image_size, train=True,
        patch_bank=patch_bank, fg_mask_dir=fg_mask_dir,
        data_root=data_root, synthetic_prob=synthetic_prob,
    )
    val_ds = SegAnomalyDataset(val_samples, image_size, train=False)

    n_norm = len(splits["train_normals"])
    n_anom = max(1, len(splits["train_anomalies"]))

    if use_synthetic:
        p_anom = real_anomaly_frac
    else:
        p_anom = ANOMALY_BATCH_FRACTION

    weights = []
    for s in train_samples:
        if s["label"] == 1:
            weights.append(p_anom / n_anom)
        else:
            weights.append((1.0 - p_anom) / max(1, n_norm))
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_samples),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return train_ds, val_ds, train_loader, val_loader
