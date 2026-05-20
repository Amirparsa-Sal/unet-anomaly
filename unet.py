# Generated from: unet.ipynb
# Converted at: 2026-05-20T16:53:02.928Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Mission Spacepresso — Test 1
# ## Classic supervised segmentation for pixel-level anomaly detection
# 
# This notebook implements the **classic segmentation-based** baseline: we treat the
# task as a per-pixel binary classification problem and train a U-Net to predict an
# anomaly probability map. We use:
# 
# - **All `train/good` images** as fully-normal supervision (mask = all zeros).
# - **All `train/anomaly_YY` images** as positive supervision, paired with the
#   ground-truth masks in `ground_truth_train/anomaly_YY/`.
# - A **U-Net** with an ImageNet-pretrained **ResNet18** encoder.
# - A combined **weighted BCE + Dice** loss to cope with the strong pixel imbalance
#   between defective and normal pixels.
# - A **WeightedRandomSampler** that oversamples the few real anomaly samples so
#   every batch contains both normal and defective images.
# - One model **per object class** (`class_01`, `class_02`, …). Each class has its
#   own normal appearance and its own defect catalogue, and per-class models are
#   the standard MVTec recipe.
# 
# The notebook structure follows the teaching-assistant lab
# `Laboratory_04_Anomaly_Detection_on_MVTec_AD_complete.ipynb`. We reuse the
# useful utilities (PIL → tensor conversion, `list_image_files`, visualization
# helpers, metric helpers) and skip the autoencoder / student-teacher /
# PatchCore / self-supervised parts because they are different anomaly-detection
# paradigms — at this stage we only implement the classic segmentation pipeline.


# Import cell — same `ensure_package` helper as the TA notebook so the
# notebook is self-contained on a fresh Colab runtime.
import importlib.util
import subprocess
import sys

def ensure_package(package_name, import_name=None):
    import_name = import_name or package_name.split(">=")[0].split("==")[0].replace("-", "_")
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

ensure_package("scikit-learn>=1.1.0", "sklearn")
ensure_package("segmentation-models-pytorch>=0.3.3", "segmentation_models_pytorch")

import os
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

import segmentation_models_pytorch as smp

# Optional Drive mount when running on Colab
try:
    from google.colab import drive  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False
print(f"Running in Colab: {IN_COLAB}")

# !gdown 1XFh0Ku0K0F1OjtGAsnoXcf6t0UU3wok2 -O "drive/MyDrive/adl-2025-2026-anomaly-detection-normal.zip"

# !unzip "drive/MyDrive/adl-2025-2026-anomaly-detection-normal.zip" -d "drive/MyDrive/adl-2025-2026-anomaly-detection-normal"

# !gdown 1Cn0c-lz_tv8MwYrC5UHG8nUVDUNwH5Qx -O "/content/drive/MyDrive/adl-2025-2026-anomaly-detection-clean.zip"

# !unzip -qq "drive/MyDrive/adl-2025-2026-anomaly-detection-clean.zip" -d "drive/MyDrive/adl-2025-2026-anomaly-detection-clean"

# Reproducibility & runtime resources (same recipe as the TA notebook)
SEED = 7

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms make some kernels slower but help reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 2 if os.name != "nt" else 0

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# ## Dataset location and global configuration
# 
# On Colab the dataset is expected to live under `MyDrive/adl-2025-2026-anomaly-detection/`.
# Locally the notebook will fall back to a sibling folder. Both layouts have the
# structure described in the project brief:
# 
# ```
# class_XX/train/good/                  -> clean training images
# class_XX/train/anomaly_YY/            -> labelled anomalous example(s)
# class_XX/ground_truth_train/anomaly_YY/ -> binary masks for those examples
# class_XX/test/                        -> unlabeled leaderboard images
# ```
# 
# Each sample is exported as **5 image files** sharing the same `sample_id`. We
# treat every view as an independent image at training and inference time.


# =========================
# User configuration
# =========================

# Where the dataset lives.
DATASET_NAME = "adl-2025-2026-anomaly-detection-normal"

if IN_COLAB:
    drive.mount("/content/drive", force_remount=False)
    DATA_ROOT = Path(f"/content/drive/MyDrive/{DATASET_NAME}")
else:
    # Local fallback: this notebook lives next to the dataset folder.
    DATA_ROOT = Path.cwd() / DATASET_NAME

!unzip "drive/MyDrive/adl-2025-2026-anomaly-detection-normal.zip" -d "drive/MyDrive/adl-2025-2026-anomaly-detection-normal"

assert DATA_ROOT.exists(), f"Dataset folder not found: {DATA_ROOT}"
print(f"DATA_ROOT = {DATA_ROOT}")

# Where to save model checkpoints and the submission CSV.
OUTPUTS_DIR = Path("/content/drive/MyDrive/outputs_test1")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

# Model / training hyperparameters
IMAGE_SIZE      = 256      # working resolution
BATCH_SIZE      = 16
EPOCHS          = 25
LR              = 1e-4
WEIGHT_DECAY    = 1e-5

# Validation split is taken from the *normal* train images. With only a handful
# of labelled anomalies we keep them all in the train set; for monitoring we
# also evaluate them as a (very small) anomaly val set.
VAL_RATIO       = 0.10

# Maximum number of normal images per class used for fitting (the dataset has
# 2k+ normals per class, which is more than we need to fit a U-Net). Using all
# of them only slows things down and skews the loss towards the normal class.
MAX_NORMALS_PER_CLASS = 800

# Oversampling: in each batch the *expected* fraction of pixels coming from
# real anomaly samples is roughly this number. With ~25 anomaly images per
# class against 800 normals, this prevents the model from collapsing to
# "predict 0 everywhere".
ANOMALY_BATCH_FRACTION = 0.5

# Loss weighting. `pos_weight` for BCE compensates the dominance of normal
# pixels even within an anomaly image (defects usually cover <5% of pixels).
BCE_POS_WEIGHT  = 50.0
DICE_WEIGHT     = 1.0
BCE_WEIGHT      = 1.0

# Final anomaly map smoothing (small Gaussian blur removes salt-and-pepper
# noise from the per-pixel sigmoid).
PRED_BLUR_SIGMA = 2.0

# Test-time augmentation: average of original + horizontal flip predictions.
USE_TTA = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ## Dataset discovery utilities
# 
# Helpers to enumerate classes, list image files, and produce a summary table.
# These are direct adaptations of the TA notebook's MVTec utilities to the
# *Spacepresso* directory layout.


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def discover_classes(root):
    root = Path(root)
    return sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("class_")])

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
                # Mask is expected at ground_truth_train/<defect>/<same_name>
                mask_path = gt_dir / defect_type / img_path.name
                if not mask_path.exists():
                    # Some MVTec-style sets append "_mask" to the filename
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
    return [{"image_path": p, "mask_path": None, "label": -1, "defect_type": "unknown"}
            for p in list_image_files(test_dir)]

def summarize_dataset(root):
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
            "class":      cls,
            "train_good": n_good,
            "train_anomaly_imgs": n_anom,
            "anomaly_types": ", ".join(name for name, _ in anomalies) or "-",
            "test":       n_test,
        })
    return pd.DataFrame(rows)

summary_df = summarize_dataset(DATA_ROOT)
display(summary_df)

CLASSES = discover_classes(DATA_ROOT)
print(f"Found {len(CLASSES)} classes: {CLASSES}")

# ## `Dataset` class
# 
# We build a `SegAnomalyDataset` that returns aligned image/mask tensors. Two
# design choices matter for a *classic segmentation* pipeline:
# 
# 1. **Mask alignment**: any geometric augmentation (flip, rotation) is applied
#    identically to the image and the mask.
# 2. **Normal-image masks** are tensors of zeros (no anomaly anywhere). This is
#    how the network learns the normal appearance from thousands of unlabeled
#    `train/good` images even when only a handful of anomaly examples exist.


def pil_to_chw_float(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

def normalize_imagenet(x):
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std

class SegAnomalyDataset(Dataset):
    """Image + binary anomaly mask for one object class."""

    def __init__(self, samples, image_size=IMAGE_SIZE, train=False):
        self.samples = list(samples)
        self.image_size = image_size
        self.train = train

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
        # Random 4-fold rotation
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
        # Photometric jitter — image only.
        if random.random() < 0.5:
            arr = np.asarray(img, dtype=np.float32) / 255.0
            brightness = 1.0 + random.uniform(-0.15, 0.15)
            contrast   = 1.0 + random.uniform(-0.15, 0.15)
            arr = arr * brightness
            arr = (arr - 0.5) * contrast + 0.5
            arr = np.clip(arr, 0.0, 1.0)
            img = Image.fromarray((arr * 255).astype(np.uint8))
        return img, mask

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img, mask = self._load(rec)
        if self.train:
            img, mask = self._augment(img, mask)

        img_t  = pil_to_chw_float(img)               # (3, H, W) in [0, 1]
        img_n  = normalize_imagenet(img_t)           # (3, H, W) ImageNet-normalized
        mask_t = pil_to_chw_float(mask)              # (1, H, W) in [0, 1]
        mask_t = (mask_t > 0.5).float()              # binarize

        return {
            "image": img_n,
            "image_raw": img_t,
            "mask": mask_t,
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "defect_type": rec["defect_type"],
            "path": str(rec["image_path"]),
        }

def build_class_splits(class_name, root=DATA_ROOT, max_normals=MAX_NORMALS_PER_CLASS, val_ratio=VAL_RATIO, seed=SEED):
    """Build train / val sample lists for a single class.

    - Subsample normals to `max_normals` to control training time.
    - Hold out `val_ratio` of normals + ~20% of anomaly samples for validation.
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
        "train_normals":   train_normals,
        "train_anomalies": train_anomalies,
        "val_normals":     val_normals,
        "val_anomalies":   val_anomalies,
    }

def build_loaders(splits, batch_size=BATCH_SIZE):
    """Train DataLoader uses a WeightedRandomSampler to balance normals and anomalies.
    Validation uses sequential ordering.
    """
    train_samples = splits["train_normals"] + splits["train_anomalies"]
    val_samples   = splits["val_normals"]   + splits["val_anomalies"]

    train_ds = SegAnomalyDataset(train_samples, IMAGE_SIZE, train=True)
    val_ds   = SegAnomalyDataset(val_samples,   IMAGE_SIZE, train=False)

    n_norm = len(splits["train_normals"])
    n_anom = max(1, len(splits["train_anomalies"]))
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

# ## Visualization helpers
# 
# Direct adaptations of the helpers in the TA notebook so we can sanity-check
# loaders and qualitatively inspect predictions.


def to_numpy_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(x, 0.0, 1.0)

def to_numpy_mask(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    return x.astype(np.float32)

def normalize_map(amap, eps=1e-8):
    amap = np.asarray(amap, dtype=np.float32)
    if amap.max() <= amap.min():
        return np.zeros_like(amap)
    return (amap - amap.min()) / (amap.max() - amap.min() + eps)

def heatmap_overlay(image, amap, alpha=0.45):
    image = to_numpy_image(image)
    heat = plt.cm.jet(normalize_map(amap))[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heat, 0.0, 1.0)

def show_train_samples(splits, class_name, n_each=3):
    rng = np.random.default_rng(SEED)
    norm_idx = rng.choice(len(splits["train_normals"]), size=min(n_each, len(splits["train_normals"])), replace=False)
    anom_idx = rng.choice(len(splits["train_anomalies"]), size=min(n_each, len(splits["train_anomalies"])), replace=False)

    fig, axes = plt.subplots(2, max(len(norm_idx), len(anom_idx)), figsize=(3.2 * max(len(norm_idx), len(anom_idx)), 6.4))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"{class_name}: normals (top) and labeled anomalies (bottom)")

    norm_ds = SegAnomalyDataset([splits["train_normals"][i] for i in norm_idx], IMAGE_SIZE, train=False)
    anom_ds = SegAnomalyDataset([splits["train_anomalies"][i] for i in anom_idx], IMAGE_SIZE, train=False)
    for col, sample in enumerate([norm_ds[i] for i in range(len(norm_idx))]):
        axes[0, col].imshow(to_numpy_image(sample["image_raw"]))
        axes[0, col].set_title("normal")
        axes[0, col].axis("off")
    for col, sample in enumerate([anom_ds[i] for i in range(len(anom_idx))]):
        ax = axes[1, col]
        ax.imshow(to_numpy_image(sample["image_raw"]))
        mask = to_numpy_mask(sample["mask"])
        if mask.max() > 0:
            ax.contour(mask, levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(sample["defect_type"])
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# ## Metrics
# 
# The leaderboard metric is **pixel-level Average Precision**. Helpers below match
# the TA notebook's metric utilities.


def safe_pixel_ap(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true.reshape(-1), scores.reshape(-1)))

# ## Model — U-Net with a pretrained ResNet18 encoder
# 
# We use `segmentation_models_pytorch` to assemble a U-Net with an
# ImageNet-pretrained ResNet18 encoder and 1 output channel (anomaly logit per
# pixel). This is the prototypical "classic segmentation" architecture.
# 
# *Why pretrained?* The brief explicitly allows pretrained models, and ImageNet
# features are crucial: with so few labeled anomalies, the encoder cannot be
# trained from scratch in a useful way.


def build_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,   # we apply sigmoid ourselves and use BCEWithLogits
    )
    return model.to(DEVICE)

_demo_model = build_model()
n_params = sum(p.numel() for p in _demo_model.parameters())
print(f"U-Net (ResNet18 encoder) — {n_params/1e6:.2f}M parameters")
del _demo_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ## Loss — weighted BCE + Dice
# 
# Defective pixels are extremely rare (often <2% of an image, and ~0% in normal
# images). A plain BCE collapses to the trivial "all zeros" solution. We combine:
# 
# - **`BCEWithLogitsLoss`** with a `pos_weight` to up-weight defective pixels.
# - **Soft Dice** to directly optimise the overlap of the predicted heatmap and
#   the ground-truth mask (insensitive to the number of negative pixels).


class SegLoss(nn.Module):
    def __init__(self, bce_pos_weight=BCE_POS_WEIGHT, bce_w=BCE_WEIGHT, dice_w=DICE_WEIGHT):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight]))
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, logits, target):
        # logits, target -> (B, 1, H, W)
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        bce_l = self.bce(logits, target)

        prob = torch.sigmoid(logits)
        # Per-image soft Dice; gracefully reduces to 0 loss when the target is
        # entirely empty (both prob and target sum to 0 -> dice = 1).
        dims = (1, 2, 3)
        inter = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * inter + 1.0) / (union + 1.0)
        dice_l = (1 - dice).mean()

        return self.bce_w * bce_l + self.dice_w * dice_l, {"bce": float(bce_l), "dice": float(dice_l)}

# ## Training & evaluation utilities


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks  = batch["mask"].to(DEVICE, non_blocking=True)
        logits = model(images)
        loss, parts = criterion(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss) * images.size(0)
        n += images.size(0)
        pbar.set_postfix(loss=f"{float(loss):.3f}", bce=f"{parts['bce']:.3f}", dice=f"{parts['dice']:.3f}")
    return total / max(1, n)

@torch.no_grad()
def evaluate(model, loader):
    """Return validation pixel AP (over all pixels of all val images)."""
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    all_y, all_s = [], []
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks  = batch["mask"].cpu().numpy()
        prob = torch.sigmoid(model(images)).cpu().numpy()
        all_y.append(masks.reshape(-1))
        all_s.append(prob.reshape(-1))
    y = np.concatenate(all_y)
    s = np.concatenate(all_s)
    return safe_pixel_ap(y, s)

# Small Gaussian blur applied to the predicted heatmap. This helps both
# pixel AP and the visual quality of the maps.
def _gaussian_kernel(sigma, channels=1, device="cpu", dtype=torch.float32):
    if sigma is None or sigma <= 0:
        return None
    ksize = int(max(3, 2 * round(4 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1
    coords = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d.expand(channels, 1, ksize, ksize).contiguous()

def gaussian_blur(x, sigma=PRED_BLUR_SIGMA):
    if sigma is None or sigma <= 0:
        return x
    k = _gaussian_kernel(sigma, channels=x.shape[1], device=x.device, dtype=x.dtype)
    pad = k.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, k, groups=x.shape[1])

@torch.no_grad()
def predict_anomaly_map(model, image_chw_normalized, use_tta=USE_TTA, blur_sigma=PRED_BLUR_SIGMA):
    """Return a (H, W) numpy array of anomaly probabilities in [0, 1].

    `image_chw_normalized` is a (3, H, W) ImageNet-normalized tensor on CPU/GPU.
    """
    model.eval()
    x = image_chw_normalized.unsqueeze(0).to(DEVICE)
    logits = model(x)
    prob = torch.sigmoid(logits)
    if use_tta:
        prob_flip = torch.sigmoid(model(torch.flip(x, dims=[3])))
        prob = 0.5 * (prob + torch.flip(prob_flip, dims=[3]))
    prob = gaussian_blur(prob, sigma=blur_sigma)
    return prob.squeeze(0).squeeze(0).clamp(0, 1).cpu().numpy()

# ## Per-class training loop
# 
# We fit one U-Net per object class. After each epoch we monitor the validation
# pixel AP and keep the best checkpoint. Each model is saved under
# `outputs_test1/checkpoints/<class>.pt`.


def fit_class(class_name, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY):
    seed_everything(SEED)
    splits = build_class_splits(class_name)

    if len(splits["train_normals"]) + len(splits["train_anomalies"]) == 0:
        print(f"[{class_name}] no training data found — skipping.")
        return None, None, None

    print(
        f"[{class_name}] train: {len(splits['train_normals'])} normals + "
        f"{len(splits['train_anomalies'])} anomalies | "
        f"val: {len(splits['val_normals'])} normals + {len(splits['val_anomalies'])} anomalies"
    )
    show_train_samples(splits, class_name, n_each=3)

    train_ds, val_ds, train_loader, val_loader = build_loaders(splits)

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SegLoss().to(DEVICE)

    history = []
    best_ap = -1.0
    ckpt_path = CHECKPOINT_DIR / f"{class_name}.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        val_ap = evaluate(model, val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pixel_ap": val_ap})
        print(f"  epoch {epoch:02d} | train_loss={train_loss:.4f} | val_pixel_ap={val_ap:.4f}")
        if not math.isnan(val_ap) and val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), ckpt_path)

    if best_ap < 0:  # no val anomalies — keep last weights
        torch.save(model.state_dict(), ckpt_path)
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"[{class_name}] best val pixel AP = {best_ap:.4f} -> {ckpt_path}")
    return model, splits, pd.DataFrame(history)

trained = {}
for cls in CLASSES:
    print("=" * 80)
    print(f"Training class: {cls}")
    print("=" * 80)
    model, splits, history = fit_class(cls)
    if model is not None:
        trained[cls] = {"model": model, "splits": splits, "history": history}

# ## Qualitative validation
# 
# For each class we plot a couple of validation predictions side-by-side with
# their ground-truth masks.


def show_class_predictions(class_name, info, n_normal=2, n_anom=2):
    model  = info["model"]
    splits = info["splits"]
    rng = np.random.default_rng(SEED + 1)

    val_norm = splits["val_normals"]
    val_anom = splits["val_anomalies"]
    norm_pick = rng.choice(len(val_norm), size=min(n_normal, len(val_norm)), replace=False) if val_norm else []
    anom_pick = rng.choice(len(val_anom), size=min(n_anom,    len(val_anom)),    replace=False) if val_anom else []

    rows = list(norm_pick) + list(anom_pick)
    if not rows:
        print(f"[{class_name}] no validation images to visualize.")
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(f"{class_name} — validation predictions (top: normals, bottom: anomalies)")

    samples = []
    samples += [val_norm[i] for i in norm_pick]
    samples += [val_anom[i] for i in anom_pick]
    val_ds = SegAnomalyDataset(samples, IMAGE_SIZE, train=False)

    for r, item in enumerate(val_ds):
        amap = predict_anomaly_map(model, item["image"])
        img  = to_numpy_image(item["image_raw"])
        mask = to_numpy_mask(item["mask"])
        axes[r, 0].imshow(img); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"{item['defect_type']}")
        axes[r, 1].imshow(mask, cmap="gray"); axes[r, 1].axis("off"); axes[r, 1].set_title("GT mask")
        im = axes[r, 2].imshow(amap, cmap="jet", vmin=0, vmax=1); axes[r, 2].axis("off")
        axes[r, 2].set_title("pred map")
        plt.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.04)
        axes[r, 3].imshow(heatmap_overlay(item["image_raw"], amap)); axes[r, 3].axis("off")
        axes[r, 3].set_title("overlay")
    plt.tight_layout()
    plt.show()

for cls, info in trained.items():
    show_class_predictions(cls, info)

# ## q8rle encoding (from the project brief)
# 
# Each prediction is a 2D array of float anomaly scores in `[0, 1]`. We encode
# it as a quantized 8-bit run-length string.


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)  # column-wise flattening

    if flat.size == 0:
        return f"q8rle {h} {w}"

    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]

    parts = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)

def q8rle_to_float_matrix(s: str) -> np.ndarray:
    t = s.split()
    h, w = int(t[1]), int(t[2])
    vals = np.array(list(map(int, t[3::2])), dtype=np.uint8)
    lens = np.array(list(map(int, t[4::2])), dtype=np.int64)
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat.astype(np.float32) / 255.0

# Self-test: encode/decode round-trip on a random matrix.
_rng = np.random.default_rng(0)
_x = (_rng.random((32, 24)) * 255).astype(np.uint8) / 255.0
_round = q8rle_to_float_matrix(float_matrix_to_q8rle(_x))
assert _round.shape == _x.shape
print("q8rle round-trip max abs diff:", float(np.max(np.abs(_round - _x))))

# ## Inference on the test set & submission
# 
# For every test image we:
# 
# 1. Load and resize to `IMAGE_SIZE`.
# 2. Predict the anomaly map at `IMAGE_SIZE × IMAGE_SIZE` using the per-class
#    model (with horizontal-flip TTA + a small Gaussian blur).
# 3. Resize the map back to the **original image resolution** before encoding
#    to q8rle. This avoids submitting downscaled predictions.
# 4. Encode with q8rle and write a row to `submission.csv`.
# 
# If a class has no trained model (e.g. its `train/` folder is missing in the
# current environment) we fall back to predicting an all-zero map so the
# submission file still covers every test image.


@torch.no_grad()
def predict_test_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
    x_raw = pil_to_chw_float(img_resized)
    x_norm = normalize_imagenet(x_raw)
    amap = predict_anomaly_map(model, x_norm)  # (H, W) in [0, 1]

    # Resize the score map back to the original resolution.
    amap_t = torch.from_numpy(amap).unsqueeze(0).unsqueeze(0)
    amap_t = F.interpolate(amap_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return amap_t.squeeze(0).squeeze(0).clamp(0, 1).numpy()

def make_submission(trained, root=DATA_ROOT, out_path=SUBMISSION_PATH):
    rows = []
    for cls in discover_classes(root):
        class_root = Path(root) / cls
        test_samples = discover_test_samples(class_root)
        info = trained.get(cls)
        if info is None:
            print(f"[{cls}] no trained model, writing all-zero predictions for {len(test_samples)} images.")
        else:
            print(f"[{cls}] predicting {len(test_samples)} test images.")
        model = info["model"] if info is not None else None

        for s in tqdm(test_samples, desc=f"infer {cls}", leave=False):
            sample_id = s["image_path"].stem  # e.g. img_xxxx_view1
            if model is None:
                with Image.open(s["image_path"]) as im:
                    w, h = im.size
                amap = np.zeros((h, w), dtype=np.float32)
            else:
                amap = predict_test_image(model, s["image_path"])
            rows.append({"ID": sample_id, "Label": float_matrix_to_q8rle(amap)})

    df = pd.DataFrame(rows, columns=["ID", "Label"])
    df.to_csv(out_path, index=False)
    print(f"Wrote submission: {out_path}  ({len(df)} rows)")
    return df

submission_df = make_submission(trained)
display(submission_df.head())

# ## Sanity check on the submission
# 
# We decode a few rows back to a 2D float matrix, verify the dimensions match
# the original test image, and sanity-check that the score range is `[0, 1]`.


def sanity_check_submission(df, root=DATA_ROOT, n_per_class=2):
    by_class = {}
    for cls in discover_classes(root):
        class_root = Path(root) / cls
        test_files = list_image_files(class_root / "test")
        by_class[cls] = {p.stem: p for p in test_files}

    id_to_path = {}
    for cls, mapping in by_class.items():
        for stem, p in mapping.items():
            id_to_path[stem] = p

    rng = np.random.default_rng(SEED)
    sampled_rows = []
    for cls, mapping in by_class.items():
        ids = list(mapping)
        if not ids:
            continue
        for sid in rng.choice(ids, size=min(n_per_class, len(ids)), replace=False):
            row = df[df["ID"] == sid]
            if len(row) == 0:
                print(f"  [warn] {sid} missing from submission")
                continue
            sampled_rows.append((cls, sid, row.iloc[0]["Label"]))

    for cls, sid, label in sampled_rows:
        amap = q8rle_to_float_matrix(label)
        with Image.open(id_to_path[sid]) as im:
            ow, oh = im.size
        ok = (amap.shape == (oh, ow)) and (0.0 <= amap.min() <= amap.max() <= 1.0)
        print(f"[{cls}] {sid}: shape={amap.shape} expected=({oh},{ow}) min={amap.min():.3f} max={amap.max():.3f} ok={ok}")

sanity_check_submission(submission_df)

# ## Notes & next steps
# 
# What this baseline does (and does not):
# 
# - **Pure supervised segmentation** — no autoencoder reconstruction error, no
#   feature memory bank, no synthetic anomaly generation. The signal comes
#   entirely from the `train/good` images and the few labelled `anomaly_YY`
#   examples per class.
# - **One model per class** with a U-Net + ResNet18 (ImageNet) encoder.
# - **Class-balanced sampling + weighted BCE + Dice** to handle the strong
#   imbalance between normal and defective pixels.
# - **Multi-view handling**: each of the 5 views is predicted independently.
#   Aggregating across views (e.g. averaging predictions of the same `sample_id`
#   in different views) is *not* used here because the leaderboard is scored
#   per view.
# 
# Ideas to try next (kept out of scope on purpose):
# 
# - A **bigger backbone** (ResNet34/50, EfficientNet-B3) and/or higher input
#   resolution.
# - **DRAEM-style synthetic anomalies** to massively grow the positive supervision.
# - A **PatchCore / FastFlow** variant for comparison (these are different
#   paradigms and live in their own notebooks).
# - **Cross-view features** (sharing information across the 5 views of a sample).