"""Shared utilities: reproducibility, metrics, q8rle encoding, prediction,
and visualisation helpers.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PRED_BLUR_SIGMA,
    SEED,
    USE_TTA,
)
from dataset import SegAnomalyDataset, normalize_imagenet, pil_to_chw_float


# ── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Metrics ──────────────────────────────────────────────────────────────────

def safe_pixel_ap(y_true, scores):
    """Pixel-level Average Precision (= AUPRC); returns NaN when only one class is present."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true.reshape(-1), scores.reshape(-1)))


def safe_pixel_auroc(y_true, scores):
    """Pixel-level AUROC; returns NaN when only one class is present."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true.reshape(-1), scores.reshape(-1)))


# ── q8rle encoding (from the project brief) ─────────────────────────────────

def float_matrix_to_q8rle(x: np.ndarray) -> str:
    """Encode a 2-D float anomaly-score map as a q8rle string."""
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)

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
    """Decode a q8rle string back to a 2-D float anomaly-score map."""
    t = s.split()
    h, w = int(t[1]), int(t[2])
    vals = np.array(list(map(int, t[3::2])), dtype=np.uint8)
    lens = np.array(list(map(int, t[4::2])), dtype=np.int64)
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat.astype(np.float32) / 255.0


# ── Gaussian blur for post-processing predicted maps ────────────────────────

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
    """Apply a small Gaussian blur to a (B, C, H, W) tensor."""
    if sigma is None or sigma <= 0:
        return x
    k = _gaussian_kernel(sigma, channels=x.shape[1], device=x.device, dtype=x.dtype)
    pad = k.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, k, groups=x.shape[1])


# ── Prediction helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def predict_anomaly_map(model, image_chw_normalized, device=DEVICE,
                        use_tta=USE_TTA, blur_sigma=PRED_BLUR_SIGMA):
    """Return a (H, W) numpy array of anomaly probabilities in [0, 1].

    *image_chw_normalized* is a (3, H, W) ImageNet-normalised tensor.
    """
    model.eval()
    x = image_chw_normalized.unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)
    if use_tta:
        prob_flip = torch.sigmoid(model(torch.flip(x, dims=[3])))
        prob = 0.5 * (prob + torch.flip(prob_flip, dims=[3]))
    prob = gaussian_blur(prob, sigma=blur_sigma)
    return prob.squeeze(0).squeeze(0).clamp(0, 1).cpu().numpy()


@torch.no_grad()
def predict_test_image(model, image_path, image_size=IMAGE_SIZE, device=DEVICE,
                       use_tta=USE_TTA, blur_sigma=PRED_BLUR_SIGMA):
    """Predict the anomaly map for a single test image, resized back to original resolution."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((image_size, image_size), resample=Image.BILINEAR)
    x_raw = pil_to_chw_float(img_resized)
    x_norm = normalize_imagenet(x_raw)
    amap = predict_anomaly_map(model, x_norm, device=device,
                               use_tta=use_tta, blur_sigma=blur_sigma)

    amap_t = torch.from_numpy(amap).unsqueeze(0).unsqueeze(0)
    amap_t = F.interpolate(amap_t, size=(orig_h, orig_w), mode="bilinear",
                           align_corners=False)
    return amap_t.squeeze(0).squeeze(0).clamp(0, 1).numpy()


# ── Visualisation helpers ───────────────────────────────────────────────────

def to_numpy_image(x):
    """Convert a (C, H, W) tensor to an (H, W, C) numpy array in [0, 1]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(x, 0.0, 1.0)


def to_numpy_mask(x):
    """Convert a (1, H, W) tensor to an (H, W) float numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    return x.astype(np.float32)


def normalize_map(amap, eps=1e-8):
    """Min-max normalise an anomaly map to [0, 1]."""
    amap = np.asarray(amap, dtype=np.float32)
    if amap.max() <= amap.min():
        return np.zeros_like(amap)
    return (amap - amap.min()) / (amap.max() - amap.min() + eps)


def heatmap_overlay(image, amap, alpha=0.45):
    """Overlay a jet-coloured anomaly heatmap on top of an image."""
    image = to_numpy_image(image)
    heat = plt.cm.jet(normalize_map(amap))[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heat, 0.0, 1.0)


def show_train_samples(splits, class_name, image_size=IMAGE_SIZE, n_each=3):
    """Plot a few normal and anomalous training samples with mask contours."""
    rng = np.random.default_rng(SEED)
    norm_idx = rng.choice(
        len(splits["train_normals"]),
        size=min(n_each, len(splits["train_normals"])),
        replace=False,
    )
    anom_idx = rng.choice(
        len(splits["train_anomalies"]),
        size=min(n_each, len(splits["train_anomalies"])),
        replace=False,
    )

    n_cols = max(len(norm_idx), len(anom_idx))
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.4))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"{class_name}: normals (top) and labeled anomalies (bottom)")

    norm_ds = SegAnomalyDataset(
        [splits["train_normals"][i] for i in norm_idx], image_size, train=False
    )
    anom_ds = SegAnomalyDataset(
        [splits["train_anomalies"][i] for i in anom_idx], image_size, train=False
    )

    for col in range(len(norm_idx)):
        sample = norm_ds[col]
        axes[0, col].imshow(to_numpy_image(sample["image_raw"]))
        axes[0, col].set_title("normal")
        axes[0, col].axis("off")
    for col in range(len(anom_idx)):
        sample = anom_ds[col]
        ax = axes[1, col]
        ax.imshow(to_numpy_image(sample["image_raw"]))
        mask = to_numpy_mask(sample["mask"])
        if mask.max() > 0:
            ax.contour(mask, levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(sample["defect_type"])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_class_predictions(class_name, model, splits, image_size=IMAGE_SIZE,
                           device=DEVICE, n_normal=2, n_anom=2):
    """Plot validation predictions side-by-side with ground-truth masks."""
    rng = np.random.default_rng(SEED + 1)

    val_norm = splits["val_normals"]
    val_anom = splits["val_anomalies"]
    norm_pick = (
        rng.choice(len(val_norm), size=min(n_normal, len(val_norm)), replace=False)
        if val_norm else []
    )
    anom_pick = (
        rng.choice(len(val_anom), size=min(n_anom, len(val_anom)), replace=False)
        if val_anom else []
    )

    rows = list(norm_pick) + list(anom_pick)
    if not rows:
        print(f"[{class_name}] no validation images to visualize.")
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(
        f"{class_name} — validation predictions (top: normals, bottom: anomalies)"
    )

    samples = [val_norm[i] for i in norm_pick] + [val_anom[i] for i in anom_pick]
    val_ds = SegAnomalyDataset(samples, image_size, train=False)

    for r, item in enumerate(val_ds):
        amap = predict_anomaly_map(model, item["image"], device=device)
        img = to_numpy_image(item["image_raw"])
        mask = to_numpy_mask(item["mask"])
        axes[r, 0].imshow(img)
        axes[r, 0].axis("off")
        axes[r, 0].set_title(f"{item['defect_type']}")
        axes[r, 1].imshow(mask, cmap="gray")
        axes[r, 1].axis("off")
        axes[r, 1].set_title("GT mask")
        im = axes[r, 2].imshow(amap, cmap="jet", vmin=0, vmax=1)
        axes[r, 2].axis("off")
        axes[r, 2].set_title("pred map")
        plt.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.04)
        axes[r, 3].imshow(heatmap_overlay(item["image_raw"], amap))
        axes[r, 3].axis("off")
        axes[r, 3].set_title("overlay")
    plt.tight_layout()
    plt.show()


# ── Save validation predictions to disk ─────────────────────────────────────

def _save_single_grid(item, amap, out_path):
    """Save a 3-panel grid (image | GT mask | predicted mask) to *out_path*.

    Both masks use a fixed ``jet`` colourmap with ``vmin=0, vmax=1`` so the
    colour scale is globally consistent across all saved images.
    """
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    img = to_numpy_image(item["image_raw"])
    mask = to_numpy_mask(item["mask"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GT mask")
    axes[1].axis("off")

    im = axes[2].imshow(amap, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Predicted mask")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_val_predictions(class_name, model, splits, output_dir,
                         image_size=IMAGE_SIZE, device=DEVICE):
    """Save a grid image for every validation sample to disk.

    Normal and anomalous samples are stored separately:
        ``<output_dir>/segmentation/<class>/normal/``
        ``<output_dir>/segmentation/<class>/anomalous/``
    """
    from pathlib import Path

    base = Path(output_dir) / "segmentation" / class_name
    normal_dir = base / "normal"
    anomalous_dir = base / "anomalous"
    normal_dir.mkdir(parents=True, exist_ok=True)
    anomalous_dir.mkdir(parents=True, exist_ok=True)

    val_norm = splits["val_normals"]
    val_anom = splits["val_anomalies"]

    if not val_norm and not val_anom:
        print(f"[{class_name}] no validation images to save.")
        return

    for tag, samples, folder in [("normal", val_norm, normal_dir),
                                  ("anomalous", val_anom, anomalous_dir)]:
        if not samples:
            continue
        ds = SegAnomalyDataset(samples, image_size, train=False)
        for i in range(len(ds)):
            item = ds[i]
            amap = predict_anomaly_map(model, item["image"], device=device)
            stem = Path(item["path"]).stem
            out_path = folder / f"{stem}.png"
            _save_single_grid(item, amap, out_path)

    total = len(val_norm) + len(val_anom)
    print(f"[{class_name}] saved {total} validation grids to {base}")
