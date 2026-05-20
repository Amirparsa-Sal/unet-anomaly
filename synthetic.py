"""Synthetic anomaly generation via patch extraction and alpha-blended pasting.

Workflow (per class):
1.  Build a *patch bank* by extracting bounding-box crops around every
    connected component in the real anomaly masks.
2.  At training time, with a configurable probability, pick a random patch,
    augment it (geometry + photometric jitter), and alpha-blend it onto a
    normal image's foreground region.  The anomaly mask is updated to match.

The foreground mask (from the ``bg_cache`` folder) constrains where patches
can be placed so they only land on the object, not on the background.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from dataset import discover_train_samples, list_image_files

# Minimum patch side length (pixels) -- tiny components are noise, skip them.
_MIN_PATCH_SIDE = 8
# Alpha-feathering blur radius (pixels) applied to the binary patch mask
# before blending so edges transition smoothly.
_FEATHER_RADIUS = 3


# ── Patch bank construction ─────────────────────────────────────────────────

def extract_patch_bank(class_root, image_size):
    """Extract bounding-box patches from every anomaly connected component.

    Returns a list of ``(image_crop, mask_crop)`` tuples where both elements
    are uint8 numpy arrays (image is H x W x 3, mask is H x W).  Crops are
    stored at the *resized* resolution (``image_size x image_size``) so they
    are ready for pasting without an extra resize at training time.
    """
    class_root = Path(class_root)
    samples = discover_train_samples(class_root)
    anomalies = [s for s in samples if s["label"] == 1]

    bank = []
    for rec in anomalies:
        img = Image.open(rec["image_path"]).convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        mask = Image.open(rec["mask_path"]).convert("L")
        mask = mask.resize((image_size, image_size), resample=Image.NEAREST)

        img_np = np.asarray(img)
        mask_np = (np.asarray(mask) > 127).astype(np.uint8)

        labeled, n_components = ndimage.label(mask_np)
        if n_components == 0:
            continue

        slices = ndimage.find_objects(labeled)
        for i, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            crop_mask = (labeled[slc] == i).astype(np.uint8) * 255
            h, w = crop_mask.shape
            if h < _MIN_PATCH_SIDE or w < _MIN_PATCH_SIDE:
                continue
            crop_img = img_np[slc].copy()
            bank.append((crop_img, crop_mask))

    return bank


# ── Patch augmentation ──────────────────────────────────────────────────────

def augment_patch(image_crop, mask_crop):
    """Apply random geometric and photometric augmentation to a patch.

    *image_crop* is an (H, W, 3) uint8 array and *mask_crop* is an (H, W)
    uint8 array (0 or 255).  Returns augmented ``(PIL.Image, PIL.Image)``
    pair (RGB image, L mask).
    """
    img = Image.fromarray(image_crop, "RGB")
    mask = Image.fromarray(mask_crop, "L")

    # Random scale 0.5x -- 1.5x
    scale = random.uniform(0.5, 1.5)
    new_w = max(_MIN_PATCH_SIDE, int(img.width * scale))
    new_h = max(_MIN_PATCH_SIDE, int(img.height * scale))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    mask = mask.resize((new_w, new_h), resample=Image.NEAREST)

    # Random rotation 0 -- 360 degrees (expand=True keeps full content)
    angle = random.uniform(0, 360)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0))
    mask = mask.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)

    # Random flips
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    # Photometric jitter (image only)
    if random.random() < 0.5:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        brightness = 1.0 + random.uniform(-0.15, 0.15)
        contrast = 1.0 + random.uniform(-0.15, 0.15)
        arr = arr * brightness
        arr = (arr - 0.5) * contrast + 0.5
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    return img, mask


# ── Foreground-mask helpers ─────────────────────────────────────────────────

def resolve_fg_mask_path(image_path, data_root, fg_mask_dir):
    """Map an image path to the corresponding foreground mask in bg_cache.

    The bg_cache folder mirrors the dataset directory structure, so the mask
    lives at ``fg_mask_dir / image_path.relative_to(data_root)``.
    """
    image_path = Path(image_path)
    data_root = Path(data_root)
    fg_mask_dir = Path(fg_mask_dir)
    try:
        rel = image_path.relative_to(data_root)
    except ValueError:
        return None
    candidate = fg_mask_dir / rel
    return candidate if candidate.exists() else None


def load_fg_mask(fg_mask_path, image_size):
    """Load and resize a foreground mask to ``(image_size, image_size)``."""
    mask = Image.open(fg_mask_path).convert("L")
    mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
    return (np.asarray(mask) > 127).astype(np.uint8)


# ── Synthetic anomaly pasting ───────────────────────────────────────────────

def paste_synthetic_anomaly(image_pil, fg_mask_np, patch_img, patch_mask,
                            image_size):
    """Alpha-blend an augmented anomaly patch onto a normal image.

    Parameters
    ----------
    image_pil : PIL.Image
        Normal RGB image at ``image_size x image_size``.
    fg_mask_np : np.ndarray
        Binary foreground mask (H x W, uint8 0/1).
    patch_img : PIL.Image
        Augmented anomaly patch (RGB).
    patch_mask : PIL.Image
        Augmented anomaly mask (L, 0/255).
    image_size : int
        Working resolution.

    Returns
    -------
    (PIL.Image, PIL.Image)
        Augmented image (RGB) and the corresponding anomaly mask (L, 0/255).
    """
    pw, ph = patch_img.size

    # Clamp patch to at most 75 % of the image so it doesn't swamp everything.
    max_side = int(image_size * 0.75)
    if pw > max_side or ph > max_side:
        ratio = max_side / max(pw, ph)
        pw = max(_MIN_PATCH_SIDE, int(pw * ratio))
        ph = max(_MIN_PATCH_SIDE, int(ph * ratio))
        patch_img = patch_img.resize((pw, ph), resample=Image.BILINEAR)
        patch_mask = patch_mask.resize((pw, ph), resample=Image.NEAREST)

    # Find candidate paste locations: foreground pixels where the patch fits.
    ys, xs = np.where(fg_mask_np > 0)
    if len(ys) == 0:
        return image_pil, Image.new("L", (image_size, image_size), 0)

    # We need the top-left corner such that the patch centre is on foreground
    # and the patch stays within image bounds.
    half_h, half_w = ph // 2, pw // 2
    valid = (
        (ys - half_h >= 0) & (ys - half_h + ph <= image_size) &
        (xs - half_w >= 0) & (xs - half_w + pw <= image_size)
    )
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        # Fall back: pick any foreground pixel, clip the patch to fit.
        idx = random.randrange(len(ys))
        cy, cx = int(ys[idx]), int(xs[idx])
        top = max(0, min(cy - half_h, image_size - ph))
        left = max(0, min(cx - half_w, image_size - pw))
    else:
        idx = valid_idx[random.randrange(len(valid_idx))]
        top = int(ys[idx]) - half_h
        left = int(xs[idx]) - half_w

    # Build feathered alpha from the binary patch mask.
    alpha_pil = patch_mask.filter(ImageFilter.GaussianBlur(radius=_FEATHER_RADIUS))
    alpha_np = np.asarray(alpha_pil, dtype=np.float32) / 255.0

    # Alpha-blend patch onto image.
    img_np = np.array(image_pil, dtype=np.float32)
    patch_np = np.asarray(patch_img, dtype=np.float32)
    region = img_np[top:top + ph, left:left + pw]
    alpha_3 = alpha_np[..., None]
    blended = region * (1.0 - alpha_3) + patch_np * alpha_3
    img_np[top:top + ph, left:left + pw] = blended
    result_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), "RGB")

    # Build the anomaly mask (binary, no feathering -- we want a hard GT).
    result_mask_np = np.zeros((image_size, image_size), dtype=np.uint8)
    binary_patch = (np.asarray(patch_mask) > 127).astype(np.uint8) * 255
    result_mask_np[top:top + ph, left:left + pw] = binary_patch
    result_mask = Image.fromarray(result_mask_np, "L")

    return result_img, result_mask
