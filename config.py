"""Default hyperparameters and runtime configuration.

Every value here can be overridden by CLI arguments in train.py / test.py.
Importing this module also sets up DEVICE, PIN_MEMORY, and NUM_WORKERS so
that downstream code can use them immediately.
"""

import os
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 7

# ── Runtime resources ────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 2 if os.name != "nt" else 0

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME = "adl-2025-2026-anomaly-detection-normal"

# ── Training hyperparameters ─────────────────────────────────────────────────
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-5

# Fraction of normal images held out for validation.
VAL_RATIO = 0.10

# Cap normal images per class to control training time and class balance.
MAX_NORMALS_PER_CLASS = 800

# Target fraction of anomaly pixels per batch (via WeightedRandomSampler).
ANOMALY_BATCH_FRACTION = 0.5

# ── Loss ─────────────────────────────────────────────────────────────────────
BCE_POS_WEIGHT = 50.0
DICE_WEIGHT = 1.0
BCE_WEIGHT = 1.0

# ── Prediction post-processing ──────────────────────────────────────────────
PRED_BLUR_SIGMA = 2.0
USE_TTA = True

# ── ImageNet normalisation constants ────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
