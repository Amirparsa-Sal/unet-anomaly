"""U-Net model construction and segmentation loss.

Uses `segmentation_models_pytorch` for a U-Net with an ImageNet-pretrained
ResNet18 encoder and 1 output channel (anomaly logit per pixel).
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import BCE_POS_WEIGHT, BCE_WEIGHT, DEVICE, DICE_WEIGHT


def build_model(device=DEVICE):
    """Build a U-Net with a pretrained ResNet18 encoder (1 output class)."""
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(device)


class SegLoss(nn.Module):
    """Combined weighted BCE + soft Dice loss for binary segmentation.

    The BCE pos_weight up-weights defective pixels to counter the strong
    imbalance (defects often cover < 5 % of an image).  Soft Dice directly
    optimises overlap and is insensitive to the number of negative pixels.
    """

    def __init__(self, bce_pos_weight=BCE_POS_WEIGHT, bce_w=BCE_WEIGHT,
                 dice_w=DICE_WEIGHT):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight]))
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, logits, target):
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        bce_l = self.bce(logits, target)

        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * inter + 1.0) / (union + 1.0)
        dice_l = (1 - dice).mean()

        return self.bce_w * bce_l + self.dice_w * dice_l, {
            "bce": float(bce_l),
            "dice": float(dice_l),
        }
