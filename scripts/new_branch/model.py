"""Wrapper cho CCNet và PIDNet-L dùng trong pipeline new_branch.

- `build_segnet`    → CCNet ResNet-101 + RCCA (paper gốc)
- `build_pidnet_l`  → PIDNet-L (Xu et al. CVPR 2023)

Cả hai đều expose cùng interface: forward(x) → logits (B,C,H,W).
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import CCNET_ROOT, NUM_CLASSES, RESNET101_PRETRAINED, SCRIPTS_ROOT

# ── CCNet ─────────────────────────────────────────────────────────────────────
if str(CCNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CCNET_ROOT))

from networks.ccnet import Seg_Model  # noqa: E402


def build_segnet(num_classes: int = NUM_CLASSES, recurrence: int = 2,
                 pretrained: bool = True, criterion=None):
    """CCNet(num_classes=4, recurrence=2) — model theo paper §III-B."""
    pretrained_path = str(RESNET101_PRETRAINED) if pretrained and RESNET101_PRETRAINED.exists() else None
    return Seg_Model(num_classes=num_classes, criterion=criterion,
                     pretrained_model=pretrained_path, recurrence=recurrence)


# ── PIDNet-L ──────────────────────────────────────────────────────────────────
class _PIDNetWrapper(nn.Module):
    """Bọc PIDNet để forward luôn trả về logits (B,C,H,W).

    PIDNet với augment=True trả về [p, x, d] khi training;
    output chính là x = out[1]. Wrapper này chuẩn hoá thành 1 tensor.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            return out[1]   # x_ = main segmentation head
        return out


def build_pidnet_l(num_classes: int = NUM_CLASSES) -> nn.Module:
    """PIDNet-L với 4 lớp đầu ra, bọc trong _PIDNetWrapper."""
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    from train_pidnet_l import build_pidnet_l as _build  # noqa: E402
    backbone = _build(num_classes=num_classes)
    return _PIDNetWrapper(backbone)


# ── Shared inference helper ────────────────────────────────────────────────────
@torch.no_grad()
def predict_segmask(model, image_tensor: torch.Tensor, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Inference – trả về argmax mask (B,H,W) ở kích thước out_hw (mặc định = input)."""
    model.eval()
    out = model(image_tensor)
    if isinstance(out, (list, tuple)):
        out = out[0]
    H, W = out_hw if out_hw is not None else (image_tensor.shape[-2], image_tensor.shape[-1])
    out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
    return out.argmax(dim=1)
