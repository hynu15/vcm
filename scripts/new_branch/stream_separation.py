"""Khối S2 (paper §III-C, Hình 4): binary masking + macroblock 16×16 + Hadamard split.

Quy ước:
- `seg_mask`: int (H,W) ∈ {0,1,2,3}, **0 = ROI**, 1/2/3 = non-ROI.
- Bước 1: gộp non-ROI {1,2,3} → 0, ROI → 1 ⇒ binary mask `M_i`.
- Bước 2: upsample về full-resolution (1024×2048) bằng nearest-neighbor.
- Bước 3: macroblock filter 16×16 — block nào có ≥1 pixel ROI → toàn block thành ROI.
- Bước 4: M_n = 1 − M_i; S_i = M_i ⊙ X; S_n = M_n ⊙ X.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .paths import FRAME_HW, NON_ROI_CLASS_IDS

MB_SIZE = 16


def seg_to_binary_roi(seg_mask: np.ndarray) -> np.ndarray:
    """Map seg id → binary ROI mask. Trả về uint8 (H,W), 1 = ROI."""
    non_roi = np.zeros_like(seg_mask, dtype=bool)
    for c in NON_ROI_CLASS_IDS:
        non_roi |= (seg_mask == c)
    return (~non_roi).astype(np.uint8)


def upsample_mask(mask: np.ndarray, target_hw: tuple[int, int] = FRAME_HW) -> np.ndarray:
    """Nearest-neighbor upsample mask {0,1} lên target_hw."""
    H, W = target_hw
    t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(H, W), mode="nearest")
    return up.squeeze().numpy().astype(np.uint8)


def macroblock_filter(mask: np.ndarray, block: int = MB_SIZE) -> np.ndarray:
    """Mọi block block×block có ≥1 pixel ROI → toàn block thành 1.

    Cài bằng max-pool kernel=block, stride=block, sau đó upsample nearest.
    Cần H, W chia hết cho `block`; nếu không, pad zero rồi crop trở lại.
    """
    H, W = mask.shape
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)))
    t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(t, kernel_size=block, stride=block)
    up = F.interpolate(pooled, scale_factor=block, mode="nearest")
    out = up.squeeze().numpy().astype(np.uint8)
    return out[:H, :W]


def split_streams(frame: np.ndarray, roi_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hadamard: S_i = M_i ⊙ X; S_n = (1 - M_i) ⊙ X.

    `frame`: uint8 (H,W,3) hoặc (H,W). `roi_mask`: uint8 (H,W) ∈ {0,1}.
    """
    if frame.ndim == 3:
        Mi = roi_mask[..., None]
    else:
        Mi = roi_mask
    Si = (frame * Mi).astype(np.uint8)
    Sn = (frame * (1 - Mi)).astype(np.uint8)
    return Si, Sn


def build_roi_mask(seg_mask_lowres: np.ndarray, target_hw: tuple[int, int] = FRAME_HW,
                   block: int = MB_SIZE) -> np.ndarray:
    """Pipeline đầy đủ S2: seg 4-class lowres → ROI mask MB-aligned ở full-res."""
    binary = seg_to_binary_roi(seg_mask_lowres)
    up = upsample_mask(binary, target_hw)
    return macroblock_filter(up, block)


# ── RA-CRF helpers (Việc 1 + 2) ──────────────────────────────────────────────

def gop_roi_ratio(masks: list[np.ndarray]) -> float:
    """Việc 1: tỷ lệ pixel ROI trung bình trên một GOP.

    Mỗi mask là uint8 (H,W) ∈ {0,1}; mean() = tỷ lệ pixel ROI.
    Trả về giá trị trong [0, 1].
    """
    return float(np.mean([m.mean() for m in masks]))


def select_delta_crf(roi_ratio: float) -> int:
    """Việc 2: rule rời rạc chọn ΔCRF từ roi_ratio của GOP.

    roi_ratio < 0.25  → ΔCRF = 5  (ROI nhỏ, ưu tiên mạnh vùng quan trọng)
    0.25 ≤ roi_ratio ≤ 0.60 → ΔCRF = 3  (ROI trung bình, cân bằng)
    roi_ratio > 0.60  → ΔCRF = 2  (ROI lớn, giảm chênh lệch tránh tăng bitrate)
    """
    if roi_ratio < 0.25:
        return 5
    elif roi_ratio <= 0.60:
        return 3
    else:
        return 2
