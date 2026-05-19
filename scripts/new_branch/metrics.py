"""Độ đo paper §III-E:
- PSNR, SSIM toàn frame (sklearn).
- Regional PSNR/SSIM trên mask ROI / non-ROI.
- SA-PSNR  (Eq.14): r_non * P_i + r_roi * P_n.
- SA-SSIM  (Eq.13): r_non * S_i + r_roi * S_n.
- mIOU     (Eq.15-16): trung bình IoU trên 4 lớp.
- iIOU     (Eq.17): IoU chỉ trên lớp ROI.

**Chú ý quan trọng** (đã ghi trong CLAUDE.md §9.3): trọng số LỚN r_non gắn với
chất lượng của ROI (P_i, S_i) – không phải gắn với P_n. Đây là dụng ý của paper:
ROI quan trọng hơn nên cần đánh trọng số nhiều hơn.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


_PSNR_MAX = 100.0  # clamp khi MSE = 0 để tránh inf-poison trung bình


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if err == 0:
        return _PSNR_MAX
    return float(10.0 * np.log10((255.0 ** 2) / err))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        return float(_ssim(a, b, channel_axis=-1, data_range=255))
    return float(_ssim(a, b, data_range=255))


def regional_psnr_ssim(orig: np.ndarray, recon: np.ndarray, mask: np.ndarray
                       ) -> tuple[float, float]:
    """PSNR/SSIM chỉ trên vùng `mask == 1` (zero-out phần ngoài, giữ chung shape).

    Phù hợp với paper: tính metric trên ảnh đã được mask, vì pixel ngoài mask = 0
    ở cả gốc và tái tạo nên không ảnh hưởng kết quả.
    """
    m3 = mask[..., None] if orig.ndim == 3 else mask
    o = (orig * m3).astype(np.uint8)
    r = (recon * m3).astype(np.uint8)
    return psnr(o, r), ssim(o, r)


def compression_ratio_indices(crf_roi: int, crf_non: int) -> tuple[float, float]:
    """r_roi, r_non theo Eq.11-12. Chú ý: r_non > r_roi vì crf_non > crf_roi."""
    s = crf_roi + crf_non
    return crf_roi / s, crf_non / s


def sa_psnr_ssim(crf_roi: int, crf_non: int, P_i: float, P_n: float,
                 S_i: float, S_n: float) -> tuple[float, float]:
    """SA-PSNR và SA-SSIM theo Eq.13-14."""
    r_roi, r_non = compression_ratio_indices(crf_roi, crf_non)
    sa_p = r_non * P_i + r_roi * P_n
    sa_s = r_non * S_i + r_roi * S_n
    return sa_p, sa_s


def per_class_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4
                  ) -> tuple[float, list[float]]:
    """Trả về (mIoU, [iou_per_class]). Bỏ qua pixel `gt == 255`."""
    ious: list[float] = []
    valid = gt != 255
    for c in range(num_classes):
        p = (pred == c) & valid
        g = (gt == c) & valid
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        ious.append(float(inter) / float(union + 1e-9) if union > 0 else float("nan"))
    valid_ious = [x for x in ious if not np.isnan(x)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return miou, ious


def iiou(pred: np.ndarray, gt: np.ndarray, roi_class_id: int = 0) -> float:
    """IoU chỉ trên ROI: |GT_roi ∩ Pred_roi| / |GT_roi ∪ Pred_roi|."""
    valid = gt != 255
    p = (pred == roi_class_id) & valid
    g = (gt == roi_class_id) & valid
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter) / float(union + 1e-9) if union > 0 else 0.0
