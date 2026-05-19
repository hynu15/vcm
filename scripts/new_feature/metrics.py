"""
Metrics for SAC paper evaluation (Wang et al., 2023).

Traditional metrics
-------------------
  PSNR   : whole-frame Peak Signal-to-Noise Ratio (dB, ↑ better)
  SSIM   : whole-frame Structural Similarity Index (↑ better)

Semantic-Aware metrics  (Equations 11-14 in paper)
---------------------------------------------------
  r_roi    = C_roi / (C_roi + C_non)
  r_non    = C_non / (C_roi + C_non)   →  r_non > r_roi  (always, since C_non > C_roi)

  SA-SSIM  = r_non · S_i + r_roi · S_n   (Eq. 13)
  SA-PSNR  = r_non · P_i + r_roi · P_n   (Eq. 14)

  where P_i / S_i are PSNR / SSIM over the ROI region only
        P_n / S_n are PSNR / SSIM over the non-ROI region only

  Cross-weighting means the metric puts MORE weight on ROI quality
  (because C_non > C_roi → r_non > r_roi).

Segmentation metrics  (Equations 15-17 in paper)
--------------------------------------------------
  mIOU  : mean IoU over all 4 classes
  iIOU  : ROI-specific IoU (class 0 only)
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
from skimage.metrics import structural_similarity  as _sk_ssim


# ─────────────────────────────────────────────────────────────────────────────
# Whole-frame metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """PSNR (dB) over the whole frame.  Arrays must be uint8 or float [0,255]."""
    return float(_sk_psnr(orig, recon, data_range=255))


def compute_ssim(orig: np.ndarray, recon: np.ndarray) -> float:
    """SSIM over the whole frame."""
    ch_axis = 2 if orig.ndim == 3 else None
    return float(_sk_ssim(orig, recon, data_range=255, channel_axis=ch_axis))


# ─────────────────────────────────────────────────────────────────────────────
# Region-specific metrics  (used inside SA-PSNR / SA-SSIM)
# ─────────────────────────────────────────────────────────────────────────────

def _regional_psnr(orig: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> float:
    """
    PSNR computed only over the pixels where mask == True.
    mask : bool array [H, W]
    orig / recon : uint8 [H, W, 3] or [H, W]
    """
    n = int(mask.sum())
    if n == 0:
        return 0.0

    diff = orig.astype(np.float64) - recon.astype(np.float64)

    if orig.ndim == 3:
        # mask [H,W] indexes first two dims of [H,W,3] → returns [N,3]
        mse = float((diff[mask] ** 2).mean())
    else:
        mse = float((diff[mask] ** 2).mean())

    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


def _regional_ssim(orig: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> float:
    """
    SSIM computed on the bounding box of the masked region.
    Fallback to 1.0 when the crop is too small for the default 7×7 window.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return 1.0
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    oc = orig[r0:r1, c0:c1]
    rc = recon[r0:r1, c0:c1]
    h, w = oc.shape[:2]
    if h < 7 or w < 7:
        return 1.0
    ch_axis = 2 if oc.ndim == 3 else None
    return float(_sk_ssim(oc, rc, data_range=255, channel_axis=ch_axis))


# ─────────────────────────────────────────────────────────────────────────────
# Semantic-Aware metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_sa_psnr(
    orig: np.ndarray,
    recon: np.ndarray,
    roi_mask: np.ndarray,
    crf_roi: int,
    crf_non: int,
) -> float:
    """
    SA-PSNR = r_non · P_i + r_roi · P_n   (Eq. 14)

    Parameters
    ----------
    orig, recon : uint8 [H, W, 3]
    roi_mask    : bool [H, W]  — True where pixel belongs to ROI (class 0)
    crf_roi     : CRF used for the ROI stream
    crf_non     : CRF used for the non-ROI stream
    """
    r_roi = crf_roi / (crf_roi + crf_non)
    r_non = crf_non / (crf_roi + crf_non)

    p_i = _regional_psnr(orig, recon, roi_mask)          # ROI PSNR
    p_n = _regional_psnr(orig, recon, ~roi_mask)          # non-ROI PSNR

    return r_non * p_i + r_roi * p_n


def compute_sa_ssim(
    orig: np.ndarray,
    recon: np.ndarray,
    roi_mask: np.ndarray,
    crf_roi: int,
    crf_non: int,
) -> float:
    """
    SA-SSIM = r_non · S_i + r_roi · S_n   (Eq. 13)
    """
    r_roi = crf_roi / (crf_roi + crf_non)
    r_non = crf_non / (crf_roi + crf_non)

    s_i = _regional_ssim(orig, recon, roi_mask)
    s_n = _regional_ssim(orig, recon, ~roi_mask)

    return r_non * s_i + r_roi * s_n


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> float:
    """
    Mean IoU over all classes.  (Eq. 16)
    pred, gt : integer arrays with values in [0, num_classes)
    """
    ious = []
    for c in range(num_classes):
        inter = int(((pred == c) & (gt == c)).sum())
        union = int(((pred == c) | (gt == c)).sum())
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_iiou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    ROI-specific IoU  (iIOU, Eq. 17).
    Measures how well the ROI class (0) is segmented.
    """
    inter = int(((pred == 0) & (gt == 0)).sum())
    union = int(((pred == 0) | (gt == 0)).sum())
    return float(inter / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute full metrics bundle for one frame
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_compression_metrics(
    orig: np.ndarray,
    recon: np.ndarray,
    roi_mask: np.ndarray,
    crf_roi: int,
    crf_non: int,
) -> dict:
    """
    Compute PSNR, SSIM, SA-PSNR, SA-SSIM for a single (orig, recon) pair.

    Returns
    -------
    dict with keys: psnr, ssim, sa_psnr, sa_ssim
    """
    return {
        "psnr":    compute_psnr(orig, recon),
        "ssim":    compute_ssim(orig, recon),
        "sa_psnr": compute_sa_psnr(orig, recon, roi_mask, crf_roi, crf_non),
        "sa_ssim": compute_sa_ssim(orig, recon, roi_mask, crf_roi, crf_non),
    }
