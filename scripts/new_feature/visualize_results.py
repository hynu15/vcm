"""
Visualize SAC compression results.

Reproduces the qualitative comparisons from the paper (Figs. 6-9):
  - Original frame + semantic segmentation mask
  - Non-ROI stream / ROI stream (separated)
  - Reconstructed frame: traditional H.264/H.265 vs SA-X264/SA-X265
  - Segmentation comparison on decompressed frames

Usage (from scripts/ dir, conda sac env):
  # Compare one frame with all 4 methods
  python new_feature/visualize_results.py

  # Specific frame index
  python new_feature/visualize_results.py --frame-idx 55

  # Custom CRF pair
  python new_feature/visualize_results.py --crf-roi 23 --crf-non 32 --codec libx265

Outputs saved to outputs/new_feature/visualization/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from new_feature.ccnet_4class import load_ccnet_4class
from new_feature.compression import (
    compress_sac,
    compress_traditional,
    macroblock_align_filter,
)
from new_feature.dataset import Cityscapes4Class
from new_feature.evaluate import _load_model, _segment, _get_roi_mask, SEG_SIZE  # noqa: E402
from new_feature.metrics import (
    compute_psnr,
    compute_ssim,
    compute_sa_psnr,
    compute_sa_ssim,
    compute_miou,
    compute_iiou,
)

PROJECT_ROOT = _HERE.parent.parent
IMG_VAL = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
LBL_VAL = PROJECT_ROOT / "data" / "gt_4class" / "val"
VIZ_OUT = PROJECT_ROOT / "outputs" / "new_feature" / "visualization"

# 4-class colour palette (BGR)
SEG_PALETTE = {
    0: (128, 64, 128),   # ROI (road purple) - Cityscapes road colour
    1: (70, 130, 180),   # sky (steel-blue)
    2: (70, 70, 70),     # construction (dark grey)
    3: (107, 142, 35),   # nature (olive-green)
}

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def seg_to_colour(seg: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Convert integer label map to colour image (BGR)."""
    H, W = seg.shape
    colour = np.zeros((H, W, 3), dtype=np.uint8)
    if num_classes == 4:
        for cls, bgr in SEG_PALETTE.items():
            colour[seg == cls] = bgr
    else:
        # Binary: 0=ROI (purple), 1=non-ROI (grey)
        colour[seg == 0] = SEG_PALETTE[0]
        colour[seg == 1] = SEG_PALETTE[2]
    return colour


def put_text(img: np.ndarray, text: str, psnr: float = None, sa_psnr: float = None) -> np.ndarray:
    """Overlay method name and metrics on image."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [text]
    if psnr is not None:
        lines.append(f"PSNR={psnr:.2f}dB")
    if sa_psnr is not None:
        lines.append(f"SA-PSNR={sa_psnr:.2f}dB")
    for i, line in enumerate(lines):
        y = 28 + i * 22
        cv2.putText(out, line, (8, y), font, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def visualize_frame(args: argparse.Namespace) -> None:
    VIZ_OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_model, model_src = _load_model(device)
    seg_model.eval()
    num_seg_classes = getattr(seg_model, "num_classes", 2)

    dataset = Cityscapes4Class(IMG_VAL, LBL_VAL, augment=False)
    if args.frame_idx >= len(dataset):
        raise ValueError(f"frame_idx={args.frame_idx} exceeds dataset size {len(dataset)}")

    img_t, lbl_gt, img_path = dataset[args.frame_idx]
    orig_rgb = np.array(Image.open(img_path).convert("RGB"))
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    H, W = orig_rgb.shape[:2]

    # Ground-truth label
    lbl_np = cv2.resize(
        lbl_gt.numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
    )

    print(f"Frame: {img_path}")
    print(f"Resolution: {W}×{H}")

    # Segment original frame
    seg_pred = _segment(seg_model, device, orig_rgb)
    roi_mask = _get_roi_mask(seg_pred)  # uint8 0/1
    roi_bool = roi_mask.astype(bool)
    roi_pct = roi_mask.mean() * 100
    print(f"ROI: {roi_pct:.1f}%  non-ROI: {100-roi_pct:.1f}%")

    tmp = tempfile.mkdtemp()
    try:
        codec = args.codec
        crf_roi = args.crf_roi
        crf_non = args.crf_non
        crf_trad = args.crf_trad if args.crf_trad is not None else (crf_roi + crf_non) // 2

        # ── Encode ───────────────────────────────────────────────────────────
        dec_trad, _ = compress_traditional([orig_bgr], codec, crf_trad, tmp, "trad", preset=args.preset)
        dec_sac,  roi_b, non_b = compress_sac([orig_bgr], [roi_mask], codec, crf_roi, crf_non, tmp, "sac", preset=args.preset)

        dec_trad_rgb = cv2.cvtColor(dec_trad[0], cv2.COLOR_BGR2RGB)
        dec_sac_rgb  = cv2.cvtColor(dec_sac[0],  cv2.COLOR_BGR2RGB)

        # ── Metrics ──────────────────────────────────────────────────────────
        psnr_trad    = compute_psnr(orig_rgb, dec_trad_rgb)
        ssim_trad    = compute_ssim(orig_rgb, dec_trad_rgb)
        sa_psnr_trad = compute_sa_psnr(orig_rgb, dec_trad_rgb, roi_bool, crf_trad, crf_trad)
        sa_ssim_trad = compute_sa_ssim(orig_rgb, dec_trad_rgb, roi_bool, crf_trad, crf_trad)

        psnr_sac     = compute_psnr(orig_rgb, dec_sac_rgb)
        ssim_sac     = compute_ssim(orig_rgb, dec_sac_rgb)
        sa_psnr_sac  = compute_sa_psnr(orig_rgb, dec_sac_rgb, roi_bool, crf_roi, crf_non)
        sa_ssim_sac  = compute_sa_ssim(orig_rgb, dec_sac_rgb, roi_bool, crf_roi, crf_non)

        codec_short = "X264" if "264" in codec else "X265"

        print(f"\n{codec_short} (CRF={crf_trad},{crf_trad}):  PSNR={psnr_trad:.2f}  SSIM={ssim_trad:.4f}  SA-PSNR={sa_psnr_trad:.2f}  SA-SSIM={sa_ssim_trad:.4f}")
        print(f"SA-{codec_short} ({crf_roi},{crf_non}):  PSNR={psnr_sac:.2f}  SSIM={ssim_sac:.4f}  SA-PSNR={sa_psnr_sac:.2f}  SA-SSIM={sa_ssim_sac:.4f}")
        print(f"SA-PSNR improvement: {sa_psnr_sac - sa_psnr_trad:+.2f} dB")
        print(f"ROI size: {roi_b:,} bytes  non-ROI size: {non_b:,} bytes")

        # ── Segmentation on decompressed frames ───────────────────────────────
        seg_trad = _segment(seg_model, device, dec_trad_rgb)
        seg_sac  = _segment(seg_model, device, dec_sac_rgb)

        if num_seg_classes == 4:
            miou_trad = compute_miou(seg_trad, lbl_np, num_classes=4)
            iiou_trad = compute_iiou(seg_trad, lbl_np)
            miou_sac  = compute_miou(seg_sac,  lbl_np, num_classes=4)
            iiou_sac  = compute_iiou(seg_sac,  lbl_np)
        else:
            roi_gt_bin = (lbl_np == 0).astype(np.uint8)
            def _bin_iou(pred, gt):
                p = (pred == 0).astype(np.uint8)
                inter = int((p & gt).sum()); union = int((p | gt).sum())
                return inter/union if union > 0 else 0.0
            miou_trad = iiou_trad = _bin_iou(seg_trad, roi_gt_bin)
            miou_sac  = iiou_sac  = _bin_iou(seg_sac,  roi_gt_bin)

        print(f"\n{codec_short} seg: mIoU={miou_trad*100:.2f}%  iIoU={iiou_trad*100:.2f}%")
        print(f"SA-{codec_short} seg: mIoU={miou_sac*100:.2f}%  iIoU={iiou_sac*100:.2f}%")

        # ── Build composite image (matches Fig. 6 in paper) ──────────────────
        # Downsample for display if large
        disp_w = min(W, 800)
        disp_h = int(H * disp_w / W)
        rs = lambda img: cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        row1 = [rs(orig_bgr),                                   # original
                rs(seg_to_colour(seg_pred, num_seg_classes))]    # segmentation

        m255 = (roi_mask * 255).astype(np.uint8)
        roi_stream = cv2.bitwise_and(orig_bgr, orig_bgr, mask=m255)
        non_stream = cv2.bitwise_and(orig_bgr, orig_bgr, mask=255 - m255)
        row2 = [rs(non_stream), rs(roi_stream)]                  # streams

        dec_trad_bgr = cv2.cvtColor(dec_trad_rgb, cv2.COLOR_RGB2BGR)
        dec_sac_bgr  = cv2.cvtColor(dec_sac_rgb,  cv2.COLOR_RGB2BGR)
        row3 = [put_text(rs(dec_trad_bgr), f"H.{codec_short[1:]}(CRF={crf_trad})", psnr_trad, sa_psnr_trad),
                put_text(rs(dec_sac_bgr),  f"SA-{codec_short}({crf_roi},{crf_non})", psnr_sac, sa_psnr_sac)]

        seg_trad_bgr = seg_to_colour(seg_trad, num_seg_classes)
        seg_sac_bgr  = seg_to_colour(seg_sac,  num_seg_classes)
        row4 = [put_text(rs(seg_trad_bgr), f"Seg {codec_short}  mIoU={miou_trad*100:.1f}%"),
                put_text(rs(seg_sac_bgr),  f"Seg SA-{codec_short}  mIoU={miou_sac*100:.1f}%")]

        labels = [
            ["Original", "Segmentation"],
            ["non-ROI stream", "ROI stream"],
            [f"H.{codec_short[1:]}(CRF={crf_trad})", f"SA-{codec_short}({crf_roi},{crf_non})"],
            [f"Seg H.{codec_short[1:]}", f"Seg SA-{codec_short}"],
        ]

        for i, (row, lbls) in enumerate(zip([row1, row2, row3, row4], labels)):
            for j, (panel, lbl) in enumerate(zip(row, lbls)):
                cv2.rectangle(panel, (0, 0), (disp_w-1, disp_h-1), (200, 200, 200), 1)

        composite = np.vstack([np.hstack(row) for row in [row1, row2, row3, row4]])

        out_name = VIZ_OUT / f"frame{args.frame_idx:04d}_{codec_short}_{crf_roi}_{crf_non}.png"
        cv2.imwrite(str(out_name), composite)
        print(f"\nVisualization saved: {out_name}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize SAC compression (paper Figs. 6-9)")
    p.add_argument("--frame-idx", type=int, default=0,  help="Val set frame index")
    p.add_argument("--codec",     default="libx265",     choices=["libx264", "libx265"])
    p.add_argument("--crf-roi",   type=int, default=23,  help="ROI CRF (low = high quality)")
    p.add_argument("--crf-non",   type=int, default=32,  help="non-ROI CRF")
    p.add_argument("--crf-trad",  type=int, default=None, help="Traditional CRF (default: mean of roi+non)")
    p.add_argument("--preset",    default="fast",
                   choices=["ultrafast","superfast","veryfast","faster","fast","medium","slow"])
    return p.parse_args()


if __name__ == "__main__":
    visualize_frame(_parse())
