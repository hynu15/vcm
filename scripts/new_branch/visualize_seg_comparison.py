"""Tạo ảnh so sánh segmentation giữa SAC và phương pháp truyền thống.

Layout 3×2 giống Fig.9 của paper:
  (a) Original Frame   | (b) Ground Truth
  (c) H.264 result     | (d) SA-X264 result
  (e) H.265 result     | (f) SA-X265 result

Hộp đỏ tự động được vẽ ở vùng SAC cải thiện nhiều nhất so với truyền thống.

Chạy:
  cd /home/huy/sac_project
  conda activate sac
  python -m scripts.new_branch.visualize_seg_comparison [--frames 210 273 251]
"""
from __future__ import annotations

if __package__ in (None, ""):
    import os, sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    __package__ = "scripts.new_branch"

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.amp import autocast

from .compression import compress_sac, compress_traditional
from .dataset import _normalize, _rgb_to_gray3
from .model import build_segnet, build_pidnet_l
from .paths import (CRF_CONFIGS, EVAL_DIR, FRAME_HW, IMG_ROOT, LBL_ROOT,
                    NUM_CLASSES, PIDNET_L_CHECKPOINT, ROI_CLASS_ID,
                    SAC_CHECKPOINT, SEG_INPUT_HW)
from .stream_separation import build_roi_mask

# ── Màu cho 4 lớp (BGR/RGB đều dùng RGB ở đây) ──────────────────────────────
# 0=ROI, 1=sky, 2=construction, 3=nature
SEG_PALETTE = np.array([
    [40,  40,  40],   # ROI        → gần đen
    [180, 210, 240],  # sky        → xanh nhạt
    [160, 140, 120],  # construction → nâu nhạt
    [ 80, 140,  80],  # nature     → xanh lá
], dtype=np.uint8)

LABEL_CAPTIONS = [
    "(a) Original Frame", "(b) Ground Truth",
    "(c) Result of H.264", "(d) Result of SA-X264",
    "(e) Result of H.265", "(f) Result of SA-X265",
]


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """(H,W) int → (H,W,3) uint8 colored."""
    rgb = SEG_PALETTE[seg.clip(0, NUM_CLASSES - 1)]
    return rgb


def overlay_seg(orig_rgb: np.ndarray, seg: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend segmentation màu lên ảnh gốc."""
    seg_rgb = seg_to_rgb(seg).astype(np.float32)
    orig_f = orig_rgb.astype(np.float32)
    blended = (alpha * seg_rgb + (1 - alpha) * orig_f).clip(0, 255).astype(np.uint8)
    return blended


@torch.no_grad()
def predict_seg(model, img_rgb_fullres: np.ndarray, device,
                gray_input: bool = True) -> np.ndarray:
    """img_rgb_fullres: (H,W,3) uint8 → (H,W) prediction full-res.

    gray_input=True cho CCNet (paper gốc dùng grayscale 3-ch),
    gray_input=False cho PIDNet (train với RGB ImageNet-normalized).
    """
    lr = np.array(Image.fromarray(img_rgb_fullres).resize(
        (SEG_INPUT_HW[1], SEG_INPUT_HW[0]), Image.BILINEAR))
    inp = _rgb_to_gray3(lr) if gray_input else lr
    t = _normalize(inp).unsqueeze(0).to(device)
    use_amp = device.type == "cuda"
    with autocast(device_type="cuda", enabled=use_amp):
        out = model(t)
        logits = out[0] if isinstance(out, (list, tuple)) else out
    logits = F.interpolate(logits.float(), size=SEG_INPUT_HW, mode="bilinear", align_corners=True)
    seg_lr = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    # Upsample về full-res
    seg_full = np.array(Image.fromarray(seg_lr).resize(
        (FRAME_HW[1], FRAME_HW[0]), Image.NEAREST))
    return seg_full


def find_best_diff_box(seg_trad: np.ndarray, seg_sac: np.ndarray,
                       block: int = 64) -> tuple[int, int, int, int]:
    """Tìm vùng block×block có nhiều pixel SAC đúng hơn truyền thống nhất.
    Trả về (x, y, w, h) pixel coords."""
    diff = (seg_trad != seg_sac).astype(np.float32)
    H, W = diff.shape
    best_score, best_yx = -1, (0, 0)
    for r in range(0, H - block, block // 2):
        for c in range(0, W - block, block // 2):
            s = diff[r:r+block, c:c+block].sum()
            if s > best_score:
                best_score, best_yx = s, (r, c)
    y, x = best_yx
    # Widen the box a bit for visibility
    pad = block // 4
    return (max(0, x - pad), max(0, y - pad),
            min(block + 2*pad, W - x), min(block + 2*pad, H - y))


def load_orig_and_gt(img_name: str, split: str = "test"):
    """Trả về (orig_fullres uint8, gt_4class uint8) full-res."""
    # img_name có dạng: strasbourg_000001_017675_leftImg8bit.png
    city = img_name.split("_")[0]
    img_path = IMG_ROOT / split / city / img_name
    lbl_name = img_name.replace("_leftImg8bit.png", "_gtFine_4class.png")
    lbl_path = LBL_ROOT / split / city / lbl_name

    orig = np.array(Image.open(img_path).convert("RGB"))
    if orig.shape[:2] != FRAME_HW:
        orig = np.array(Image.fromarray(orig).resize(
            (FRAME_HW[1], FRAME_HW[0]), Image.BILINEAR))
    gt = np.array(Image.open(lbl_path))
    if gt.shape[:2] != FRAME_HW:
        gt = np.array(Image.fromarray(gt).resize(
            (FRAME_HW[1], FRAME_HW[0]), Image.NEAREST))
    return orig, gt


def compress_and_decode(orig: np.ndarray, roi_mask: np.ndarray,
                        codec: str, crf_roi: int, crf_non: int) -> np.ndarray:
    """Nén + giải nén 1 frame, trả về frame tái tạo."""
    is_sac = (crf_roi != crf_non)
    if is_sac:
        recon_list, _, _ = compress_sac([orig], [roi_mask], codec, crf_roi, crf_non)
    else:
        recon_list, _ = compress_traditional([orig], codec, crf_roi)
    return recon_list[0]


def compute_iiou(seg: np.ndarray, gt: np.ndarray, roi_id: int = ROI_CLASS_ID) -> float:
    inter = np.logical_and(seg == roi_id, gt == roi_id).sum()
    union = np.logical_or(seg == roi_id, gt == roi_id).sum()
    return float(inter / (union + 1e-9))


def make_figure(orig: np.ndarray, gt: np.ndarray,
                panels: dict[str, np.ndarray],  # method → recon_rgb
                segs: dict[str, np.ndarray],     # method → seg full-res
                iious: dict[str, float],
                out_path: Path,
                fig_title: str = ""):
    """Vẽ grid 3×2 giống Fig.9 của paper và lưu file."""
    methods_order = ["H264", "SA-X264", "H265", "SA-X265"]

    # Thumbnail size cho display (giữ tỷ lệ 2:1)
    DISP_W, DISP_H = 1024, 512

    def thumb(arr: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(arr).resize((DISP_W, DISP_H), Image.BILINEAR))

    def thumb_seg(seg: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(seg_to_rgb(seg)).resize((DISP_W, DISP_H), Image.NEAREST))

    orig_t = thumb(orig)
    gt_t = thumb_seg(gt)

    # Tìm box khác biệt (so sánh H264 vs SA-X264 trên thumbnail)
    seg_h264_t = np.array(Image.fromarray(segs["H264"]).resize((DISP_W, DISP_H), Image.NEAREST))
    seg_sax264_t = np.array(Image.fromarray(segs["SA-X264"]).resize((DISP_W, DISP_H), Image.NEAREST))
    bx, by, bw, bh = find_best_diff_box(seg_h264_t, seg_sax264_t, block=128)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.08, wspace=0.04)
    if fig_title:
        fig.suptitle(fig_title, fontsize=11, y=0.98)

    cells = [
        (0, 0, orig_t,   "(a) Original Frame", False),
        (0, 1, gt_t,     "(b) Ground Truth",    False),
        (1, 0, thumb_seg(segs["H264"]),   f"(c) Result of H.264  [iIoU={iious['H264']:.4f}]",    True),
        (1, 1, thumb_seg(segs["SA-X264"]),f"(d) Result of SA-X264 [iIoU={iious['SA-X264']:.4f}]",True),
        (2, 0, thumb_seg(segs["H265"]),   f"(e) Result of H.265  [iIoU={iious['H265']:.4f}]",    True),
        (2, 1, thumb_seg(segs["SA-X265"]),f"(f) Result of SA-X265 [iIoU={iious['SA-X265']:.4f}]",True),
    ]

    for (row, col, img_arr, caption, draw_box) in cells:
        ax = axes[row][col]
        ax.imshow(img_arr)
        ax.set_title(caption, fontsize=9, pad=3)
        ax.axis("off")
        if draw_box:
            rect = patches.Rectangle(
                (bx, by), bw, bh,
                linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

    # Legend cho màu lớp
    legend_labels = ["ROI", "Sky", "Construction", "Nature"]
    legend_colors = [tuple(c/255 for c in SEG_PALETTE[i]) for i in range(4)]
    legend_patches = [patches.Patch(facecolor=lc, label=ll)
                      for lc, ll in zip(legend_colors, legend_labels)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", type=int,
                    default=[273, 210, 251],
                    help="Dataset indices của frame muốn visualize (từ per_frame.csv).")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", choices=("ccnet", "pidnet"), default="ccnet",
                    help="Backbone segmentation: ccnet (paper gốc) hoặc pidnet (PIDNet-L).")
    ap.add_argument("--ckpt", default=None,
                    help="Override checkpoint path; mặc định auto theo --model.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}")

    if args.model == "pidnet":
        model = build_pidnet_l(num_classes=NUM_CLASSES)
        default_ckpt = PIDNET_L_CHECKPOINT
        gray_input = False
    else:
        model = build_segnet(num_classes=NUM_CLASSES, recurrence=2, pretrained=False)
        default_ckpt = SAC_CHECKPOINT
        gray_input = True

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(default_ckpt)
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=False)
        print(f"Loaded: {ckpt_path}")
    else:
        print(f"⚠  Checkpoint không tồn tại: {ckpt_path}. Dùng weight hiện tại.")
    model = model.to(device).eval()

    out_dir = Path(args.out_dir) if args.out_dir else EVAL_DIR / "seg_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset index để ánh xạ idx → filename
    from .dataset import _list_split
    pairs = _list_split(args.split)
    idx_to_name = {i: pairs[i][0].name for i in range(len(pairs))}

    for frame_idx in args.frames:
        if frame_idx not in idx_to_name:
            print(f"⚠  Index {frame_idx} không có trong split={args.split}, bỏ qua.")
            continue
        img_name = idx_to_name[frame_idx]
        print(f"\n[Frame {frame_idx}] {img_name}")

        orig, gt = load_orig_and_gt(img_name, args.split)

        # ROI mask từ seg trên ảnh gốc (dùng cho SAC stream split)
        seg_orig = predict_seg(model, orig, device, gray_input=gray_input)
        roi_mask = build_roi_mask(
            np.array(Image.fromarray(seg_orig).resize(
                (SEG_INPUT_HW[1], SEG_INPUT_HW[0]), Image.NEAREST)),
            target_hw=FRAME_HW)

        segs: dict[str, np.ndarray] = {}
        iious: dict[str, float] = {}

        for method, cfg in CRF_CONFIGS.items():
            codec, crf_r, crf_n = cfg["codec"], cfg["crf_roi"], cfg["crf_non"]
            print(f"  {method}: codec={codec} CRF=({crf_r},{crf_n})", end=" ... ", flush=True)
            recon = compress_and_decode(orig, roi_mask, codec, crf_r, crf_n)
            seg = predict_seg(model, recon, device, gray_input=gray_input)
            segs[method] = seg
            iious[method] = compute_iiou(seg, gt)
            print(f"iIoU={iious[method]:.4f}")

        out_path = out_dir / f"seg_compare_idx{frame_idx}_{args.model}.png"
        make_figure(orig, gt, {}, segs, iious, out_path,
                    fig_title=f"Segmentation Comparison ({args.model.upper()}) — {img_name}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
