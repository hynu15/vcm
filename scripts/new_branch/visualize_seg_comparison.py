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


def _load_model(name: str, ckpt_override: str | None, device):
    """Load một model theo tên ('ccnet' | 'pidnet'), trả về (model, gray_input)."""
    if name == "pidnet":
        model = build_pidnet_l(num_classes=NUM_CLASSES)
        default_ckpt = PIDNET_L_CHECKPOINT
        gray_input = False
    else:
        model = build_segnet(num_classes=NUM_CLASSES, recurrence=2, pretrained=False)
        default_ckpt = SAC_CHECKPOINT
        gray_input = True

    ckpt_path = Path(ckpt_override) if ckpt_override else Path(default_ckpt)
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        if name == "pidnet":
            model.backbone.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(sd, strict=False)
        print(f"  Loaded {name.upper():7s}  ←  {ckpt_path.name}")
    else:
        print(f"  ⚠  Checkpoint không tồn tại: {ckpt_path}")
    return model.to(device).eval(), gray_input


# H.264 / H.265 truyền thống
_CRF_TRAD = {
    "H264": {"codec": "libx264", "crf_roi": 23, "crf_non": 23},
    "H265": {"codec": "libx265", "crf_roi": 28, "crf_non": 28},
}

# SA-X264 / SA-X265 dùng backbone để tạo ROI mask
_CRF_SA = {
    "SA-X264": {"codec": "libx264", "crf_roi": 18, "crf_non": 27},
    "SA-X265": {"codec": "libx265", "crf_roi": 23, "crf_non": 32},
}

# Thứ tự hiển thị 4 codec trong figure
_ALL_CODECS = ["H264", "H265", "SA-X264", "SA-X265"]


def make_figure_models(orig: np.ndarray, gt: np.ndarray,
                       segs_left: dict[str, np.ndarray],
                       segs_right: dict[str, np.ndarray],
                       iious_left: dict[str, float],
                       iious_right: dict[str, float],
                       out_path: Path,
                       fig_title: str = "") -> None:
    """Layout 5×2: H.264/H.265 truyền thống + SA-X264/SA-X265.

    Cột trái  = CCNet  (trad: CCNet seg; SAC: PIDNet eval trên CCNet-ROI frame)
    Cột phải  = PIDNet (trad: PIDNet seg; SAC: PIDNet eval trên PIDNet-ROI frame)

    (a) Original Frame        | (b) Ground Truth
    (c) H.264   — CCNet       | (d) H.264   — PIDNet-L
    (e) H.265   — CCNet       | (f) H.265   — PIDNet-L
    (g) SA-X264, CCNet bbone  | (h) SA-X264, PIDNet bbone
    (i) SA-X265, CCNet bbone  | (j) SA-X265, PIDNet bbone
    """
    DISP_W, DISP_H = 1024, 512

    def thumb(arr: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(arr).resize((DISP_W, DISP_H), Image.BILINEAR))

    def thumb_seg(seg: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(seg_to_rgb(seg)).resize(
            (DISP_W, DISP_H), Image.NEAREST))

    letters = "cdefghij"
    row_specs = [
        ("H264",    "H.264",   "CCNet",         "PIDNet-L"),
        ("H265",    "H.265",   "CCNet",         "PIDNet-L"),
        ("SA-X264", "SA-X264", "CCNet backbone","PIDNet backbone"),
        ("SA-X265", "SA-X265", "CCNet backbone","PIDNet backbone"),
    ]

    # Tính red box riêng cho từng row (vùng trái/phải khác nhau nhiều nhất)
    def row_box(key):
        sl = np.array(Image.fromarray(segs_left[key]).resize(
            (DISP_W, DISP_H), Image.NEAREST))
        sr = np.array(Image.fromarray(segs_right[key]).resize(
            (DISP_W, DISP_H), Image.NEAREST))
        return find_best_diff_box(sl, sr, block=128)

    row_boxes = {key: row_box(key) for key, *_ in row_specs}

    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.10, wspace=0.04)
    if fig_title:
        fig.suptitle(fig_title, fontsize=11, y=0.995)

    # Row 0: Original + GT
    axes[0][0].imshow(thumb(orig))
    axes[0][0].set_title("(a) Original Frame", fontsize=9, pad=3)
    axes[0][0].axis("off")
    axes[0][1].imshow(thumb_seg(gt))
    axes[0][1].set_title("(b) Ground Truth", fontsize=9, pad=3)
    axes[0][1].axis("off")

    # Rows 1-4: 4 codec, mỗi row có red box riêng
    for r_off, (key, label, lname, rname) in enumerate(row_specs):
        row = r_off + 1
        ltr_l = letters[r_off * 2]
        ltr_r = letters[r_off * 2 + 1]
        bx, by, bw, bh = row_boxes[key]

        ax_l = axes[row][0]
        ax_r = axes[row][1]

        ax_l.imshow(thumb_seg(segs_left[key]))
        ax_l.set_title(
            f"({ltr_l}) {label} — {lname}  [iIoU={iious_left[key]:.4f}]",
            fontsize=9, pad=3)
        ax_l.axis("off")

        ax_r.imshow(thumb_seg(segs_right[key]))
        ax_r.set_title(
            f"({ltr_r}) {label} — {rname} [iIoU={iious_right[key]:.4f}]",
            fontsize=9, pad=3)
        ax_r.axis("off")

        for ax in (ax_l, ax_r):
            ax.add_patch(patches.Rectangle(
                (bx, by), bw, bh,
                linewidth=2, edgecolor="red", facecolor="none"))

    legend_labels = ["ROI", "Sky", "Construction", "Nature"]
    legend_colors = [tuple(c / 255 for c in SEG_PALETTE[i]) for i in range(4)]
    legend_patches = [patches.Patch(facecolor=lc, label=ll)
                      for lc, ll in zip(legend_colors, legend_labels)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.005))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def make_figure_codec_family(
    orig: np.ndarray, gt: np.ndarray,
    seg_trad: np.ndarray,
    seg_cc_sac: np.ndarray,
    seg_pid_sac: np.ndarray,
    iiou_trad: float,
    iiou_cc_sac: float,
    iiou_pid_sac: float,
    codec_label: str,   # "H.264" hoặc "H.265"
    sa_label: str,      # "SA-X264" hoặc "SA-X265"
    out_path: Path,
    fig_title: str = "",
) -> None:
    """Layout 2×3: so sánh Traditional < CCNet SAC < PIDNet SAC.
    Tất cả iIoU đo bằng PIDNet-L.

    Row 0: (a) Original Frame  | (b) Ground Truth      | [class legend]
    Row 1: (c) {codec} trad    | (d) {sa} CCNet bbone  | (e) {sa} PIDNet bbone
    """
    DISP_W, DISP_H = 1024, 512

    def thumb(arr: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(arr).resize((DISP_W, DISP_H), Image.BILINEAR))

    def thumb_seg(seg: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(seg_to_rgb(seg)).resize(
            (DISP_W, DISP_H), Image.NEAREST))

    # Red box: vùng traditional vs PIDNet SAC khác nhau nhiều nhất (cải thiện nhiều nhất)
    st = np.array(Image.fromarray(seg_trad).resize((DISP_W, DISP_H), Image.NEAREST))
    sp = np.array(Image.fromarray(seg_pid_sac).resize((DISP_W, DISP_H), Image.NEAREST))
    bx, by, bw, bh = find_best_diff_box(st, sp, block=128)

    fig, axes = plt.subplots(2, 3, figsize=(20, 9))
    plt.subplots_adjust(hspace=0.12, wspace=0.04)
    if fig_title:
        fig.suptitle(fig_title, fontsize=11, y=0.99)

    # Row 0: Original | GT | legend cell
    axes[0][0].imshow(thumb(orig))
    axes[0][0].set_title("(a) Original Frame", fontsize=10, pad=4)
    axes[0][0].axis("off")

    axes[0][1].imshow(thumb_seg(gt))
    axes[0][1].set_title("(b) Ground Truth", fontsize=10, pad=4)
    axes[0][1].axis("off")

    # ax[0][2]: legend
    axes[0][2].axis("off")
    legend_labels = ["ROI", "Sky", "Construction", "Nature"]
    legend_colors = [tuple(c / 255 for c in SEG_PALETTE[i]) for i in range(4)]
    legend_patches = [patches.Patch(facecolor=lc, label=ll)
                      for lc, ll in zip(legend_colors, legend_labels)]
    axes[0][2].legend(handles=legend_patches, loc="center",
                      fontsize=11, frameon=True, ncol=1)
    axes[0][2].set_title("Class legend", fontsize=10, pad=4)

    # Row 1: 3 kết quả so sánh
    result_cells = [
        (axes[1][0], thumb_seg(seg_trad),
         f"(c) {codec_label} Traditional\n[PIDNet iIoU={iiou_trad:.4f}]"),
        (axes[1][1], thumb_seg(seg_cc_sac),
         f"(d) {sa_label} — CCNet backbone\n[PIDNet iIoU={iiou_cc_sac:.4f}]"),
        (axes[1][2], thumb_seg(seg_pid_sac),
         f"(e) {sa_label} — PIDNet backbone\n[PIDNet iIoU={iiou_pid_sac:.4f}]"),
    ]
    for ax, img_arr, caption in result_cells:
        ax.imshow(img_arr)
        ax.set_title(caption, fontsize=9, pad=3)
        ax.axis("off")
        ax.add_patch(patches.Rectangle(
            (bx, by), bw, bh,
            linewidth=2, edgecolor="red", facecolor="none"))

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
    ap.add_argument("--compare-models", action="store_true",
                    help="So sánh CCNet vs PIDNet-L trên H.264/H.265 (3×2 layout).")
    ap.add_argument("--ckpt-ccnet", default=None, help="Override CCNet checkpoint.")
    ap.add_argument("--ckpt-pidnet", default=None, help="Override PIDNet checkpoint.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir) if args.out_dir else EVAL_DIR / "seg_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    from .dataset import _list_split
    pairs = _list_split(args.split)
    idx_to_name = {i: pairs[i][0].name for i in range(len(pairs))}

    # ── Chế độ compare-models: H.264/H.265 trad + SA-X264/SA-X265, 5×2 ─────────
    if args.compare_models:
        print(f"Device: {device}  |  Mode: compare-models (5×2 layout)")
        cc_model,  cc_gray  = _load_model("ccnet",  args.ckpt_ccnet,  device)
        pid_model, pid_gray = _load_model("pidnet", args.ckpt_pidnet, device)

        for frame_idx in args.frames:
            if frame_idx not in idx_to_name:
                print(f"⚠  Index {frame_idx} không có trong split={args.split}, bỏ qua.")
                continue
            img_name = idx_to_name[frame_idx]
            print(f"\n[Frame {frame_idx}] {img_name}")

            orig, gt = load_orig_and_gt(img_name, args.split)

            # ROI mask từ CCNet và PIDNet trên ảnh gốc (dùng cho SAC)
            seg_cc_orig  = predict_seg(cc_model,  orig, device, gray_input=cc_gray)
            seg_pid_orig = predict_seg(pid_model, orig, device, gray_input=pid_gray)
            roi_mask_cc  = build_roi_mask(
                np.array(Image.fromarray(seg_cc_orig).resize(
                    (SEG_INPUT_HW[1], SEG_INPUT_HW[0]), Image.NEAREST)),
                target_hw=FRAME_HW)
            roi_mask_pid = build_roi_mask(
                np.array(Image.fromarray(seg_pid_orig).resize(
                    (SEG_INPUT_HW[1], SEG_INPUT_HW[0]), Image.NEAREST)),
                target_hw=FRAME_HW)

            segs_left:  dict[str, np.ndarray] = {}
            segs_right: dict[str, np.ndarray] = {}
            iious_left:  dict[str, float] = {}
            iious_right: dict[str, float] = {}

            # Truyền thống H.264 / H.265: cùng 1 frame nén, CCNet vs PIDNet seg
            for method, cfg in _CRF_TRAD.items():
                codec, crf_r, crf_n = cfg["codec"], cfg["crf_roi"], cfg["crf_non"]
                print(f"  {method} (trad) CRF={crf_r}", end=" ... ", flush=True)
                recon = compress_and_decode(orig, roi_mask_cc, codec, crf_r, crf_n)
                seg_l = predict_seg(cc_model,  recon, device, gray_input=cc_gray)
                seg_r = predict_seg(pid_model, recon, device, gray_input=pid_gray)
                segs_left[method]  = seg_l
                segs_right[method] = seg_r
                iious_left[method]  = compute_iiou(seg_l, gt)
                iious_right[method] = compute_iiou(seg_r, gt)
                print(f"CCNet={iious_left[method]:.4f}  PIDNet={iious_right[method]:.4f}")

            # SAC SA-X264 / SA-X265: 2 frame nén riêng (khác ROI mask), PIDNet eval
            for method, cfg in _CRF_SA.items():
                codec, crf_r, crf_n = cfg["codec"], cfg["crf_roi"], cfg["crf_non"]
                print(f"  {method} (SAC) CRF=({crf_r},{crf_n})", end=" ... ", flush=True)
                recon_cc  = compress_and_decode(orig, roi_mask_cc,  codec, crf_r, crf_n)
                recon_pid = compress_and_decode(orig, roi_mask_pid, codec, crf_r, crf_n)
                seg_l = predict_seg(pid_model, recon_cc,  device, gray_input=pid_gray)
                seg_r = predict_seg(pid_model, recon_pid, device, gray_input=pid_gray)
                segs_left[method]  = seg_l
                segs_right[method] = seg_r
                iious_left[method]  = compute_iiou(seg_l, gt)
                iious_right[method] = compute_iiou(seg_r, gt)
                print(f"CCNet-bbone={iious_left[method]:.4f}  PIDNet-bbone={iious_right[method]:.4f}")

            out_path = out_dir / f"seg_compare_idx{frame_idx}_models.png"
            make_figure_models(orig, gt, segs_left, segs_right, iious_left, iious_right,
                               out_path,
                               fig_title=f"CCNet vs PIDNet-L (eval by PIDNet) — {img_name}")

        print(f"\nDone. Output dir: {out_dir}")
        return

    # ── Chế độ cũ: một model, 4 phương pháp nén ──────────────────────────────
    print(f"Device: {device}  |  Model: {args.model}")
    model, gray_input = _load_model(args.model, args.ckpt, device)

    for frame_idx in args.frames:
        if frame_idx not in idx_to_name:
            print(f"⚠  Index {frame_idx} không có trong split={args.split}, bỏ qua.")
            continue
        img_name = idx_to_name[frame_idx]
        print(f"\n[Frame {frame_idx}] {img_name}")

        orig, gt = load_orig_and_gt(img_name, args.split)

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
