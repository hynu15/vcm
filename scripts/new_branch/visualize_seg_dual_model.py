"""So sánh CCNet vs PIDNet-L (4-class) trong CÙNG MỘT figure.

Layout 5x2:
  Row 1:  (a) Original Frame                 | (b) Ground Truth
  Row 2:  (c) H.264   — CCNet                | (d) H.264   — PIDNet-L
  Row 3:  (e) SA-X264 — CCNet                | (f) SA-X264 — PIDNet-L
  Row 4:  (g) H.265   — CCNet                | (h) H.265   — PIDNet-L
  Row 5:  (i) SA-X265 — CCNet                | (j) SA-X265 — PIDNet-L

Chạy:
  cd /home/huy/sac_project
  conda activate sac
  python -m scripts.new_branch.visualize_seg_dual_model --frames 210
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
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .model import build_segnet, build_pidnet_l
from .paths import (CRF_CONFIGS, EVAL_DIR, FRAME_HW, NUM_CLASSES,
                    PIDNET_L_CHECKPOINT, ROI_CLASS_ID, SAC_CHECKPOINT,
                    SEG_INPUT_HW)
from .stream_separation import build_roi_mask
from .visualize_seg_comparison import (SEG_PALETTE, compress_and_decode,
                                       compute_iiou, find_best_diff_box,
                                       load_orig_and_gt, predict_seg,
                                       seg_to_rgb)


DISP_W, DISP_H = 1024, 512


def thumb(arr: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((DISP_W, DISP_H), Image.BILINEAR))


def thumb_seg(seg: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(seg_to_rgb(seg)).resize(
        (DISP_W, DISP_H), Image.NEAREST))


def load_model(name: str, device: torch.device):
    """name in {'ccnet', 'pidnet'} → (model.eval(), gray_input_flag)."""
    if name == "ccnet":
        model = build_segnet(num_classes=NUM_CLASSES, recurrence=2, pretrained=False)
        ckpt_path = Path(SAC_CHECKPOINT)
        gray_input = True
    else:
        model = build_pidnet_l(num_classes=NUM_CLASSES)
        ckpt_path = Path(PIDNET_L_CHECKPOINT)
        gray_input = False
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    # PIDNet checkpoint lưu state-dict RAW (không có prefix "backbone.") →
    # load thẳng vào .backbone bên trong wrapper để key khớp.
    if name == "pidnet":
        missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [{name}] missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    print(f"  Loaded {name.upper():7s}  ←  {ckpt_path.name}")
    return model, gray_input


def make_dual_figure(orig: np.ndarray, gt: np.ndarray,
                     segs_cc: dict[str, np.ndarray],
                     segs_pid: dict[str, np.ndarray],
                     iious_cc: dict[str, float],
                     iious_pid: dict[str, float],
                     out_path: Path,
                     fig_title: str = "") -> None:
    """Layout 5x2, mỗi MODEL chiếm 2 hàng liền nhau (giữ logic 'một model
    đánh giá nhiều codec' như file seg_compare_idx*.png gốc).

        Row 0:  Original           | Ground Truth
        Row 1:  H.264   (CCNet)    | SA-X264  (CCNet)
        Row 2:  H.265   (CCNet)    | SA-X265  (CCNet)
        Row 3:  H.264   (PIDNet-L) | SA-X264  (PIDNet-L)
        Row 4:  H.265   (PIDNet-L) | SA-X265  (PIDNet-L)
    """
    pretty = {"H264": "H.264", "SA-X264": "SA-X264",
              "H265": "H.265", "SA-X265": "SA-X265"}
    # Thứ tự codec trong mỗi block model: 2 hàng × 2 cột
    pair_grid = [["H264", "SA-X264"], ["H265", "SA-X265"]]

    # Vùng khác biệt: so H264 vs SA-X264 trên CCNet (giữ nhất quán với bản gốc)
    seg_h264_t = thumb_seg(segs_cc["H264"])
    seg_sax_t  = thumb_seg(segs_cc["SA-X264"])
    bx, by, bw, bh = find_best_diff_box(
        seg_h264_t[..., 0], seg_sax_t[..., 0], block=128)

    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.18, wspace=0.04)
    if fig_title:
        fig.suptitle(fig_title, fontsize=12, y=0.995)

    # ── Row 0: Original + GT ────────────────────────────────────────────────
    axes[0][0].imshow(thumb(orig))
    axes[0][0].set_title("(a) Original Frame", fontsize=10, pad=3)
    axes[0][0].axis("off")

    axes[0][1].imshow(thumb_seg(gt))
    axes[0][1].set_title("(b) Ground Truth", fontsize=10, pad=3)
    axes[0][1].axis("off")

    # ── Hai block model: mỗi block 2x2 codec ────────────────────────────────
    letters = iter("cdefghij")
    blocks = [
        ("CCNet",     segs_cc,  iious_cc,  1),  # row 1, 2
        ("PIDNet-L",  segs_pid, iious_pid, 3),  # row 3, 4
    ]
    for model_name, segs, iious, base_row in blocks:
        for r_off in range(2):
            for c in range(2):
                method = pair_grid[r_off][c]
                ax = axes[base_row + r_off][c]
                ax.imshow(thumb_seg(segs[method]))
                ax.set_title(
                    f"({next(letters)}) {pretty[method]}  "
                    f"[iIoU={iious[method]:.4f}]",
                    fontsize=10, pad=3)
                ax.axis("off")
                ax.add_patch(patches.Rectangle(
                    (bx, by), bw, bh,
                    linewidth=2, edgecolor="red", facecolor="none"))

        # Nhãn model ở mép trái của hàng đầu trong block
        axes[base_row][0].set_ylabel(model_name, fontsize=12,
                                     fontweight="bold", labelpad=10)
        axes[base_row][0].axis("on")
        for s in ("top", "right", "bottom", "left"):
            axes[base_row][0].spines[s].set_visible(False)
        axes[base_row][0].set_xticks([])
        axes[base_row][0].set_yticks([])

    # ── Đường phân cách giữa hai block model ────────────────────────────────
    # Vẽ một line ngang nhẹ giữa hàng 2 (CCNet cuối) và hàng 3 (PIDNet đầu)
    y_sep = (axes[2][0].get_position().y0 + axes[3][0].get_position().y1) / 2
    fig.add_artist(plt.Line2D(
        [0.05, 0.95], [y_sep, y_sep],
        color="gray", linewidth=0.8, linestyle="--",
        transform=fig.transFigure, alpha=0.6))

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_labels = ["ROI", "Sky", "Construction", "Nature"]
    legend_colors = [tuple(c / 255 for c in SEG_PALETTE[i]) for i in range(4)]
    legend_patches = [patches.Patch(facecolor=lc, label=ll)
                      for lc, ll in zip(legend_colors, legend_labels)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.005))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", type=int, default=[210])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cc_model,  cc_gray  = load_model("ccnet",  device)
    pid_model, pid_gray = load_model("pidnet", device)

    out_dir = Path(args.out_dir) if args.out_dir else EVAL_DIR / "seg_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

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

        # ROI mask cho stream split: dùng CCNet (ổn định theo paper gốc)
        seg_orig = predict_seg(cc_model, orig, device, gray_input=cc_gray)
        roi_mask = build_roi_mask(
            np.array(Image.fromarray(seg_orig).resize(
                (SEG_INPUT_HW[1], SEG_INPUT_HW[0]), Image.NEAREST)),
            target_hw=FRAME_HW)

        segs_cc:  dict[str, np.ndarray] = {}
        segs_pid: dict[str, np.ndarray] = {}
        iious_cc:  dict[str, float] = {}
        iious_pid: dict[str, float] = {}

        for method, cfg in CRF_CONFIGS.items():
            codec, crf_r, crf_n = cfg["codec"], cfg["crf_roi"], cfg["crf_non"]
            print(f"  {method}: codec={codec} CRF=({crf_r},{crf_n})")
            recon = compress_and_decode(orig, roi_mask, codec, crf_r, crf_n)

            seg_c = predict_seg(cc_model,  recon, device, gray_input=cc_gray)
            seg_p = predict_seg(pid_model, recon, device, gray_input=pid_gray)
            segs_cc[method]  = seg_c
            segs_pid[method] = seg_p
            iious_cc[method]  = compute_iiou(seg_c, gt)
            iious_pid[method] = compute_iiou(seg_p, gt)
            print(f"     CCNet iIoU={iious_cc[method]:.4f}  |  "
                  f"PIDNet iIoU={iious_pid[method]:.4f}")

        out_path = out_dir / f"seg_compare_idx{frame_idx}_dual.png"
        make_dual_figure(orig, gt, segs_cc, segs_pid, iious_cc, iious_pid,
                         out_path,
                         fig_title=f"Segmentation Comparison: CCNet vs PIDNet-L — {img_name}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
