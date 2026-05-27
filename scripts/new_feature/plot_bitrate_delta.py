"""
So sánh bitrate giữa 3 cấu hình tại cùng mức QP truyền thống:
  • Traditional        : crf = QP  (single stream)
  • SAC delta=7 (hiện tại): crf_roi = QP-3, crf_non = QP+4
  • SAC delta=4 (mới)  : crf_roi = QP-2, crf_non = QP+2

Chỉ encode video, không tính metrics segmentation → chạy nhanh.

Chạy từ scripts/:
  conda run -n sac python3 new_feature/plot_bitrate_delta.py
  conda run -n sac python3 new_feature/plot_bitrate_delta.py --qp 22 25 28 31 34 --num-frames 30
"""

from __future__ import annotations

import argparse
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from new_feature.compression import (
    compress_sac,
    compress_traditional,
    macroblock_align_filter,
)
from new_feature.dataset import Cityscapes4Class

PROJECT_ROOT       = _HERE.parent.parent
MODEL_PATH_DEFAULT = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
FALLBACK_MODELS    = [
    PROJECT_ROOT / "models" / "best_pidnet_l.pth",
    PROJECT_ROOT / "models" / "best_ccnet.pth",
]
IMG_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
LBL_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "val"
IMG_TEST = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "test"
LBL_TEST = PROJECT_ROOT / "data" / "gt_4class" / "test"
TEST_CITIES = ["strasbourg", "ulm"]

SEG_SIZE   = (512, 1024)
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DEFAULT_QPS = [22, 25, 28, 31, 34]


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation (chỉ cần để tạo ROI mask cho SAC)
# ─────────────────────────────────────────────────────────────────────────────

def _load_seg_model(device, model_path=None):
    candidates = [Path(model_path)] if model_path else [MODEL_PATH_DEFAULT] + FALLBACK_MODELS
    for p in candidates:
        if not p.is_file():
            continue
        ckpt      = torch.load(str(p), map_location=device, weights_only=False)
        meta      = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        state     = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        num_cls   = int(meta.get("num_classes", 4))
        model_name = meta.get("model_name", "")
        keys      = list(state.keys())
        is_new    = any(k.startswith("stem.") or k.startswith("layer1.") for k in keys)

        if model_name.startswith("pidnet") or (
            not is_new and any(k.startswith("conv1.") for k in keys)
        ):
            from train_segmentation import load_segmentation_model as _ll
            model, _ = _ll(str(p), device, num_classes=num_cls)
        elif is_new:
            from new_feature.ccnet_4class import CCNet4Class
            model = CCNet4Class(num_classes=num_cls, pretrained=False).to(device)
            model.load_state_dict(state)
        else:
            from train_segmentation import load_segmentation_model as _ll
            model, _ = _ll(str(p), device, num_classes=num_cls)

        model.eval()
        print(f"  Seg model ({num_cls}-class): {p.name}")
        return model, num_cls
    raise FileNotFoundError("Không tìm thấy checkpoint. Đã thử: " +
                            ", ".join(str(c) for c in candidates))


@torch.no_grad()
def _segment(model, device, frame_rgb: np.ndarray) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    pil  = Image.fromarray(frame_rgb).resize((SEG_SIZE[1], SEG_SIZE[0]), Image.BILINEAR)
    t    = _TRANSFORM(pil).unsqueeze(0).to(device)
    out  = model(t)
    if isinstance(out, (list, tuple)):
        out = out[1]
    pred = out.argmax(1)[0]
    return F.interpolate(
        pred.float().unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest",
    )[0, 0].byte().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Encode helpers — chỉ lấy total_bytes, không decode metrics
# ─────────────────────────────────────────────────────────────────────────────

def _encode_trad(frames_bgr, codec, crf, work_dir, fps, preset, tag):
    _, total_b = compress_traditional(
        frames_bgr, codec=codec, crf=crf,
        work_dir=str(work_dir), tag=tag, fps=fps, preset=preset,
    )
    return total_b


def _encode_sac(frames_bgr, roi_masks, codec, crf_roi, crf_non,
                work_dir, fps, preset, tag):
    _, roi_b, non_b = compress_sac(
        frames_bgr, roi_masks, codec=codec,
        crf_roi=crf_roi, crf_non=crf_non,
        work_dir=str(work_dir), tag=tag, fps=fps, preset=preset,
    )
    return roi_b + non_b


def bytes_to_kbps(total_bytes: int, n_frames: int, fps: int) -> float:
    return (total_bytes * 8.0) / (n_frames / fps) / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = [
    # (label, color, marker, linestyle)
    ("H.264 Traditional",   "#1f77b4", "o",  "-"),
    ("H.264 SAC delta=7",   "#ff7f0e", "s",  "--"),
    ("H.264 SAC delta=6",   "#e377c2", "D",  "-."),
    ("H.264 SAC delta=4",   "#9467bd", "^",  ":"),
    ("H.265 Traditional",   "#2ca02c", "o",  "-"),
    ("H.265 SAC delta=7",   "#d62728", "s",  "--"),
    ("H.265 SAC delta=6",   "#8c564b", "D",  "-."),
    ("H.265 SAC delta=4",   "#17becf", "^",  ":"),
]


def _bar_panel(ax, qp_list, kbps_a, kbps_b, label_a, label_b, color_pos, color_neg, title):
    """Vẽ bar chart Δ = kbps_a − kbps_b với nhãn phần trăm."""
    diff = [a - b for a, b in zip(kbps_a, kbps_b)]
    pct  = [d / b * 100 for d, b in zip(diff, kbps_b)]
    bars = ax.bar(qp_list, diff,
                  color=[color_pos if d >= 0 else color_neg for d in diff],
                  alpha=0.75, width=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    max_d, min_d = max(diff), min(diff)
    span = max_d - min_d if max_d != min_d else 1
    for bar, p in zip(bars, pct):
        offset = span * 0.04 if bar.get_height() >= 0 else -span * 0.04
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{p:+.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("QP", fontsize=10)
    ax.set_ylabel("Δ Bitrate (kbps)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(qp_list)
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")


def _plot(qp_list: List[int], data: dict, out_png: Path, n_frames: int, fps: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    labels  = [c[0] for c in CONFIGS]
    colors  = {c[0]: c[1] for c in CONFIGS}
    markers = {c[0]: c[2] for c in CONFIGS}
    ls_map  = {c[0]: c[3] for c in CONFIGS}

    kbps = {lbl: [bytes_to_kbps(data[lbl][qp], n_frames, fps) for qp in qp_list]
            for lbl in labels if lbl in data}

    fig = plt.figure(figsize=(16, 14), dpi=150)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ── Panel 1 (full width): Bitrate tuyệt đối ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for lbl in labels:
        if lbl not in kbps:
            continue
        ax1.plot(qp_list, kbps[lbl],
                 color=colors[lbl], marker=markers[lbl], ls=ls_map[lbl],
                 linewidth=2, markersize=8, label=lbl)
        for qp, val in zip(qp_list, kbps[lbl]):
            ax1.annotate(f"{val:.0f}", (qp, val),
                         textcoords="offset points", xytext=(0, 6),
                         fontsize=6.5, ha="center", color=colors[lbl])
    ax1.set_xlabel("QP (traditional baseline)", fontsize=11)
    ax1.set_ylabel("Bitrate (kbps)", fontsize=11)
    ax1.set_title("Bitrate tuyệt đối: Traditional vs SAC delta=7 vs delta=6 vs delta=4",
                  fontsize=12)
    ax1.set_xticks(qp_list)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8.5, ncol=2, loc="upper right")

    # ── Panel 2 & 3: delta=6 vs delta=7 ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if "H.264 SAC delta=6" in kbps and "H.264 SAC delta=7" in kbps:
        _bar_panel(ax2, qp_list,
                   kbps["H.264 SAC delta=6"], kbps["H.264 SAC delta=7"],
                   "delta=6", "delta=7", "#e377c2", "#1f77b4",
                   "H.264: delta=6 − delta=7\n(+ = delta=6 dùng nhiều hơn)")

    ax3 = fig.add_subplot(gs[1, 1])
    if "H.265 SAC delta=6" in kbps and "H.265 SAC delta=7" in kbps:
        _bar_panel(ax3, qp_list,
                   kbps["H.265 SAC delta=6"], kbps["H.265 SAC delta=7"],
                   "delta=6", "delta=7", "#8c564b", "#2ca02c",
                   "H.265: delta=6 − delta=7\n(+ = delta=6 dùng nhiều hơn)")

    # ── Panel 4 & 5: delta=4 vs delta=7 ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if "H.264 SAC delta=4" in kbps and "H.264 SAC delta=7" in kbps:
        _bar_panel(ax4, qp_list,
                   kbps["H.264 SAC delta=4"], kbps["H.264 SAC delta=7"],
                   "delta=4", "delta=7", "#9467bd", "#1f77b4",
                   "H.264: delta=4 − delta=7\n(+ = delta=4 dùng nhiều hơn)")

    ax5 = fig.add_subplot(gs[2, 1])
    if "H.265 SAC delta=4" in kbps and "H.265 SAC delta=7" in kbps:
        _bar_panel(ax5, qp_list,
                   kbps["H.265 SAC delta=4"], kbps["H.265 SAC delta=7"],
                   "delta=4", "delta=7", "#17becf", "#2ca02c",
                   "H.265: delta=4 − delta=7\n(+ = delta=4 dùng nhiều hơn)")

    fig.suptitle(
        f"Ảnh hưởng của delta (crf_non − crf_roi) lên Bitrate\n"
        f"({n_frames} frames, fps={fps}, QP {{{','.join(map(str, qp_list))}}})",
        fontsize=13,
    )
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "new_feature" / "bitrate_delta" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "tmp"
    work_dir.mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Output : {out_dir}")
    print(f"QP     : {args.qp}")

    # 1. Load segmentation model (chỉ để build ROI mask)
    seg_model, _ = _load_seg_model(device, args.model_path)

    # 2. Load frames
    if args.split == "test":
        img_dir, lbl_dir = IMG_TEST, LBL_TEST
        cities = TEST_CITIES
    else:
        img_dir, lbl_dir = IMG_VAL, LBL_VAL
        cities = None

    dataset  = Cityscapes4Class(img_dir, lbl_dir, augment=False, cities=cities)
    n        = min(args.num_frames, len(dataset))
    print(f"Frames : {n} / {len(dataset)}")

    frames_bgr: list = []
    roi_masks:  list = []

    print("\nLoad frames + build ROI masks ...")
    for i in tqdm(range(n)):
        _, _, img_path = dataset[i]
        rgb = np.array(Image.open(img_path).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pred = _segment(seg_model, device, rgb)
        mask = macroblock_align_filter((pred == 0).astype(np.uint8))
        frames_bgr.append(bgr)
        roi_masks.append(mask)

    # 3. Encode mỗi (codec, config, QP)
    #    config: traditional / sac_d7 / sac_d6 / sac_d4
    codecs = [("libx264", "H.264"), ("libx265", "H.265")]

    # SAC configs: (suffix, label_fmt, roi_offset, non_offset)
    sac_configs = [
        ("d7", "SAC delta=7", -3, +4),   # delta = 7
        ("d6", "SAC delta=6", -3, +3),   # delta = 6
        ("d4", "SAC delta=4", -2, +2),   # delta = 4
    ]

    # data[label][qp] = total_bytes
    data: dict = {}

    total_jobs = len(args.qp) * len(codecs) * (1 + len(sac_configs))
    job = 0

    print(f"\nEncode {total_jobs} jobs ...")
    for qp in args.qp:
        for codec, codec_name in codecs:
            # Traditional
            lbl_trad = f"{codec_name} Traditional"
            if lbl_trad not in data:
                data[lbl_trad] = {}
            job += 1
            tag = f"trad_{codec}_{qp}"
            print(f"[{job}/{total_jobs}] {lbl_trad}  QP{qp}")
            data[lbl_trad][qp] = _encode_trad(
                frames_bgr, codec, qp, work_dir, args.fps, args.preset, tag)

            # SAC variants
            for suffix, sac_label, roi_off, non_off in sac_configs:
                lbl = f"{codec_name} {sac_label}"
                if lbl not in data:
                    data[lbl] = {}
                job += 1
                crf_roi = qp + roi_off
                crf_non = qp + non_off
                tag = f"sac{suffix}_{codec}_{qp}"
                print(f"[{job}/{total_jobs}] {lbl:<28} QP{qp}  "
                      f"(roi={crf_roi}, non={crf_non})")
                data[lbl][qp] = _encode_sac(
                    frames_bgr, roi_masks, codec, crf_roi, crf_non,
                    work_dir, args.fps, args.preset, tag)

    # 4. In bảng kết quả
    W = 10
    sep = "=" * (28 + W * len(args.qp))
    print("\n" + sep)
    print(f"{'BẢNG BITRATE (kbps)':^{len(sep)}}")
    print(sep)
    header = f"{'Config':<28}" + "".join(f"{'QP'+str(q):>{W}}" for q in args.qp)
    print(header)
    print("-" * len(sep))
    for lbl in [c[0] for c in CONFIGS]:
        if lbl not in data:
            continue
        row_vals = "".join(
            f"{bytes_to_kbps(data[lbl][q], n, args.fps):>{W}.0f}"
            for q in args.qp
        )
        print(f"{lbl:<28}{row_vals}")

    # So sánh từng delta vs delta=7
    for d_label, d_lbl_fmt in [("delta=6", "SAC delta=6"), ("delta=4", "SAC delta=4")]:
        print(f"\n── Δ Bitrate ({d_label} − delta=7, kbps / %) ──")
        for codec_name in ["H.264", "H.265"]:
            lbl_d7 = f"{codec_name} SAC delta=7"
            lbl_dx = f"{codec_name} {d_lbl_fmt}"
            if lbl_d7 not in data or lbl_dx not in data:
                continue
            print(f"\n  {codec_name}:")
            for qp in args.qp:
                d7 = bytes_to_kbps(data[lbl_d7][qp], n, args.fps)
                dx = bytes_to_kbps(data[lbl_dx][qp], n, args.fps)
                diff = dx - d7
                pct  = diff / d7 * 100
                print(f"    QP{qp}: delta=7={d7:7.0f}  {d_label}={dx:7.0f}  "
                      f"diff={diff:+7.0f} kbps ({pct:+.1f}%)")

    # So sánh delta=6 vs delta=4
    print(f"\n── Δ Bitrate (delta=6 − delta=4, kbps / %) ──")
    for codec_name in ["H.264", "H.265"]:
        lbl_d6 = f"{codec_name} SAC delta=6"
        lbl_d4 = f"{codec_name} SAC delta=4"
        if lbl_d6 not in data or lbl_d4 not in data:
            continue
        print(f"\n  {codec_name}:")
        for qp in args.qp:
            d6 = bytes_to_kbps(data[lbl_d6][qp], n, args.fps)
            d4 = bytes_to_kbps(data[lbl_d4][qp], n, args.fps)
            diff = d6 - d4
            pct  = diff / d4 * 100
            print(f"    QP{qp}: delta=6={d6:7.0f}  delta=4={d4:7.0f}  "
                  f"diff={diff:+7.0f} kbps ({pct:+.1f}%)")
    print(sep)

    # 5. Save CSV
    csv_path = out_dir / "bitrate_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["config"] + [f"QP{q}_kbps" for q in args.qp]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for lbl in [c[0] for c in CONFIGS]:
            if lbl not in data:
                continue
            row = {"config": lbl}
            row.update({f"QP{q}_kbps": f"{bytes_to_kbps(data[lbl][q], n, args.fps):.1f}"
                        for q in args.qp})
            w.writerow(row)
    print(f"\nCSV  : {csv_path}")

    # 6. Plot
    import shutil
    _plot(args.qp, data, out_dir / "bitrate_delta_comparison.png", n, args.fps)
    shutil.rmtree(str(work_dir), ignore_errors=True)
    print(f"Outputs: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="So sánh bitrate: Traditional vs SAC delta=7 vs SAC delta=4")
    p.add_argument("--qp", type=int, nargs="+", default=DEFAULT_QPS,
                   help="Danh sách QP traditional (default: 22 25 28 31 34)")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--preset", default="medium",
                   choices=["ultrafast", "superfast", "veryfast", "faster",
                            "fast", "medium", "slow", "slower", "veryslow"])
    p.add_argument("--model-path", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
