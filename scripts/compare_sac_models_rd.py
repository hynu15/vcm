#!/usr/bin/env python3
"""So sánh RD giữa H.265 baseline, SAC-CCNet_4class và SAC-PIDNet_4class.

Dựa trên run_segmentation_rd_pipeline.py.

Logic:
  - PIDNet_L_4class là evaluator CỐ ĐỊNH để tính mIoU cho cả 3 phương pháp
  - CCNet_4class tạo ROI mask → SAC-CCNet encoding
  - PIDNet_L_4class tạo ROI mask → SAC-PIDNet encoding (dùng lại evaluator)
  - H.265 baseline: encode nguyên frame với crf_trad = (crf_roi+crf_non)//2
  - Bitrate SAC = roi.mp4 + nonroi.mp4 (tổng 2 stream)
  - mIoU đo trên decoded video bởi evaluator (PIDNet)

Chạy từ thư mục scripts/:
  conda activate sac
  python compare_sac_models_rd.py --num-frames 30 --split test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.interpolate import PchipInterpolator
from torchvision import transforms

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

PROJECT_ROOT    = _SCRIPTS.parent
IMAGE_ROOT_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
LABEL_ROOT_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "val"
IMAGE_ROOT_TEST = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "test"
LABEL_ROOT_TEST = PROJECT_ROOT / "data" / "gt_4class" / "test"

CCNET_CKPT_DEFAULT  = PROJECT_ROOT / "models" / "best_ccnet_4class.pth"
PIDNET_CKPT_DEFAULT = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sac_models_comparison"

NON_ROI_IDS      = [11, 12, 13, 14, 15, 16, 21, 22, 23]
SKY_IDS          = [23]
CONSTRUCTION_IDS = [11, 12, 13, 14, 15, 16]
NATURE_IDS       = [21, 22]
CLASS_NAMES_4    = ["ROI", "sky", "construction", "nature"]


@dataclass(frozen=True)
class OperatingPoint:
    label: str
    crf_trad: int   # CRF cho H.265 baseline  ← tùy chỉnh độc lập
    crf_roi: int    # CRF cho SAC ROI stream
    crf_non: int    # CRF cho SAC non-ROI stream


# ─── Tùy chỉnh 4 operating points tại đây ────────────────────────────────────
#   label   : tên hiển thị trên RD curve
#   crf_trad: CRF của H.265 baseline  (4 giá trị độc lập)
#   crf_roi : CRF stream ROI của SAC
#   crf_non : CRF stream non-ROI của SAC
REQUESTED_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("OP1", crf_trad=22, crf_roi=22, crf_non=22),
    OperatingPoint("OP2", crf_trad=27, crf_roi=27, crf_non=27),
    OperatingPoint("OP3", crf_trad=32, crf_roi=32, crf_non=32),
    OperatingPoint("OP4", crf_trad=37, crf_roi=37, crf_non=37),
)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (giống run_segmentation_rd_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def macroblock_align_filter(mask_2d: np.ndarray, block_size: int = 16) -> np.ndarray:
    h, w = mask_2d.shape
    ph = (h + block_size - 1) // block_size * block_size
    pw = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((ph, pw), dtype=mask_2d.dtype)
    padded[:h, :w] = mask_2d
    blocks = padded.reshape(ph // block_size, block_size, pw // block_size, block_size)
    roi_max = blocks.max(axis=(1, 3))
    aligned = np.repeat(np.repeat(roi_max, block_size, axis=0), block_size, axis=1)
    return aligned[:h, :w]


def run_ffmpeg(command: Sequence[str], step_name: str) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed at {step_name}:\n{result.stderr}")


def list_eval_pairs(
    image_root: Path,
    label_root: Path,
    num_frames: int,
    city_filter: Optional[List[str]] = None,
) -> List[Tuple[Path, Optional[Path]]]:
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for city in sorted(os.listdir(image_root)):
        if city_filter is not None and city not in city_filter:
            continue
        image_city_dir = image_root / city
        label_city_dir = label_root / city
        if not image_city_dir.is_dir():
            continue
        for image_path in sorted(image_city_dir.glob("*_leftImg8bit.png")):
            stem = image_path.name.replace("_leftImg8bit.png", "")
            label_path: Optional[Path] = None
            if label_city_dir.is_dir():
                cand = label_city_dir / f"{stem}_gtFine_4class.png"
                if cand.is_file():
                    label_path = cand
            pairs.append((image_path, label_path))
            if len(pairs) >= num_frames:
                return pairs
    return pairs


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def to_np_label(path: Optional[Path]) -> np.ndarray:
    if path is None:
        return np.zeros((512, 1024), dtype=np.uint8)
    return np.array(Image.open(path), dtype=np.uint8)


def read_video_frames(video_path: Path, expected_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    while len(frames) < expected_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def bitrate_mbps(video_path: Path, duration_sec: float) -> float:
    return (video_path.stat().st_size * 8.0 / duration_sec) / 1_000_000.0 if duration_sec > 0 else 0.0


def mean_iou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    num_classes: int = 4,
) -> Tuple[float, Dict[int, float]]:
    class_scores: Dict[int, float] = {}
    for cls in range(num_classes):
        inter = sum(int(np.logical_and(p == cls, g == cls).sum())
                    for p, g in zip(pred_masks, gt_masks))
        union = sum(int(np.logical_or(p == cls, g == cls).sum())
                    for p, g in zip(pred_masks, gt_masks))
        class_scores[cls] = float(inter / union) if union > 0 else 0.0
    return float(np.mean(list(class_scores.values()))), class_scores


def bd_rate_and_acc(
    baseline_df: pd.DataFrame,
    propose_df: pd.DataFrame,
) -> Tuple[float, float, Tuple[float, float]]:
    """BD-Rate và BD-Accuracy bằng PCHIP (giống run_segmentation_rd_pipeline.py)."""
    bl = baseline_df.groupby("bitrate_kbps", as_index=False)[["bitrate_kbps", "accuracy_mean"]]\
                    .mean().sort_values("bitrate_kbps").reset_index(drop=True)
    pr = propose_df.groupby("bitrate_kbps", as_index=False)[["bitrate_kbps", "accuracy_mean"]]\
                   .mean().sort_values("bitrate_kbps").reset_index(drop=True)

    x1, y1 = np.log(bl["bitrate_kbps"].to_numpy(float)), bl["accuracy_mean"].to_numpy(float)
    x2, y2 = np.log(pr["bitrate_kbps"].to_numpy(float)), pr["accuracy_mean"].to_numpy(float)

    def enforce_mono(x, y):
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        xk, yk = [x[0]], [y[0]]
        for ix, iy in zip(x[1:], y[1:]):
            if ix > xk[-1] and iy > yk[-1]:
                xk.append(ix); yk.append(iy)
        return np.array(xk), np.array(yk)

    x1, y1 = enforce_mono(x1, y1)
    x2, y2 = enforce_mono(x2, y2)

    acc_min = max(float(y1.min()), float(y2.min()))
    acc_max = min(float(y1.max()), float(y2.max()))
    if not np.isfinite(acc_min) or acc_min >= acc_max:
        raise RuntimeError("No valid metric overlap for BD calculation")

    acc_grid   = np.linspace(acc_min, acc_max, 200)
    bl_log_br  = PchipInterpolator(y1, x1)(acc_grid)
    pr_log_br  = PchipInterpolator(y2, x2)(acc_grid)
    bd_rate    = float((np.exp(np.trapz(pr_log_br - bl_log_br, acc_grid) / (acc_max - acc_min)) - 1) * 100)

    rate_min = max(float(x1.min()), float(x2.min()))
    rate_max = min(float(x1.max()), float(x2.max()))
    rate_grid = np.linspace(rate_min, rate_max, 200)
    bd_acc = float(np.trapz(
        PchipInterpolator(x2, y2)(rate_grid) - PchipInterpolator(x1, y1)(rate_grid),
        rate_grid,
    ) / (rate_max - rate_min))

    return bd_rate, bd_acc, (float(np.exp(rate_min)), float(np.exp(rate_max)))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_ckpt(path: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(path), map_location=device)
    return ckpt


def load_pidnet_l_4class(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from train_pidnet_l import build_pidnet_l
    ckpt = _load_ckpt(ckpt_path, device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = build_pidnet_l(num_classes=4).to(device)
    model.load_state_dict(state_dict)
    if hasattr(model, "augment"):
        model.augment = False
    model.eval()
    return model


def load_ccnet_4class(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from new_feature.ccnet_4class import CCNet4Class
    ckpt = _load_ckpt(ckpt_path, device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = CCNet4Class(num_classes=4).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ROI mask và frame split (giống run_segmentation_rd_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_roi_masks(
    model: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose,
    images: List[np.ndarray],
) -> List[np.ndarray]:
    roi_masks: List[np.ndarray] = []
    for frame in images:
        h, w = frame.shape[:2]
        inp = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)
        mask = torch.argmax(pred, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_masks.append(macroblock_align_filter((mask == 0).astype(np.uint8), 16))
    return roi_masks


def build_split_frames(
    images: List[np.ndarray],
    roi_masks: List[np.ndarray],
    frame_dir: Path,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, (frame, roi_mask) in enumerate(zip(images, roi_masks)):
        roi_255 = (roi_mask * 255).astype(np.uint8)
        non_255 = 255 - roi_255
        roi_img = cv2.bitwise_and(frame, frame, mask=roi_255)
        non_img = cv2.bitwise_and(frame, frame, mask=non_255)
        cv2.imwrite(str(frame_dir / f"frame_{idx:04d}_orig.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_dir / f"frame_{idx:04d}_roi.png"),
                    cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_dir / f"frame_{idx:04d}_non.png"),
                    cv2.cvtColor(non_img, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# Encoding (giống run_segmentation_rd_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def encode_h265_baseline(
    frame_dir: Path, out_dir: Path, fps: int, crf: int, preset: str
) -> Path:
    vid = out_dir / "h265_baseline.mp4"
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d_orig.png"),
        "-c:v", "libx265", "-crf", str(crf), "-preset", preset, str(vid),
    ], f"h265_baseline crf={crf}")
    return vid


def encode_sac_streams(
    frame_dir: Path, out_dir: Path, prefix: str,
    fps: int, crf_roi: int, crf_non: int, crf_trad: int, preset: str,
) -> Tuple[Path, Path, Path]:
    """Trả về (roi.mp4, nonroi.mp4, sac_merged.mp4)."""
    roi_vid = out_dir / f"{prefix}_roi.mp4"
    non_vid = out_dir / f"{prefix}_nonroi.mp4"
    sac_vid = out_dir / f"{prefix}_sac.mp4"
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d_roi.png"),
        "-c:v", "libx265", "-crf", str(crf_roi), "-preset", preset, str(roi_vid),
    ], f"{prefix} roi crf={crf_roi}")
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d_non.png"),
        "-c:v", "libx265", "-crf", str(crf_non), "-preset", preset, str(non_vid),
    ], f"{prefix} nonroi crf={crf_non}")
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(roi_vid), "-i", str(non_vid),
        "-filter_complex", "[0:v][1:v]blend=all_mode=addition",
        "-c:v", "libx265", "-crf", str(crf_trad), "-preset", preset, str(sac_vid),
    ], f"{prefix} merge")
    return roi_vid, non_vid, sac_vid


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (giống run_segmentation_rd_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def predict_masks(
    model: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose,
    frames: List[np.ndarray],
) -> List[np.ndarray]:
    preds: List[np.ndarray] = []
    for frame in frames:
        h, w = frame.shape[:2]
        inp = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
        mask = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        preds.append(mask)
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# RD plot (3 curves: H.265, SAC-CCNet, SAC-PIDNet)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rd_curves(
    h265_df: pd.DataFrame,
    ccnet_df: pd.DataFrame,
    pidnet_df: pd.DataFrame,
    output_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib không khả dụng, bỏ qua plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    for df, label, fmt, color in [
        (h265_df,   "H.265 baseline",     "o-",  "#4c72b0"),
        (ccnet_df,  "SAC-CCNet_4class",   "s--", "#dd8452"),
        (pidnet_df, "SAC-PIDNet_4class",  "^--", "#55a868"),
    ]:
        df_s = df.sort_values("bitrate_kbps")
        ax.plot(df_s["bitrate_kbps"], df_s["accuracy_mean"],
                fmt, linewidth=2.5, markersize=7, color=color, label=label)
        for _, row in df_s.iterrows():
            ax.annotate(
                f"QP={row['qp']}",
                (row["bitrate_kbps"], row["accuracy_mean"]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=7,
            )
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("iIoU (ROI, class 0)")
    ax.set_title("Rate–iIoU RD Curve: H.265 vs SAC-CCNet vs SAC-PIDNet\n(evaluated by PIDNet_L_4class)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_png))
    plt.close(fig)
    print(f"RD curve saved: {output_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="So sánh H.265 vs SAC-CCNet_4class vs SAC-PIDNet_4class "
                    "(evaluator cố định: PIDNet_L_4class)"
    )
    parser.add_argument("-n", "--num-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", type=str, default="slow")
    parser.add_argument("--split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--ccnet-ckpt", type=str, default=None,
                        help="CCNet_4class checkpoint (mặc định: models/best_ccnet_4class.pth)")
    parser.add_argument("--pidnet-ckpt", type=str, default=None,
                        help="PIDNet_L_4class checkpoint (mặc định: models/best_pidnet_l_4class.pth)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cities", type=str, default=None,
                        help="City filter cách nhau dấu phẩy, ví dụ: strasbourg,ulm")
    parser.add_argument("--crf-points", type=str, default=None,
                        help="Override REQUESTED_POINTS từ CLI: "
                             "format 'label:trad:roi:non,...' "
                             "vd 'OP1:22:19:26,OP2:27:24:31,OP3:32:29:36,OP4:37:34:41'")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    ccnet_ckpt  = Path(args.ccnet_ckpt)  if args.ccnet_ckpt  else CCNET_CKPT_DEFAULT
    pidnet_ckpt = Path(args.pidnet_ckpt) if args.pidnet_ckpt else PIDNET_CKPT_DEFAULT
    for p in (ccnet_ckpt, pidnet_ckpt):
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint không tìm thấy: {p}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg không tìm thấy trong PATH")

    if args.crf_points:
        op_points: Sequence[OperatingPoint] = []
        for entry in args.crf_points.split(","):
            parts = entry.strip().split(":")
            if len(parts) != 4:
                raise ValueError(
                    f"Format sai: '{entry}'. Cần 'label:trad:roi:non', "
                    "vd 'OP1:22:19:26'"
                )
            lbl, t, r, n = parts
            op_points.append(OperatingPoint(lbl, int(t), int(r), int(n)))
        print("Custom operating points:")
        for o in op_points:
            print(f"  {o.label}: crf_trad={o.crf_trad}  crf_roi={o.crf_roi}  crf_non={o.crf_non}")
    else:
        op_points = REQUESTED_POINTS

    image_root = IMAGE_ROOT_TEST if args.split == "test" else IMAGE_ROOT_VAL
    label_root = LABEL_ROOT_TEST if args.split == "test" else LABEL_ROOT_VAL
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root không tồn tại: {image_root}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = (Path(args.output_dir) if args.output_dir
               else OUTPUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    city_filter = [c.strip() for c in args.cities.split(",")] if args.cities else None
    if city_filter:
        print(f"City filter: {city_filter}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading PIDNet_L_4class (evaluator + SAC-PIDNet mask): {pidnet_ckpt.name}")
    pidnet_model = load_pidnet_l_4class(pidnet_ckpt, device)
    print(f"Loading CCNet_4class (SAC-CCNet mask):                   {ccnet_ckpt.name}")
    ccnet_model = load_ccnet_4class(ccnet_ckpt, device)
    transform = build_transform()

    pairs = list_eval_pairs(image_root, label_root, args.num_frames, city_filter)
    if not pairs:
        raise RuntimeError(f"Không tìm thấy ảnh trong {image_root}")

    images = [np.array(Image.open(p).convert("RGB")) for p, _ in pairs]
    gts    = [to_np_label(lp) for _, lp in pairs]

    valid_gt_idx = [i for i, g in enumerate(gts) if int(g.max()) > 0]
    skip_miou    = len(valid_gt_idx) == 0
    if skip_miou:
        print("WARNING: Không có GT label hợp lệ → bỏ qua mIoU.")
    else:
        print(f"GT labels: {len(valid_gt_idx)}/{len(gts)} frame hợp lệ")

    duration_sec = len(images) / float(args.fps)
    print(f"Frames: {len(images)}  duration: {duration_sec:.2f}s")

    # Tạo ROI masks
    print("\n[CCNet_4class]  Tạo ROI masks ...")
    ccnet_masks  = build_roi_masks(ccnet_model,  device, transform, images)
    ccnet_ratios = [m.mean() * 100 for m in ccnet_masks]
    print(f"  ROI ratio: mean={np.mean(ccnet_ratios):.1f}%  "
          f"min={np.min(ccnet_ratios):.1f}%  max={np.max(ccnet_ratios):.1f}%")

    print("[PIDNet_L_4class] Tạo ROI masks ...")
    pidnet_masks  = build_roi_masks(pidnet_model, device, transform, images)
    pidnet_ratios = [m.mean() * 100 for m in pidnet_masks]
    print(f"  ROI ratio: mean={np.mean(pidnet_ratios):.1f}%  "
          f"min={np.min(pidnet_ratios):.1f}%  max={np.max(pidnet_ratios):.1f}%")

    # Ghi split frames
    frame_dir_ccnet  = run_dir / "tmp_frames_ccnet"
    frame_dir_pidnet = run_dir / "tmp_frames_pidnet"
    print("\nGhi split frames ...")
    build_split_frames(images, ccnet_masks,  frame_dir_ccnet)
    build_split_frames(images, pidnet_masks, frame_dir_pidnet)

    # Kết quả tổng hợp
    h265_rows:   List[Dict] = []
    ccnet_rows:  List[Dict] = []
    pidnet_rows: List[Dict] = []
    detail_rows: List[Dict] = []

    for op in op_points:
        op_dir = run_dir / f"op_{op.label}"
        op_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n── {op.label}  "
              f"(crf_trad={op.crf_trad} | crf_roi={op.crf_roi}, crf_non={op.crf_non}) ──")

        # ── H.265 baseline ──
        print(f"  H.265 baseline (crf={op.crf_trad}) ...")
        h265_vid = encode_h265_baseline(frame_dir_ccnet, op_dir, args.fps, op.crf_trad, args.preset)
        h265_br  = bitrate_mbps(h265_vid, duration_sec)

        # ── SAC-CCNet ──
        print(f"  SAC-CCNet (roi={op.crf_roi}, non={op.crf_non}) ...")
        roi_c, non_c, sac_c = encode_sac_streams(
            frame_dir_ccnet, op_dir, "ccnet",
            args.fps, op.crf_roi, op.crf_non, op.crf_trad, args.preset)
        ccnet_br = bitrate_mbps(roi_c, duration_sec) + bitrate_mbps(non_c, duration_sec)

        # ── SAC-PIDNet ──
        print(f"  SAC-PIDNet (roi={op.crf_roi}, non={op.crf_non}) ...")
        roi_p, non_p, sac_p = encode_sac_streams(
            frame_dir_pidnet, op_dir, "pidnet",
            args.fps, op.crf_roi, op.crf_non, op.crf_trad, args.preset)
        pidnet_br = bitrate_mbps(roi_p, duration_sec) + bitrate_mbps(non_p, duration_sec)

        # ── mIoU (evaluator = PIDNet_L_4class) ──
        if skip_miou:
            h265_miou, ccnet_miou, pidnet_miou = float("nan"), float("nan"), float("nan")
            h265_iiou, ccnet_iiou, pidnet_iiou = float("nan"), float("nan"), float("nan")
            h265_class_iou, ccnet_class_iou, pidnet_class_iou = {}, {}, {}
        else:
            valid_gts = [gts[i] for i in valid_gt_idx]

            h265_frames  = read_video_frames(h265_vid, len(images))
            sac_c_frames = read_video_frames(sac_c,    len(images))
            sac_p_frames = read_video_frames(sac_p,    len(images))

            for name, frames in [("h265", h265_frames), ("sac_c", sac_c_frames), ("sac_p", sac_p_frames)]:
                if len(frames) != len(images):
                    raise RuntimeError(f"Decoded frame count mismatch for {name} at QP={op.label}")

            print("  Evaluating mIoU (PIDNet_L_4class) ...")
            h265_preds  = predict_masks(pidnet_model, device, transform,
                                        [h265_frames[i]  for i in valid_gt_idx])
            ccnet_preds = predict_masks(pidnet_model, device, transform,
                                        [sac_c_frames[i] for i in valid_gt_idx])
            pidnet_preds= predict_masks(pidnet_model, device, transform,
                                        [sac_p_frames[i] for i in valid_gt_idx])

            h265_miou,   h265_class_iou   = mean_iou(h265_preds,   valid_gts, 4)
            ccnet_miou,  ccnet_class_iou  = mean_iou(ccnet_preds,  valid_gts, 4)
            pidnet_miou, pidnet_class_iou = mean_iou(pidnet_preds, valid_gts, 4)
            # iIoU = IoU của class 0 (ROI) — chỉ số chính được dùng để vẽ RD curve
            h265_iiou   = float(h265_class_iou.get(0,   0.0))
            ccnet_iiou  = float(ccnet_class_iou.get(0,  0.0))
            pidnet_iiou = float(pidnet_class_iou.get(0, 0.0))

        # ── Print per-op summary ──
        print(f"  H.265:    {h265_br:.4f} Mbps  |  iIoU={h265_iiou:.4f}  mIoU={h265_miou:.4f}")
        print(f"  CCNet:    {ccnet_br:.4f} Mbps  |  iIoU={ccnet_iiou:.4f}  mIoU={ccnet_miou:.4f}"
              f"  (Δbr={((ccnet_br-h265_br)/h265_br*100):+.1f}%,"
              f" ΔiIoU={ccnet_iiou-h265_iiou:+.4f})")
        print(f"  PIDNet:   {pidnet_br:.4f} Mbps  |  iIoU={pidnet_iiou:.4f}  mIoU={pidnet_miou:.4f}"
              f"  (Δbr={((pidnet_br-h265_br)/h265_br*100):+.1f}%,"
              f" ΔiIoU={pidnet_iiou-h265_iiou:+.4f})")
        if not skip_miou:
            print("  Class IoU (H.265 / CCNet / PIDNet):")
            for c, cname in enumerate(CLASS_NAMES_4):
                print(f"    {cname:14s}: "
                      f"{h265_class_iou.get(c,0):.4f} / "
                      f"{ccnet_class_iou.get(c,0):.4f} / "
                      f"{pidnet_class_iou.get(c,0):.4f}")

        # ── Lưu rows cho BD (accuracy_mean = iIoU = class 0 ROI) ──
        h265_rows.append({"qp": op.label,
                          "bitrate_kbps": h265_br * 1000,
                          "accuracy_mean": h265_iiou})
        ccnet_rows.append({"qp": op.label,
                           "bitrate_kbps": ccnet_br * 1000,
                           "accuracy_mean": ccnet_iiou})
        pidnet_rows.append({"qp": op.label,
                            "bitrate_kbps": pidnet_br * 1000,
                            "accuracy_mean": pidnet_iiou})

        detail_rows.append({
            "qp":                       op.label,
            "crf_roi":                  op.crf_roi,
            "crf_non":                  op.crf_non,
            "crf_trad":                 op.crf_trad,
            # bitrate
            "h265_bitrate_mbps":        round(h265_br,   6),
            "sac_ccnet_bitrate_mbps":   round(ccnet_br,  6),
            "sac_pidnet_bitrate_mbps":  round(pidnet_br, 6),
            "sac_ccnet_bitrate_kbps":   round(ccnet_br  * 1000, 3),
            "sac_pidnet_bitrate_kbps":  round(pidnet_br * 1000, 3),
            "sac_ccnet_delta_br_pct":   round((ccnet_br  - h265_br) / h265_br * 100, 2),
            "sac_pidnet_delta_br_pct":  round((pidnet_br - h265_br) / h265_br * 100, 2),
            # iIoU (chỉ số chính — IoU class 0 = ROI)
            "h265_iiou":                round(h265_iiou,   4),
            "sac_ccnet_iiou":           round(ccnet_iiou,  4),
            "sac_pidnet_iiou":          round(pidnet_iiou, 4),
            "sac_ccnet_delta_iiou":     round(ccnet_iiou  - h265_iiou, 4),
            "sac_pidnet_delta_iiou":    round(pidnet_iiou - h265_iiou, 4),
            # mIoU (tham khảo)
            "h265_miou":                round(h265_miou,   4),
            "sac_ccnet_miou":           round(ccnet_miou,  4),
            "sac_pidnet_miou":          round(pidnet_miou, 4),
            "sac_ccnet_delta_miou":     round(ccnet_miou  - h265_miou, 4),
            "sac_pidnet_delta_miou":    round(pidnet_miou - h265_miou, 4),
            # per-class IoU
            "h265_class_iou":           json.dumps(h265_class_iou,   ensure_ascii=False),
            "sac_ccnet_class_iou":      json.dumps(ccnet_class_iou,  ensure_ascii=False),
            "sac_pidnet_class_iou":     json.dumps(pidnet_class_iou, ensure_ascii=False),
        })

    # ── DataFrames ──
    h265_df   = pd.DataFrame(h265_rows).sort_values("qp")
    ccnet_df  = pd.DataFrame(ccnet_rows).sort_values("qp")
    pidnet_df = pd.DataFrame(pidnet_rows).sort_values("qp")
    detail_df = pd.DataFrame(detail_rows).sort_values("qp")

    detail_csv = run_dir / "combined_rd_detail.csv"
    detail_df.to_csv(detail_csv, index=False)

    # ── BD metrics ──
    bd_ccnet_rate = bd_ccnet_acc = bd_pidnet_rate = bd_pidnet_acc = None
    bd_error_ccnet = bd_error_pidnet = None
    if not skip_miou:
        try:
            bd_ccnet_rate, bd_ccnet_acc, ol_c = bd_rate_and_acc(h265_df, ccnet_df)
        except RuntimeError as e:
            bd_error_ccnet = str(e)
        try:
            bd_pidnet_rate, bd_pidnet_acc, ol_p = bd_rate_and_acc(h265_df, pidnet_df)
        except RuntimeError as e:
            bd_error_pidnet = str(e)
        plot_rd_curves(h265_df, ccnet_df, pidnet_df, run_dir / "rd_curve_iiou.png")

    summary = {
        "split":                        args.split,
        "num_frames":                   len(images),
        "evaluator_model":              "pidnet_l_4class",
        "ccnet_ckpt":                   str(ccnet_ckpt),
        "pidnet_ckpt":                  str(pidnet_ckpt),
        "operating_points": [
            {"qp": op.label, "crf_roi": op.crf_roi,
             "crf_non": op.crf_non, "crf_trad": op.crf_trad}
            for op in op_points
        ],
        "mean_h265_bitrate_mbps":       round(float(detail_df["h265_bitrate_mbps"].mean()), 4),
        "mean_sac_ccnet_bitrate_mbps":  round(float(detail_df["sac_ccnet_bitrate_mbps"].mean()), 4),
        "mean_sac_pidnet_bitrate_mbps": round(float(detail_df["sac_pidnet_bitrate_mbps"].mean()), 4),
        "mean_sac_ccnet_delta_br_pct":  round(float(detail_df["sac_ccnet_delta_br_pct"].mean()), 2),
        "mean_sac_pidnet_delta_br_pct": round(float(detail_df["sac_pidnet_delta_br_pct"].mean()), 2),
        "mean_h265_iiou":               round(float(detail_df["h265_iiou"].mean()), 4),
        "mean_sac_ccnet_iiou":          round(float(detail_df["sac_ccnet_iiou"].mean()), 4),
        "mean_sac_pidnet_iiou":         round(float(detail_df["sac_pidnet_iiou"].mean()), 4),
        "mean_h265_miou":               round(float(detail_df["h265_miou"].mean()), 4),
        "mean_sac_ccnet_miou":          round(float(detail_df["sac_ccnet_miou"].mean()), 4),
        "mean_sac_pidnet_miou":         round(float(detail_df["sac_pidnet_miou"].mean()), 4),
        # BD metrics: accuracy_mean = iIoU
        "bd_rate_ccnet_vs_h265_pct":    bd_ccnet_rate,
        "bd_iiou_ccnet_vs_h265":        bd_ccnet_acc,
        "bd_rate_pidnet_vs_h265_pct":   bd_pidnet_rate,
        "bd_iiou_pidnet_vs_h265":       bd_pidnet_acc,
        "bd_error_ccnet":               bd_error_ccnet,
        "bd_error_pidnet":              bd_error_pidnet,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Final print ──
    print("\n" + "=" * 72)
    print("COMBINED RD DETAIL TABLE (so sánh chính: bitrate vs iIoU):")
    print(detail_df[[
        "qp", "h265_bitrate_mbps", "sac_ccnet_bitrate_mbps", "sac_pidnet_bitrate_mbps",
        "sac_ccnet_delta_br_pct", "sac_pidnet_delta_br_pct",
        "h265_iiou", "sac_ccnet_iiou", "sac_pidnet_iiou",
        "sac_ccnet_delta_iiou", "sac_pidnet_delta_iiou",
    ]].to_string(index=False))

    print(f"\nMean bitrate  H.265={summary['mean_h265_bitrate_mbps']:.4f} Mbps  "
          f"CCNet={summary['mean_sac_ccnet_bitrate_mbps']:.4f} Mbps  "
          f"PIDNet={summary['mean_sac_pidnet_bitrate_mbps']:.4f} Mbps")
    print(f"Mean delta br CCNet={summary['mean_sac_ccnet_delta_br_pct']:+.1f}%  "
          f"PIDNet={summary['mean_sac_pidnet_delta_br_pct']:+.1f}%")
    print(f"Mean iIoU     H.265={summary['mean_h265_iiou']:.4f}  "
          f"CCNet={summary['mean_sac_ccnet_iiou']:.4f}  "
          f"PIDNet={summary['mean_sac_pidnet_iiou']:.4f}")
    if bd_ccnet_rate is not None:
        print(f"\nBD-Rate (CCNet vs H.265):  {bd_ccnet_rate:+.2f}%   "
              f"BD-iIoU: {bd_ccnet_acc:+.4f}")
    if bd_pidnet_rate is not None:
        print(f"BD-Rate (PIDNet vs H.265): {bd_pidnet_rate:+.2f}%   "
              f"BD-iIoU: {bd_pidnet_acc:+.4f}")
    if bd_error_ccnet:
        print(f"BD CCNet error: {bd_error_ccnet}")
    if bd_error_pidnet:
        print(f"BD PIDNet error: {bd_error_pidnet}")
    print(f"\nOutput dir: {run_dir}")

    if not args.keep_artifacts:
        shutil.rmtree(frame_dir_ccnet,  ignore_errors=True)
        shutil.rmtree(frame_dir_pidnet, ignore_errors=True)


if __name__ == "__main__":
    main()
