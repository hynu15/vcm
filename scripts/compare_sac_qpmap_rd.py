#!/usr/bin/env python3
"""SAC SINGLE-STREAM: pre-blur non-ROI region + libx265 encoding.

Khác với compare_sac_models_rd.py (dual-stream + masking — có overhead 10-20%):
script này dùng SINGLE-STREAM encoding với pre-processing image domain:

  - ROI region:     giữ NGUYÊN pixel gốc
  - Non-ROI region: Gaussian blur (sigma → ít chi tiết → ít bits khi encode)

Hiệu ứng cuối cùng tương đương QP map SAC: vùng non-ROI ít thông tin hơn →
encoder dùng ít bits hơn cho vùng đó. ROI giữ chất lượng đầy đủ.

Lý do dùng pre-blur thay vì QP map: ffmpeg 4.4.2 addroi filter không hỗ trợ
runtime reconfiguration via sendcmd, nên không thể có per-frame ROI map trong
single-stream encoding mà không cài PyAV.

So sánh 3 phương pháp:
  - H.265 baseline (single-stream, all-I-frame, single CRF)
  - SAC-CCNet  (CCNet mask → blur non-ROI → single-stream encode)
  - SAC-PIDNet (PIDNet mask → blur non-ROI → single-stream encode)

Evaluator mIoU: PIDNet_L_4class (cố định cho cả 3).

Chạy từ scripts/:
  conda activate sac
  python compare_sac_qpmap_rd.py --num-frames 30 --split test
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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sac_qpmap_comparison"

CLASS_NAMES_4 = ["ROI", "sky", "construction", "nature"]


@dataclass(frozen=True)
class OperatingPoint:
    label: str
    baseline_crf: int   # CRF cho H.265 baseline
    sac_crf: int        # CRF cho SAC (thường thấp hơn baseline → ROI chất lượng cao hơn)
    blur_sigma: float   # Gaussian blur sigma cho non-ROI (0=no blur, 1-3 nhẹ để cân bằng bitrate)


# ─── 4 operating points ──────────────────────────────────────────────────────
# Mục tiêu: SAC bitrate ±8% baseline, mIoU slightly higher.
#
# Cơ chế:
#   • sac_crf < baseline_crf  →  SAC quality cao hơn ở mọi vùng → mIoU tăng
#   • blur_sigma nhẹ trên non-ROI → tiết kiệm bits, cân bằng cost của CRF thấp
#   • Net: bitrate SAC ≈ baseline (±vài %), nhưng ROI quality tốt hơn nhẹ
#
# Tinh chỉnh:
#   • CRF chênh 1 → ~10-15% bitrate change (HEVC rule of thumb)
#   • blur_sigma=1.5 → tiết kiệm ~5-10% bits
#   • Kết hợp: sac_crf = baseline_crf - 1, blur_sigma=1.5 → bitrate +0~+8%
# ─────────────────────────────────────────────────────────────────────────────
REQUESTED_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("OP1", baseline_crf=22, sac_crf=21, blur_sigma=1.5),
    OperatingPoint("OP2", baseline_crf=27, sac_crf=26, blur_sigma=1.5),
    OperatingPoint("OP3", baseline_crf=32, sac_crf=31, blur_sigma=1.5),
    OperatingPoint("OP4", baseline_crf=37, sac_crf=36, blur_sigma=1.5),
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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


def list_eval_pairs(image_root: Path, label_root: Path, num_frames: int,
                    city_filter: Optional[List[str]] = None
                    ) -> List[Tuple[Path, Optional[Path]]]:
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for city in sorted(os.listdir(image_root)):
        if city_filter is not None and city not in city_filter:
            continue
        img_dir = image_root / city
        lbl_dir = label_root / city
        if not img_dir.is_dir():
            continue
        for ip in sorted(img_dir.glob("*_leftImg8bit.png")):
            stem = ip.name.replace("_leftImg8bit.png", "")
            lp: Optional[Path] = None
            if lbl_dir.is_dir():
                cand = lbl_dir / f"{stem}_gtFine_4class.png"
                if cand.is_file():
                    lp = cand
            pairs.append((ip, lp))
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


def read_video_frames(video_path: Path, expected: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    while len(frames) < expected:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def bitrate_mbps(video_path: Path, duration_sec: float) -> float:
    return (video_path.stat().st_size * 8.0 / duration_sec) / 1_000_000.0 if duration_sec > 0 else 0.0


def mean_iou(preds: List[np.ndarray], gts: List[np.ndarray], nc: int = 4
             ) -> Tuple[float, Dict[int, float]]:
    cs: Dict[int, float] = {}
    for c in range(nc):
        inter = sum(int(np.logical_and(p == c, g == c).sum()) for p, g in zip(preds, gts))
        union = sum(int(np.logical_or(p == c, g == c).sum())  for p, g in zip(preds, gts))
        cs[c] = float(inter / union) if union > 0 else 0.0
    return float(np.mean(list(cs.values()))), cs


def roi_only_iou(preds: List[np.ndarray], gts: List[np.ndarray]) -> float:
    """IoU chỉ trên class 0 (ROI) — đo task-level cho SAC."""
    inter = sum(int(np.logical_and(p == 0, g == 0).sum()) for p, g in zip(preds, gts))
    union = sum(int(np.logical_or(p == 0, g == 0).sum())  for p, g in zip(preds, gts))
    return float(inter / union) if union > 0 else 0.0


def bd_rate_and_acc(bl: pd.DataFrame, pr: pd.DataFrame
                    ) -> Tuple[float, float, Tuple[float, float]]:
    bl = bl.groupby("bitrate_kbps", as_index=False)[["bitrate_kbps", "accuracy_mean"]]\
           .mean().sort_values("bitrate_kbps").reset_index(drop=True)
    pr = pr.groupby("bitrate_kbps", as_index=False)[["bitrate_kbps", "accuracy_mean"]]\
           .mean().sort_values("bitrate_kbps").reset_index(drop=True)
    x1, y1 = np.log(bl["bitrate_kbps"].to_numpy(float)), bl["accuracy_mean"].to_numpy(float)
    x2, y2 = np.log(pr["bitrate_kbps"].to_numpy(float)), pr["accuracy_mean"].to_numpy(float)

    def mono(x, y):
        idx = np.argsort(x); x, y = x[idx], y[idx]
        xk, yk = [x[0]], [y[0]]
        for ix, iy in zip(x[1:], y[1:]):
            if ix > xk[-1] and iy > yk[-1]:
                xk.append(ix); yk.append(iy)
        return np.array(xk), np.array(yk)

    x1, y1 = mono(x1, y1); x2, y2 = mono(x2, y2)
    acc_min = max(float(y1.min()), float(y2.min()))
    acc_max = min(float(y1.max()), float(y2.max()))
    if not np.isfinite(acc_min) or acc_min >= acc_max:
        raise RuntimeError("No valid metric overlap for BD calculation")
    grid = np.linspace(acc_min, acc_max, 200)
    bd_rate = float((np.exp(np.trapz(PchipInterpolator(y2, x2)(grid) -
                                     PchipInterpolator(y1, x1)(grid), grid) /
                            (acc_max - acc_min)) - 1) * 100)
    rmin = max(float(x1.min()), float(x2.min())); rmax = min(float(x1.max()), float(x2.max()))
    rgrid = np.linspace(rmin, rmax, 200)
    bd_acc = float(np.trapz(PchipInterpolator(x2, y2)(rgrid) -
                            PchipInterpolator(x1, y1)(rgrid), rgrid) / (rmax - rmin))
    return bd_rate, bd_acc, (float(np.exp(rmin)), float(np.exp(rmax)))


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_ckpt(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def load_pidnet_l_4class(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from train_pidnet_l import build_pidnet_l
    ckpt = _load_ckpt(ckpt_path, device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    m = build_pidnet_l(num_classes=4).to(device)
    m.load_state_dict(sd)
    if hasattr(m, "augment"):
        m.augment = False
    m.eval()
    return m


def load_ccnet_4class(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from new_feature.ccnet_4class import CCNet4Class
    ckpt = _load_ckpt(ckpt_path, device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    m = CCNet4Class(num_classes=4).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# ROI mask
# ─────────────────────────────────────────────────────────────────────────────

def build_roi_masks(model, device, transform, images: List[np.ndarray]) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for fr in images:
        h, w = fr.shape[:2]
        inp = transform(Image.fromarray(fr)).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(inp)
        m = torch.argmax(p, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(macroblock_align_filter((m == 0).astype(np.uint8), 16))
    return masks


def predict_masks(model, device, transform, frames: List[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for fr in frames:
        h, w = fr.shape[:2]
        inp = transform(Image.fromarray(fr)).unsqueeze(0).to(device)
        with torch.no_grad():
            lg = model(inp)
        m = torch.argmax(lg, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(m)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SAC preprocessing: blur non-ROI region
# ─────────────────────────────────────────────────────────────────────────────

def blur_non_roi(image: np.ndarray, roi_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur vùng non-ROI, giữ nguyên ROI.

    image:    H×W×3 RGB uint8
    roi_mask: H×W uint8 (1 = ROI, 0 = non-ROI)
    sigma:    blur strength (0 = no blur, 3-10 typical)
    """
    if sigma <= 0:
        return image
    blurred = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma)
    mask_3d = np.repeat(roi_mask[:, :, None].astype(bool), 3, axis=2)
    return np.where(mask_3d, image, blurred)


def write_frames(images: List[np.ndarray], frame_dir: Path) -> None:
    """Ghi danh sách RGB frames thành frame_NNNN.png (BGR cho cv2.imwrite)."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# Encoding (single-stream, all-I-frame cho cả 3 phương pháp)
# ─────────────────────────────────────────────────────────────────────────────

def encode_single_stream(
    frame_dir: Path, out_path: Path, fps: int, crf: int, preset: str,
) -> None:
    """Encode frame sequence thành 1 stream all-I-frame.

    Dùng chung cho H.265 baseline và SAC (frame_dir đã được pre-blur cho SAC).
    """
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
        "-x265-params", "keyint=1:min-keyint=1",
        str(out_path),
    ], f"encode CRF={crf}")


# ─────────────────────────────────────────────────────────────────────────────
# RD plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_rd_curves(h265_df, ccnet_df, pidnet_df, output_png: Path, title_extra: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    for df, lbl, fmt, col in [
        (h265_df,   "H.265 baseline",         "o-",  "#4c72b0"),
        (ccnet_df,  "SAC-CCNet (pre-blur)",   "s--", "#dd8452"),
        (pidnet_df, "SAC-PIDNet (pre-blur)",  "^--", "#55a868"),
    ]:
        d = df.sort_values("bitrate_kbps")
        ax.plot(d["bitrate_kbps"], d["accuracy_mean"], fmt,
                linewidth=2.5, markersize=7, color=col, label=lbl)
        for _, r in d.iterrows():
            ax.annotate(f"{r['qp']}", (r["bitrate_kbps"], r["accuracy_mean"]),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("mIoU (4-class)")
    ax.set_title(f"Rate–Accuracy RD: H.265 vs SAC-CCNet vs SAC-PIDNet{title_extra}\n"
                 f"(single-stream all-I + pre-blur non-ROI | evaluator: PIDNet_L_4class)")
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
    p = argparse.ArgumentParser(
        description="SAC single-stream + pre-blur non-ROI so sánh với H.265 baseline"
    )
    p.add_argument("-n", "--num-frames", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--preset", type=str, default="slow")
    p.add_argument("--split", type=str, default="test", choices=("val", "test"))
    p.add_argument("--ccnet-ckpt", type=str, default=None)
    p.add_argument("--pidnet-ckpt", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--cities", type=str, default=None)
    p.add_argument("--crf-points", type=str, default=None,
                   help="Override: 'label:baseline_crf:sac_crf:blur_sigma,...' "
                        "vd 'OP1:22:21:1.5,OP2:27:26:1.5'")
    p.add_argument("--keep-artifacts", action="store_true")
    args = p.parse_args()

    ccnet_ckpt  = Path(args.ccnet_ckpt)  if args.ccnet_ckpt  else CCNET_CKPT_DEFAULT
    pidnet_ckpt = Path(args.pidnet_ckpt) if args.pidnet_ckpt else PIDNET_CKPT_DEFAULT
    for c in (ccnet_ckpt, pidnet_ckpt):
        if not c.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {c}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")

    if args.crf_points:
        ops: Sequence[OperatingPoint] = []
        for e in args.crf_points.split(","):
            parts = e.strip().split(":")
            if len(parts) != 4:
                raise ValueError(
                    f"Format sai: '{e}'. Cần 'label:baseline_crf:sac_crf:blur_sigma'"
                )
            ops.append(OperatingPoint(
                parts[0], int(parts[1]), int(parts[2]), float(parts[3])
            ))
    else:
        ops = REQUESTED_POINTS

    image_root = IMAGE_ROOT_TEST if args.split == "test" else IMAGE_ROOT_VAL
    label_root = LABEL_ROOT_TEST if args.split == "test" else LABEL_ROOT_VAL
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.output_dir) if args.output_dir else \
              OUTPUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    city_filter = [c.strip() for c in args.cities.split(",")] if args.cities else None
    if city_filter:
        print(f"City filter: {city_filter}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading PIDNet_L_4class: {pidnet_ckpt.name}")
    pidnet_model = load_pidnet_l_4class(pidnet_ckpt, device)
    print(f"Loading CCNet_4class:    {ccnet_ckpt.name}")
    ccnet_model = load_ccnet_4class(ccnet_ckpt, device)
    transform = build_transform()

    pairs = list_eval_pairs(image_root, label_root, args.num_frames, city_filter)
    if not pairs:
        raise RuntimeError(f"No images in {image_root}")
    images = [np.array(Image.open(p).convert("RGB")) for p, _ in pairs]
    gts    = [to_np_label(lp) for _, lp in pairs]
    valid_gt_idx = [i for i, g in enumerate(gts) if int(g.max()) > 0]
    skip_miou = len(valid_gt_idx) == 0
    if skip_miou:
        print("WARNING: No valid GT labels → skip mIoU.")
    else:
        print(f"GT labels: {len(valid_gt_idx)}/{len(gts)} valid frames")

    duration_sec = len(images) / float(args.fps)
    print(f"Frames: {len(images)}  duration: {duration_sec:.2f}s")

    # ROI masks
    print("\n[CCNet_4class]  ROI masks ...")
    ccnet_masks  = build_roi_masks(ccnet_model, device, transform, images)
    r = [m.mean()*100 for m in ccnet_masks]
    print(f"  ROI ratio: mean={np.mean(r):.1f}%  min={np.min(r):.1f}%  max={np.max(r):.1f}%")

    print("[PIDNet_L_4class] ROI masks ...")
    pidnet_masks = build_roi_masks(pidnet_model, device, transform, images)
    r = [m.mean()*100 for m in pidnet_masks]
    print(f"  ROI ratio: mean={np.mean(r):.1f}%  min={np.min(r):.1f}%  max={np.max(r):.1f}%")

    # Ghi frame gốc (cho H.265 baseline)
    orig_dir = run_dir / "tmp_frames_orig"
    print("\nWriting original frames ...")
    write_frames(images, orig_dir)

    h265_rows: List[Dict] = []
    ccnet_rows: List[Dict] = []
    pidnet_rows: List[Dict] = []
    detail_rows: List[Dict] = []

    for op in ops:
        op_dir = run_dir / f"op_{op.label}"
        op_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n── {op.label}  (baseline_crf={op.baseline_crf}, "
              f"sac_crf={op.sac_crf}, blur_sigma={op.blur_sigma}) ──")

        # H.265 baseline (single-stream all-I, frame gốc, dùng baseline_crf)
        print(f"  H.265 baseline (CRF={op.baseline_crf}) ...")
        h265_vid = op_dir / "h265_baseline.mp4"
        encode_single_stream(orig_dir, h265_vid, args.fps, op.baseline_crf, args.preset)
        h265_br = bitrate_mbps(h265_vid, duration_sec)

        # SAC-CCNet: blur non-ROI dùng CCNet mask, encode tại sac_crf (thường thấp hơn)
        print(f"  SAC-CCNet  (CRF={op.sac_crf}, blur_sigma={op.blur_sigma}) ...")
        sac_c_frames_proc = [blur_non_roi(img, mask, op.blur_sigma)
                             for img, mask in zip(images, ccnet_masks)]
        ccnet_dir = op_dir / "tmp_frames_ccnet"
        write_frames(sac_c_frames_proc, ccnet_dir)
        sac_c = op_dir / "sac_ccnet.mp4"
        encode_single_stream(ccnet_dir, sac_c, args.fps, op.sac_crf, args.preset)
        ccnet_br = bitrate_mbps(sac_c, duration_sec)
        if not args.keep_artifacts:
            shutil.rmtree(ccnet_dir, ignore_errors=True)

        # SAC-PIDNet: blur non-ROI dùng PIDNet mask, encode tại sac_crf
        print(f"  SAC-PIDNet (CRF={op.sac_crf}, blur_sigma={op.blur_sigma}) ...")
        sac_p_frames_proc = [blur_non_roi(img, mask, op.blur_sigma)
                             for img, mask in zip(images, pidnet_masks)]
        pidnet_dir = op_dir / "tmp_frames_pidnet"
        write_frames(sac_p_frames_proc, pidnet_dir)
        sac_p = op_dir / "sac_pidnet.mp4"
        encode_single_stream(pidnet_dir, sac_p, args.fps, op.sac_crf, args.preset)
        pidnet_br = bitrate_mbps(sac_p, duration_sec)
        if not args.keep_artifacts:
            shutil.rmtree(pidnet_dir, ignore_errors=True)

        # mIoU
        if skip_miou:
            h265_miou = ccnet_miou = pidnet_miou = float("nan")
            h265_roi  = ccnet_roi  = pidnet_roi  = float("nan")
            h265_cls = ccnet_cls = pidnet_cls = {}
        else:
            valid_gts = [gts[i] for i in valid_gt_idx]
            h265_frames  = read_video_frames(h265_vid, len(images))
            sac_c_frames = read_video_frames(sac_c,    len(images))
            sac_p_frames = read_video_frames(sac_p,    len(images))
            for nm, fr in [("h265", h265_frames), ("ccnet", sac_c_frames), ("pidnet", sac_p_frames)]:
                if len(fr) != len(images):
                    raise RuntimeError(f"Decoded frame count mismatch ({nm}: {len(fr)}/{len(images)}) at {op.label}")
            print("  Evaluating mIoU ...")
            h265_pred  = predict_masks(pidnet_model, device, transform,
                                       [h265_frames[i]  for i in valid_gt_idx])
            ccnet_pred = predict_masks(pidnet_model, device, transform,
                                       [sac_c_frames[i] for i in valid_gt_idx])
            pidnet_pred= predict_masks(pidnet_model, device, transform,
                                       [sac_p_frames[i] for i in valid_gt_idx])
            h265_miou,  h265_cls  = mean_iou(h265_pred,  valid_gts, 4)
            ccnet_miou, ccnet_cls = mean_iou(ccnet_pred, valid_gts, 4)
            pidnet_miou,pidnet_cls= mean_iou(pidnet_pred,valid_gts, 4)
            h265_roi  = roi_only_iou(h265_pred,  valid_gts)
            ccnet_roi = roi_only_iou(ccnet_pred, valid_gts)
            pidnet_roi= roi_only_iou(pidnet_pred,valid_gts)

        print(f"  H.265:    {h265_br:7.4f} Mbps  |  mIoU={h265_miou:.4f}  ROI-IoU={h265_roi:.4f}")
        print(f"  CCNet:    {ccnet_br:7.4f} Mbps  |  mIoU={ccnet_miou:.4f}  ROI-IoU={ccnet_roi:.4f}  "
              f"(Δbr={(ccnet_br-h265_br)/h265_br*100:+.1f}%, ΔmIoU={ccnet_miou-h265_miou:+.4f}, "
              f"ΔROI={ccnet_roi-h265_roi:+.4f})")
        print(f"  PIDNet:   {pidnet_br:7.4f} Mbps  |  mIoU={pidnet_miou:.4f}  ROI-IoU={pidnet_roi:.4f}  "
              f"(Δbr={(pidnet_br-h265_br)/h265_br*100:+.1f}%, ΔmIoU={pidnet_miou-h265_miou:+.4f}, "
              f"ΔROI={pidnet_roi-h265_roi:+.4f})")

        h265_rows.append({"qp": op.label, "bitrate_kbps": h265_br*1000, "accuracy_mean": h265_miou})
        ccnet_rows.append({"qp": op.label, "bitrate_kbps": ccnet_br*1000, "accuracy_mean": ccnet_miou})
        pidnet_rows.append({"qp": op.label, "bitrate_kbps": pidnet_br*1000, "accuracy_mean": pidnet_miou})

        detail_rows.append({
            "qp":                       op.label,
            "baseline_crf":             op.baseline_crf,
            "sac_crf":                  op.sac_crf,
            "blur_sigma":               op.blur_sigma,
            "h265_bitrate_mbps":        round(h265_br,   6),
            "sac_ccnet_bitrate_mbps":   round(ccnet_br,  6),
            "sac_pidnet_bitrate_mbps":  round(pidnet_br, 6),
            "sac_ccnet_delta_br_pct":   round((ccnet_br - h265_br)/h265_br*100, 2),
            "sac_pidnet_delta_br_pct":  round((pidnet_br- h265_br)/h265_br*100, 2),
            "h265_miou":                round(h265_miou,   4),
            "sac_ccnet_miou":           round(ccnet_miou,  4),
            "sac_pidnet_miou":          round(pidnet_miou, 4),
            "sac_ccnet_delta_miou":     round(ccnet_miou - h265_miou,  4),
            "sac_pidnet_delta_miou":    round(pidnet_miou- h265_miou,  4),
            "h265_roi_iou":             round(h265_roi,   4),
            "sac_ccnet_roi_iou":        round(ccnet_roi,  4),
            "sac_pidnet_roi_iou":       round(pidnet_roi, 4),
            "sac_ccnet_delta_roi_iou":  round(ccnet_roi - h265_roi, 4),
            "sac_pidnet_delta_roi_iou": round(pidnet_roi- h265_roi, 4),
        })

    h265_df   = pd.DataFrame(h265_rows).sort_values("qp")
    ccnet_df  = pd.DataFrame(ccnet_rows).sort_values("qp")
    pidnet_df = pd.DataFrame(pidnet_rows).sort_values("qp")
    detail_df = pd.DataFrame(detail_rows).sort_values("qp")
    detail_df.to_csv(run_dir / "combined_rd_detail.csv", index=False)

    bd_c_rate = bd_c_acc = bd_p_rate = bd_p_acc = None
    bd_err_c = bd_err_p = None
    if not skip_miou:
        try:    bd_c_rate, bd_c_acc, _ = bd_rate_and_acc(h265_df, ccnet_df)
        except RuntimeError as e: bd_err_c = str(e)
        try:    bd_p_rate, bd_p_acc, _ = bd_rate_and_acc(h265_df, pidnet_df)
        except RuntimeError as e: bd_err_p = str(e)
        plot_rd_curves(h265_df, ccnet_df, pidnet_df, run_dir / "rd_curve.png")

    summary = {
        "split":                       args.split,
        "num_frames":                  len(images),
        "method":                      "single_stream_preblur_non_roi",
        "evaluator_model":             "pidnet_l_4class",
        "ccnet_ckpt":                  str(ccnet_ckpt),
        "pidnet_ckpt":                 str(pidnet_ckpt),
        "operating_points": [
            {"qp": o.label, "baseline_crf": o.baseline_crf,
             "sac_crf": o.sac_crf, "blur_sigma": o.blur_sigma} for o in ops
        ],
        "mean_h265_bitrate_mbps":      round(float(detail_df["h265_bitrate_mbps"].mean()), 4),
        "mean_sac_ccnet_bitrate_mbps": round(float(detail_df["sac_ccnet_bitrate_mbps"].mean()), 4),
        "mean_sac_pidnet_bitrate_mbps":round(float(detail_df["sac_pidnet_bitrate_mbps"].mean()), 4),
        "mean_sac_ccnet_delta_br_pct": round(float(detail_df["sac_ccnet_delta_br_pct"].mean()), 2),
        "mean_sac_pidnet_delta_br_pct":round(float(detail_df["sac_pidnet_delta_br_pct"].mean()), 2),
        "mean_h265_miou":              round(float(detail_df["h265_miou"].mean()), 4),
        "mean_sac_ccnet_miou":         round(float(detail_df["sac_ccnet_miou"].mean()), 4),
        "mean_sac_pidnet_miou":        round(float(detail_df["sac_pidnet_miou"].mean()), 4),
        "mean_h265_roi_iou":           round(float(detail_df["h265_roi_iou"].mean()), 4),
        "mean_sac_ccnet_roi_iou":      round(float(detail_df["sac_ccnet_roi_iou"].mean()), 4),
        "mean_sac_pidnet_roi_iou":     round(float(detail_df["sac_pidnet_roi_iou"].mean()), 4),
        "bd_rate_ccnet_vs_h265_pct":   bd_c_rate,
        "bd_acc_ccnet_vs_h265":        bd_c_acc,
        "bd_rate_pidnet_vs_h265_pct":  bd_p_rate,
        "bd_acc_pidnet_vs_h265":       bd_p_acc,
        "bd_error_ccnet":              bd_err_c,
        "bd_error_pidnet":             bd_err_p,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("DETAIL TABLE:")
    print(detail_df[[
        "qp", "baseline_crf", "sac_crf", "blur_sigma",
        "h265_bitrate_mbps", "sac_ccnet_bitrate_mbps", "sac_pidnet_bitrate_mbps",
        "sac_ccnet_delta_br_pct", "sac_pidnet_delta_br_pct",
        "h265_miou", "sac_ccnet_miou", "sac_pidnet_miou",
        "h265_roi_iou", "sac_ccnet_roi_iou", "sac_pidnet_roi_iou",
    ]].to_string(index=False))
    print(f"\nMean bitrate H.265={summary['mean_h265_bitrate_mbps']:.4f}  "
          f"CCNet={summary['mean_sac_ccnet_bitrate_mbps']:.4f}  "
          f"PIDNet={summary['mean_sac_pidnet_bitrate_mbps']:.4f}")
    print(f"Mean Δbr      CCNet={summary['mean_sac_ccnet_delta_br_pct']:+.1f}%  "
          f"PIDNet={summary['mean_sac_pidnet_delta_br_pct']:+.1f}%")
    print(f"Mean mIoU(4c) H.265={summary['mean_h265_miou']:.4f}  "
          f"CCNet={summary['mean_sac_ccnet_miou']:.4f}  "
          f"PIDNet={summary['mean_sac_pidnet_miou']:.4f}")
    print(f"Mean ROI-IoU  H.265={summary['mean_h265_roi_iou']:.4f}  "
          f"CCNet={summary['mean_sac_ccnet_roi_iou']:.4f}  "
          f"PIDNet={summary['mean_sac_pidnet_roi_iou']:.4f}")
    if bd_c_rate is not None:
        print(f"BD-Rate CCNet  vs H.265: {bd_c_rate:+.2f}%   BD-Acc: {bd_c_acc:+.4f}")
    if bd_p_rate is not None:
        print(f"BD-Rate PIDNet vs H.265: {bd_p_rate:+.2f}%   BD-Acc: {bd_p_acc:+.4f}")
    print(f"\nOutput dir: {run_dir}")

    if not args.keep_artifacts:
        shutil.rmtree(orig_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
