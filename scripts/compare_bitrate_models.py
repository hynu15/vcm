#!/usr/bin/env python3
"""So sánh bitrate + PSNR giữa H.265, SAC-CCNet_4class và SAC-PIDNet_4class.

Chạy trên test split (không cần GT label).  Với mỗi operating point (CRF pair):
  • H.265 baseline   – encode nguyên frame với single CRF = (crf_roi+crf_non)//2
  • SAC-CCNet4class  – dual-stream, mask từ CCNet_4class
  • SAC-PIDNet4class – dual-stream, mask từ PIDNet_L_4class

Kết quả: bitrate_comparison_detail.csv, bitrate_comparison_summary.csv,
          bitrate_rd_curve.png (bitrate vs PSNR cho 3 phương pháp).

Chạy từ thư mục scripts/:
  conda activate sac
  python compare_bitrate_models.py --num-frames 30 --split test
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
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision import transforms

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

PROJECT_ROOT = _SCRIPTS.parent
IMAGE_ROOT_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
IMAGE_ROOT_TEST = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "test"
LABEL_ROOT_VAL  = PROJECT_ROOT / "data" / "gt_4class" / "val"
LABEL_ROOT_TEST = PROJECT_ROOT / "data" / "gt_4class" / "test"

CCNET_CKPT_DEFAULT  = PROJECT_ROOT / "models" / "best_ccnet_4class.pth"
PIDNET_CKPT_DEFAULT = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "bitrate_comparison"


@dataclass(frozen=True)
class OperatingPoint:
    label: str
    crf_roi: int
    crf_non: int

    @property
    def crf_trad(self) -> int:
        return (self.crf_roi + self.crf_non) // 2


DEFAULT_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("22", 19, 26),
    OperatingPoint("27", 24, 31),
    OperatingPoint("32", 29, 36),
    OperatingPoint("37", 34, 41),
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def run_ffmpeg(cmd: List[str], step: str) -> None:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed at '{step}':\n{r.stderr}")


def macroblock_align_filter(mask_2d: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Bất kỳ block 16×16 nào chứa ít nhất 1 pixel ROI đều được coi là ROI."""
    h, w = mask_2d.shape
    ph = (h + block_size - 1) // block_size * block_size
    pw = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((ph, pw), dtype=mask_2d.dtype)
    padded[:h, :w] = mask_2d
    blocks = padded.reshape(ph // block_size, block_size, pw // block_size, block_size)
    roi_max = blocks.max(axis=(1, 3))
    aligned = np.repeat(np.repeat(roi_max, block_size, axis=0), block_size, axis=1)
    return aligned[:h, :w]


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def bitrate_mbps(video_path: Path, duration_sec: float) -> float:
    return (video_path.stat().st_size * 8.0 / duration_sec) / 1_000_000.0 if duration_sec > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ccnet_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from new_feature.ccnet_4class import CCNet4Class
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = CCNet4Class(num_classes=4).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_pidnet_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from train_pidnet_l import build_pidnet_l
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = build_pidnet_l(num_classes=4).to(device)
    model.load_state_dict(state_dict)
    if hasattr(model, "augment"):
        model.augment = False
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Frame loading
# ─────────────────────────────────────────────────────────────────────────────

def list_images(image_root: Path, num_frames: int, city_filter: Optional[List[str]] = None) -> List[Path]:
    paths: List[Path] = []
    for city in sorted(os.listdir(image_root)):
        if city_filter and city not in city_filter:
            continue
        city_dir = image_root / city
        if not city_dir.is_dir():
            continue
        for img in sorted(city_dir.glob("*_leftImg8bit.png")):
            paths.append(img)
            if len(paths) >= num_frames:
                return paths
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# ROI mask building
# ─────────────────────────────────────────────────────────────────────────────

def build_roi_masks(model: torch.nn.Module, device: torch.device, transform: transforms.Compose,
                    images: List[np.ndarray]) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for frame in images:
        h, w = frame.shape[:2]
        inp = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)
        mask = torch.argmax(pred, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(macroblock_align_filter((mask == 0).astype(np.uint8), 16))
    roi_ratios = [m.mean() * 100 for m in masks]
    print(f"    ROI ratio: mean={np.mean(roi_ratios):.1f}%  min={np.min(roi_ratios):.1f}%  max={np.max(roi_ratios):.1f}%")
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_frames(images: List[np.ndarray], roi_masks: List[np.ndarray], frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, roi) in enumerate(zip(images, roi_masks)):
        roi_255 = (roi * 255).astype(np.uint8)
        non_255 = 255 - roi_255
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}_orig.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}_roi.png"),
                    cv2.cvtColor(cv2.bitwise_and(img, img, mask=roi_255), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}_non.png"),
                    cv2.cvtColor(cv2.bitwise_and(img, img, mask=non_255), cv2.COLOR_RGB2BGR))


def encode_h265_baseline(orig_frame_dir: Path, out_dir: Path, fps: int, crf: int, preset: str) -> Path:
    vid = out_dir / "h265_baseline.mp4"
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(orig_frame_dir / "frame_%04d_orig.png"),
        "-c:v", "libx265", "-crf", str(crf), "-preset", preset, str(vid),
    ], f"h265_baseline crf={crf}")
    return vid


def encode_sac_streams(frame_dir: Path, out_dir: Path, prefix: str,
                       fps: int, crf_roi: int, crf_non: int, preset: str) -> Tuple[Path, Path, Path]:
    roi_vid = out_dir / f"{prefix}_roi.mp4"
    non_vid = out_dir / f"{prefix}_non.mp4"
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
    ], f"{prefix} non crf={crf_non}")
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(roi_vid), "-i", str(non_vid),
        "-filter_complex", "[0:v][1:v]blend=all_mode=addition",
        "-c:v", "libx265", "-crf", str((crf_roi + crf_non) // 2), "-preset", preset, str(sac_vid),
    ], f"{prefix} merge")
    return roi_vid, non_vid, sac_vid


# ─────────────────────────────────────────────────────────────────────────────
# PSNR of decoded video vs originals
# ─────────────────────────────────────────────────────────────────────────────

def mean_psnr_decoded(video_path: Path, originals: List[np.ndarray]) -> float:
    cap = cv2.VideoCapture(str(video_path))
    psnrs: List[float] = []
    for orig in originals:
        ok, frame = cap.read()
        if not ok:
            break
        decoded = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = orig.shape[:2]
        if decoded.shape[:2] != (h, w):
            decoded = cv2.resize(decoded, (w, h))
        psnrs.append(compare_psnr(orig, decoded, data_range=255))
    cap.release()
    return float(np.mean(psnrs)) if psnrs else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_bitrate_comparison(detail_df: pd.DataFrame, output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # ── Bitrate comparison ──
    ax = axes[0]
    x = np.arange(len(detail_df))
    w = 0.28
    ax.bar(x - w, detail_df["h265_bitrate_mbps"], w, label="H.265 baseline", color="#4c72b0")
    ax.bar(x,      detail_df["sac_ccnet_bitrate_mbps"], w, label="SAC-CCNet4class", color="#dd8452")
    ax.bar(x + w,  detail_df["sac_pidnet_bitrate_mbps"], w, label="SAC-PIDNet4class", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels([f"QP={row['qp']}\n(roi={row['crf_roi']},non={row['crf_non']})"
                        for _, row in detail_df.iterrows()], fontsize=8)
    ax.set_ylabel("Bitrate (Mbps)")
    ax.set_title("Bitrate Comparison per Operating Point")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── PSNR vs Bitrate (RD curve) ──
    ax2 = axes[1]
    if detail_df["h265_psnr"].notna().any():
        df_s = detail_df.sort_values("h265_bitrate_mbps")
        ax2.plot(df_s["h265_bitrate_mbps"], df_s["h265_psnr"],
                 "o-", color="#4c72b0", label="H.265 baseline", linewidth=2)
        df_s2 = detail_df.sort_values("sac_ccnet_bitrate_mbps")
        ax2.plot(df_s2["sac_ccnet_bitrate_mbps"], df_s2["sac_ccnet_psnr"],
                 "s--", color="#dd8452", label="SAC-CCNet4class", linewidth=2)
        df_s3 = detail_df.sort_values("sac_pidnet_bitrate_mbps")
        ax2.plot(df_s3["sac_pidnet_bitrate_mbps"], df_s3["sac_pidnet_psnr"],
                 "^--", color="#55a868", label="SAC-PIDNet4class", linewidth=2)
        ax2.set_xlabel("Bitrate (Mbps)")
        ax2.set_ylabel("PSNR (dB)")
        ax2.set_title("Rate-Distortion Curve (PSNR vs Bitrate)")
        ax2.legend()
        ax2.grid(linestyle="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "PSNR not available", ha="center", va="center", transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(str(output_png))
    plt.close(fig)
    print(f"Plot saved: {output_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh bitrate H.265 vs SAC-CCNet4class vs SAC-PIDNet4class")
    parser.add_argument("-n", "--num-frames", type=int, default=30,
                        help="Số frame dùng để encode (mặc định: 30)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", type=str, default="slow")
    parser.add_argument("--split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--ccnet-ckpt", type=str, default=None,
                        help="Path đến CCNet_4class checkpoint (mặc định: models/best_ccnet_4class.pth)")
    parser.add_argument("--pidnet-ckpt", type=str, default=None,
                        help="Path đến PIDNet_L_4class checkpoint (mặc định: models/best_pidnet_l_4class.pth)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cities", type=str, default=None,
                        help="City filter, cách nhau dấu phẩy, ví dụ: berlin,munich")
    parser.add_argument("--crf-pairs", type=str, default=None,
                        help="Ghi đè operating points: 'roi:non,...' vd '19:26,24:31,29:36,34:41'")
    parser.add_argument("--keep-artifacts", action="store_true",
                        help="Giữ lại thư mục frame/video tạm thời")
    args = parser.parse_args()

    ccnet_ckpt  = Path(args.ccnet_ckpt)  if args.ccnet_ckpt  else CCNET_CKPT_DEFAULT
    pidnet_ckpt = Path(args.pidnet_ckpt) if args.pidnet_ckpt else PIDNET_CKPT_DEFAULT
    for p in (ccnet_ckpt, pidnet_ckpt):
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    if args.crf_pairs:
        op_points: Sequence[OperatingPoint] = []
        for pair in args.crf_pairs.split(","):
            r, n = pair.strip().split(":")
            op_points.append(OperatingPoint(str((int(r) + int(n)) // 2), int(r), int(n)))
    else:
        op_points = DEFAULT_POINTS

    image_root = IMAGE_ROOT_TEST if args.split == "test" else IMAGE_ROOT_VAL
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

    print(f"\nLoading CCNet_4class from {ccnet_ckpt.name} ...")
    ccnet_model = load_ccnet_model(ccnet_ckpt, device)
    print(f"Loading PIDNet_L_4class from {pidnet_ckpt.name} ...")
    pidnet_model = load_pidnet_model(pidnet_ckpt, device)
    transform = build_transform()

    img_paths = list_images(image_root, args.num_frames, city_filter)
    if not img_paths:
        raise RuntimeError(f"No images found in {image_root}")
    print(f"\nLoaded {len(img_paths)} frames from split='{args.split}'")

    images = [np.array(Image.open(p).convert("RGB")) for p in img_paths]
    duration_sec = len(images) / float(args.fps)

    print("\nBuilding ROI masks with CCNet_4class ...")
    ccnet_masks  = build_roi_masks(ccnet_model,  device, transform, images)
    print("Building ROI masks with PIDNet_L_4class ...")
    pidnet_masks = build_roi_masks(pidnet_model, device, transform, images)

    # Write split frames for each model (original frames shared via symlink-free copy trick)
    frame_dir_ccnet  = run_dir / "tmp_frames_ccnet"
    frame_dir_pidnet = run_dir / "tmp_frames_pidnet"
    print("\nWriting split frames ...")
    write_frames(images, ccnet_masks,  frame_dir_ccnet)
    write_frames(images, pidnet_masks, frame_dir_pidnet)

    # Write original frames once (shared between models for baseline)
    frame_dir_orig = run_dir / "tmp_frames_orig"
    frame_dir_orig.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(str(frame_dir_orig / f"frame_{i:04d}_orig.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    detail_rows: List[Dict] = []

    for op in op_points:
        combo = f"op_{op.label}"
        op_dir = run_dir / combo
        op_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n── Operating point QP={op.label}  (crf_roi={op.crf_roi}, crf_non={op.crf_non}, crf_trad={op.crf_trad}) ──")

        # H.265 baseline
        print(f"  Encoding H.265 baseline (crf={op.crf_trad}) ...")
        h265_vid = encode_h265_baseline(frame_dir_ccnet, op_dir, args.fps, op.crf_trad, args.preset)
        h265_br  = bitrate_mbps(h265_vid, duration_sec)
        h265_psnr = mean_psnr_decoded(h265_vid, images)
        print(f"  H.265:           {h265_br:.4f} Mbps  PSNR={h265_psnr:.2f} dB")

        # SAC with CCNet
        print(f"  Encoding SAC-CCNet  (crf_roi={op.crf_roi}, crf_non={op.crf_non}) ...")
        roi_c, non_c, sac_c = encode_sac_streams(
            frame_dir_ccnet, op_dir, "ccnet", args.fps, op.crf_roi, op.crf_non, args.preset)
        ccnet_br  = bitrate_mbps(roi_c, duration_sec) + bitrate_mbps(non_c, duration_sec)
        ccnet_psnr = mean_psnr_decoded(sac_c, images)
        ccnet_saving = (ccnet_br - h265_br) / h265_br * 100
        print(f"  SAC-CCNet4class: {ccnet_br:.4f} Mbps  PSNR={ccnet_psnr:.2f} dB  (Δbr={ccnet_saving:+.1f}%)")

        # SAC with PIDNet
        print(f"  Encoding SAC-PIDNet (crf_roi={op.crf_roi}, crf_non={op.crf_non}) ...")
        roi_p, non_p, sac_p = encode_sac_streams(
            frame_dir_pidnet, op_dir, "pidnet", args.fps, op.crf_roi, op.crf_non, args.preset)
        pidnet_br  = bitrate_mbps(roi_p, duration_sec) + bitrate_mbps(non_p, duration_sec)
        pidnet_psnr = mean_psnr_decoded(sac_p, images)
        pidnet_saving = (pidnet_br - h265_br) / h265_br * 100
        print(f"  SAC-PIDNet4class:{pidnet_br:.4f} Mbps  PSNR={pidnet_psnr:.2f} dB  (Δbr={pidnet_saving:+.1f}%)")

        detail_rows.append({
            "qp":                     op.label,
            "crf_roi":                op.crf_roi,
            "crf_non":                op.crf_non,
            "crf_trad":               op.crf_trad,
            "h265_bitrate_mbps":      round(h265_br, 6),
            "sac_ccnet_bitrate_mbps": round(ccnet_br, 6),
            "sac_pidnet_bitrate_mbps":round(pidnet_br, 6),
            "sac_ccnet_delta_pct":    round(ccnet_saving, 2),
            "sac_pidnet_delta_pct":   round(pidnet_saving, 2),
            "h265_psnr":              round(h265_psnr, 4),
            "sac_ccnet_psnr":         round(ccnet_psnr, 4),
            "sac_pidnet_psnr":        round(pidnet_psnr, 4),
            "sac_ccnet_delta_psnr":   round(ccnet_psnr - h265_psnr, 4),
            "sac_pidnet_delta_psnr":  round(pidnet_psnr - h265_psnr, 4),
        })

    detail_df = pd.DataFrame(detail_rows)
    detail_csv = run_dir / "bitrate_comparison_detail.csv"
    detail_df.to_csv(detail_csv, index=False)

    # Summary: mean over all operating points
    summary = {
        "split":                       args.split,
        "num_frames":                  len(images),
        "num_operating_points":        len(op_points),
        "ccnet_ckpt":                  str(ccnet_ckpt),
        "pidnet_ckpt":                 str(pidnet_ckpt),
        "mean_h265_bitrate_mbps":      round(float(detail_df["h265_bitrate_mbps"].mean()), 4),
        "mean_sac_ccnet_bitrate_mbps": round(float(detail_df["sac_ccnet_bitrate_mbps"].mean()), 4),
        "mean_sac_pidnet_bitrate_mbps":round(float(detail_df["sac_pidnet_bitrate_mbps"].mean()), 4),
        "mean_sac_ccnet_delta_pct":    round(float(detail_df["sac_ccnet_delta_pct"].mean()), 2),
        "mean_sac_pidnet_delta_pct":   round(float(detail_df["sac_pidnet_delta_pct"].mean()), 2),
        "mean_h265_psnr":              round(float(detail_df["h265_psnr"].mean()), 4),
        "mean_sac_ccnet_psnr":         round(float(detail_df["sac_ccnet_psnr"].mean()), 4),
        "mean_sac_pidnet_psnr":        round(float(detail_df["sac_pidnet_psnr"].mean()), 4),
        "mean_sac_ccnet_delta_psnr":   round(float(detail_df["sac_ccnet_delta_psnr"].mean()), 4),
        "mean_sac_pidnet_delta_psnr":  round(float(detail_df["sac_pidnet_delta_psnr"].mean()), 4),
    }
    summary_csv = run_dir / "bitrate_comparison_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    plot_bitrate_comparison(detail_df, run_dir / "bitrate_rd_curve.png")

    print("\n" + "=" * 70)
    print("DETAIL TABLE:")
    print(detail_df.to_string(index=False))
    print("\nSUMMARY:")
    for k, v in summary.items():
        if k not in ("ccnet_ckpt", "pidnet_ckpt"):
            print(f"  {k}: {v}")
    print(f"\nOutput dir: {run_dir}")

    if not args.keep_artifacts:
        for d in (frame_dir_ccnet, frame_dir_pidnet, frame_dir_orig):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
