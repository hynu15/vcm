#!/usr/bin/env python3
"""Run Single-Stream Semantic-Aware Compression RD pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.interpolate import PchipInterpolator
from torchvision import transforms

from train_segmentation import load_segmentation_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
LABEL_ROOT = PROJECT_ROOT / "data" / "gt_4class" / "val"
MODEL_PATH = PROJECT_ROOT / "models" / "best_pidnet.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "segmentation_bd_single"

@dataclass(frozen=True)
class OperatingPoint:
    label: str
    qp: int

REQUESTED_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("22", 22),
    OperatingPoint("27", 27),
    OperatingPoint("32", 32),
    OperatingPoint("37", 37),
)

def run_ffmpeg(command: Sequence[str], step_name: str) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed at {step_name}:\n{result.stderr}")

def list_eval_pairs(num_frames: int) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for city in sorted(os.listdir(IMAGE_ROOT)):
        image_city_dir = IMAGE_ROOT / city
        label_city_dir = LABEL_ROOT / city
        if not image_city_dir.is_dir() or not label_city_dir.is_dir():
            continue
        for image_path in sorted(image_city_dir.glob("*_leftImg8bit.png")):
            label_candidates = [
                image_path.name.replace("_leftImg8bit.png", "_gtFine_4class.png"),
                image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png"),
            ]
            label_path = next((label_city_dir / candidate for candidate in label_candidates if (label_city_dir / candidate).is_file()), None)
            if label_path is not None:
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

def to_np_label(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint8)

def dilate_roi_mask(roi_mask: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return roi_mask.astype(np.uint8)
    kernel_size = dilation_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(roi_mask.astype(np.uint8), kernel, iterations=1)
    return (dilated > 0).astype(np.uint8)

def encode_single_streams(
    frame_dir: Path,
    output_dir: Path,
    fps: int,
    qp: int,
    preset: str,
) -> Tuple[Path, Path]:
    
    sac_video = output_dir / "sac_x265.mp4"
    trad_video = output_dir / "traditional_x265.mp4"

    # Encode Semantic Blur (Propose)
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-start_number", "1", "-i", str(frame_dir / "frame_%04d_blur.png"),
            "-c:v", "libx265",
            "-crf", str(qp),
            "-preset", preset,
            str(sac_video),
        ],
        f"encode propose qp={qp}",
    )

    # Encode Traditional (Baseline)
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-start_number", "1", "-i", str(frame_dir / "frame_%04d_orig.png"),
            "-c:v", "libx265",
            "-crf", str(qp),
            "-preset", preset,
            str(trad_video),
        ],
        f"encode trad qp={qp}",
    )
    return sac_video, trad_video

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

def build_roi_masks(model, device, transform, images: List[np.ndarray], roi_dilation_px: int) -> List[np.ndarray]:
    roi_masks: List[np.ndarray] = []
    for frame in images:
        h, w = frame.shape[:2]
        pil = Image.fromarray(frame)
        input_tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        mask = torch.argmax(pred, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_masks.append(dilate_roi_mask((mask == 0).astype(np.uint8), roi_dilation_px))
    return roi_masks

def build_split_frames(
    images: List[np.ndarray],
    roi_masks: List[np.ndarray],
    frame_dir: Path,
    nonroi_keep_weight: float,
    blur_ksize: int,
    blur_sigma: float,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    kernel_size = max(3, int(blur_ksize))
    if kernel_size % 2 == 0:
        kernel_size += 1
    for idx, (frame, roi_mask) in enumerate(zip(images, roi_masks)):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        roi_3 = np.repeat(roi_mask[..., None], 3, axis=2)
        blurred = cv2.GaussianBlur(frame_bgr, (kernel_size, kernel_size), blur_sigma)
        soft_nonroi = cv2.addWeighted(frame_bgr, nonroi_keep_weight, blurred, 1.0 - nonroi_keep_weight, 0.0)
        hybrid = np.where(roi_3 == 1, frame_bgr, soft_nonroi).astype(np.uint8)
        
        cv2.imwrite(str(frame_dir / f"frame_{idx+1:04d}_orig.png"), frame_bgr)
        cv2.imwrite(str(frame_dir / f"frame_{idx+1:04d}_blur.png"), hybrid)

def predict_masks(model, device, transform, frames: List[np.ndarray]) -> List[np.ndarray]:
    preds: List[np.ndarray] = []
    for frame in frames:
        h, w = frame.shape[:2]
        pil = Image.fromarray(frame)
        input_tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
        mask = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        preds.append(mask)
    return preds

def mean_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], num_classes: int = 4) -> Tuple[float, Dict[int, float]]:
    class_scores: Dict[int, float] = {}
    for cls in range(num_classes):
        intersections = 0
        unions = 0
        for pred, gt in zip(pred_masks, gt_masks):
            pred_cls = pred == cls
            gt_cls = gt == cls
            intersections += int(np.logical_and(pred_cls, gt_cls).sum())
            unions += int(np.logical_or(pred_cls, gt_cls).sum())
        class_scores[cls] = float(intersections / unions) if unions > 0 else 0.0
    miou = float(np.mean(list(class_scores.values())))
    return miou, class_scores

def bitrate_mbps(video_path: Path, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return (video_path.stat().st_size * 8.0 / duration_sec) / 1_000_000.0

def bd_rate_and_acc(baseline_df: pd.DataFrame, propose_df: pd.DataFrame) -> Tuple[float, float, Tuple[float, float]]:
    baseline_df = baseline_df.groupby("bitrate_kbps", as_index=False)[["bitrate_kbps", "accuracy_mean"]].mean().sort_values("bitrate_kbps").reset_index(drop=True)
    propose_df = propose_df.groupby("bitrate_kbps", as_index=False)[["bitrate_kbps", "accuracy_mean"]].mean().sort_values("bitrate_kbps").reset_index(drop=True)

    x1 = np.log(baseline_df["bitrate_kbps"].to_numpy(dtype=float))
    y1 = baseline_df["accuracy_mean"].to_numpy(dtype=float)
    x2 = np.log(propose_df["bitrate_kbps"].to_numpy(dtype=float))
    y2 = propose_df["accuracy_mean"].to_numpy(dtype=float)

    def enforce_monotonic(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        keep_x = [x[0]]
        keep_y = [y[0]]
        for xi, yi in zip(x[1:], y[1:]):
            if xi > keep_x[-1] and yi > keep_y[-1]:
                keep_x.append(xi)
                keep_y.append(yi)
        return np.asarray(keep_x, dtype=float), np.asarray(keep_y, dtype=float)

    x1, y1 = enforce_monotonic(x1, y1)
    x2, y2 = enforce_monotonic(x2, y2)

    if len(x1) < 2 or len(x2) < 2:
        raise RuntimeError("Insufficient monotonic points for BD calculation")

    acc_min = max(float(y1.min()), float(y2.min()))
    acc_max = min(float(y1.max()), float(y2.max()))
    if not np.isfinite(acc_min) or not np.isfinite(acc_max) or acc_min >= acc_max:
        raise RuntimeError("No valid metric overlap for BD calculation")

    baseline_inv = PchipInterpolator(y1, x1)
    propose_inv = PchipInterpolator(y2, x2)
    acc_grid = np.linspace(acc_min, acc_max, 200)
    bl_log_rate = baseline_inv(acc_grid)
    pr_log_rate = propose_inv(acc_grid)
    avg_log_rate_delta = float(np.trapz(pr_log_rate - bl_log_rate, acc_grid) / (acc_max - acc_min))
    bd_rate = float((np.exp(avg_log_rate_delta) - 1.0) * 100.0)

    rate_min = max(float(x1.min()), float(x2.min()))
    rate_max = min(float(x1.max()), float(x2.max()))
    if not np.isfinite(rate_min) or not np.isfinite(rate_max) or rate_min >= rate_max:
        raise RuntimeError("No valid bitrate overlap for BD calculation")
    rate_grid = np.linspace(rate_min, rate_max, 200)
    baseline_fwd = PchipInterpolator(x1, y1)
    propose_fwd = PchipInterpolator(x2, y2)
    avg_delta_acc = float(np.trapz(propose_fwd(rate_grid) - baseline_fwd(rate_grid), rate_grid) / (rate_max - rate_min))

    return bd_rate, avg_delta_acc, (float(np.exp(rate_min)), float(np.exp(rate_max)))

def plot_rd_curve(baseline_df: pd.DataFrame, propose_df: pd.DataFrame, output_png: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    baseline_df = baseline_df.sort_values("bitrate_kbps")
    propose_df = propose_df.sort_values("bitrate_kbps")
    ax.plot(
        baseline_df["bitrate_kbps"], baseline_df["accuracy_mean"], "o-", linewidth=2.5, markersize=7, label="Baseline",
    )
    ax.plot(
        propose_df["bitrate_kbps"], propose_df["accuracy_mean"], "s--", linewidth=2.5, markersize=7, label="Propose (SAC Blur)",
    )
    try:
        common_min = max(float(baseline_df["bitrate_kbps"].min()), float(propose_df["bitrate_kbps"].min()))
        common_max = min(float(baseline_df["bitrate_kbps"].max()), float(propose_df["bitrate_kbps"].max()))
        if np.isfinite(common_min) and np.isfinite(common_max) and common_min < common_max:
            grid = np.linspace(common_min, common_max, 200)
            baseline_curve = np.interp(grid, baseline_df["bitrate_kbps"].to_numpy(dtype=float), baseline_df["accuracy_mean"].to_numpy(dtype=float))
            propose_curve = np.interp(grid, propose_df["bitrate_kbps"].to_numpy(dtype=float), propose_df["accuracy_mean"].to_numpy(dtype=float))
            ax.fill_between(grid, baseline_curve, propose_curve, where=propose_curve >= baseline_curve, color="#2ca02c", alpha=0.15, interpolate=True, label="Propose better")
            ax.fill_between(grid, baseline_curve, propose_curve, where=propose_curve < baseline_curve, color="#d62728", alpha=0.12, interpolate=True, label="Baseline better")
    except Exception:
        pass
    for _, row in baseline_df.iterrows():
        ax.annotate(f"{row['qp']}", (row["bitrate_kbps"], row["accuracy_mean"]), xytext=(0, 10), textcoords="offset points", ha="center")
    for _, row in propose_df.iterrows():
        ax.annotate(f"{row['qp']}*", (row["bitrate_kbps"], row["accuracy_mean"]), xytext=(0, -14), textcoords="offset points", ha="center")
    ax.set_xlabel("Rate (kbps)")
    ax.set_ylabel("Acc (mIoU)")
    ax.set_title("Acc - Rate RD Curve (Semantic Blur)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Single Stream RD pipeline")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", type=str, default="slow")
    parser.add_argument("--roi-dilation", type=int, default=20, help="Dilate ROI mask in pixels to preserve context around object boundaries")
    parser.add_argument("--nonroi-keep-weight", type=float, default=0.75, help="How much original signal to keep in non-ROI areas (0-1)")
    parser.add_argument("--blur-ksize", type=int, default=21, help="Gaussian blur kernel size for non-ROI areas")
    parser.add_argument("--blur-sigma", type=float, default=4.0, help="Gaussian blur sigma for non-ROI areas")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = load_segmentation_model(str(MODEL_PATH), device=device, num_classes=4)
    model.eval()
    transform = build_transform()

    pairs = list_eval_pairs(args.num_frames)
    if not pairs:
        raise RuntimeError("No Cityscapes validation pairs found")
    images = [np.array(Image.open(img_path).convert("RGB")) for img_path, _ in pairs]
    gts = [to_np_label(label_path) for _, label_path in pairs]
    roi_masks = build_roi_masks(model, device, transform, images, max(0, int(args.roi_dilation)))

    frame_dir = output_dir / "tmp_frames"
    print('Writing split frames...')
    build_split_frames(
        images,
        roi_masks,
        frame_dir,
        nonroi_keep_weight=float(np.clip(args.nonroi_keep_weight, 0.0, 1.0)),
        blur_ksize=max(3, int(args.blur_ksize)),
        blur_sigma=max(0.0, float(args.blur_sigma)),
    )
    print(f'Done writing. Found {len(list(frame_dir.glob("*")))} files in dir')

    duration_sec = len(images) / float(args.fps)
    baseline_rows: List[Dict[str, float]] = []
    propose_rows: List[Dict[str, float]] = []
    detail_rows: List[Dict[str, float]] = []

    for op in REQUESTED_POINTS:
        combo_dir = output_dir / f"qp_{op.label}"
        combo_dir.mkdir(parents=True, exist_ok=True)
        
        sac_video, trad_video = encode_single_streams(
            frame_dir, combo_dir, args.fps, op.qp, args.preset
        )

        sac_frames = read_video_frames(sac_video, len(images))
        trad_frames = read_video_frames(trad_video, len(images))
        
        sac_miou, _ = mean_iou(predict_masks(model, device, transform, sac_frames), gts)
        trad_miou, _ = mean_iou(predict_masks(model, device, transform, trad_frames), gts)

        sac_bitrate = bitrate_mbps(sac_video, duration_sec)
        trad_bitrate = bitrate_mbps(trad_video, duration_sec)

        baseline_rows.append({"qp": op.label, "bitrate_kbps": trad_bitrate * 1000.0, "accuracy_mean": trad_miou})
        propose_rows.append({"qp": op.label, "bitrate_kbps": sac_bitrate * 1000.0, "accuracy_mean": sac_miou})
        detail_rows.append(
            {
                "qp": op.label,
                "trad_bitrate_mbps": trad_bitrate,
                "sac_bitrate_mbps": sac_bitrate,
                "trad_miou": trad_miou,
                "sac_miou": sac_miou,
                "delta_miou": sac_miou - trad_miou,
                "delta_bitrate_mbps": sac_bitrate - trad_bitrate,
            }
        )
        print(f"{op.label}: trad {trad_bitrate:.3f} Mbps / mIoU {trad_miou:.4f} | sac {sac_bitrate:.3f} Mbps / mIoU {sac_miou:.4f}")

    baseline_df = pd.DataFrame(baseline_rows).sort_values("qp")
    propose_df = pd.DataFrame(propose_rows).sort_values("qp")
    detail_df = pd.DataFrame(detail_rows).sort_values("qp")

    baseline_csv = output_dir / "baseline.csv"
    propose_csv = output_dir / "propose.csv"
    detail_csv = output_dir / "decoded_metrics_detail.csv"
    baseline_df.to_csv(baseline_csv, index=False)
    propose_df.to_csv(propose_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    bd_error = None
    try:
        bd_rate, bd_acc, overlap = bd_rate_and_acc(baseline_df, propose_df)
    except Exception as exc:
        bd_rate = None
        bd_acc = None
        overlap = None
        bd_error = str(exc)

    plot_rd_curve(baseline_df, propose_df, output_dir / "rd_curve.png")

    summary = {
        "task": "segmentation",
        "metric": "mIoU",
        "comparison_mode": "baseline_original_vs_context_preserving_propose",
        "qp_labels": [op.label for op in REQUESTED_POINTS],
        "roi_dilation_px": int(args.roi_dilation),
        "nonroi_keep_weight": float(np.clip(args.nonroi_keep_weight, 0.0, 1.0)),
        "blur_ksize": int(max(3, int(args.blur_ksize))),
        "blur_sigma": float(max(0.0, float(args.blur_sigma))),
        "bitrate_overlap_kbps": overlap,
        "bd_rate_percent": bd_rate,
        "bd_accuracy": bd_acc,
        "bd_error": bd_error,
        "baseline_mean_miou": float(baseline_df["accuracy_mean"].mean()),
        "propose_mean_miou": float(propose_df["accuracy_mean"].mean()),
        "baseline_mean_bitrate_kbps": float(baseline_df["bitrate_kbps"].mean()),
        "propose_mean_bitrate_kbps": float(propose_df["bitrate_kbps"].mean()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRD baseline CSV:")
    print(baseline_df.to_string(index=False))
    print("\nRD propose CSV:")
    print(propose_df.to_string(index=False))
    print("\nDecoded metrics detail:")
    print(detail_df.to_string(index=False))
    if bd_error:
        print(f"\nBD metric calculation failed: {bd_error}")
    else:
        print(f"\nBD-Rate: {bd_rate:+.2f}%")
        print(f"BD-Accuracy: {bd_acc:+.4f}")
    print(f"Output dir: {output_dir}")

    if not args.keep_artifacts:
        shutil.rmtree(frame_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
