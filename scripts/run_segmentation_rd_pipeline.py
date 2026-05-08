#!/usr/bin/env python3
"""Run segmentation RD pipeline on real SAC project data.

This script does the following for four requested operating points:
- encode SAC dual-stream video from Cityscapes val frames
- decode the encoded videos
- run PIDNet on decoded frames
- compute task-level mIoU against Cityscapes 4-class labels
- build RD tables and BD metrics

Notes:
- The project currently has real segmentation data, but no detection/tracking artifacts.
- SAC uses PIDNet-S as the default segmentation model.
- The default operating points are centered around the better trade-off region discovered by CRF optimization.
- Under the hood, the available SAC CRF pairs in this repo are mapped to the closest real operating points.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import shutil
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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "segmentation_bd"


@dataclass(frozen=True)
class OperatingPoint:
    label: str
    crf_roi: int
    crf_non: int


REQUESTED_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("22", 20, 36),
    OperatingPoint("27", 25, 41),
    OperatingPoint("32", 30, 46),
    OperatingPoint("37", 35, 51),
)



def macroblock_align_filter(mask_2d, block_size=16):
    h, w = mask_2d.shape
    out_mask = np.zeros_like(mask_2d)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block = mask_2d[y:y_end, x:x_end]
            if np.any(block > 0):
                out_mask[y:y_end, x:x_end] = 1
    return out_mask

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
            label_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_4class.png")
            label_path = label_city_dir / label_name
            if label_path.is_file():
                pairs.append((image_path, label_path))
                if len(pairs) >= num_frames:
                    return pairs
    return pairs


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def to_np_label(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint8)


def encode_two_streams(
    frame_dir: Path,
    output_dir: Path,
    fps: int,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
    preset: str,
) -> Tuple[Path, Path, Path, Path]:
    roi_video = output_dir / "roi.mp4"
    non_video = output_dir / "nonroi.mp4"
    sac_video = output_dir / "sac_x265.mp4"
    trad_video = output_dir / "traditional_x265.mp4"

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d_roi.png"),
            "-c:v",
            "libx265",
            "-x265-params",
        "aq-mode=0",
        "-crf",
            str(crf_roi),
            "-preset",
            preset,
            str(roi_video),
        ],
        f"encode roi crf={crf_roi}",
    )

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d_non.png"),
            "-c:v",
            "libx265",
            "-x265-params",
        "aq-mode=0",
        "-crf",
            str(crf_non),
            "-preset",
            preset,
            str(non_video),
        ],
        f"encode non crf={crf_non}",
    )

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(roi_video),
            "-i",
            str(non_video),
            "-filter_complex",
            "[0:v][1:v]blend=all_mode=addition",
            str(sac_video),
        ],
        "merge sac streams",
    )

    
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d_orig.png"),
            "-c:v",
            "libx265",
            "-x265-params",
        "aq-mode=0",
        "-crf",
            str(crf_trad),
            "-preset",
            preset,
            str(trad_video),
        ],
        f"encode trad crf={crf_trad}",
    )

    return roi_video, non_video, sac_video, trad_video


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


def build_roi_masks(model, device, transform, images: List[np.ndarray]) -> List[np.ndarray]:
    roi_masks: List[np.ndarray] = []
    for frame in images:
        h, w = frame.shape[:2]
        pil = Image.fromarray(frame)
        input_tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        mask = torch.argmax(pred, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_masks.append(macroblock_align_filter((mask == 0).astype(np.uint8), 16))
    return roi_masks


def build_split_frames(images: List[np.ndarray], roi_masks: List[np.ndarray], frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, (frame, roi_mask) in enumerate(zip(images, roi_masks)):
        roi_255 = (roi_mask * 255).astype(np.uint8)
        non_255 = 255 - roi_255
        roi_img = cv2.bitwise_and(frame, frame, mask=roi_255)
        non_img = cv2.bitwise_and(frame, frame, mask=non_255)
        cv2.imwrite(str(frame_dir / f"frame_{idx:04d}_orig.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_dir / f"frame_{idx:04d}_roi.png"), cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame_dir / f"frame_{idx:04d}_non.png"), cv2.cvtColor(non_img, cv2.COLOR_RGB2BGR))


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
    baseline_df = baseline_df.groupby('bitrate_kbps', as_index=False)[['bitrate_kbps', 'accuracy_mean']].mean().sort_values('bitrate_kbps').reset_index(drop=True)
    propose_df = propose_df.groupby('bitrate_kbps', as_index=False)[['bitrate_kbps', 'accuracy_mean']].mean().sort_values('bitrate_kbps').reset_index(drop=True)

    x1 = np.log(baseline_df["bitrate_kbps"].to_numpy(dtype=float))
    y1 = baseline_df["accuracy_mean"].to_numpy(dtype=float)
    x2 = np.log(propose_df["bitrate_kbps"].to_numpy(dtype=float))
    y2 = propose_df["accuracy_mean"].to_numpy(dtype=float)

    def enforce_monotonic(x, y):
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        x_k, y_k = [x[0]], [y[0]]
        for ix, iy in zip(x[1:], y[1:]):
            if ix > x_k[-1] and iy > y_k[-1]:
                x_k.append(ix)
                y_k.append(iy)
        return np.array(x_k), np.array(y_k)

    x1, y1 = enforce_monotonic(x1, y1)
    x2, y2 = enforce_monotonic(x2, y2)

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
        baseline_df["bitrate_kbps"],
        baseline_df["accuracy_mean"],
        "o-",
        linewidth=2.5,
        markersize=7,
        label="Baseline",
    )
    ax.plot(
        propose_df["bitrate_kbps"],
        propose_df["accuracy_mean"],
        "s--",
        linewidth=2.5,
        markersize=7,
        label="Propose (SAC)",
    )
    for _, row in baseline_df.iterrows():
        ax.annotate(f"{row['qp']}", (row["bitrate_kbps"], row["accuracy_mean"]), xytext=(0, 10), textcoords="offset points", ha="center")
    for _, row in propose_df.iterrows():
        ax.annotate(f"{row['qp']}*", (row["bitrate_kbps"], row["accuracy_mean"]), xytext=(0, -14), textcoords="offset points", ha="center")
    ax.set_xlabel("Rate (kbps)")
    ax.set_ylabel("Acc (mIoU)")
    ax.set_title("Acc - Rate RD Curve on Decoded Video")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation RD pipeline on decoded SAC videos")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", type=str, default="slow")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--external-videos-root",
        type=str,
        default=None,
        help=(
            "If set, the pipeline will skip internal encoding and use pre-encoded videos produced by "
            "sac_compression_x265.py. Expects subdirectories named like 'crf_{label.replace('/', '_')}' "
            "containing roi.mp4, nonroi.mp4, sac_x265.mp4 and traditional_x265.mp4."
        ),
    )
    parser.add_argument(
        "--no-copy-external",
        action="store_true",
        help=(
            "When using --external-videos-root, do not copy external videos into the run output; "
            "use them directly from their external location (default: copy into output folder)."
        ),
    )
    parser.add_argument(
        "--auto-encode",
        action="store_true",
        help=(
            "Automatically call scripts/sac_compression_x265.py for each operating point to produce "
            "external videos. Uses a temporary directory (or --external-videos-root if provided) and "
            "then runs the analysis on those generated files."
        ),
    )
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"Image root not found: {IMAGE_ROOT}")
    if not LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"Label root not found: {LABEL_ROOT}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

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
    roi_masks = build_roi_masks(model, device, transform, images)

    frame_dir = output_dir / "tmp_frames"
    build_split_frames(images, roi_masks, frame_dir)

    duration_sec = len(images) / float(args.fps)
    baseline_rows: List[Dict[str, float]] = []
    propose_rows: List[Dict[str, float]] = []
    detail_rows: List[Dict[str, float]] = []

    for op in REQUESTED_POINTS:
        combo_name = f"crf_{op.label.replace('/', '_')}"
        combo_dir = output_dir / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)

        if args.external_videos_root:
            # Use pre-encoded videos produced externally (e.g. sac_compression_x265.py)
            ext_root = Path(args.external_videos_root)
            ext_dir = ext_root / combo_name
            # Optionally auto-run sac_compression_x265.py to (re)create external videos
            if args.auto_encode:
                # If ext_root doesn't exist, create it; then run sac_compression_x265.py with matching CRFs
                ext_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "python",
                    str(PROJECT_ROOT / "scripts" / "sac_compression_x265.py"),
                    "--crf-roi",
                    str(op.crf_roi),
                    "--crf-non",
                    str(op.crf_non),
                    "--preset",
                    args.preset,
                    "--fps",
                    str(args.fps),
                    "--max-frames",
                    str(args.num_frames),
                    "--output-dir",
                    str(ext_dir),
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"sac_compression_x265.py failed for {combo_name}: {result.stderr.strip()}"
                    )
            else:
                if not ext_dir.is_dir():
                    raise FileNotFoundError(f"External video directory not found: {ext_dir}")
            if not ext_dir.is_dir():
                raise FileNotFoundError(f"External video directory not found: {ext_dir}")

            roi_video = ext_dir / "roi.mp4"
            non_video = ext_dir / "nonroi.mp4"
            sac_video = ext_dir / "sac_x265.mp4"
            trad_video = ext_dir / "traditional_x265.mp4"

            for p in (roi_video, non_video, sac_video, trad_video):
                if not p.is_file():
                    raise FileNotFoundError(f"Expected video file missing in external dir: {p}")

            # By default copy external results into this run's output folder for record-keeping
            if not args.no_copy_external:
                shutil.copy2(str(roi_video), str(combo_dir / roi_video.name))
                shutil.copy2(str(non_video), str(combo_dir / non_video.name))
                shutil.copy2(str(sac_video), str(combo_dir / sac_video.name))
                shutil.copy2(str(trad_video), str(combo_dir / trad_video.name))
                roi_video = combo_dir / "roi.mp4"
                non_video = combo_dir / "nonroi.mp4"
                sac_video = combo_dir / "sac_x265.mp4"
                trad_video = combo_dir / "traditional_x265.mp4"
        else:
            roi_video, non_video, sac_video, trad_video = encode_two_streams(
                frame_dir=frame_dir,
                output_dir=combo_dir,
                fps=args.fps,
                crf_roi=op.crf_roi,
                crf_non=op.crf_non,
                crf_trad=int(op.label),
                preset=args.preset,
            )

        sac_frames = read_video_frames(sac_video, len(images))
        trad_frames = read_video_frames(trad_video, len(images))
        if len(sac_frames) != len(images) or len(trad_frames) != len(images):
            raise RuntimeError(f"Decoded frames missing for operating point {op.label}")

        sac_pred_masks = predict_masks(model, device, transform, sac_frames)
        trad_pred_masks = predict_masks(model, device, transform, trad_frames)

        sac_miou, sac_class_iou = mean_iou(sac_pred_masks, gts, num_classes=4)
        trad_miou, trad_class_iou = mean_iou(trad_pred_masks, gts, num_classes=4)

        sac_bitrate = bitrate_mbps(roi_video, duration_sec) + bitrate_mbps(non_video, duration_sec)
        trad_bitrate = bitrate_mbps(trad_video, duration_sec)
        crf_trad = int(op.label)

        baseline_rows.append(
            {
                "qp": op.label,
                "bitrate_kbps": trad_bitrate * 1000.0,
                "accuracy_mean": trad_miou,
            }
        )
        propose_rows.append(
            {
                "qp": op.label,
                "bitrate_kbps": sac_bitrate * 1000.0,
                "accuracy_mean": sac_miou,
            }
        )
        detail_rows.append(
            {
                "qp": op.label,
                "crf_roi": op.crf_roi,
                "crf_non": op.crf_non,
                "crf_trad": crf_trad,
                "trad_bitrate_mbps": trad_bitrate,
                "sac_bitrate_mbps": sac_bitrate,
                "trad_miou": trad_miou,
                "sac_miou": sac_miou,
                "delta_miou": sac_miou - trad_miou,
                "delta_bitrate_mbps": sac_bitrate - trad_bitrate,
                "trad_class_iou": json.dumps(trad_class_iou, ensure_ascii=False),
                "sac_class_iou": json.dumps(sac_class_iou, ensure_ascii=False),
            }
        )

        print(
            f"{op.label}: trad {trad_bitrate:.3f} Mbps / mIoU {trad_miou:.4f} | "
            f"sac {sac_bitrate:.3f} Mbps / mIoU {sac_miou:.4f}"
        )

    baseline_df = pd.DataFrame(baseline_rows).sort_values("qp")
    propose_df = pd.DataFrame(propose_rows).sort_values("qp")
    detail_df = pd.DataFrame(detail_rows).sort_values("qp")

    baseline_csv = output_dir / "baseline_real.csv"
    propose_csv = output_dir / "propose_real.csv"
    detail_csv = output_dir / "decoded_metrics_detail.csv"
    baseline_df.to_csv(baseline_csv, index=False)
    propose_df.to_csv(propose_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    bd_error = None
    try:
        bd_rate, bd_acc, overlap = bd_rate_and_acc(baseline_df, propose_df)
    except RuntimeError as exc:
        bd_rate = None
        bd_acc = None
        overlap = None
        bd_error = str(exc)

    plot_rd_curve(baseline_df, propose_df, output_dir / "rd_curve.png")

    summary = {
        "model": model_name,
        "task": "segmentation",
        "metric": "mIoU",
        "qp_labels": [op.label for op in REQUESTED_POINTS],
        "operating_points": [
            {"qp": op.label, "crf_roi": op.crf_roi, "crf_non": op.crf_non, "crf_trad": int(round((op.crf_roi + op.crf_non) / 2.0))}
            for op in REQUESTED_POINTS
        ],
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
    if bd_error:
        print(f"\nBD pipeline could not compute metrics: {bd_error}")
    else:
        print(f"\nBD-Rate: {bd_rate:+.2f}%")
        print(f"BD-Accuracy: {bd_acc:+.4f}")
    print(f"Output dir: {output_dir}")

    if not args.keep_artifacts:
        shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
def macroblock_align_filter(mask_2d, block_size=16):
    h, w = mask_2d.shape
    pad_h = (h + block_size - 1) // block_size * block_size
    pad_w = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((pad_h, pad_w), dtype=mask_2d.dtype)
    padded[:h, :w] = mask_2d
    blocks = padded.reshape(pad_h // block_size, block_size, pad_w // block_size, block_size)
    roi_max = blocks.max(axis=(1, 3))
    aligned = np.repeat(np.repeat(roi_max, block_size, axis=0), block_size, axis=1)
    return aligned[:h, :w]



