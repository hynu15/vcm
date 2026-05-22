#!/usr/bin/env python3
"""Run segmentation RD pipeline – SAC single-stream variant.

Khác biệt so với run_segmentation_rd_pipeline.py:
  - KHÔNG có SAC_MIOU_OFFSET / SAC_BITRATE_REDUCTION (không offset giả tạo).
  - SAC được encode thành MỘT luồng duy nhất (sac_single.mp4):
      1. Encode full video ở crf_non  → decode → frame chất lượng thấp.
      2. Composite mỗi frame: vùng ROI lấy từ ảnh gốc, vùng non-ROI lấy từ
         frame đã decode (chất lượng thấp).
      3. Encode chuỗi composite ở crf_roi → sac_single.mp4.
    Bitrate SAC = kích thước thực của sac_single.mp4, không nhân hệ số nào.
  - Giữ nguyên toàn bộ logic: mIoU, BD-Rate, BD-Acc, plot.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.interpolate import PchipInterpolator
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT_VAL = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
LABEL_ROOT_VAL = PROJECT_ROOT / "data" / "gt_4class" / "val"
IMAGE_ROOT_TEST = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "test"
LABEL_ROOT_TEST = PROJECT_ROOT / "data" / "gt_4class" / "test"
MODEL_PATH_DEFAULT = PROJECT_ROOT / "models" / "best_pidnet_l_4class.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "segmentation_bd_single"
GT_FINE_ROOT = PROJECT_ROOT / "data" / "gt_4class" / "gtFine_trainvaltest" / "gtFine"

# Cityscapes labelId -> 4-class mapping (đồng bộ với prepare_4class_labels.py)
# Class 0 = background (catch-all)  → BA stream
# Class 1 = road        (7–10)       → IA stream
# Class 2 = vehicle     (26–33)      → IA stream
# Class 3 = pedestrian  (24–25)      → IA stream
ROAD_IDS        = [7, 8, 9, 10]
VEHICLE_IDS     = [26, 27, 28, 29, 30, 31, 32, 33]
PEDESTRIAN_IDS  = [24, 25]
CLASS_NAMES_4   = ["background", "road", "vehicle", "pedestrian"]

ROI_IDS_2CLASS  = ROAD_IDS + VEHICLE_IDS + PEDESTRIAN_IDS

VEHICLE_PEDESTRIAN_IDS = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]


@dataclass(frozen=True)
class OperatingPoint:
    label: str
    crf_roi: int              # CRF encode SAC composite (final)
    crf_non: int              # CRF pre-compress non-ROI — dùng khi blur_sigma=None và global sigma=0
    crf_trad: int             # CRF encode traditional — ĐỘC LẬP
    blur_sigma: Optional[float] = None  # ghi đè global --non-roi-blur-sigma cho OP này


REQUESTED_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("16", 13, 19, 16),
    OperatingPoint("19", 16, 22, 19),
    OperatingPoint("22", 19, 25, 22),
    OperatingPoint("25", 22, 28, 25),
    OperatingPoint("28", 25, 31, 28),
)

NARROW_ROI_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("20", 18, 22, 20),
    OperatingPoint("24", 22, 26, 24),
    OperatingPoint("28", 26, 30, 28),
    OperatingPoint("32", 30, 34, 32),
)


def macroblock_align_filter(mask_2d, block_size=64):
    """Any 64×64 block containing at least one ROI pixel is treated as a full ROI block."""
    h, w = mask_2d.shape
    pad_h = (h + block_size - 1) // block_size * block_size
    pad_w = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((pad_h, pad_w), dtype=mask_2d.dtype)
    padded[:h, :w] = mask_2d
    blocks = padded.reshape(pad_h // block_size, block_size, pad_w // block_size, block_size)
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
    num_classes: int = 2,
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
                if num_classes == 4:
                    cand = label_city_dir / f"{stem}_gtFine_4class.png"
                    if cand.is_file():
                        label_path = cand
                if label_path is None:
                    cand = label_city_dir / f"{stem}_gtFine_2class.png"
                    if cand.is_file():
                        label_path = cand
                if label_path is None:
                    cand = label_city_dir / f"{stem}_gtFine_labelIds.png"
                    if cand.is_file():
                        label_path = cand
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


def convert_label_ids_to_2class(label: np.ndarray) -> np.ndarray:
    """0 = background, 1 = road+vehicle+pedestrian (ROI)."""
    mask = np.zeros_like(label, dtype=np.uint8)
    for rid in ROI_IDS_2CLASS:
        mask[label == rid] = 1
    return mask


def convert_label_ids_to_4class(label: np.ndarray) -> np.ndarray:
    """0 = background, 1 = road, 2 = vehicle, 3 = pedestrian."""
    mask = np.zeros_like(label, dtype=np.uint8)
    for rid in ROAD_IDS:
        mask[label == rid] = 1
    for vid in VEHICLE_IDS:
        mask[label == vid] = 2
    for pid in PEDESTRIAN_IDS:
        mask[label == pid] = 3
    return mask


def load_model_auto(model_path: Path, device: torch.device, num_classes_hint: int = 4):
    try:
        ckpt = torch.load(str(model_path), map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(model_path), map_location=device)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt)
        meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta"), dict) else {}
    else:
        state_dict = ckpt
        meta = {}

    model_name = meta.get("model_name", "pidnet_l")
    num_classes = int(meta.get("num_classes", num_classes_hint))

    if model_name == "pidnet_l":
        from train_pidnet_l import build_pidnet_l
        model = build_pidnet_l(num_classes=num_classes).to(device)
    else:
        from train_segmentation import build_segmentation_model
        model = build_segmentation_model(
            model_name=model_name, num_classes=num_classes).to(device)

    model.load_state_dict(state_dict)
    if hasattr(model, "augment"):
        model.augment = False
    model.eval()
    return model, model_name, num_classes


def to_np_label(path: Optional[Path], num_classes: int = 2) -> np.ndarray:
    if path is None:
        return np.zeros((512, 1024), dtype=np.uint8)
    label = np.array(Image.open(path), dtype=np.uint8)
    if path.name.endswith("_gtFine_4class.png"):
        return label
    if path.name.endswith("_labelIds.png"):
        return convert_label_ids_to_4class(label) if num_classes == 4 else convert_label_ids_to_2class(label)
    return label


def all_zero_labels(labels: Sequence[np.ndarray]) -> bool:
    return all(int(label.max()) == 0 for label in labels)


def build_orig_frames(images: List[np.ndarray], frame_dir: Path) -> None:
    """Chỉ lưu frame gốc (không cần tách ROI/non-ROI cho phương pháp single-stream)."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(images):
        cv2.imwrite(
            str(frame_dir / f"frame_{idx:04d}_orig.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )


def encode_single_stream_sac(
    frame_dir: Path,
    output_dir: Path,
    images: List[np.ndarray],
    roi_masks: List[np.ndarray],
    fps: int,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
    preset: str,
    mode: str = "twostream",
    non_roi_blur_sigma: float = 3.0,
    sac_final_crf_offset: int = 0,
) -> Tuple[Path, Path]:
    """Encode SAC dưới dạng 1 luồng duy nhất.

    mode="twostream" (mặc định — theo reference project):
      1. Tách IA (ROI=gốc, non-ROI=đen) → encode tại crf_roi → decode
      2. Tách BA (non-ROI=gốc, ROI=đen) → encode tại crf_non → decode
      3. Combine: decoded_IA + decoded_BA  (cộng pixel, clip 0-255)
      4. Encode combined tại crf_trad → sac_single.mp4
      → crf_roi, crf_non, crf_trad đều có tác dụng thực sự.
      → Bitrate SAC ≈ traditional (cùng encode tại crf_trad, content có entropy thấp hơn).

    mode="blur":
      1. Gaussian blur vùng non-ROI với sigma=non_roi_blur_sigma
      2. Composite ROI=gốc + non-ROI=blurred → encode tại (crf_roi + sac_final_crf_offset)
      → crf_non KHÔNG dùng; crf_trad chỉ dùng cho traditional.

    mode="precompress":
      1. Encode toàn frame tại crf_non → decode → lấy vùng non-ROI
      2. Composite → encode tại (crf_roi + sac_final_crf_offset)
      → crf_non dùng cho pre-compress.
    """
    sac_video  = output_dir / "sac_single.mp4"
    trad_video = output_dir / "traditional_x265.mp4"

    if mode == "twostream":
        _encode_twostream(
            output_dir, images, roi_masks, fps, crf_roi, crf_non, crf_trad, preset, sac_video
        )
    elif mode == "blur":
        _encode_blur(
            frame_dir, output_dir, images, roi_masks, fps,
            crf_roi, crf_trad, preset, non_roi_blur_sigma, sac_final_crf_offset, sac_video
        )
    elif mode == "precompress":
        _encode_precompress(
            frame_dir, output_dir, images, roi_masks, fps,
            crf_roi, crf_non, crf_trad, preset, sac_final_crf_offset, sac_video
        )
    else:
        raise ValueError(f"mode phải là 'twostream', 'blur', hoặc 'precompress', nhận được: '{mode}'")

    # Traditional: encode frame gốc tại crf_trad
    run_ffmpeg(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(frame_dir / "frame_%04d_orig.png"),
         "-c:v", "libx265", "-crf", str(crf_trad), "-preset", preset,
         str(trad_video)],
        f"encode trad crf={crf_trad}",
    )
    return sac_video, trad_video


def _write_frames(frames_rgb: List[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames_rgb):
        cv2.imwrite(
            str(out_dir / f"frame_{idx:04d}.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )


def _encode_twostream(
    output_dir: Path,
    images: List[np.ndarray],
    roi_masks: List[np.ndarray],
    fps: int,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
    preset: str,
    sac_video: Path,
) -> None:
    """Two-stream → single-stream (theo logic TwoStream_generate + combine.py của reference project).

    IA + BA encode riêng, decode, cộng pixel, encode lần cuối thành 1 luồng.
    """
    ia_dir       = output_dir / "_ia_frames"
    ba_dir       = output_dir / "_ba_frames"
    combined_dir = output_dir / "_combined_frames"
    ia_video     = output_dir / "_ia_tmp.mp4"
    ba_video     = output_dir / "_ba_tmp.mp4"

    # ── Bước 1a: tạo IA frames (ROI=gốc, non-ROI=đen) ──────────────────────
    ia_frames: List[np.ndarray] = []
    ba_frames: List[np.ndarray] = []
    for orig, roi_mask in zip(images, roi_masks):
        roi_255 = (roi_mask * 255).astype(np.uint8)
        non_255 = (255 - roi_255).astype(np.uint8)
        ia_frames.append(cv2.bitwise_and(orig, orig, mask=roi_255))
        ba_frames.append(cv2.bitwise_and(orig, orig, mask=non_255))

    _write_frames(ia_frames, ia_dir)
    _write_frames(ba_frames, ba_dir)

    # ── Bước 1b: encode IA tại crf_roi, encode BA tại crf_non ───────────────
    run_ffmpeg(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(ia_dir / "frame_%04d.png"),
         "-c:v", "libx265", "-crf", str(crf_roi), "-preset", preset, str(ia_video)],
        f"encode IA crf={crf_roi}",
    )
    run_ffmpeg(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(ba_dir / "frame_%04d.png"),
         "-c:v", "libx265", "-crf", str(crf_non), "-preset", preset, str(ba_video)],
        f"encode BA crf={crf_non}",
    )

    # ── Bước 2: decode cả hai ────────────────────────────────────────────────
    ia_decoded = read_video_frames(ia_video, len(images))
    ba_decoded = read_video_frames(ba_video, len(images))
    if len(ia_decoded) < len(images) or len(ba_decoded) < len(images):
        raise RuntimeError("Decoded IA/BA frames ít hơn expected")

    # ── Bước 3: combine IA + BA (cộng pixel, clip 0-255) ────────────────────
    combined_frames: List[np.ndarray] = []
    for ia, ba in zip(ia_decoded, ba_decoded):
        if ia.shape != ba.shape:
            ba = cv2.resize(ba, (ia.shape[1], ia.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined = np.clip(ia.astype(np.int32) + ba.astype(np.int32), 0, 255).astype(np.uint8)
        combined_frames.append(combined)
    _write_frames(combined_frames, combined_dir)

    # ── Bước 4: encode combined thành 1 luồng SAC tại crf_trad ──────────────
    run_ffmpeg(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(combined_dir / "frame_%04d.png"),
         "-c:v", "libx265", "-crf", str(crf_trad), "-preset", preset, str(sac_video)],
        f"encode SAC combined crf={crf_trad}",
    )

    # dọn dẹp
    for d in (ia_dir, ba_dir, combined_dir):
        shutil.rmtree(d, ignore_errors=True)
    ia_video.unlink(missing_ok=True)
    ba_video.unlink(missing_ok=True)


def _encode_blur(
    frame_dir: Path,
    output_dir: Path,
    images: List[np.ndarray],
    roi_masks: List[np.ndarray],
    fps: int,
    crf_roi: int,
    crf_trad: int,
    preset: str,
    blur_sigma: float,
    sac_final_crf_offset: int,
    sac_video: Path,
) -> None:
    composite_dir = output_dir / "_composite_tmp"
    composite_dir.mkdir(parents=True, exist_ok=True)
    for idx, (orig, roi_mask) in enumerate(zip(images, roi_masks)):
        blurred = cv2.GaussianBlur(orig, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        roi_255 = (roi_mask * 255).astype(np.uint8)
        non_255 = (255 - roi_255).astype(np.uint8)
        composite = cv2.add(
            cv2.bitwise_and(orig,    orig,    mask=roi_255),
            cv2.bitwise_and(blurred, blurred, mask=non_255),
        )
        cv2.imwrite(str(composite_dir / f"frame_{idx:04d}.png"),
                    cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    sac_final_crf = max(0, crf_roi + sac_final_crf_offset)
    run_ffmpeg(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(composite_dir / "frame_%04d.png"),
         "-c:v", "libx265", "-crf", str(sac_final_crf), "-preset", preset, str(sac_video)],
        f"encode SAC blur(σ={blur_sigma}) crf={sac_final_crf}",
    )
    shutil.rmtree(composite_dir, ignore_errors=True)


def _encode_precompress(
    frame_dir: Path,
    output_dir: Path,
    images: List[np.ndarray],
    roi_masks: List[np.ndarray],
    fps: int,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
    preset: str,
    sac_final_crf_offset: int,
    sac_video: Path,
) -> None:
    tmp_video = output_dir / "_precompress_tmp.mp4"
    run_ffmpeg(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(frame_dir / "frame_%04d_orig.png"),
         "-c:v", "libx265", "-crf", str(crf_non), "-preset", preset, str(tmp_video)],
        f"pre-compress crf={crf_non}",
    )
    low_frames = read_video_frames(tmp_video, len(images))
    composite_dir = output_dir / "_composite_tmp"
    composite_dir.mkdir(parents=True, exist_ok=True)
    for idx, (orig, low, roi_mask) in enumerate(zip(images, low_frames, roi_masks)):
        if low.shape[:2] != orig.shape[:2]:
            low = cv2.resize(low, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        roi_255 = (roi_mask * 255).astype(np.uint8)
        non_255 = (255 - roi_255).astype(np.uint8)
        composite = cv2.add(
            cv2.bitwise_and(orig, orig, mask=roi_255),
            cv2.bitwise_and(low,  low,  mask=non_255),
        )
        cv2.imwrite(str(composite_dir / f"frame_{idx:04d}.png"),
                    cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    sac_final_crf = max(0, crf_roi + sac_final_crf_offset)
    run_ffmpeg(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(composite_dir / "frame_%04d.png"),
         "-c:v", "libx265", "-crf", str(sac_final_crf), "-preset", preset, str(sac_video)],
        f"encode SAC precompress crf={sac_final_crf}",
    )
    tmp_video.unlink(missing_ok=True)
    shutil.rmtree(composite_dir, ignore_errors=True)


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
        roi_masks.append(macroblock_align_filter((mask != 0).astype(np.uint8), 16))
    return roi_masks


def build_roi_masks_narrow_gt(
    pairs: List[Tuple[Path, Optional[Path]]],
    images: List[np.ndarray],
) -> List[np.ndarray]:
    roi_masks: List[np.ndarray] = []
    missing = 0
    for (img_path, _label_path), frame in zip(pairs, images):
        h, w = frame.shape[:2]
        city = img_path.parent.name
        stem_base = img_path.stem.replace("_leftImg8bit", "")
        labelids_path: Optional[Path] = None
        for split_name in ("train", "val", "test"):
            candidate = GT_FINE_ROOT / split_name / city / f"{stem_base}_gtFine_labelIds.png"
            if candidate.is_file():
                labelids_path = candidate
                break
        if labelids_path is None:
            missing += 1
            roi_masks.append(np.zeros((h, w), dtype=np.uint8))
            continue
        label = np.array(Image.open(labelids_path))
        roi_mask = np.zeros(label.shape, dtype=np.uint8)
        for vid in VEHICLE_PEDESTRIAN_IDS:
            roi_mask[label == vid] = 1
        roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_masks.append(macroblock_align_filter(roi_mask, 16))
    if missing:
        print(f"  WARNING: {missing}/{len(pairs)} frames thiếu _gtFine_labelIds.png → ROI mask = zeros")
    ratios = [m.mean() * 100 for m in roi_masks]
    print(f"  Narrow-GT ROI ratio: mean={np.mean(ratios):.1f}%  "
          f"min={np.min(ratios):.1f}%  max={np.max(ratios):.1f}%")
    return roi_masks


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


def mean_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], num_classes: int = 2) -> Tuple[float, Dict[int, float]]:
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rd_curve(baseline_df: pd.DataFrame, propose_df: pd.DataFrame, output_png: Path,
                  bd_rate: Optional[float] = None, bd_acc: Optional[float] = None) -> None:
    import matplotlib.pyplot as plt

    BLUE   = "#1a6faf"
    ORANGE = "#e07b39"
    GRAY   = "#888888"

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor("#f8f9fb")
    ax.set_facecolor("#f8f9fb")

    baseline_df = baseline_df.sort_values("bitrate_kbps").reset_index(drop=True)
    propose_df  = propose_df.sort_values("bitrate_kbps").reset_index(drop=True)

    try:
        x_min = max(baseline_df["bitrate_kbps"].min(), propose_df["bitrate_kbps"].min())
        x_max = min(baseline_df["bitrate_kbps"].max(), propose_df["bitrate_kbps"].max())
        x_grid = np.linspace(x_min, x_max, 300)
        bl_interp = PchipInterpolator(baseline_df["bitrate_kbps"], baseline_df["accuracy_mean"])
        pr_interp = PchipInterpolator(propose_df["bitrate_kbps"],  propose_df["accuracy_mean"])
        ax.fill_between(x_grid, bl_interp(x_grid), pr_interp(x_grid),
                        alpha=0.12, color=ORANGE, label="_nolegend_")
    except Exception:
        pass

    ax.plot(baseline_df["bitrate_kbps"], baseline_df["accuracy_mean"],
            "o-", color=BLUE, linewidth=2.5, markersize=8,
            markerfacecolor="white", markeredgewidth=2.5, label="Traditional x265")
    ax.plot(propose_df["bitrate_kbps"], propose_df["accuracy_mean"],
            "s--", color=ORANGE, linewidth=2.5, markersize=8,
            markerfacecolor="white", markeredgewidth=2.5, label="SAC single-stream (proposed)")

    for _, row in baseline_df.iterrows():
        ax.annotate(f"QP {row['qp']}",
                    (row["bitrate_kbps"], row["accuracy_mean"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
    for _, row in propose_df.iterrows():
        ax.annotate(f"QP {row['qp']}*",
                    (row["bitrate_kbps"], row["accuracy_mean"]),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8.5, color=ORANGE, fontweight="bold")

    if bd_rate is not None and bd_acc is not None:
        sign = "−" if bd_rate < 0 else "+"
        bd_text = (
            f"BD-Rate:  {sign}{abs(bd_rate):.2f}%\n"
            f"BD-Acc:  {bd_acc:+.4f} mIoU"
        )
        ax.text(0.97, 0.05, bd_text,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=10, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=GRAY, alpha=0.85))

    ax.set_xscale("log")
    ax.set_xlabel("Bitrate (kbps) — log scale", fontsize=12)
    ax.set_ylabel("mIoU", fontsize=12)
    ax.set_title("Rate–Distortion Curve: SAC single-stream vs Traditional x265\n"
                 "(no offset, single-stream bitrate, evaluated on decoded video)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.35, color=GRAY)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor=GRAY)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"  [plot] RD curve saved → {output_png}")


def plot_per_class_miou(detail_df: pd.DataFrame, class_names: List[str],
                        output_png: Path) -> None:
    import matplotlib.pyplot as plt

    num_classes = len(class_names)
    qp_labels   = detail_df["qp"].tolist()
    n_ops       = len(qp_labels)

    trad_class = [json.loads(r) for r in detail_df["trad_class_iou"]]
    sac_class  = [json.loads(r) for r in detail_df["sac_class_iou"]]

    GRAY   = "#888888"
    CLASS_COLORS_TRAD = ["#1a6faf", "#2196a6", "#1a7a4a", "#6a5acd"]
    CLASS_COLORS_SAC  = ["#e07b39", "#e0a839", "#e05a39", "#c07acc"]

    fig, axes = plt.subplots(1, num_classes, figsize=(4.5 * num_classes, 6), dpi=150,
                             sharey=False)
    fig.patch.set_facecolor("#f8f9fb")

    x = np.arange(n_ops)
    bar_w = 0.35

    for c, (ax, cname) in enumerate(zip(axes, class_names)):
        ax.set_facecolor("#f8f9fb")
        trad_vals = [trad_class[i].get(str(c), trad_class[i].get(c, 0.0)) for i in range(n_ops)]
        sac_vals  = [sac_class[i].get(str(c),  sac_class[i].get(c, 0.0))  for i in range(n_ops)]

        bars_t = ax.bar(x - bar_w / 2, trad_vals, bar_w,
                        label="Traditional", color=CLASS_COLORS_TRAD[c % len(CLASS_COLORS_TRAD)],
                        alpha=0.85, edgecolor="white", linewidth=0.8)
        bars_s = ax.bar(x + bar_w / 2, sac_vals, bar_w,
                        label="SAC", color=CLASS_COLORS_SAC[c % len(CLASS_COLORS_SAC)],
                        alpha=0.85, edgecolor="white", linewidth=0.8)

        for bar in bars_t:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")
        for bar in bars_s:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels([f"QP {q}" for q in qp_labels], fontsize=9)
        ax.set_ylabel("IoU" if c == 0 else "", fontsize=10)
        ax.set_title(f"Class: {cname}", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(0, min(1.05, max(max(trad_vals + sac_vals) + 0.08, 0.2)))
        ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRAY)
        ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=GRAY)
        ax.tick_params(labelsize=9)

    fig.suptitle("Per-class IoU: SAC single-stream vs Traditional x265 (decoded video)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"  [plot] Per-class mIoU bar chart saved → {output_png}")


def plot_bd_summary(bd_rate: float, bd_acc: float, output_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#f8f9fb")

    GREEN = "#27ae60"
    RED   = "#e74c3c"
    GRAY  = "#888888"

    ax1 = axes[0]
    ax1.set_facecolor("#f8f9fb")
    color_rate = GREEN if bd_rate < 0 else RED
    bar1 = ax1.bar(["BD-Rate (%)"], [bd_rate], color=color_rate, alpha=0.85,
                   edgecolor="white", linewidth=1.2, width=0.45)
    ax1.axhline(0, color=GRAY, linewidth=1.2, linestyle="--")
    ax1.set_ylabel("BD-Rate (%)", fontsize=11)
    ax1.set_title("Bitrate Savings\n(negative = SAC saves bitrate)", fontsize=11,
                  fontweight="bold", pad=8)
    for bar in bar1:
        h = bar.get_height()
        sign = "−" if h < 0 else "+"
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 h + (0.2 if h >= 0 else -0.5),
                 f"{sign}{abs(h):.2f}%",
                 ha="center", va="bottom" if h >= 0 else "top",
                 fontsize=13, fontweight="bold", color=color_rate)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color=GRAY)
    ax1.tick_params(labelsize=10)

    ax2 = axes[1]
    ax2.set_facecolor("#f8f9fb")
    color_acc = GREEN if bd_acc > 0 else RED
    bar2 = ax2.bar(["BD-Accuracy (mIoU)"], [bd_acc], color=color_acc, alpha=0.85,
                   edgecolor="white", linewidth=1.2, width=0.45)
    ax2.axhline(0, color=GRAY, linewidth=1.2, linestyle="--")
    ax2.set_ylabel("BD-Accuracy (mIoU)", fontsize=11)
    ax2.set_title("Accuracy Gain\n(positive = SAC gains mIoU)", fontsize=11,
                  fontweight="bold", pad=8)
    for bar in bar2:
        h = bar.get_height()
        sign = "+" if h >= 0 else "−"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 h + (0.0005 if h >= 0 else -0.001),
                 f"{sign}{abs(h):.4f}",
                 ha="center", va="bottom" if h >= 0 else "top",
                 fontsize=13, fontweight="bold", color=color_acc)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color=GRAY)
    ax2.tick_params(labelsize=10)

    fig.suptitle("BD Metrics Summary: SAC single-stream vs Traditional x265",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"  [plot] BD summary bar chart saved → {output_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segmentation RD pipeline – SAC single-stream (no offset)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path tới file YAML config (ví dụ: scripts/config_sac.yaml). "
             "Định nghĩa operating_points với crf_roi, crf_trad, crf_non độc lập. "
             "Nếu dùng đồng thời với --crf-pairs thì --crf-pairs được ưu tiên.",
    )
    parser.add_argument("-n", "--num-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", type=str, default="slow")
    parser.add_argument("--split", type=str, default="val", choices=("val", "test"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=4, choices=(2, 4))
    parser.add_argument("--cities", type=str, default=None)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument(
        "--roi-mode", type=str, default="model", choices=("model", "narrow-gt"),
    )
    parser.add_argument("--crf-pairs", type=str, default=None)
    parser.add_argument(
        "--sac-crf-offset", type=int, default=0,
        help=(
            "Offset cộng thêm vào crf_roi khi encode SAC composite. "
            "Mặc định 0: SAC encode đúng tại crf_roi (= crf_trad - 3), "
            "bù chính xác ~28%% bitrate tiết kiệm từ blur non-ROI (0.72 × 1.12³ ≈ 1.0). "
            "Dùng giá trị âm để tiết kiệm thêm bitrate, dương để tăng chất lượng hơn nữa."
        ),
    )
    parser.add_argument(
        "--non-roi-blur-sigma", type=float, default=3.0,
        help=(
            "Sigma của Gaussian blur áp vào vùng non-ROI trước khi composite. "
            "Mặc định 3.0: làm mịn non-ROI mà không tạo codec artifact "
            "(khác với pre-compress dễ gây double-compression artifact). "
            "Đặt 0 để dùng chế độ pre-compress cũ (legacy)."
        ),
    )
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else MODEL_PATH_DEFAULT
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.crf_pairs:
        # format: "roi:non:trad,roi:non:trad,..." hoặc "roi:non,..." (trad = (roi+non)//2)
        op_points: Sequence[OperatingPoint] = []
        for pair_str in args.crf_pairs.split(","):
            parts = [p.strip() for p in pair_str.strip().split(":")]
            if len(parts) == 3:
                roi_crf, non_crf, trad_crf = int(parts[0]), int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                roi_crf, non_crf = int(parts[0]), int(parts[1])
                trad_crf = (roi_crf + non_crf) // 2
            else:
                raise ValueError(f"--crf-pairs: mỗi cặp phải là 'roi:non' hoặc 'roi:non:trad', nhận được: '{pair_str}'")
            label = str(trad_crf)
            op_points.append(OperatingPoint(label, roi_crf, non_crf, trad_crf))
        print(f"Custom CRF pairs: {[(o.crf_roi, o.crf_non, o.crf_trad) for o in op_points]}")
    elif args.config:
        try:
            import yaml
            with open(args.config, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except ImportError:
            raise RuntimeError("PyYAML chưa cài. Chạy: conda run -n sac pip install pyyaml")
        if "operating_points" not in cfg:
            raise ValueError(f"Config '{args.config}' thiếu mục 'operating_points'")
        op_points = [
            OperatingPoint(
                str(op["label"]),
                int(op["crf_roi"]),
                int(op["crf_non"]),
                int(op["crf_trad"]),
                float(op["non_roi_blur_sigma"]) if "non_roi_blur_sigma" in op else None,
            )
            for op in cfg["operating_points"]
        ]
        print(f"Config loaded: {args.config}")
        for o in op_points:
            sigma_str = f"{o.blur_sigma}" if o.blur_sigma is not None else f"global({args.non_roi_blur_sigma})"
            print(f"  {o.label}: crf_roi={o.crf_roi}  crf_trad={o.crf_trad}  "
                  f"crf_non={o.crf_non}  blur_sigma={sigma_str}")
    elif args.roi_mode == "narrow-gt":
        op_points = NARROW_ROI_POINTS
        print("Operating points: NARROW_ROI_POINTS (optimised for ~10% ROI)")
    else:
        op_points = REQUESTED_POINTS

    if args.split == "val":
        image_root = IMAGE_ROOT_VAL
        label_root = LABEL_ROOT_VAL
    else:
        image_root = IMAGE_ROOT_TEST
        label_root = LABEL_ROOT_TEST

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else OUTPUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_path}")
    model, model_name, num_classes = load_model_auto(model_path, device, args.num_classes)
    class_names = CLASS_NAMES_4 if num_classes == 4 else ["ROI", "non_ROI"]
    print(f"Model: {model_name}  |  num_classes: {num_classes}  |  classes: {class_names}")
    transform = build_transform()

    city_filter = [c.strip() for c in args.cities.split(",")] if args.cities else None
    if city_filter:
        print(f"City filter: {city_filter}")

    pairs = list_eval_pairs(image_root, label_root, args.num_frames, num_classes, city_filter)
    if not pairs:
        raise RuntimeError(f"No Cityscapes {args.split} images found in {image_root}")

    images = [np.array(Image.open(img_path).convert("RGB")) for img_path, _ in pairs]
    gts    = [to_np_label(label_path, num_classes) for _, label_path in pairs]

    valid_gt_indices = [i for i, gt in enumerate(gts) if int(gt.max()) > 0]
    skip_miou = len(valid_gt_indices) == 0
    if skip_miou:
        print("WARNING: Không tìm thấy frame nào có GT label hợp lệ → mIoU bị bỏ qua.")
    else:
        invalid_count = len(gts) - len(valid_gt_indices)
        if invalid_count > 0:
            print(f"  GT info: {len(valid_gt_indices)}/{len(gts)} frames có label hợp lệ "
                  f"({invalid_count} frame toàn zeros bị loại)")

    if args.roi_mode == "narrow-gt":
        print("ROI mode: narrow-gt (vehicles+pedestrians, IDs 24-33)")
        roi_masks = build_roi_masks_narrow_gt(pairs, images)
    else:
        print("ROI mode: model (class 0 = broad catch-all ROI)")
        roi_masks = build_roi_masks(model, device, transform, images)
        ratios = [m.mean() * 100 for m in roi_masks]
        print(f"  Broad ROI ratio: mean={np.mean(ratios):.1f}%  "
              f"min={np.min(ratios):.1f}%  max={np.max(ratios):.1f}%")

    frame_dir = output_dir / "tmp_frames"
    build_orig_frames(images, frame_dir)

    duration_sec = len(images) / float(args.fps)
    baseline_rows: List[Dict] = []
    propose_rows:  List[Dict] = []
    detail_rows:   List[Dict] = []

    for op in op_points:
        combo_name = f"crf_{op.label.replace('/', '_')}"
        combo_dir  = output_dir / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)

        # blur_sigma: ưu tiên per-OP → nếu không có thì dùng global CLI arg
        effective_sigma = op.blur_sigma if op.blur_sigma is not None else args.non_roi_blur_sigma
        crf_trad = op.crf_trad
        sac_video, trad_video = encode_single_stream_sac(
            frame_dir=frame_dir,
            output_dir=combo_dir,
            images=images,
            roi_masks=roi_masks,
            fps=args.fps,
            crf_roi=op.crf_roi,
            crf_non=op.crf_non,
            crf_trad=crf_trad,
            preset=args.preset,
            sac_final_crf_offset=args.sac_crf_offset,
            non_roi_blur_sigma=effective_sigma,
        )

        if skip_miou:
            sac_miou = float("nan")
            trad_miou = float("nan")
            sac_class_iou: Dict[int, float] = {}
            trad_class_iou: Dict[int, float] = {}
        else:
            sac_frames  = read_video_frames(sac_video,  len(images))
            trad_frames = read_video_frames(trad_video, len(images))
            if len(sac_frames) != len(images) or len(trad_frames) != len(images):
                raise RuntimeError(f"Decoded frames missing for operating point {op.label}")

            sac_pred_masks  = predict_masks(model, device, transform, sac_frames)
            trad_pred_masks = predict_masks(model, device, transform, trad_frames)

            valid_gts        = [gts[i]             for i in valid_gt_indices]
            valid_sac_preds  = [sac_pred_masks[i]  for i in valid_gt_indices]
            valid_trad_preds = [trad_pred_masks[i] for i in valid_gt_indices]

            sac_miou,  sac_class_iou  = mean_iou(valid_sac_preds,  valid_gts, num_classes=num_classes)
            trad_miou, trad_class_iou = mean_iou(valid_trad_preds, valid_gts, num_classes=num_classes)

        # bitrate SAC = 1 luồng duy nhất, không offset
        sac_bitrate  = bitrate_mbps(sac_video,  duration_sec)
        trad_bitrate = bitrate_mbps(trad_video, duration_sec)

        baseline_rows.append({
            "qp": op.label,
            "bitrate_kbps": trad_bitrate * 1000.0,
            "accuracy_mean": trad_miou,
        })
        propose_rows.append({
            "qp": op.label,
            "bitrate_kbps": sac_bitrate * 1000.0,
            "accuracy_mean": sac_miou,
        })
        detail_rows.append({
            "qp": op.label,
            "crf_roi": op.crf_roi,
            "crf_non": op.crf_non,
            "crf_trad": crf_trad,
            "trad_bitrate_mbps": trad_bitrate,
            "sac_bitrate_mbps":  sac_bitrate,
            "trad_miou":         trad_miou,
            "sac_miou":          sac_miou,
            "delta_miou":        sac_miou - trad_miou,
            "delta_bitrate_mbps": sac_bitrate - trad_bitrate,
            "trad_class_iou":    json.dumps({str(k): v for k, v in trad_class_iou.items()}, ensure_ascii=False),
            "sac_class_iou":     json.dumps({str(k): v for k, v in sac_class_iou.items()}, ensure_ascii=False),
        })

        if skip_miou:
            print(f"{op.label}: trad {trad_bitrate:.3f} Mbps / mIoU N/A | "
                  f"sac {sac_bitrate:.3f} Mbps / mIoU N/A")
        else:
            print(f"{op.label}: trad {trad_bitrate:.3f} Mbps / mIoU {trad_miou:.4f} | "
                  f"sac {sac_bitrate:.3f} Mbps / mIoU {sac_miou:.4f}  "
                  f"(Δ={sac_miou - trad_miou:+.4f})")
            if num_classes == 4:
                for c, cname in enumerate(class_names):
                    print(f"    {cname:15s}: trad={trad_class_iou.get(c, 0):.4f}  "
                          f"sac={sac_class_iou.get(c, 0):.4f}")

    baseline_df = pd.DataFrame(baseline_rows).sort_values("qp")
    propose_df  = pd.DataFrame(propose_rows).sort_values("qp")
    detail_df   = pd.DataFrame(detail_rows).sort_values("qp")

    baseline_df.to_csv(output_dir / "baseline_real.csv", index=False)
    propose_df.to_csv(output_dir / "propose_real.csv",   index=False)
    detail_df.to_csv(output_dir / "decoded_metrics_detail.csv", index=False)

    bd_rate = bd_acc = overlap = None
    bd_error = None

    if skip_miou:
        bd_error = "mIoU skipped – test labels all zero"
    else:
        try:
            bd_rate, bd_acc, overlap = bd_rate_and_acc(baseline_df, propose_df)
        except RuntimeError as exc:
            bd_error = str(exc)

        plot_rd_curve(baseline_df, propose_df,
                      output_dir / "rd_curve.png",
                      bd_rate=bd_rate, bd_acc=bd_acc)
        plot_per_class_miou(detail_df, class_names,
                            output_dir / "per_class_miou.png")
        if bd_rate is not None and bd_acc is not None:
            plot_bd_summary(bd_rate, bd_acc, output_dir / "bd_summary.png")

    summary = {
        "method": "SAC single-stream (composite: ROI=original, non-ROI=decoded crf_non)",
        "offsets_applied": {"sac_miou_offset": 0.0, "sac_bitrate_reduction": 0.0},
        "sac_final_crf_offset": args.sac_crf_offset,
        "non_roi_blur_sigma": args.non_roi_blur_sigma,
        "model": model_name,
        "model_path": str(model_path),
        "num_classes": num_classes,
        "class_names": class_names,
        "split": args.split,
        "task": "segmentation",
        "metric": "mIoU",
        "qp_labels": [op.label for op in op_points],
        "operating_points": [
            {"qp": op.label, "crf_roi": op.crf_roi, "crf_non": op.crf_non, "crf_trad": op.crf_trad}
            for op in op_points
        ],
        "bitrate_overlap_kbps": overlap,
        "bd_rate_percent": bd_rate,
        "bd_accuracy": bd_acc,
        "bd_error": bd_error,
        "baseline_mean_miou": float(baseline_df["accuracy_mean"].mean()),
        "propose_mean_miou":  float(propose_df["accuracy_mean"].mean()),
        "baseline_mean_bitrate_kbps": float(baseline_df["bitrate_kbps"].mean()),
        "propose_mean_bitrate_kbps":  float(propose_df["bitrate_kbps"].mean()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRD baseline CSV:")
    print(baseline_df.to_string(index=False))
    print("\nRD propose CSV:")
    print(propose_df.to_string(index=False))
    if bd_error:
        print(f"\nBD pipeline could not compute metrics: {bd_error}")
    else:
        print(f"\nBD-Rate:     {bd_rate:+.2f}%")
        print(f"BD-Accuracy: {bd_acc:+.4f}")
    print(f"\nOutput dir: {output_dir}")
    print("Plots: rd_curve.png  |  per_class_miou.png  |  bd_summary.png")

    if not args.keep_artifacts:
        shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
