"""
Compression utilities for the SAC pipeline.

Supports four methods from the paper (Table I):
  H.264    — libx264, uniform CRF
  H.265    — libx265, uniform CRF
  SA-X264  — libx264, semantic-aware (ROI at low CRF, non-ROI at high CRF)
  SA-X265  — libx265, semantic-aware

All codec calls disable adaptive quantisation (aq-mode=0) so the codec uses
the CRF value uniformly — region-of-interest quality is controlled entirely
by stream splitting, not by codec-level AQ.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_ffmpeg(cmd: List[str], step: str) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed at '{step}':\n{result.stderr[-3000:]}")


def macroblock_align_filter(mask: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Snap ROI mask boundaries to the codec macroblock grid (16×16 for H.264/H.265).

    Any 16×16 block that contains at least one ROI pixel is promoted to a full
    ROI block.  This prevents quality bleeding across block boundaries.

    Parameters
    ----------
    mask       : uint8 [H, W], values 0 or 1
    block_size : must be 16 for H.264/H.265

    Returns
    -------
    aligned mask : uint8 [H, W], values 0 or 1
    """
    h, w = mask.shape
    ph = (h + block_size - 1) // block_size * block_size
    pw = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((ph, pw), dtype=np.uint8)
    padded[:h, :w] = mask
    # reshape so axis (1,3) are the within-block dims
    blocks = padded.reshape(ph // block_size, block_size, pw // block_size, block_size)
    roi_max = blocks.max(axis=(1, 3))                          # 1 if any pixel is ROI
    aligned = np.repeat(np.repeat(roi_max, block_size, axis=0), block_size, axis=1)
    return aligned[:h, :w]


# ─────────────────────────────────────────────────────────────────────────────
# Encode / decode helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_frames(frames_bgr: List[np.ndarray], tmp_dir: str) -> None:
    for i, f in enumerate(frames_bgr):
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i:04d}.png"), f)


def _encode(
    tmp_dir: str,
    out_path: str,
    codec: str,
    crf: int,
    fps: int,
    preset: str,
) -> int:
    """Encode PNG sequence in tmp_dir → out_path.  Returns file size (bytes)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # disable adaptive quant so CRF is applied uniformly
    if codec == "libx264":
        extra = ["-x264-params", "aq-mode=0"]
    elif codec == "libx265":
        extra = ["-x265-params", "aq-mode=0"]
    else:
        extra = []

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%04d.png"),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
    ] + extra + [out_path]

    _run_ffmpeg(cmd, f"encode {codec} crf={crf}")
    return os.path.getsize(out_path)


def _decode(video_path: str) -> List[np.ndarray]:
    """Decode video → list of BGR uint8 frames."""
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compress_traditional(
    frames_bgr: List[np.ndarray],
    codec: str,
    crf: int,
    work_dir: str,
    tag: str = "trad",
    fps: int = 1,
    preset: str = "medium",
) -> Tuple[List[np.ndarray], int]:
    """
    Traditional (uniform) compression.

    Parameters
    ----------
    frames_bgr : list of BGR uint8 frames
    codec      : 'libx264' or 'libx265'
    crf        : constant rate factor  (lower = higher quality)
    work_dir   : directory for intermediate files
    tag        : filename prefix

    Returns
    -------
    (decoded_frames, total_bytes)
    """
    tmp = tempfile.mkdtemp(dir=work_dir)
    out = os.path.join(work_dir, f"{tag}_{codec.replace('lib','')}_{crf}.mp4")
    try:
        _write_frames(frames_bgr, tmp)
        size = _encode(tmp, out, codec, crf, fps, preset)
        return _decode(out), size
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compress_sac(
    frames_bgr: List[np.ndarray],
    roi_masks: List[np.ndarray],
    codec: str,
    crf_roi: int,
    crf_non: int,
    work_dir: str,
    tag: str = "sac",
    fps: int = 1,
    preset: str = "medium",
) -> Tuple[List[np.ndarray], int, int]:
    """
    Semantic-Aware Compression (S2 + S3 in paper).

    Steps
    -----
    1. Split each frame into ROI stream and non-ROI stream using mask.
    2. Encode ROI stream at low CRF (high quality).
    3. Encode non-ROI stream at high CRF (low quality).
    4. Decode both streams.
    5. Merge via addition (blend=all_mode=addition equivalent, works because
       the two streams have non-overlapping non-zero regions).

    Parameters
    ----------
    frames_bgr : list of BGR uint8 frames
    roi_masks  : list of uint8 [H,W] masks, 1=ROI 0=non-ROI
                 (should be macroblock-aligned before calling this function)
    codec      : 'libx264' or 'libx265'
    crf_roi    : low CRF for ROI (higher quality)
    crf_non    : high CRF for non-ROI (lower quality)

    Returns
    -------
    (merged_frames_bgr, roi_size_bytes, non_size_bytes)
    """
    assert len(frames_bgr) == len(roi_masks), "frames and masks must have equal length"

    tmp_roi = tempfile.mkdtemp(dir=work_dir)
    tmp_non = tempfile.mkdtemp(dir=work_dir)
    out_roi = os.path.join(work_dir, f"{tag}_{codec.replace('lib','')}_{crf_roi}_{crf_non}_roi.mp4")
    out_non = os.path.join(work_dir, f"{tag}_{codec.replace('lib','')}_{crf_roi}_{crf_non}_non.mp4")

    try:
        roi_frames_bgr: List[np.ndarray] = []
        non_frames_bgr: List[np.ndarray] = []

        for frame, mask in zip(frames_bgr, roi_masks):
            m255 = (mask * 255).astype(np.uint8)
            roi_frames_bgr.append(cv2.bitwise_and(frame, frame, mask=m255))
            non_frames_bgr.append(cv2.bitwise_and(frame, frame, mask=255 - m255))

        _write_frames(roi_frames_bgr, tmp_roi)
        _write_frames(non_frames_bgr, tmp_non)

        roi_size = _encode(tmp_roi, out_roi, codec, crf_roi, fps, preset)
        non_size = _encode(tmp_non, out_non, codec, crf_non, fps, preset)

        dec_roi = _decode(out_roi)
        dec_non = _decode(out_non)

        # Merge (addition = correct because regions are disjoint)
        merged = [cv2.add(r, n) for r, n in zip(dec_roi, dec_non)]

        return merged, roi_size, non_size

    finally:
        shutil.rmtree(tmp_roi, ignore_errors=True)
        shutil.rmtree(tmp_non, ignore_errors=True)
