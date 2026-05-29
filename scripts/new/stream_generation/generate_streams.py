"""
Step 2: Two-stream generation.
Uses segmentation masks to split each frame into ROI and non-ROI streams.
Supports pixel-level and 16×16 block-level masks.

Usage:
    python generate_streams.py --config ../../config.yaml \
        --frames_dir /data/frames \
        --masks_dir  /data/masks \
        --roi_dir    /data/roi \
        --nonroi_dir /data/nonroi
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.new.utils.io import load_config, setup_logger, ensure_dirs, sorted_frame_paths


def build_block_mask(pixel_mask: np.ndarray, block_size: int) -> np.ndarray:
    """
    Coarsen a pixel-level binary mask to block-level.
    A block is ROI if any pixel in it is ROI.
    Returns a binary mask (H, W) with same shape as pixel_mask.
    """
    h, w = pixel_mask.shape
    block_mask = np.zeros_like(pixel_mask, dtype=np.uint8)
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            block = pixel_mask[r:r + block_size, c:c + block_size]
            if np.any(block):
                block_mask[r:r + block_size, c:c + block_size] = 1
    return block_mask


def split_frame(frame_bgr: np.ndarray, label_mask: np.ndarray,
                roi_class_ids: list, block_size: int):
    """
    Split frame into (roi_frame, nonroi_frame) using block-level mask.
    """
    pixel_roi = np.isin(label_mask, roi_class_ids).astype(np.uint8)
    block_roi = build_block_mask(pixel_roi, block_size)

    roi_mask3 = block_roi[..., None]       # (H, W, 1) broadcast
    nonroi_mask3 = (1 - block_roi)[..., None]

    roi_frame = (frame_bgr * roi_mask3).astype(np.uint8)
    nonroi_frame = (frame_bgr * nonroi_mask3).astype(np.uint8)
    return roi_frame, nonroi_frame


def main():
    parser = argparse.ArgumentParser(description="Generate ROI and non-ROI streams")
    parser.add_argument("--config", default="../../config.yaml")
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--roi_dir", required=True)
    parser.add_argument("--nonroi_dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("stream_generation")
    ensure_dirs(args.roi_dir, args.nonroi_dir)

    roi_class_ids = cfg["segmentation"]["roi_class_ids"]
    block_size = cfg["segmentation"]["block_size"]

    frames = sorted_frame_paths(args.frames_dir)
    logger.info(f"Processing {len(frames)} frames | block_size={block_size} "
                f"| roi_classes={roi_class_ids}")

    for frame_path in tqdm(frames, desc="Splitting streams"):
        mask_path = Path(args.masks_dir) / frame_path.name
        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}, skipping.")
            continue

        frame_bgr = cv2.imread(str(frame_path))
        label_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame_bgr is None or label_mask is None:
            logger.warning(f"Cannot read {frame_path.name}, skipping.")
            continue

        roi_frame, nonroi_frame = split_frame(frame_bgr, label_mask,
                                              roi_class_ids, block_size)

        cv2.imwrite(str(Path(args.roi_dir) / frame_path.name), roi_frame)
        cv2.imwrite(str(Path(args.nonroi_dir) / frame_path.name), nonroi_frame)

    logger.info(f"ROI frames    -> {args.roi_dir}")
    logger.info(f"Non-ROI frames-> {args.nonroi_dir}")


if __name__ == "__main__":
    main()
