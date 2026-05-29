"""
Step 4: Reconstruct frames by combining decoded ROI and non-ROI streams.

Usage:
    python combine_streams.py --config ../../config.yaml \
        --roi_video    /data/compressed/crf22_28/roi.mp4 \
        --nonroi_video /data/compressed/crf22_28/nonroi.mp4 \
        --output_dir   /data/reconstructed/crf22_28
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.new.utils.io import load_config, setup_logger, ensure_dirs


def combine_videos(roi_video: str, nonroi_video: str, output_dir: str, logger):
    ensure_dirs(output_dir)
    cap_roi = cv2.VideoCapture(roi_video)
    cap_nonroi = cv2.VideoCapture(nonroi_video)

    frame_count = int(cap_roi.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Combining {frame_count} frames from {roi_video} + {nonroi_video}")

    idx = 0
    pbar = tqdm(total=frame_count, desc="Combining")
    while True:
        ret_r, roi_frame = cap_roi.read()
        ret_n, nonroi_frame = cap_nonroi.read()
        if not ret_r or not ret_n:
            break

        # Saturated add: valid because ROI/nonROI pixels don't overlap
        combined = cv2.add(roi_frame, nonroi_frame)

        out_name = f"{idx:06d}.png"
        cv2.imwrite(str(Path(output_dir) / out_name), combined)
        idx += 1
        pbar.update(1)

    pbar.close()
    cap_roi.release()
    cap_nonroi.release()
    logger.info(f"Reconstructed {idx} frames -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Combine ROI and non-ROI decoded streams")
    parser.add_argument("--config", default="../../config.yaml")
    parser.add_argument("--roi_video", required=True)
    parser.add_argument("--nonroi_video", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("reconstruction")

    combine_videos(args.roi_video, args.nonroi_video, args.output_dir, logger)


if __name__ == "__main__":
    main()
